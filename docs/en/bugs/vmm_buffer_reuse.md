# VMM buffer reuse across weight-update iterations

## Context

`update_weights_from_tensor` (the colocated path) sends Megatron's HF-converted
weights to SGLang via a VMM-backed shared buffer plus an SCM_RIGHTS fd transport
over UDS. See [_send_to_colocated_engine](../../../slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py)
and [vmm_ipc.py](../../../../sglang/python/sglang/srt/weight_sync/vmm_ipc.py).

Today, the producer allocates and frees one VMM buffer **per chunk per iteration**:

- `alloc_vmm_buffer(total_bytes, device)` in `_send_to_colocated_engine`
  (one `cuMemCreate(POSIX_FD)` + `cuMemExportToShareableHandle`).
- A fresh UUID-named UDS path is bound, fd is sent once, listener is closed.
- After `ray.get(refs)` and the `_ipc_gather_group` barrier,
  `_cleanup_vmm_resources` calls `free_vmm_buffer` (`cuMemUnmap` →
  `cuMemAddressFree` → `cuMemRelease` → `os.close(fd)`).

The consumer mirrors this: `import_vmm_buffer` → `wrap_as_torch_uint8` →
`load_weights` → `cuda.synchronize` → `free_vmm_buffer` (model_runner.py:1492).

## Why reuse it

- Hot path becomes a single `torch.cat` into a pre-mapped buffer plus one Ray
  RPC per chunk. Drops per-chunk `cuMemCreate` / `cuMemMap` /
  `cuMemExportToShareableHandle` / `socket(AF_UNIX)` / `bind` / `accept` /
  `cuMemImportFromShareableHandle` / `cuMemUnmap` / `cuMemRelease`.
- Fewer master_daemon keys held by GCR → faster `Cmd::Ckpt` / `Cmd::Restore`
  during suspend/resume cycles (cost is O(N keys)).

## What needs to be made persistent

1. **Producer VMM buffer** — currently per-chunk at `update_weight_from_tensor.py:250`.
2. **UDS handshake** — currently a fresh `uuid` socket bound + accepted per
   chunk at `update_weight_from_tensor.py:257-267`.
3. **Consumer import** — currently `import_vmm_buffer` + `free_vmm_buffer` per
   chunk at `model_runner.py:1516,1538`.

## Lifecycle plan

### One-time, in `UpdateWeightFromTensor.connect_rollout_engines`

After the IPC gather group is established:

1. Compute `max_chunk_bytes` per rank. Walk `self._hf_weight_iterator
   .get_hf_weight_chunks(...)` shape metadata (no tensor copies) or derive
   from a model-level upper bound. Take the per-rank max because chunk sizes
   are rank-local.
2. `self._vmm_alloc = alloc_vmm_buffer(max_chunk_bytes, device)` — one
   persistent allocation per train rank.
3. `self._vmm_buf = wrap_as_torch_uint8(self._vmm_alloc)`.
4. Bind a UDS listener at a stable path (derive from `device_uuid` + engine id;
   no per-chunk uuid). Keep the listener alive for the actor's lifetime.
5. Add `attach_vmm_buffer(uds_paths, buffer_size)` to SGLang's
   `update_weights_*` request set. Rank 0 calls it once per colocated engine
   in `connect_rollout_engines`. The handler opens a sidecar client, receives
   fd, calls `import_vmm_buffer`, stashes `(alloc, buf)` on the worker, and
   returns. **Does not free.**

### Per chunk, in `_send_to_colocated_engine`

Replace the alloc / UDS / cleanup with:

```python
torch.cat(flat_parts, out=self._vmm_buf[:total_bytes])
torch.cuda.synchronize(device)
dist.gather_object(metadata_only_dict, ..., group=ipc_gather_group)
if rank == ipc_gather_src:
    refs.append(ipc_engine.update_weights_from_tensor_vmm.remote(
        tensor_metadata=metadata,
        used_bytes=total_bytes,
        weight_version=str(weight_version),
    ))
```

Drop `_cleanup_vmm_resources`. The persistent buffer is owned by the
`UpdateWeightFromTensor` instance.

### Per chunk, on SGLang's `update_weights_from_tensor_vmm`

Replace per-call import/free with a lookup of the cached `(alloc, buf)`:

```python
alloc, buf = self._persistent_vmm  # set by attach_vmm_buffer
named_tensors = [...]  # slice buf[start_idx:end_idx] per metadata entry
self.model.load_weights(named_tensors)
torch.cuda.synchronize(device)
del named_tensors
# do not free
```

### Teardown

On engine restart / fault tolerance, call `free_vmm_buffer` on both sides
before re-running `attach_vmm_buffer`.

## Synchronization

The existing `dist.barrier(group=self._ipc_gather_group)` after `ray.get(refs)`
([update_weight_from_tensor.py:175](../../../slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py))
already provides the producer↔consumer fence. Rank 0's `ray.get(refs)` only
returns after the engine RPC's post-`load_weights` `cuda.synchronize`, so by
the time the barrier completes, every TP rank has finished consuming chunk N
and the producer is safe to overwrite for chunk N+1.

## Suspend / resume interaction

The persistent VMM allocation must survive `gcr_suspend` / `gcr_resume`.
Sanity check: log the producer's `va` and the master key (from GCR trace) on
both sides of a cycle. They should be identical, and the consumer's import
handle should still be valid post-restore.

## Sizing

`max_chunk_bytes` is bounded by Megatron's HF chunking policy (see
`HfWeightIteratorBase`). Worst case: largest individual layer's HF-converted
parameter count × max bytes-per-param across dtypes.

For a 0.5B model the buffer is ~hundreds of MB; for production 70B+ models it
will be in the multi-GB range. If memory pressure is a concern, the buffer can
be sized to the largest single-chunk payload (not the full param count) — chunks
are bounded above by Megatron's `_hf_weight_iterator` chunking strategy.
