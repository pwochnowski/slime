# SLIME VMM IPC Integration

Replace legacy CUDA IPC weight transfer (`MultiprocessingSerializer` / `cudaIpcGetMemHandle`) with VMM IPC (`cuMemCreate` + fd transport over UDS).

## Background

SGLang (on a separate branch) now has a complete VMM IPC weight update endpoint. The VMM path replaces `cudaIpcGetMemHandle` (which only works on `cudaMalloc` memory) with `cuMemCreate` + `cuMemExportToShareableHandle`, transporting the fd over a Unix domain socket. The consumer maps the same physical GPU memory via `cuMemImportFromShareableHandle`. See [vmm_ipc_research.md](vmm_ipc_research.md) for the full technical background.

### SGLang VMM endpoint (already implemented)

| Layer | SGLang location | API |
|-------|----------------|-----|
| VMM primitives | `sglang.srt.weight_sync.vmm_ipc` | `alloc_vmm_buffer`, `import_vmm_buffer`, `wrap_as_torch_uint8`, `free_vmm_buffer`, `send_fd`/`recv_fd`, `open_sidecar_listener` |
| Request dataclass | `sglang.srt.managers.io_struct` | `UpdateWeightsFromTensorVMMReqInput(uds_paths, buffer_sizes, tensor_metadata, flush_cache)` |
| HTTP endpoint | `POST /update_weights_from_tensor_vmm` | Accepts the dataclass fields as JSON |
| Consumer logic | `model_runner.update_weights_from_tensor_vmm()` | recv_fd -> import -> reconstruct tensors -> `model.load_weights()` -> free |

## Current slime weight transfer flow (legacy)

```
HfWeightIterator.get_hf_weight_chunks()    # yields list[tuple[name, tensor]] per bucket
  -> UpdateWeightFromTensor._send_hf_params()
    -> _send_to_colocated_engine()           # the function we're replacing
      1. FlattenedTensorBucket(named_tensors)       # flatten to contiguous uint8
      2. MultiprocessingSerializer.serialize(...)    # ForkingPickler -> cudaIpcGetMemHandle -> base64
      3. dist.gather_object(serialized, ...)         # Gloo gather to src rank
      4. ipc_engine.update_weights_from_tensor.remote(serialized_named_tensors=...)
         -> SGLangEngine.update_weights_from_tensor()  # HTTP POST to SGLang
         -> SGLang: MultiprocessingSerializer.deserialize -> cudaIpcOpenMemHandle -> load_weights
    -> update_weights_from_distributed()     # NCCL path for non-colocated engines (unchanged)
```

Steps 2-4 are the legacy CUDA IPC path that breaks with VMM-allocated memory.

## Files that change in slime

### 1. `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

**Primary change.** Replace `_send_to_colocated_engine()` (lines 209-267) with a VMM-based equivalent.

Current function:
- Creates `FlattenedTensorBucket` from named tensors
- Serializes via `MultiprocessingSerializer.serialize()` (triggers `cudaIpcGetMemHandle`)
- Gathers serialized base64 strings via `dist.gather_object()` to src rank
- Src rank calls `ipc_engine.update_weights_from_tensor.remote()`

New function (`_send_to_colocated_engine_vmm`):
- Flatten tensors to uint8 manually (no `FlattenedTensorBucket` needed)
- Allocate VMM buffer via `alloc_vmm_buffer()`, copy flattened data in
- Start UDS listener thread to serve the fd
- Gather lightweight metadata (uds_path, device_uuid, buffer_size) via `dist.gather_object()` -- same Gloo group, but gathering small dicts instead of large base64 strings
- Src rank calls `ipc_engine.update_weights_from_tensor_vmm.remote()`
- After completion: join thread, free VMM buffer, unlink socket

Also in `update_weights()` (line 138):
- Remove `torch.cuda.ipc_collect()` calls (lines 165, 170) -- these clean up `cudaIpcGetMemHandle` cache entries, which VMM doesn't use

### 2. `slime/backends/sglang_utils/sglang_engine.py`

**Add HTTP wrapper.** Add `update_weights_from_tensor_vmm()` method to `SGLangEngine` class, mirroring the existing `update_weights_from_tensor()` (line 266).

The new method posts to `/update_weights_from_tensor_vmm` with the `UpdateWeightsFromTensorVMMReqInput` fields as JSON payload. The `uds_paths` dict maps GPU device UUIDs to UDS socket paths, `buffer_sizes` maps UUIDs to allocation sizes, and `tensor_metadata` is a list of `{name, shape, dtype, start_idx, end_idx}` dicts.

### 3. `slime/backends/megatron_utils/sglang.py`

**Remove legacy imports.** This module centralizes SGLang imports for the megatron backend. Changes:
- `MultiprocessingSerializer`: no longer needed for tensor transfer. Check if anything else in slime uses it -- if not, remove entirely.
- `FlattenedTensorBucket`: no longer needed if we flatten manually. Can be removed from imports.
- `monkey_patch_torch_reductions`: only needed to fix device ordinal mismatches in `reduce_tensor`/`rebuild_cuda_tensor` (the `ForkingPickler` path). VMM uses device UUIDs natively. Can be removed.

### 4. `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`

**Remove monkey patch call.** Line 48 calls `monkey_patch_torch_reductions()` inside `_get_megatron_full_params()`. This patches torch's IPC serialization to use device UUIDs instead of ordinals. With VMM, IPC serialization is no longer used. Remove this call and the import (line 14).

### 5. `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py`

**Update comment.** Line 32 says `# Move merged result back to GPU for CUDA IPC serialization`. The operation itself (moving to GPU) is still needed -- the VMM path also needs GPU tensors to copy into the VMM buffer -- but the comment should say "for VMM buffer copy" instead.

## What does NOT change

- **HfWeightIterator / conversion pipeline**: Everything up to and including HF-named tensor generation is unchanged. The VMM path consumes `list[tuple[str, torch.Tensor]]` just like the legacy path.
- **Distributed (NCCL) path**: `update_weights_from_distributed()` uses NCCL broadcast, not CUDA IPC. Completely unaffected.
- **Gloo gather groups**: The `_ipc_gather_group` / `_ipc_gather_src` infrastructure in `UpdateWeightFromTensor.__init__` and `connect_rollout_engines()` is reused. The only change is what gets gathered (small metadata dicts vs large base64 strings).
- **Bucket iteration**: `update_weights()` iterates buckets via `_hf_weight_iterator.get_hf_weight_chunks()`. Each bucket call allocates a fresh VMM buffer, transfers, and frees. No persistent state between buckets.
- **`model.load_weights()`**: The final `copy_()` into model parameters on the SGLang side is identical regardless of IPC mechanism.
- **`pause_generation` / `continue_generation` / `flush_cache`**: Lifecycle management around weight updates is unchanged.

## New VMM flow (detailed)

```python
def _send_to_colocated_engine_vmm(
    hf_named_tensors,  # list[tuple[str, torch.Tensor]] -- already on GPU, HF format
    *,
    ipc_engine,        # Ray actor handle for SGLangEngine
    ipc_gather_src,    # rank that drives the engine call
    ipc_gather_group,  # Gloo group for metadata gather
    weight_version,
):
    if ipc_gather_group is None:
        return [], None

    device = torch.cuda.current_device()

    # 1. Flatten to uint8 + build metadata
    metadata, flat_parts, total_bytes = [], [], 0
    for name, tensor in hf_named_tensors:
        flat = tensor.flatten().view(torch.uint8)
        metadata.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "start_idx": total_bytes,
            "end_idx": total_bytes + flat.numel(),
        })
        flat_parts.append(flat)
        total_bytes += flat.numel()

    # 2. Allocate VMM buffer, copy data
    alloc = alloc_vmm_buffer(total_bytes, device)
    buf = wrap_as_torch_uint8(alloc)
    torch.cat(flat_parts, out=buf[:total_bytes])
    torch.cuda.synchronize(device)

    # 3. UDS listener (one connection, serves the fd to SGLang's consumer)
    uds_path = f"/tmp/slime-vmm-{uuid.uuid4()}.sock"
    listener = open_sidecar_listener(uds_path)
    def _serve():
        conn, _ = listener.accept()
        send_fd(conn, alloc.fd)
        conn.close()
        listener.close()
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # 4. Gather per-rank info to src rank (small dict, not serialized tensors)
    device_props = torch.cuda.get_device_properties(device)
    my_info = {
        "uds_path": uds_path,
        "device_uuid": f"GPU-{device_props.uuid!s}",
        "buffer_size": alloc.size,
    }
    gather_list = (
        [None] * dist.get_world_size(ipc_gather_group)
        if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(my_info, object_gather_list=gather_list,
                       dst=ipc_gather_src, group=ipc_gather_group)

    # 5. Src rank calls VMM endpoint on engine
    refs = []
    if dist.get_rank() == ipc_gather_src:
        refs.append(ipc_engine.update_weights_from_tensor_vmm.remote(
            uds_paths={g["device_uuid"]: g["uds_path"] for g in gather_list},
            buffer_sizes={g["device_uuid"]: g["buffer_size"] for g in gather_list},
            tensor_metadata=metadata,
            weight_version=str(weight_version),
        ))

    # 6. Cleanup after engine returns
    # (caller does ray.get(refs), then we clean up)
    # The alloc and thread must stay alive until the consumer has recv'd the fd,
    # so cleanup is deferred to the caller or done after ray.get.
    return refs, (thread, alloc, buf, uds_path)
```

Cleanup in the caller (`_send_hf_params` or `update_weights`) after `ray.get(refs)`:
```python
thread.join(timeout=30)
del buf
free_vmm_buffer(alloc, close_fd=True)
try:
    os.unlink(uds_path)
except FileNotFoundError:
    pass
```

## SGLangEngine HTTP wrapper

```python
# In slime/backends/sglang_utils/sglang_engine.py

def update_weights_from_tensor_vmm(
    self,
    uds_paths: dict[str, str],
    buffer_sizes: dict[str, int],
    tensor_metadata: list[dict],
    flush_cache: bool = False,
    weight_version: str | None = None,
):
    payload = {
        "uds_paths": uds_paths,
        "buffer_sizes": buffer_sizes,
        "tensor_metadata": tensor_metadata,
        "flush_cache": flush_cache,
    }
    if weight_version is not None:
        payload["weight_version"] = weight_version
    return self._make_request("update_weights_from_tensor_vmm", payload)
```

## Legacy IPC usage inventory (within slime)

| File | What uses legacy IPC | Action |
|------|---------------------|--------|
| `update_weight_from_tensor.py:243` | `MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)` | Replace with VMM buffer + fd |
| `update_weight_from_tensor.py:165,170` | `torch.cuda.ipc_collect()` | Remove (VMM has no IPC cache) |
| `update_weight_from_tensor.py:236` | `FlattenedTensorBucket(named_tensors=named_tensors)` | Replace with manual uint8 flattening |
| `sglang.py:16` | `from sglang.srt.utils import MultiprocessingSerializer` | Remove import |
| `sglang.py:20-22` | `from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket` | Remove import |
| `sglang.py:11-13` | `from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions` | Remove import |
| `hf_weight_iterator_direct.py:14` | `from ..sglang import monkey_patch_torch_reductions` | Remove import |
| `hf_weight_iterator_direct.py:48` | `monkey_patch_torch_reductions()` | Remove call |
| `hf_weight_iterator_bridge.py:32` | Comment referencing "CUDA IPC serialization" | Update comment |
| `sglang_engine.py:266-289` | `update_weights_from_tensor()` HTTP wrapper | Keep (backward compat), add VMM sibling |
