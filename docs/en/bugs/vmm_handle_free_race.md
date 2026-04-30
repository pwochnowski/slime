# Bug: VMM handle freed before SGLang TP1 finishes importing (no_matching_pinned_fd)

## Symptom

During `update_weights_from_tensor_vmm`, the GCR master daemon emits repeated:

```
Cmd::Claim from_pid=<sglang-tp1> fd=<N> no_matching_pinned_fd
```

followed by `claim_failed` after several seconds of retries.  Weight update
either silently corrupts or raises an exception, depending on how GCR surfaces
the failure.

## Observed timeline (gcr_logs/master_daemon-1053954.log)

| t (s) | pid | event |
|-------|-----|-------|
| 141.701 | 1055553 (Megatron rank 1) | `Cmd::Alloc` key=573, 528 MB, daemon fd=724 |
| 141.719 | 1055553 | `Cmd::Export` key=573 → pinned_fd=727 |
| **143.665** | 1055553 | `Cmd::Free` key=573, refcount=0 → daemon drops pinned_fd=727 |
| **144.108** | 1053303 (SGLang TP1) | `Cmd::Claim fd=724` → `no_matching_pinned_fd` (retries ~2000/s for several seconds) |

The free arrives ~0.4 s before the claim.

## Root cause

### SGLang TP architecture

Each TP rank runs its own `Scheduler` process.  The HTTP request arrives at TP0,
which uses `broadcast_pyobj` to fan it out to TP1.  Both schedulers independently
connect to their respective Megatron rank's UDS socket, call
`cuMemImportFromShareableHandle` (via `import_vmm_buffer`), load weights, then
rendezvous on a `tp_cpu_group` Gloo barrier **before** TP0 sends the HTTP 200.

### Megatron side: src vs non-src ranks

In `_send_to_colocated_engine` only the `ipc_gather_src` rank (rank 0) submits
the Ray RPC:

```python
# slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
if dist.get_rank() == ipc_gather_src:
    refs.append(ipc_engine.update_weights_from_tensor_vmm.remote(...))
# non-src ranks → refs = []
return refs, (fd_thread, alloc, buf, uds_path)
```

Back in `update_weights` (line 167):

```python
refs, vmm_resources = self._send_hf_params(hf_named_tensors)
ray.get(refs)                          # rank 0: waits for HTTP ack
                                       # rank 1: refs=[], returns instantly
_cleanup_vmm_resources(vmm_resources)  # rank 1: runs IMMEDIATELY
```

`_cleanup_vmm_resources` (line 286):

```python
fd_thread.join(timeout=30)       # waits only for send_fd() over UDS to finish
del buf
free_vmm_buffer(alloc, close_fd=True)  # ← drops pinned_fd from GCR daemon
```

`fd_thread.join()` unblocks the moment the UDS server completes `send_fd()`.
That is **before** the consumer calls `cuMemImportFromShareableHandle`, which is
where GCR issues `Cmd::Claim`.  By then the daemon has already processed
`Cmd::Free` and dropped the pinned fd.

### Why rank 0 is unaffected

Rank 0's `ray.get(refs)` blocks until the SGLang HTTP response, which is only
sent after the `tp_cpu_group` barrier in
`scheduler_update_weights_mixin.py:122`.  That barrier requires **both** TP0 and
TP1 to finish importing, so rank 0 never frees early.  Rank 1 has no equivalent
wait.

### Missing synchronization

There is no barrier, ACK, or shared event between:
- rank 1's `free_vmm_buffer()`, and
- SGLang TP1's `cuMemImportFromShareableHandle` completing.

## Fix

Add a Gloo barrier on `_ipc_gather_group` between `ray.get(refs)` and
`_cleanup_vmm_resources`.  This forces all non-src ranks to wait until rank 0's
`ray.get` confirms the engine acknowledged the import.

```python
# slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
# update_weights(), around line 167

for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, vmm_resources = self._send_hf_params(hf_named_tensors)
    ray.get(refs)
    # rank 0: ray.get() returned only after SGLang's tp_cpu_group barrier,
    # meaning all TP workers finished cuMemImportFromShareableHandle.
    # rank 1+: refs=[], so ray.get() returned instantly — hold them here
    # until rank 0 confirms import is done.
    if self._ipc_gather_group is not None:
        dist.barrier(group=self._ipc_gather_group)
    _cleanup_vmm_resources(vmm_resources)
    del hf_named_tensors
```

The `ipc_gather_group` is the same Gloo group used for `gather_object`, so the
barrier is already a valid collective on exactly the colocated Megatron ranks.
Cost is a few microseconds per weight-update bucket.

## Relevant files

| File | Role |
|------|------|
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | exporter: alloc / export / free |
| `python/sglang/srt/model_executor/model_runner.py` | importer: `import_vmm_buffer` / `load_weights` |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | SGLang-side dispatch + `tp_cpu_group` barrier |
| `python/sglang/srt/managers/scheduler.py:1309` | `broadcast_pyobj` fans request to all TP ranks |
| `gcr_logs/master_daemon-1053954.log` | daemon log showing the exact free/claim sequence |
