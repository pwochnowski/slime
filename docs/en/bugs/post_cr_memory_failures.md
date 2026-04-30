# Post-C/R memory access failures in slime/sglang colocate workload

## TL;DR

After a GCR suspend/resume cycle, two distinct CUDA memory-access failures have
been observed on the same workload, both in code paths that rely on driver-level
out-of-band mappings (CUDA VMM and `cudaHostRegister`-pinned host memory).
Forcing host-side synchronization changes which failure surfaces but does not
eliminate the underlying issue. GCR's master daemon log explicitly reports that
`cudaHostRegister` is failing post-init.

## Workload

- **Repo/test**: [tests/test_qwen2.5_0.5B_gsm8k_short.py](../../../tests/test_qwen2.5_0.5B_gsm8k_short.py)
- **Model**: Qwen2.5-0.5B-Instruct, TP=2, 2 GPUs
- **Mode**: slime + sglang **colocate** mode, where sglang and Megatron train
  share GPUs and use GCR (`/root/GCR/GCR/libpreload.so` + `libcuda.so`) for
  periodic suspend/resume to swap which engine has live GPU state
- **NCCL**: 2.27.5, with `NCCL_CUMEM_ENABLE=1`
- **PyTorch allocator**: `PYTORCH_ALLOC_CONF=expandable_segments:False` (so
  PyTorch uses plain `cudaMalloc`, not VMM via expandable segments)

## Observed failures

### Failure A — NCCL surfaces sticky 719 after first post-C/R `update_weights`

**Run**: `logs/1777463001/output.log` (no canaries, no extra debug flags)

- **Wall-clock**: 11:35:00 suspend → 11:35:24 resume → 11:36:18–11:37:28
  `actor_train` (succeeded) → 11:38:31 first post-C/R `update_weights` starts →
  11:38:32 crash.
- **Last successful CUDA op**: `vmm_ipc.py:152 - VMM alloc: rounded 528303616 -> 528482304`
  (line 333)
- **Traceback** (line 380-400):
  ```
  megatron/bridge/.../param_mapping.py:521 in gather_from_tp_ranks
    torch.distributed.all_gather(gathered, tensor, group=self.tp_group)
  torch.distributed.DistBackendError: NCCL error ... unhandled cuda error
  Cuda failure 719 'unspecified launch failure'
  ```
- **Notes**:
  - `actor_train` (also NCCL-heavy) ran cleanly post-C/R, so the TP NCCL
    communicator itself is intact.
  - GCR `mapped`/`handles` counts logged identically pre- and post-C/R (501
    allocations both times) per the `[gpu-mem]` lines.
  - 719 is sticky and surfaces on the next CUDA op — actual offending kernel is
    not directly identified by the traceback. Suspect kernel is the
    `torch.cat(flat_parts, out=buf[:total_bytes])` at
    [update_weight_from_tensor.py](../../../slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py),
    which writes into a freshly VMM-mapped buffer, but this is **inferred not
    proven**.

### Failure B — `cudaErrorIllegalAddress` on pinned-CPU → GPU async copy

**Run**: `logs/1777516790/output.log` (canaries enabled; exact worker-side env
vars in this run are unknown — see note below)

- **Wall-clock**: iter 0 `update_weights` succeeds (canaries fire cleanly
  through chunk loop and IPC). Crash occurs in **iter 1**, after iter 0's C/R
  cycle, during `_switch_model("actor")`.
- **Traceback** (line 604-619):
  ```
  slime/backends/megatron_utils/actor.py:225 in _switch_model
    self.weights_backuper.restore(target_tag)
  slime/utils/tensor_backper.py:73 in restore
    param.copy_(backup_dict[name], non_blocking=True)
  torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  ```
- **What that copy is**: `param` is a GPU model parameter; `backup_dict[name]`
  is allocated as `torch.empty_like(param, device="cpu", pin_memory=True)`
  ([tensor_backper.py:59](../../../slime/utils/tensor_backper.py#L59)). So
  it's an **async DMA from page-locked host memory to GPU**.

### Run that did **not** crash

**Run**: `logs/1777515570/output.log` (canaries enabled; exact worker-side env
vars in this run are unknown)

- Job succeeded end-to-end. Failure A did not surface; Failure B did not occur.
- This run alone does not prove the canaries fix anything (could be flaky). We
  have not re-run the original failing config (no canaries) to confirm Failure
  A is deterministic.

## GCR-side log signal worth investigating

In **both** Failure A and Failure B logs (also visible in the canaries-pass
log), GCR's master daemon prints at init:

```
master_daemon ... ensure_ckpt_init cudaHostRegister(shm_fs, 42949672960) failed: invalid argument
  — async D2H/H2D will be sync
```

This is GCR itself reporting that its attempt to `cudaHostRegister` a 40 GiB
shared-memory region failed. The fallback ("async D2H/H2D will be sync")
implies GCR has a pinned-host fast path that's silently degraded — and this
directly relates to Failure B's pinned-CPU→GPU copy path.

## Possible root causes (to investigate in GCR)

In rough order of "most likely given the evidence":

1. **GCR may not correctly track or restore VMM driver allocations.** The
   slime VMM IPC path uses `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap`
   / `cuMemSetAccess` / `cuMemExportToShareableHandle`
   ([sglang/.../vmm_ipc.py](../../../../sglang/python/sglang/srt/weight_sync/vmm_ipc.py)).
   On the importer side, `cuMemImportFromShareableHandle` is used. If GCR's
   interposer hooks `cudaMalloc`/`cudaFree` but not these `cuMem*` driver
   entry points, mappings may dangle after restore. An async kernel writing
   into a dangling VMM mapping would produce 700/719. A prior memory note in
   this project (`project_gcr_memory_doubling`) already hypothesized GCR
   doesn't track expandable-segments / `cuMemCreate`.

2. **GCR may not be re-registering pinned host memory after restore.** The
   `ensure_ckpt_init cudaHostRegister(...) failed: invalid argument` line is
   direct evidence that GCR's own pinned region didn't register. If
   `tensor_backper`'s pinned tensors survive a C/R cycle in name only — i.e.,
   the page-lock is gone but PyTorch still treats them as pinned and issues a
   `non_blocking=True` async DMA — the DMA engine may dereference an
   unregistered host page and raise `cudaErrorIllegalAddress`. This matches
   Failure B exactly.

3. **GCR may successfully track these allocations but its restore logic has a
   bug remapping them.** The `[gpu-mem]` lines in run `1777463001` show
   `mapped` and `handles` counts unchanged across C/R, but counts staying
   constant doesn't prove the per-allocation virtual→physical mapping is
   correct.

4. **GCR's preload may interfere with internal CUDA bookkeeping (streams,
   contexts).** Less likely given that NCCL collectives and `actor_train`
   work fine post-restore, but worth ruling out — both failures involve the
   *first* large post-restore use of a specific subsystem (VMM IPC, then
   pinned DMA). It's possible some lazy state (caches inside the driver,
   stream-priority tables) is initialized incorrectly post-restore.

5. **Race between consumer (sglang)'s `cuMemUnmap`/`cuMemRelease` and
   producer (Megatron)'s next kernel.** Adding canaries (forced sync) hid
   Failure A — consistent with a missing-sync race somewhere in the cleanup
   path. But this would be a slime-side bug, not a GCR bug. Worth eliminating
   before deep-diving GCR.

## What would help narrow it down on the GCR side

- Confirm whether GCR's preload hooks any of: `cuMemCreate`, `cuMemMap`,
  `cuMemSetAccess`, `cuMemExportToShareableHandle`,
  `cuMemImportFromShareableHandle`, `cuMemUnmap`, `cuMemRelease`,
  `cudaHostRegister`, `cudaHostUnregister`. Anything in this list that's *not*
  hooked is a candidate for stale state.
- Determine why `cudaHostRegister(shm_fs, 40 GiB)` returns `invalid argument`
  at GCR init — that error is visible in every log we have and is independent
  of the slime-level failures.
- After restore, dump `cudaPointerGetAttributes` for: (a) one of slime's VMM-
  allocated regions, (b) one of the `pin_memory=True` backup tensors. Compare
  with pre-suspend values. Any change in `type`, `hostPointer`, or
  `devicePointer` is significant.

## What we have *not* established (avoid over-claiming)

- We have **not** confirmed Failure A is deterministic. Only one failing run
  (`1777463001`) and one not-failing run (`1777515570`); the canary run differed
  in instrumentation, not just by chance.
- The exact worker-side environment variables in `1777515570` and `1777516790`
  are **unknown** — the runtime-env-json shown in the logs reflects what was
  passed to Ray, not necessarily what was exported into the worker processes.
- We have **not** identified the actual offending kernel for either failure (no
  compute-sanitizer access).
- We have **not** verified the suspected `torch.cat`-into-VMM kernel is what
  triggered 719. That's a hypothesis based on code-path proximity to the last
  successful log line, not direct evidence.
- We have **not** verified that GCR's `mapped`/`handles` counts being unchanged
  across C/R means the underlying mappings are *correct* — only that GCR's
  bookkeeping shows the same totals.
