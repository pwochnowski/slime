# GCR tagged offload vs PyTorch caching allocator: conflicts and remedies

## Background

GCR manages GPU memory at the VMM (Virtual Memory Management) level — it can
unmap physical pages from virtual address ranges (`offload_tag`) and remap them
later (`restore_tag`). PyTorch's caching allocator sits on top of the same
virtual address space and maintains its own bookkeeping of segments, blocks, and
free lists. Neither layer knows about the other's operations.

This document catalogues the conflicts that arise when the two interact and the
remedies we've found.

## Conflict 1: headroom leaking across allocation boundaries

### Symptom

After `gcr_offload_tag`, the next unrelated CUDA allocation (e.g. a byte tensor
inside `torch.distributed.all_gather_object`) crashes with
`cudaErrorInvalidValue` or `cudaErrorIllegalAddress`.

### Root cause

PyTorch's caching allocator groups allocations into expandable segments. A
segment may have **headroom** — a few MB of free space beyond the last active
allocation. When GCR unmaps the VMM pages backing that segment, the headroom
becomes dead memory at a now-invalid VA. But PyTorch still tracks it as
available in its free list. The next general-purpose allocation can be satisfied
from that headroom, handing out a pointer into unmapped memory.

### Remedy: `torch.cuda.MemPool` isolation

Wrap the allocation phase of the system whose memory will be offloaded in a
dedicated `torch.cuda.MemPool`:

```python
self._pool = torch.cuda.MemPool()
with torch.cuda.use_mem_pool(self._pool), gcr.tagged(tag=TAG):
    model, optimizer, ... = initialize(...)
```

PyTorch's allocator never satisfies a request from one pool using another pool's
free list. After `gcr_offload_tag` unmaps the tagged pool's segments, the dead
headroom stays trapped inside that pool. General-purpose allocations (default
pool) pull from their own untouched segments.

**Important details:**

- Store the `MemPool` object on a long-lived reference (e.g. `self`). If it gets
  garbage collected, the pool is destroyed.
- You do **not** need to use the pool context during training forward/backward.
  Only the initial allocation of persistent tensors (parameters, optimizer
  states) needs to land in the pool. Temporary activations and gradients can use
  the default pool — they're freed before offload happens.
- The model tensors are still dead between offload and resume. As long as
  nothing dereferences them in that window (weight sync reads from the CPU
  mirror), this is fine. On `gcr_resume` / `restore_tag` the original VAs get
  re-mapped to fresh physical memory and the tensors come back to life in-place.

### When MemPool does NOT help

`torch.cuda.empty_cache()` is **not** pool-specific — it walks every pool's
free list and returns blocks to the driver. If any code calls `empty_cache()`
while a pool's segments are unmapped, it will attempt to release those dead
blocks and crash. See Conflict 2.

## Conflict 2: `empty_cache` touching unmapped segments

### Symptom

`torch.cuda.empty_cache()` (or any code that calls it, such as SGLang's
`flush_cache` endpoint) crashes with `cudaErrorIllegalAddress` after segments
have been offloaded and then restored via GCR.

### Root cause

`empty_cache()` iterates over **all** pools and tries to return freed blocks to
the CUDA driver. If segments were offloaded (VMM pages unmapped) and the
allocator's metadata is stale, it touches invalid memory.

Even after `restore_tag` remaps the pages, the allocator's internal state may
not perfectly match the restored VMM layout. Calling `empty_cache()` at this
point can still encounter inconsistencies.

### Remedy: defer `empty_cache` until after restore, or skip it

If `empty_cache` is called as part of a larger operation (e.g. SGLang's
`flush_cache` which also resets the radix tree and KV cache pools), move that
call to a point in the lifecycle where memory is guaranteed to be live:

- **After** `gcr_restore_tag` / `gcr_resume` completes — the VMM pages are
  remapped, so `empty_cache` is safe.
- **Before** `gcr_offload_tag` / `gcr_suspend` — memory is still live.

**Do not** call `empty_cache` in the window between offload and restore.

In our case, `update_weights` was calling `flush_cache` (which internally calls
`empty_cache`) on the SGLang engines while the KV cache segments were in a
post-offload/post-restore state. The fix was to:

1. Comment out `flush_cache` inside `update_weights`.
2. Call `flush_engines_cache` on the rollout manager **after**
   `gcr_restore_tag` brings the KV cache segments back, at a point where all
   memory is live.

## Conflict 3: stale allocator metadata after restore

### Symptom

After `gcr_restore_tag`, the system doesn't crash but produces incorrect results
(e.g. degenerate model outputs, all-zero rewards). No CUDA errors are reported.

### Root cause

The offload/restore cycle unmaps and remaps VMM pages, but PyTorch's allocator
retains its pre-offload bookkeeping. Depending on what the allocator or
application code does between restore and the next use, internal data structures
(free lists, block splits, cached metadata) may be subtly inconsistent. This can
lead to:

- Allocations that silently overlap with live tensors.
- Blocks that appear free but contain stale data.
- Forward passes that read from memory with corrupted contents.

### Remedy

After restoring tagged memory, flush the caches and pools that operated over
that memory region before using it for inference or training. For SGLang this
means calling `flush_cache` (which resets the radix tree, token pools, and KV
cache allocator) **after** `gcr_restore_tag` completes.

## General principles

1. **Isolate offloadable allocations.** Use `torch.cuda.MemPool` to keep
   GCR-managed tensors in their own pool. This prevents headroom from leaking
   into the default pool's free list.

2. **Never call `empty_cache` on unmapped memory.** Any operation that walks
   the allocator's segment list (`empty_cache`, certain debugging/profiling
   calls) must only run when all segments are live.

3. **Flush after restore, not before offload.** If you need to reset caches
   (KV cache, radix tree, etc.), do it after `restore_tag` when memory is live,
   not during `update_weights` when segments may be in a transitional state.

4. **Treat offloaded tensors as dead.** Between offload and restore, do not
   read, write, or pass offloaded tensors to any CUDA operation. Use CPU mirrors
   for any data that needs to be accessed in that window (e.g. weight sync reads
   from a CPU backup, not from the dead GPU tensors).

5. **Test with `CUDA_LAUNCH_BLOCKING=1` for diagnosis.** Async CUDA errors get
   reported at the next synchronization point, making stack traces misleading.
   `CUDA_LAUNCH_BLOCKING=1` gives accurate stack traces but is too slow for
   production.