# Untracked 1.16 GB VMM region causes post-C/R illegal memory access

## TL;DR

A 1.16 GB device VA region at `0x5822000000` is mapped in the Megatron actor process but **not tracked by GCR's hook**. The hook sees `cuMemUnmap` calls against this region but never sees a corresponding `cuMemMap`. Across a C/R cycle, GCR cannot save or restore this region, so the post-restore mapping is stale. Subsequent kernels accessing offsets within this region trigger Xid 31 (`cudaErrorIllegalAddress`).

The most likely owner of the region is **NCCL with `NCCL_CUMEM_ENABLE=1`**, using a CUmem entry point not covered by GCR's hook (e.g. `cuMemMapV2` or another variant).

## Evidence

### Crash signature
- `cudaErrorIllegalAddress` (Xid 31, write fault) surfacing in NCCL watchdog and other downstream kernels.
- Same crash class observed across multiple runs (slime jobs 1777463001, 1777516790, 1777518601), each time post first or second C/R cycle. Crash site shifts (TP all_gather → tensor_backper → empty_cache after compute_log_prob) but the underlying class is consistent.
- All Xid 31 fault VAs cluster in `0x58_xxxxxxxx` — same range across distinct run instances.

### Specific fault VAs collected from dmesg
| Kernel-ts | Unix | UTC | VA | RW | Suspected run |
|-----------|------|-----|-----|------|---------------|
| 1551765 | 1777516795 | Apr 30 02:39:55 | `0x58_433b3000` | W | Run 3 (1777516790) — assumption |
| 1551765 | 1777516795 | Apr 30 02:39:55 | `0x58_27299000` | W | Run 3 — assumption |
| 1553564 | 1777518594 | Apr 30 03:09:54 | `0x58_358d9000` | W | Run 3 — assumption |
| 1553564 | 1777518594 | Apr 30 03:09:54 | `0x58_21b10000` | W | Run 3 — assumption |
| 1564548 | 1777529579 | Apr 30 06:12:59 | `0x58_44f3a000` | W | newer run, not yet identified |
| 1564549 | 1777529580 | Apr 30 06:13:00 | `0x58_5006c000` | W | newer run, not yet identified |

### Containment check vs GCR's tracking (run 1777518601)

Method: for each fault VA, search every `va=…size=…` line across `gcr_imports_*.log` (snapshot), `master_daemon-*.log`, `hook-*.log`, `data_proc-*.log` for an entry where `va ≤ fault_addr < va + size`.

Run 4 (1777518601):

| Fault VA | Status |
|----------|--------|
| `0x58_21b10000` | Inside tracked alloc: `REM va=0x5821a00000 size=0x200000 key=2168 shared=1` (a 2 MB region) |
| `0x58_27299000` | NOT in any tracked alloc. Inside the unmapped 1.16 GB region. |
| `0x58_358d9000` | NOT in any tracked alloc. Inside the unmapped 1.16 GB region. |
| `0x58_433b3000` | NOT in any tracked alloc. Inside the unmapped 1.16 GB region. |
| `0x58_44f3a000` | NOT in any tracked alloc. Inside the unmapped 1.16 GB region. |
| `0x58_5006c000` | NOT in any tracked alloc. Inside the unmapped 1.16 GB region. |

### The unmap-without-map pattern

In `slime/logs/1777518601/gcr_logs/hook-3838641.log` (Megatron actor pid 3838641):

```
421.506088 hook 3838641 3838641 cuMemUnmap va=0x5822000000 size=1245708288   # 1.16 GB
697.542772 hook 3838641 3838641 cuMemUnmap va=0x5822000000 size=1090519040   # 1.04 GB
```

**No `cuMemMap` event in any component log targets `va=0x5822000000`.** The hook's normal pattern is `cuMemMap entry … / cuMemMap … key=N` per allocation; it sees neither for this region. There is also no `untracked` or `claim_failed` annotation on the unmap line — the hook treated it as if it had been a tracked map, but the underlying record is missing.

### What's nearby in the tracked range

GCR-tracked VAs in the `0x58_xxxxxxxx` band run from `0x5820400000` to `0x58b9e00000`, mostly 2 MB-aligned `REM` allocations of size `0x200000` each. The `0x5822000000` region begins immediately after a tracked 2 MB region at `0x5821e00000`. The unmapped region is *contiguous* with — and follows — the tracked allocations.

### What's hooked vs not (counts of `0x58_…` mentions per component log, run 1777518601)

| Component | Mentions of `0x58_…` |
|-----------|----------------------|
| `master_daemon-3837248.log` | 0 |
| `hook-3838641.log` (rank 0 actor) | 90+ (mostly small 2 MB REM/IMP) + the two 1.16/1.04 GB unmaps |
| `hook-3839065.log` (rank 1 actor) | 144 |
| `hook-3836727.log` / `3836728.log` (sglang) | 0 |
| `data_proc-0-3837062.log` | 314 |
| `data_proc-{1,4,5}-*.log` | 0 |

The 1.16 GB region only ever appears in the rank-0 actor's hook log, and only as unmaps. master_daemon doesn't see it (consistent with it being a process-local, non-shared allocation). data_proc doesn't see it.

## Hypothesis (corrected)

**Some caller is invoking `cuMemMap` via a path that bypasses both the LD_PRELOAD interposer and the `cuGetProcAddress` hook.** The GCR hook *does* cover all the relevant `cuMem*` entry points (verified by reading [GCR/cuda_hooks_entrypoint.cpp](../../../../GCR/GCR/cuda_hooks_entrypoint.cpp) and [GCR/cuda.cpp:179-307](../../../../GCR/GCR/cuda.cpp#L179)) — so the earlier "missing-symbol" framing was wrong.

What we actually observe in `hook-3838641.log` for handle=0x6639b970 (key=272, size=1245708288):

| Event | Hook log? |
|---|---|
| `cuMemCreate` size=1245708288 → key=272 | YES (line 5762, t=391.047) |
| `cuMemMap` (would map handle to `va=0x5822000000`) | **NO** |
| `cuMemSetAccess` for the 1.16 GB region | **NO** |
| `cuMemUnmap va=0x5822000000 size=1245708288` | YES (line 7979, t=421.506) |
| `cuMemRelease handle=0x6639b970 key=272` | YES (line 7980, t=421.506) |

Aggregate counts in `hook-3838641.log`:
- 15 `cuMemCreate` of size 1245708288
- **0** `cuMemMap` of size 1245708288
- 14 `cuMemUnmap` of size 1245708288
- Largest size in any logged `cuMemMap` is 528482304 (~504 MB) — never 1.16 GB

Same asymmetry in rank-1's `hook-3839065.log` (15 creates / 0 maps / 14 unmaps).

The asymmetry is the puzzle. If the hook were missing a symbol entirely we'd lose all three of (create, map, unmap). Instead it only loses *map* and *setAccess*. This means the caller is using *different lookup mechanisms* for different cuMem entry points within the same logical operation.

### Likely cause: dlsym shortcut for cuMemMap (assumption)

Looking at [GCR/preload.c:51-67](../../../../GCR/GCR/preload.c#L51), the `dlsym` hook only redirects `cuGetProcAddress` / `cuGetProcAddress_v2`. For any other symbol, `dlsym(libcuda, "cuMemMap")` returns the *real* unhooked `cuMemMap` from libcuda. So a library that resolves `cuMemMap` by direct dlsym lookup at startup and stores the raw pointer would never be intercepted by GCR — neither by the LD_PRELOAD PLT-interpose nor by the cuGetProcAddress dispatch.

Why would the same library use cuGetProcAddress for cuMemCreate (intercepted) but dlsym for cuMemMap (not intercepted)? Most likely it doesn't — it's two *different* libraries or call sites with different lookup conventions, sharing the same handle:
- One path (linked against libcuda's PLT, or using cuGetProcAddress) calls cuMemCreate → intercepted, key issued.
- A different code path (using direct dlsym, possibly inside libtorch or NCCL) calls cuMemMap on that handle → bypasses GCR.
- The same first path calls cuMemUnmap → intercepted, but GCR has no record of the va.

This needs verification — see "Suggested next steps" below.

### Other candidates considered
- **PyTorch expandable_segments**: ruled out because the test sets `PYTORCH_ALLOC_CONF=expandable_segments:False` (unverified that this propagates to all child processes — worth a sanity check).
- **CUDA graph internal mempool**: possible but typically smaller and the test uses `--attention-backend flash`. Less likely.
- **sglang KV cache**: lives in the SGLang scheduler processes, not the Megatron actor. Wrong process.
- **NCCL** with `NCCL_CUMEM_ENABLE=1`: still plausible but reframed — the question isn't whether the hook covers NCCL's symbol, it's whether NCCL bypasses dlsym interposition with cached function pointers.

## Assumptions used in mapping dmesg pids to runs

- The dmesg pids (32863, 33512, 87428, 88063, 352716, 351995) **don't match** the Megatron actor pids in `gcr_imports_*.log` (3802228/3802653 for Run 3, 3838641/3839065 for Run 4). I have **assumed** these dmesg pids are TIDs of NCCL worker threads inside the actor processes, not separate process pids. Justification: the system uptime at investigation time was ~18 days so Linux PID space may have wrapped; pid pairs share similar deltas (~600-700) suggesting paired rank0+rank1 of one actor instance; the kernel's Xid logging reports the offending thread ID, not parent process ID. **This has not been independently verified.**
- I have **assumed** that the dmesg events at kernel-ts 1551765 and 1553564 correspond to Run 3 (1777516790) based on the 30-minute gap matching Run 3's wall-clock duration and Run 3's crash time of ~03:09 UTC. **Not verified by direct pid-matching.**

## Suggested next steps

1. **Identify the caller of cuMemMap for handle 0x6639b970** at hook time ~391-421s. The cuMemUnmap at 421.506088 is a `(va, size)` pair — gdb on a fresh run with a breakpoint on the *real* `cuMemMap` (or strace/ltrace if available) will show which library is calling it. Specifically:
   ```
   gdb -p <megatron actor pid>
   (gdb) break libcuda.so.1:cuMemMap_v2
   (gdb) commands
   > bt 15
   > continue
   > end
   (gdb) continue
   ```
   The backtrace identifies whether the caller is libtorch, libnccl, or something else.

2. **Extend the dlsym hook** ([GCR/preload.c:51-67](../../../../GCR/GCR/preload.c#L51)) to redirect more than just cuGetProcAddress. If a library bypasses cuGetProcAddress and calls `dlsym(libcuda, "cuMemMap")` directly, GCR currently returns the unhooked symbol. Adding the `cuMem*` entry points to the dlsym redirect table would close that hole. Risk: this might break things that legitimately want the raw symbol — measure first.

3. **Sanity check `PYTORCH_ALLOC_CONF=expandable_segments:False`** actually propagates to the Megatron actor process. Just `print(os.environ.get("PYTORCH_ALLOC_CONF"))` from inside the actor at startup. If it's not set, expandable_segments is on by default, and PyTorch's allocator could be the cuMemMap caller.

4. **Try `NCCL_CUMEM_ENABLE=0`** for one run. If the 1.16 GB cuMemUnmaps disappear from the hook log and the crash goes away, NCCL is confirmed as the caller (and is using a dlsym shortcut for cuMemMap). If they persist, NCCL is not the source and PyTorch is the more likely candidate.

5. **Verify the dmesg-pid-to-actor-pid assumption** by collecting `ps -eLo pid,tid,comm` mid-run before crash. Find which TIDs belong to the Megatron actor pids. If the TIDs match the dmesg-reported pids, that confirms the kernel is reporting NCCL worker thread TIDs.

6. **Identify the newer run** that produced Xid 31 events at kernel-ts 1564548/1564549 (UTC 06:12:59-06:13:00). Its fault VAs (`0x58_44f3a000`, `0x58_5006c000`) are also inside the 1.16 GB region, suggesting the same root cause across runs.

## Files referenced

- `/home/paul/repos/slime/logs/1777518601/output.log` — Run 4 output (NCCL watchdog crash)
- `/home/paul/repos/slime/logs/1777518601/gcr_imports_3838641.log` — Run 4 rank 0 actor snapshot
- `/home/paul/repos/slime/logs/1777518601/gcr_logs/hook-3838641.log` — Run 4 rank 0 hook trace (contains the smoking-gun unmaps)
- `/home/paul/repos/slime/logs/1777518601/gcr_logs/master_daemon-3837248.log` — Run 4 daemon
- `/home/paul/repos/GCR/docs/log-file-guide.md` — log format reference