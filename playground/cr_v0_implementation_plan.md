# C/R v0 Implementation Plan

## Goal

Replace both offload mechanisms — `torch_memory_saver` (MT-side) and SGLang's
`release_memory_occupation` / `resume_memory_occupation` (SG-side) — with a
single whole-process C/R primitive. Delete all patches that exist to work
around what C/R now handles natively (NCCL teardown/rebuild, captured-graph
recapture, `ReloadableProcessGroup`).

## Why
We have a C/R primitive that freezes a process while preserving its full CUDA context (device memory, IPC mappings, NCCL communicators, captured graphs, library handles) and on thaw it's bit-identical.
I want to reclaim the GPU memory each side holds while it's idle (MT during rollout, SG during training) without paying the things slime currently pays: NCCL teardown/rebuild, captured-graph re-capture, multi-stage wake-up, the ReloadableProcessGroup Megatron patch.

### Constraint
My C/R hooks the CUDA VMM API. So it cannot coexist with torch_memory_saver (both share the VMM hook layer) which is currently used to offload MT/SG. Therefore any reclamation can only come from the entire process being frozen.

KV pool sized conservatively in v0 to fit MT's full residency. No optimizer
offload, no syncer process. Those come in later iterations.

## Background

### Phase ordering

```
Phase A (train):   MT alive,  SG frozen
  → thaw SG
Phase B (sync):    MT alive,  SG alive       ← both fully resident
  → freeze MT
Phase C (rollout): MT frozen, SG alive
  → freeze SG, thaw MT
Phase A (train):   MT alive,  SG frozen
```

### Training loop
Current loop body:
```
generate 
→ SG.offload 
→ train 
→ offload_train 
→ SG.onload_weights
→ update_weights 
→ SG.onload_kv
```

v0 loop body:

```
generate
→ freeze_SG & thaw_MT 
→ train 
→ save 
→ thaw_SG
→ update_weights 
→ freeze_MT
```

## Sketch

GCR operates **outside the application** via LD_PRELOAD interposition. It traces
CUDA driver API calls to build up the state needed to suspend and restore all
device memory. From verl's perspective, GCR is a black box with two operations:

| Operation | Command | Effect |
|-----------|---------|--------|
| **Suspend** | `cr -d -p <PID>` | All GPU memory for the process is offloaded to host. The process remains alive but holds no device allocations. |
| **Resume** | `cr -r -p <PID>` | All GPU memory is restored at the **same virtual addresses**. The process continues as if nothing happened. |

Because virtual addresses are preserved, all in-process state that references
GPU pointers — including NCCL communicator internals, CuPy/PyTorch tensor
metadata, and KV cache allocations — remains valid after resume.

### Key interface details

- **Mechanism**: LD_PRELOAD (`libpreload.so` + `libcuda.so`) hooks CUDA driver calls.
- **Control plane**: Shared memory at `/mnt/huge/control-<PID>` + `SIGUSR1` signal.
- **Signal values**: `2` = suspend (dump), `3` = resume (restore), `0` = idle/complete.
- **Synchronization**: Controller polls `control->signal` until it returns to `0`.
- **Multi-process**: `cr` accepts multiple `-p <PID>` flags for batch operations.

### New module: orchestrator

A small wrapper around the C/R primitive: `freeze(actor_handle)` /
`thaw(actor_handle)`. Exposed as a Ray actor or as direct calls from the
driver process. One-shot API per process; nothing slime-specific.

### [train.py](../train.py) — replace the four phase-boundary calls

- `offload_train(rollout_id)` becomes a no-op (or deleted).
- `SG.offload()` → `freeze_SG` (whole-process).
- `SG.onload_weights()` + `SG.onload_kv()` collapse into one `thaw_SG` (no tag
  distinction; whole process restored bit-identical).
- New `freeze_MT` after `update_weights`, new `thaw_MT` before `train`.
- The pre-loop `update_weights()` at [train.py:31](../train.py#L31) needs the
  same setup: SG must be thawed for it. Use the same `thaw_SG` / `freeze_MT`
  bookends.

### [slime/backends/megatron_utils/actor.py](../slime/backends/megatron_utils/actor.py)

- Delete `torch_memory_saver` import + all `if args.offload_train:` branches
  (lines 81-84, 99-100, 147-150, 521-522, 539-540, 556-557, 594-595).
- `sleep()` / `wake_up()` become no-ops or are deleted entirely.
- Remove the `with torch_memory_saver.disable()` wrapper at
  [actor.py:570](../slime/backends/megatron_utils/actor.py#L570) — drop to
  plain `update_weights()` call.
- `destroy_process_groups()` / `reload_process_groups()` calls all go away.
  NCCL is preserved by C/R.

### [slime/backends/megatron_utils/__init__.py](../slime/backends/megatron_utils/__init__.py)

- Delete the `tms_set_interesting_region` monkey-patch around
  `deep_ep.Buffer.__init__`
  ([lines 5-21](../slime/backends/megatron_utils/__init__.py#L5-L21)). With no
  saver, every allocation is a normal allocation.

### [slime/ray/actor_group.py](../slime/ray/actor_group.py)

- Remove the `LD_PRELOAD` setup at
  [lines 62-73](../slime/ray/actor_group.py#L62-L73). MT processes start
  clean — no saver injected.
- `offload()` / `onload()`
  ([lines 123-127](../slime/ray/actor_group.py#L123-L127)) replaced by C/R
  freeze/thaw routed through the orchestrator.

### [slime/utils/reloadable_process_group.py](../slime/utils/reloadable_process_group.py)

- Delete the file. Remove all import sites and any `monkey_patch_torch_dist()`
  calls. With C/R preserving NCCL state, this entire layer is unnecessary.

### [slime/backends/megatron_utils/update_weight/common.py](../slime/backends/megatron_utils/update_weight/common.py)

- Remove `_maybe_get_cpu_backup` and the `translate_gpu_to_cpu` codepath
  ([lines 127-139](../slime/backends/megatron_utils/update_weight/common.py#L127-L139)).
  It only existed to read tensors out of `torch_memory_saver`'s CPU shadow
  when GPU pages were paused. With C/R, GPU tensors are simply on the GPU
  when MT is alive.

### [slime/ray/rollout.py](../slime/ray/rollout.py)

- Replace `release_memory_occupation` / `resume_memory_occupation` calls with
  C/R freeze/thaw of the SG engine processes.
- The tag distinctions (`WEIGHTS`, `KV_CACHE`, `CUDA_GRAPH`) are dropped —
  C/R is whole-process.
- `needs_offload` predicate stays as the "is this engine co-tenant with MT"
  check; its meaning is just "should we C/R this engine."

### [slime/backends/sglang_utils/sglang_engine.py](../slime/backends/sglang_utils/sglang_engine.py)

- `enable_memory_saver=args.offload_rollout` at
  [line 524](../slime/backends/sglang_utils/sglang_engine.py#L524) → set to
  `False` unconditionally. SGLang's saver must be off so it doesn't conflict
  with the VMM hook used by the C/R primitive.
- KV pool size launch parameter: pick a value that fits in
  `total_GPU − MT_full_residency − safety`. v0 just exposes this as a flag
  the user sets manually.

### Build / dependencies

- [build_conda.sh:52](../build_conda.sh#L52) and Dockerfiles: drop
  `torch_memory_saver` install. Add C/R primitive install.

## Investigation

Hidden assumptions about the saver / `ReloadableProcessGroup` that aren't
obvious from a grep:

1. **First-cycle MT↔SG NCCL group setup.**
   [`update_weight_cls`](../slime/backends/megatron_utils/actor.py#L135-L142)
   and `connect_rollout_engines` at
   [actor.py:560-565](../slime/backends/megatron_utils/actor.py#L560-L565).
   This is where the cross-process communicator is established. v0 must
   guarantee it's established *before* the first `freeze_MT` so C/R has a
   real communicator to preserve. Verify it only runs when
   `num_new_engines > 0` and that engines don't churn under steady-state.

2. **`async_train` semantics in
   [slime/ray/train_actor.py](../slime/ray/train_actor.py).** Need to confirm
   it returns at a fully quiescent point (no in-flight CUDA work, no pending
   NCCL ops) before freezing MT. C/R requires a clean stopping point.

3. **`save_model` path** at
   [actor.py:516-540](../slime/backends/megatron_utils/actor.py#L516-L540).
   It currently brackets itself with `reload/destroy_process_groups`. Verify
   save works correctly without the bracket once NCCL is C/R-preserved; the
   `async_save` codepath in particular uses background threads — confirm none
   of them run *during* a C/R freeze window.

4. **`args.use_fault_tolerance` path** at
   [actor.py:547-550](../slime/backends/megatron_utils/actor.py#L547-L550).
   Recovery re-fetches engine handles via `recover_updatable_engines`. If an
   engine is replaced mid-run, the C/R-preserved NCCL group becomes stale.
   Either disable fault tolerance in v0 or carve out a path to re-establish
   the group post-recovery.

5. **Critic model branches** in train.py and
   [actor.py:86-101](../slime/backends/megatron_utils/actor.py#L86-L101).
   Critic is a second MT-process that follows the same freeze/thaw lifecycle;
   verify the orchestrator handles two MT actors symmetrically and the phase
   ordering still makes sense in `use_critic` and `critic_train_only` configs.

6. **`eval()` mid-loop** at
   [train.py:74-75, 101-102](../train.py#L74-L75). Eval uses SG; MT can be
   frozen during eval. Confirm phase ordering: `freeze_MT` must complete
   before eval starts, and `thaw_MT` must happen after eval before next
   training.

7. **`keep_old_actor` and multi-tag TensorBackuper** at
   [actor.py:122-127](../slime/backends/megatron_utils/actor.py#L122-L127).
   Multiple weight versions on CPU. C/R preserves them across freezes (CPU
   memory is part of the process image). Verify no codepath assumes
   `torch_memory_saver`'s CPU-backup table.

8. **Initialization-time `if args.offload_rollout: rollout_manager.offload()`**
   at [placement_group.py:200-201](../slime/ray/placement_group.py#L200-L201).
   Triggered before MT is set up. Decide whether to call C/R `freeze_SG` here
   or simply leave SG running until first iteration.

9. **The `if num_new_engines > 0` branch** in
   [actor.py:559-568](../slime/backends/megatron_utils/actor.py#L559-L568).
   Should not fire after first iteration in steady state. If it does (e.g.,
   due to fault tolerance), v0 will likely break. Add an assertion to catch
   it.

10. **`NCCL_CUMEM_ENABLE` handling** at
    [actor_group.py:55-56](../slime/ray/actor_group.py#L55-L56). The comment
    says SGLang forces it to 0 to prevent NCCL errors. Check whether this is
    still needed once SGLang's saver is off and the C/R primitive is in place.

## Out of scope for v0, planned for later

| Iteration | Adds |
|---|---|
| v2 | Per-rank syncer process. MT is fully C/R'd through Phase B; syncer drives the bucket sync to SG. |
| v3 | Tight KV pool sizing tooling: launcher computes the appropriate KV pool size from MT/SG memory profiles instead of requiring a manual flag. |
| v4 | Async overlap: with the syncer in place, allow MT's next training step to begin while the syncer is still shipping weights to SG (where the rollout doesn't gate on sync completion). |
| v5 | Re-introduce fault tolerance and engine-churn handling under C/R: a path to re-establish the MT↔SG NCCL group when SG engines are replaced. |

The v0 deliverable is intentionally a *subtraction* from the current codebase
plus a thin orchestrator layer — most of the work is deletion. The
further-investigation list is where surprises will come from, not the deletion
itself.
