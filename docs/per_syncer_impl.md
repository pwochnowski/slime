# Per-Syncer Implementation Plan

## Context

In colocated RLHF, the current weight sync (Phase B) requires both the trainer and SGL to be alive on GPU simultaneously. Even with TMS offload, 2.5-5 GB of trainer residual (NCCL state, CUDA context, allocator slack) stays on GPU throughout rollout, reducing KV cache headroom.

The per-syncer introduces a lightweight process that takes over the sync role after the trainer is fully frozen via GCR. Its GPU footprint is ~1.7 GB (CUDA context + one bucket buffer), freeing 2.5-5 GB for KV cache during both sync and rollout phases.

## Current Phase Cycle (train.py:63-114)

```
C (rollout):  MT frozen, SGL alive      -> generate
C->A:         freeze SGL, thaw MT
A (train):    MT alive, SGL frozen       -> train
A->B:         thaw SGL (MT stays alive)
B (sync):     both alive                 -> update_weights
B->C:         freeze MT
```

## New Phase Cycle with Per-Syncer

```
C (rollout):  MT frozen, SY frozen, SGL alive   -> generate
C->A:         freeze SGL, thaw MT
A (train):    MT alive, SY frozen, SGL frozen    -> train
A->handoff:   MT runs HF iterator, D2H to memfd
handoff->sync: freeze MT, thaw SY + SGL
sync:         MT frozen, SY alive, SGL alive     -> syncer pushes weights via VMM IPC
sync->C:      freeze SY
```

## Key Decisions

- **Gloo groups**: Syncer creates its own independent `torch.distributed` world and Gloo gather groups (same topology as trainer, separate process groups). No shared state between trainer and syncer processes.
- **No trainer+SGL coexistence**: The syncer handles ALL weight pushes, including the initial one at startup. The trainer never calls `update_weights()`. Instead: trainer does `prepare_sync_handoff()` (D2H to memfd) → freeze trainer → thaw syncer + SGL → syncer pushes weights → freeze syncer. This applies uniformly to the initial push and every subsequent iteration.
- **Scope**: Colocated-only. If `use_distribute=True` (mixed colocated + distributed engines), fall back to existing trainer-driven path. Per-syncer only handles VMM IPC to colocated engines.
- **NCCL broadcast**: Always use intra-syncer NCCL broadcast (rank 0 H2D, broadcast to others). On systems without NVLink, NCCL falls back to PCIe automatically — no bandwidth win, but still correct. Colocated setups are single-node with NVLink in practice.

## Implementation Steps

### Step 1: memfd Handoff Module

**New file: `slime/utils/memfd_handoff.py`**

Provides primitives for the trainer to write post-HF weight buckets into a memfd, and the syncer to read them back.

Data structures:
- `TensorSlice(name, shape, dtype, offset, nbytes)` - per-tensor metadata within a bucket
- `BucketManifest(bucket_index, tensors: list[TensorSlice], total_bytes)` - one bucket's layout
- `SyncManifest(buckets: list[BucketManifest], total_bytes, weight_version, quantization_config)` - full manifest

Functions:
- `create_memfd(name, total_bytes) -> (fd, mmap)` - `os.memfd_create` + `ftruncate` + `mmap`
- `write_bucket(mapping, offset, hf_named_tensors) -> BucketManifest` - D2H each tensor into the mmap region, record slice metadata
- `read_bucket_tensors(mapping, bucket_manifest) -> list[(name, cpu_tensor)]` - reconstruct CPU tensors from mmap
- `serialize_manifest(manifest) -> bytes` / `deserialize_manifest(data) -> SyncManifest` - JSON encoding

The memfd is a kernel inode (tmpfs-backed) that survives GCR freeze because GCR only checkpoints CUDA allocations. Total size = sum of all post-HF weight bytes (same as what TensorBackuper holds in pinned CPU memory today).

### Step 2: Syncer Ray Actor

**New file: `slime/ray/syncer_actor.py`**

Minimal Ray actor class — one instance per trainer rank, pinned to the same GPU.

```python
class SyncerRayActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        # init torch.distributed (gloo backend), set CUDA device

    def setup(self, engine_gpu_counts, engine_gpu_offsets, rollout_engines, quantization_config):
        # Create Gloo IPC gather groups (same topology as trainer's _ipc_gather_group)
        # Create NCCL intra-syncer broadcast group (for rank 0 H2D -> NVLink to others)
        # Store rollout engine Ray handles
        # Allocate pinned host bounce buffer (max_bucket_bytes)

    def receive_memfd(self, ctrl_uds_path, manifest_bytes):
        # Connect to trainer's UDS, recv memfd fd via SCM_RIGHTS
        # mmap the fd, deserialize manifest

    def sync_weights(self):
        # Per-bucket loop:
        # 1. Rank 0: read bucket from memfd -> pinned bounce -> H2D to CUDA tensor
        # 2. NCCL broadcast to other syncer ranks
        # 3. All ranks: call _send_to_colocated_engine() (reused from update_weight_from_tensor.py)
        # 4. ray.get(refs), barrier, cleanup VMM resources
        # Bookended by pause_generation/flush_cache/post_process_weights/continue_generation RPCs

    def get_pid(self) -> int:
        return os.getpid()

    def gcr_suspend(self): ...
    def gcr_resume(self): ...
```

Key reuse: `_send_to_colocated_engine()` and `_cleanup_vmm_resources()` from [update_weight_from_tensor.py](slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py) are already free functions that accept explicit parameters. The syncer imports and calls them directly with GPU tensors it has reconstructed from the memfd. No changes needed to those functions.

NCCL broadcast rationale: Without it, each syncer rank independently H2Ds the full bucket from memfd (N x PCIe reads, contending on host memory bandwidth). With NCCL: rank 0 H2Ds once, broadcasts over NVLink (or PCIe fallback) to others. For TP=8 with NVLink this saves ~7x PCIe bandwidth. Cost: ~50-150 MB NCCL state per rank.

### Step 3: Syncer Group Manager

**New file: `slime/ray/syncer_group.py`**

Analogous to `RayTrainGroup` ([actor_group.py](slime/ray/actor_group.py)) but for syncer actors.

```python
class SyncerGroup:
    def __init__(self, args, num_nodes, num_gpus_per_node, pg, num_gpus_per_actor=0.1):
        # Spawn SyncerRayActor instances on same placement group bundles as trainer
        # With LD_PRELOAD for GCR (same env as actor_group.py:64-72)

    def setup(self, rollout_engines, engine_gpu_counts, engine_gpu_offsets, quantization_config):
        # ray.get([actor.setup.remote(...) for actor in self._syncer_handlers])

    def receive_memfd(self, ctrl_data: list[tuple[str, bytes]]):
        # Pass UDS paths + manifests to each syncer rank

    def sync_weights(self):
        # ray.get([actor.sync_weights.remote() for actor in self._syncer_handlers])

    def gcr_suspend(self):
        from gcr import suspend
        suspend(self._get_syncer_pids())

    def gcr_resume(self):
        from gcr import resume
        resume(self._get_syncer_pids())
```

Ray GPU scheduling: `num_gpus` is a Ray scheduling token that tells the scheduler how to pack actors onto GPU bundles — it does NOT limit actual GPU memory. A process claiming `num_gpus=0.4` can use all 80 GB of VRAM. Actual GPU memory isolation is handled by GCR (freeze/thaw ensures mutual exclusion). The fractions just ensure Ray places trainer (0.4, [placement_group.py:128](slime/ray/placement_group.py#L128)), SGL (0.2, [rollout.py:99](slime/ray/rollout.py#L99)), and syncer (0.1) on the same physical GPU without exceeding the 1.0 per-bundle budget ([placement_group.py:43](slime/ray/placement_group.py#L43)).

### Step 4: Trainer-Side Handoff

**Modified: [actor.py](slime/backends/megatron_utils/actor.py)**

Add `prepare_sync_handoff()` method to `MegatronTrainRayActor` (after line 469):

```python
def prepare_sync_handoff(self):
    """Run HF weight iterator while GPU is alive, D2H post-HF buckets into memfd.
    Returns (ctrl_uds_path, serialized_manifest) for the syncer to consume.
    Must be called before GCR freeze.
    """
    self.weight_updater.weight_version += 1
    megatron_local_weights = self.weights_backuper.get("actor")

    # Iterate HF weight chunks (PP/EP broadcast, TP all-gather, HF convert on GPU)
    all_buckets_data = []
    for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
        all_buckets_data.append(hf_named_tensors)

    # Compute total memfd size, create memfd, write all buckets
    # ...
    # Open UDS listener to serve the memfd fd to the syncer
    # Return (uds_path, manifest_bytes)
```

The `_hf_weight_iterator` is already a member of `UpdateWeightFromTensor` (line 77). We access it via `self.weight_updater._hf_weight_iterator`. The iterator calls `_get_megatron_full_params` which does PP/EP broadcasts and TP all-gather via NCCL — these require the trainer's NCCL communicators to be alive, which they are (trainer is still on GPU at this point).

TensorBackuper continues to serve duty (1) — model switching between actor/ref/old_actor. Its duty (2) — providing sync source after TMS unmap — is replaced by the memfd path.

**Modified: [actor_group.py](slime/ray/actor_group.py)**

Add `prepare_sync_handoff()` wrapper:
```python
def prepare_sync_handoff(self):
    return ray.get([actor.prepare_sync_handoff.remote() for actor in self._actor_handlers])
```

### Step 5: Coordinator Integration

**Modified: [train.py](train.py)**

New phase transitions when `args.use_per_syncer` is True:

```python
# After create_training_models() — create and initialize syncer group
if args.use_per_syncer:
    syncer_group = SyncerGroup(args, ...)
    syncer_group.setup(rollout_engines, engine_gpu_counts, engine_gpu_offsets, quantization_config)

# Initial weight push via syncer (trainer and SGL never coexist)
if args.use_per_syncer:
    handoff_data = actor_model.prepare_sync_handoff()  # HF iterate + D2H to memfd
    syncer_group.receive_memfd(handoff_data)
    actor_model.gcr_suspend()                          # freeze trainer
    syncer_group.gcr_resume()                          # thaw syncer
    # SGL is already alive (just started)
    syncer_group.sync_weights()                        # push initial weights
    syncer_group.gcr_suspend()                         # freeze syncer
    # Now: trainer frozen, syncer frozen, SGL alive → begin rollout
else:
    actor_model.update_weights()                       # existing: both alive
    actor_model.gcr_suspend()

# In the loop, replace Phase A->B->C with:
if args.use_per_syncer:
    # After train completes (trainer still alive, SGL frozen):
    handoff_data = actor_model.prepare_sync_handoff()
    syncer_group.receive_memfd(handoff_data)

    # Freeze trainer, thaw syncer + SGL
    actor_model.gcr_suspend()
    if critic_model: critic_model.gcr_suspend()
    ray.get(rollout_manager.gcr_resume.remote())
    syncer_group.gcr_resume()

    # Sync (syncer alive, SGL alive, trainer frozen)
    syncer_group.sync_weights()

    # Freeze syncer (SGL stays alive for rollout)
    syncer_group.gcr_suspend()
else:
    # existing Phase A->B->C code unchanged
```

**Modified: [placement_group.py](slime/ray/placement_group.py)**

Update `create_training_models()` to also create the syncer group when `args.use_per_syncer`. The syncer reuses the same placement group as the actor with `num_gpus_per_actor=0.1`.

**Modified: [arguments.py](slime/utils/arguments.py)**

Add `--use-per-syncer` flag near `--colocate` (after line 82):
```python
parser.add_argument(
    "--use-per-syncer",
    action="store_true",
    default=False,
    help="Use a lightweight syncer process for weight sync (colocate only). "
         "Reduces GPU footprint during sync and rollout.",
)
```

### Step 6: memfd fd Passing via UDS

Reuse the existing UDS infrastructure from [vmm_ipc.py](python/sglang/srt/weight_sync/vmm_ipc.py) (`open_sidecar_listener`, `send_fd`, `open_sidecar_client`, `recv_fd`). No new module needed — the trainer opens a listener, the syncer connects and receives the memfd fd. Same pattern as the VMM buffer fd handoff to SGL.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `slime/utils/memfd_handoff.py` | **New** | memfd create/write/read + manifest types |
| `slime/ray/syncer_actor.py` | **New** | SyncerRayActor class |
| `slime/ray/syncer_group.py` | **New** | SyncerGroup lifecycle manager |
| `slime/utils/arguments.py` | Modify | Add `--use-per-syncer` flag |
| `slime/backends/megatron_utils/actor.py` | Modify | Add `prepare_sync_handoff()` |
| `slime/ray/actor_group.py` | Modify | Add `prepare_sync_handoff()` wrapper |
| `train.py` | Modify | New phase transitions with syncer |
| `slime/ray/placement_group.py` | Modify | Create syncer group alongside actor |

Unchanged: `update_weight_from_tensor.py` (`_send_to_colocated_engine` reused as-is), `hf_weight_iterator_direct.py`, `tensor_backper.py`, `vmm_ipc.py`.

## Verification

1. **Unit test**: `tests/test_memfd_handoff.py` — write tensors to memfd, read back, verify bit-exact match. No GPU needed.

2. **VMM IPC after C/R**: Extend `tests/test_vmm_ipc_after_cr.py` to include a syncer-like flow: memfd -> H2D -> VMM -> consumer, with GCR suspend/resume of the producer between phases.

3. **Integration test**: Small model (Qwen2.5-0.5B) with `--use-per-syncer --colocate`, verify:
   - Weights correctly reach SGL (weight version matches)
   - Multiple RL iterations work (memfd recreated each iter)
   - GCR suspend/resume of trainer, syncer, and SGL all work correctly
   - Loss convergence matches non-per-syncer baseline

4. **Existing tests pass**: All current colocate tests unchanged (per-syncer is opt-in behind `--use-per-syncer`).

## Implementation Order

1. `memfd_handoff.py` + unit test (no GPU dependency)
2. `syncer_actor.py` + `syncer_group.py` (syncer process skeleton)
3. `actor.py` `prepare_sync_handoff()` (trainer handoff path)
4. `train.py` + `placement_group.py` + `arguments.py` (coordinator wiring)
5. Integration test with small model
6. Profile and tune (memfd sizing, NCCL broadcast vs individual H2D)
