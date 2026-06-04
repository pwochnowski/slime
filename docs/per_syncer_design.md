# Per-Syncer Design: Weight Sync with GCR

## Problem Statement

In colocated RLHF, training and inference alternate on the same GPUs. During weight sync, the trainer holds the source weights and the inference engine (SGL) needs to receive them. Both must be GPU-resident simultaneously.

GCR offloads the entire trainer process (freeing 100% of GPU), but it's all-or-nothing — it can't selectively keep some tensors alive. So the trainer can't drive sync after it's frozen.

The per-syncer introduces a lightweight process that takes over the sync role after the trainer freezes. Its GPU footprint is minimal (CUDA context + one bucket buffer), giving SGL significantly more KV cache headroom than either:
- Keeping the full trainer alive during sync (current slime, no offload)
- TMS-based selective offload (frees weights but NCCL state, CUDA context, and allocator slack remain)

## Phase Diagram

```
phase            trainer (MT)   syncer (SY)   inference (SGL)
─────            ────────────   ───────────   ───────────────
rollout          frozen         frozen        live  (large KV pool)
train            live           frozen        frozen
handoff          D2H post-HF weights → memfd; signal syncer
                 frozen         ─             ─
sync             frozen         live          live
                                ↓ bucket H2D from memfd → VMM → SGL imports
post-sync        frozen         frozen        live  (rollout continues)
```

## Slime Weight Sync Path (TMS offload)

Source: `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

In the TMS-based flow, `torch_memory_saver` hooks `cudaMalloc` and tags Megatron's model weight and optimizer allocations. At end-of-training, TMS unmaps their physical GPU pages (`cuMemUnmap`) while preserving virtual addresses — so `model.layer.weight` is still a valid Python object, but its backing GPU memory is released. SGL can then resume with that freed space.

Because the GPU pages are gone, the trainer cannot read from `model.named_parameters()` during sync. Instead it reads from `TensorBackuper`'s pinned CPU mirror, which was populated at end-of-train via `backup("actor")`.

### Phase sequence

1. **End of training.** `TensorBackuper.backup("actor")` D2H-copies each Megatron-local shard into a per-tensor pinned host buffer. Total: W bytes across all ranks (sharded — each rank copies only its own TP/PP/EP shard, W/N per rank).
2. **TMS pause.** Megatron's tagged GPU allocations are unmapped. Physical pages freed. Megatron's VAs kept for later resume.
3. **SGL resumes.** Uploads its model weights and KV cache into the freed GPU space.
4. **Sync loop.** Per bucket:
   - `_hf_weight_iterator.get_hf_weight_chunks()` yields one bucket. Source: `weights_getter` → `TensorBackuper.get("actor")` → pinned CPU dict.
   - H2D: each rank reads its Megatron-local shard from TB's pinned host → GPU. Size: shard bytes, parallel across ranks.
   - PP/EP broadcast (NCCL): every rank gets the full set of Megatron-sharded params for this bucket.
   - TP all-gather (NCCL) + `torch.cat`: every rank gets the fully-aggregated param.
   - HF convert: Megatron layout → HuggingFace naming/shape (GLU rechunk, MoE dim fix, etc).
   - `_send_to_colocated_engine()`:
     - Flatten bucket tensors to uint8 → alloc VMM buffer → `torch.cat` into it.
     - Open UDS listener, serve fd via SCM_RIGHTS.
     - `dist.gather_object` metadata to src rank over gloo.
     - Src rank calls `engine.update_weights_from_tensor_vmm.remote(...)` via Ray.
     - SGL TP workers import fd, remap, load weights, hit tp_cpu_group barrier.
   - After `ray.get` returns, barrier across gather group, free VMM buffer.
   - Next bucket.
5. **Post-sync.** SGL rolls out. Megatron weights remain unmapped until next train phase.

### What TMS leaves on GPU

TMS only offloads allocations it tagged via `cudaMalloc` interception — specifically model weights and optimizer state. Everything else stays resident:

- **NCCL communicator state**: persistent send/recv buffers for each DP/TP/PP/EP group. 1-3 GB at high parallelism.
- **Megatron allocator slack**: caching allocator pools, fragmented blocks not holding tagged tensors. 0.5-2 GB.
- **CUDA context**: driver/runtime overhead. ~1.2 GB.
- **Megatron metadata**: parameter structs, parallel-state bookkeeping. ~100-500 MB.

This residual (2.5-5 GB) is the irreducible trainer footprint that TMS cannot reclaim and is what GCR is designed to eliminate.

### Bucket Sizing

Defined in `hf_weight_iterator_direct.py:_get_megatron_local_param_info_buckets()`.
- Greedy bin-packing into buckets capped at `--update-weight-buffer-size` (default 512 MiB).
- Cap is in terms of post-aggregation (full-tensor) bytes: `info.size * tp_size`.
- A single param exceeding the cap gets its own bucket (soft cap, not hard).
- Computed once at init, reused every step.

### GPU Peak Per Bucket (Trainer Side)

Because aggregation runs on the trainer GPU at sync time, each bucket creates transient allocations:

- PP/EP broadcast recv slots: ~bucket_size / tp_size
- TP all-gather partitions: bucket_size (tp_size × shard buffers)
- torch.cat output: bucket_size
- HF convert (GLU split etc): up to bucket_size
- VMM staging buffer: bucket_size
- **Peak: ~2-3× bucket_size transient** (with overlap/free between stages)

These transients exist on top of the TMS residual (~2.5-5 GB). SGL must be sized to coexist with both.

## Per-Syncer Design

### Key Principle

The syncer replaces the trainer during sync. It does NOT run:
- PP/EP broadcasts (aggregation already happened on trainer pre-freeze)
- TP all-gather (same)
- HF conversion (same)

The syncer only does:
- H2D from memfd → VMM buffer (one bucket at a time)
- Existing IPC handoff to SGL (unchanged from today)

This is why its GPU peak is exactly **1× bucket_size + CUDA context** (~1.7 GB with default 512 MiB bucket).

### Data Handoff: Trainer → Syncer via memfd

The trainer runs `_hf_weight_iterator` at end of training (weights still on GPU) and D2H's each bucket's post-HF output into a memfd. This replaces TensorBackuper's role as the CPU-side source for sync.

Why memfd:
- Kernel-resident inode; survives trainer freeze (not in trainer's CUDA memory)
- No filesystem path needed (no cleanup)
- Syncer mmaps the same fd
- Pinnable via `cudaHostRegister` for DMA-efficient H2D on syncer side

Layout: contiguous per-bucket regions, same byte layout that `torch.cat(flat_parts, out=buf[:total_bytes])` would produce today. A manifest (list of `{name, shape, dtype, start_idx, end_idx}` per tensor per bucket) is written alongside.

### TensorBackuper's Role

TB has two duties:
1. **Tag swap** (actor/ref/teacher/old_actor) — `_switch_model("ref")` → `restore("ref")` puts ref weights on GPU. Still needed.
2. **Sync source** — after TMS unmaps the model weights from GPU, `weights_getter` returns `TB.get("actor")` to provide the pinned CPU copy as the sync source. Replaced by the memfd path.

In the TMS flow, duty (2) is essential: TMS unmaps the GPU pages, so `model.named_parameters()` would segfault. The only readable copy of the actor weights is TB's pinned CPU dict. In the per-syncer flow, duty (2) is replaced by the memfd: the trainer writes post-HF aggregated bytes into the memfd while weights are still on GPU (before freeze), so TB's CPU mirror is not needed as a sync source.

TB stays for duty (1). The sync path bypasses it by running `_hf_weight_iterator` on live GPU weights at end of training (before freeze), writing the output to the memfd. This is safe because the iterator runs while the trainer is still active and weights are still GPU-resident.

### Syncer Process Requirements

Per syncer rank (one per trainer rank, pinned to same GPU):
- torch.distributed world for intra-syncer gloo gather group (same topology as trainer's `_ipc_gather_group`)
- Ray actor handle to its corresponding rollout engine (via `ray.get_actor(name)`)
- memfd descriptor (received from trainer via control socket before trainer freezes)
- Pinned host bounce buffer ≥ max bucket bytes (allocated once at startup)

Does NOT need:
- Megatron imports or knowledge
- NCCL communicators for model-parallelism collectives
- Model state

### Lifecycle (Coordinator Driven)

1. End of training: trainer runs `_hf_weight_iterator`, D2Hs post-HF buckets into memfd, sends fd + manifest to syncer via control socket, acks.
2. Coordinator: `cr ckpt <trainer_pid>` (freeze trainer, ~all GPU freed).
3. Coordinator: `cr restore <syncer_pid>` (thaw syncer).
4. Syncer: per-bucket H2D from memfd → VMM buffer → existing IPC path to SGL → free. Calls `engine.continue_generation.remote()` on completion.
5. Coordinator: `cr ckpt <syncer_pid>` (freeze syncer).
6. Rollout proceeds on SGL with enlarged KV pool.

## Comparison: TMS-Offload vs Per-Syncer+GCR

### GPU during sync (non-SGL footprint)

| Component                 | TMS-offload (baseline) | Per-syncer + GCR |
|---|---|---|
| Megatron weights          | 0 (unmapped)           | 0 (frozen)       |
| Megatron optimizer        | 0 (unmapped)           | 0 (frozen)       |
| NCCL communicator state   | 1-3 GB (untagged)      | 0 (frozen)       |
| Megatron allocator slack  | 0.5-2 GB (untagged)    | 0 (frozen)       |
| CUDA context              | 1.2 GB                 | 1.2 GB (syncer)  |
| Bucket transient          | 2-3× bucket            | 1× bucket        |
| **Total**                 | **3-7 GB**             | **~1.7 GB**      |

The delta (1-5 GB) is what becomes available for KV cache during sync.

### GPU during rollout (non-SGL footprint → subtracted from KV pool)

| | TMS-offload | Per-syncer + GCR |
|---|---|---|
| Residual | 2.5-5 GB (NCCL + CUDA + allocator slack) | 0 (both trainer and syncer frozen) |

This is the primary win: 2.5-5 GB more KV cache throughout the entire rollout phase.

### Host bandwidth per sync step (W = full model HF bytes, N = ranks/node)

| | TMS-offload | Per-syncer + GCR |
|---|---|---|
| D2H | W (TB backup, sharded W/N per rank, end of train) | W (iterator → memfd, single rank or sharded) |
| H2D | W/N per rank (from TB pinned dict, parallel) | W/N per rank (if all-gather before hand to SGL) or W per rank (naive) |
| D2D | bucket_size per rank (flat_parts → VMM) | 0 (H2D goes directly into VMM buffer) |

With NCCL broadcast on the syncer side, per-GPU bus traffic is comparable. Without it (each syncer rank reads memfd independently), per-GPU H2D is N× worse due to host RAM bandwidth contention.

### Additional sync latency

| | TMS-offload | Per-syncer + GCR |
|---|---|---|
| TMS pause/resume | Sub-second (unmap/remap is driver-level) | N/A |
| GCR freeze trainer | N/A | Several seconds (DMA-bound, proportional to trainer state size) |
| GCR thaw syncer | N/A | Sub-second (syncer state is small) |
| GCR freeze syncer | N/A | Sub-second |
| **Total overhead** | **~0** | **Several seconds per RL step** |

Amortized over a rollout phase that typically dominates wall-clock, this is small. For fast RL steps (small models, short generations) it may be significant.

## Open Questions

1. **Pinning the memfd.** `cudaHostRegister` on the memfd mapping enables async DMA for the syncer's H2D. For large models the memfd may be too large to pin entirely; a fixed-size pinned bounce buffer (= max bucket bytes) is the fallback.

2. **Syncer NCCL.** *Decided: include an NCCL communicator on the syncer.* Without it, each syncer rank H2Ds the full W from memfd — with TP=8 that's 8× more PCIe traffic than necessary. An intra-syncer NCCL broadcast (rank 0 H2Ds W, broadcasts over NVLink to ranks 1-7) brings per-link PCIe back to W/8, matching TMS efficiency. The NCCL communicator costs ~50-150 MB per rank, which is subtracted from SGL's KV cache at init time, but this is small compared to the bandwidth savings (~7× W/8 PCIe avoided per sync step). Gloo remains for the IPC metadata gather group (CPU-side only, unchanged).

3. **GCR freeze latency.** The trainer state can be large (model + optimizer + activations). Freeze time is DMA-bound. For very fast RL steps this overhead may dominate. Amortization depends on rollout length.

4. **Quantization post-process.** *Resolved: no additional state needed.* The `post_process_weights` calls are pure RPCs to the rollout engine — they operate entirely on SGL's own model weights (e.g., Marlin unpack/repack for `compressed-tensors` quant). The syncer already holds Ray actor handles to the engines for the IPC handoff; it issues `engine.post_process_weights.remote(restore_weights_before_load=True)` before the bucket loop and `engine.post_process_weights.remote(post_process_quantization=True)` after. No Megatron imports or trainer state required.

5. **Distributed engines.** `_send_hf_params` also handles non-colocated engines via `update_weights_from_distributed`. The syncer would need equivalent capability if distributed rollout engines exist.
