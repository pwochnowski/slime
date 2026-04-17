# GPU Checkpoint/Restore Offload — Integration Design

Replace slime's offload primitives (`torch_memory_saver` in MT and SG, plus the
`ReloadableProcessGroup` Megatron patch) with a **whole-process GPU
checkpoint/restore (C/R)** mechanism applied to **both Megatron and SGLang** in
the co-located strategy.

## Constraint

The C/R mechanism hooks the CUDA VMM API. So in any given process, it cannot
coexist with another VMM-level offload mechanism:

- **In SG's process**: cannot coexist with SGLang's
  `release_memory_occupation` / `resume_memory_occupation` (same hook layer).
  SG must stay resident from its own perspective; reclamation comes only from
  the C/R freeze.
- **In MT's process**: cannot coexist with `torch_memory_saver`. So slime's
  `--offload-train` (saver-based) is removed, and the optimizer-state CPU
  offload that `torch_memory_saver` does implicitly has to be reimplemented.

## Legend

<table>
<tr>
  <td><span style="display:inline-block;width:1.2em;height:1.2em;background:#cfe8cf;border:1px solid #6aa66a;"></span></td>
  <td>steady state — the loop can rest here</td>
</tr>
<tr>
  <td><span style="display:inline-block;width:1.2em;height:1.2em;background:#fff;border:1px solid #888;"></span></td>
  <td>triggered action (RPC, allocator call, freeze, thaw, intra-MT collective)</td>
</tr>
<tr>
  <td><span style="display:inline-block;width:1.2em;height:1.2em;background:#e0eaff;border:1px solid #4a6dbf;"></span></td>
  <td>behavior change vs. today (planned flowchart only)</td>
</tr>
<tr>
  <td><span style="color:#0a7d2c;font-weight:bold;">✚</span></td>
  <td>new code</td>
</tr>
<tr>
  <td><span style="color:#a31515;font-weight:bold;">✖</span></td>
  <td>removed</td>
</tr>
</table>

GPU residency shading (per process):

<table>
<tr>
  <td><code>░░</code></td><td>frozen / paused (0 GB on GPU)</td>
</tr>
<tr>
  <td><code>▒▒░░</code></td><td>alive but minimal/CPU-only GPU footprint</td>
</tr>
<tr>
  <td><code>▒▒██</code></td><td>lightly resident (one bucket)</td>
</tr>
<tr>
  <td><code>▓▓</code></td><td>weights only (KV pool not loaded)</td>
</tr>
<tr>
  <td><code>████</code></td><td>fully resident</td>
</tr>
</table>

---

## Flowchart A — Today's Loop

Anchored at [train.py:73-103](../train.py#L73-L103). For each line in the loop,
what actually happens internally and the GPU state on each side.

<table style="border-collapse:collapse;width:100%;font-family:monospace;">
<tr><td style="background:#cfe8cf;border:1px solid #6aa66a;padding:8px;">
<strong>for rollout_id in range(start, num_rollout):</strong>
</td></tr>

<tr><td style="padding:6px 8px;color:#666;">(iter 0 only) <code>eval_before_train</code></td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>rollout_manager.generate(rollout_id)</strong><br/>
&nbsp;&nbsp;SG serves generations via Ray RPC<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>████</code> (model+KV+CG) &nbsp;&nbsp; MT=<code>░░</code> (saver-paused)
</td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>rollout_manager.offload()</strong> &nbsp;<em>[if offload_rollout]</em><br/>
&nbsp;&nbsp;<code>SG.release_memory_occupation()</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;· saver.pause inside SG: model + KV + CG pages released<br/>
&nbsp;&nbsp;&nbsp;&nbsp;· captured cudagraphs torn down<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>░░</code> &nbsp;&nbsp; MT=<code>░░</code>
</td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>actor_model.async_train(rollout_id, rollout_data_ref)</strong><br/>
&nbsp;&nbsp;<code>actor.train()</code> →<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>wake_up()</code>:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>torch_memory_saver.resume()</code> · pages remapped + saver CPU mirror restored<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>reload_process_groups()</code> · NCCL bootstrap (cost: latency + buffers)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>train(...)</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>weights_backuper.backup("actor")</code> · GPU weights → pinned CPU<br/>
&nbsp;&nbsp;<strong>STATE</strong> during train: SG=<code>░░</code> &nbsp;&nbsp; MT=<code>████</code> (model+opt+grad+act)
</td></tr>

<tr><td style="padding:6px 8px;color:#666;"><code>save(rollout_id)</code> &nbsp;<em>[periodic]</em></td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>offload_train(rollout_id)</strong> → <code>actor.sleep()</code><br/>
&nbsp;&nbsp;<code>clear_memory(clear_host_memory=True)</code><br/>
&nbsp;&nbsp;<code>destroy_process_groups()</code> · NCCL teardown (frees buffers)<br/>
&nbsp;&nbsp;<code>torch_memory_saver.pause()</code> · model+opt → saver CPU mirror; GPU pages released<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>░░</code> &nbsp;&nbsp; MT=<code>░░</code>
</td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>rollout_manager.onload_weights()</strong> &nbsp;<em>[if offload_rollout]</em><br/>
&nbsp;&nbsp;<code>SG.resume_memory_occupation(tags=[WEIGHTS])</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;· model pages remapped + content restored<br/>
&nbsp;&nbsp;&nbsp;&nbsp;· KV pool intentionally NOT restored (saves headroom for bucket)<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>▓▓</code> (weights only) &nbsp;&nbsp; MT=<code>░░</code>
</td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>actor_model.update_weights()</strong><br/>
&nbsp;&nbsp;<code>weight_updater.update_weights()</code> →<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>SG.flush_cache</code> + <code>pause_generation</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;for bucket in buckets:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MT bucket pinned-CPU → GPU<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;intra-MT NCCL: PP/EP bcast, TP all-gather<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>convert_to_hf</code>, <code>MultiprocessingSerializer</code>, gloo gather<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ray RPC → <code>SG.update_weights_from_tensor</code> → <code>load_weights</code> in place<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;del bucket; <code>ipc_collect</code><br/>
&nbsp;&nbsp;<strong>STATE</strong> during sync: SG=<code>▓▓</code> &nbsp;&nbsp; MT=<code>▒▒██</code> (1 bucket on GPU)
</td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>rollout_manager.onload_kv()</strong> &nbsp;<em>[if offload_rollout]</em><br/>
&nbsp;&nbsp;<code>SG.resume_memory_occupation(tags=[KV_CACHE])</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;· KV pool pages remapped (empty contents)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;· cudagraphs re-captured / restored as needed<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>████</code> &nbsp;&nbsp; MT=<code>░░</code>
</td></tr>

<tr><td style="padding:6px 8px;color:#666;">periodic eval</td></tr>
</table>

<blockquote style="border-left:4px solid #888;padding:8px 12px;background:#f5f5f5;margin-top:12px;">
<strong>Per cycle today, you pay:</strong>
<ul style="margin:4px 0 0 0;">
<li>2× NCCL teardown/rebuild (MT side: <code>destroy_process_groups</code> + <code>reload_process_groups</code>)</li>
<li>1× cudagraph teardown + 1× re-capture (SG side, on resume)</li>
<li>saver pause/resume work in both processes</li>
<li>maintenance of the <code>ReloadableProcessGroup</code> Megatron patch</li>
</ul>
</blockquote>

---

## Flowchart B — Planned Loop (C/R for both MT and SG)

Same `train.py` structure. Boxes shaded blue indicate behavior changes;
<span style="color:#0a7d2c;font-weight:bold;">✚</span> marks new code,
<span style="color:#a31515;font-weight:bold;">✖</span> marks removed code.

<table style="border-collapse:collapse;width:100%;font-family:monospace;">
<tr><td style="background:#cfe8cf;border:1px solid #6aa66a;padding:8px;">
<strong>for rollout_id in range(start, num_rollout):</strong>
</td></tr>

<tr><td style="padding:6px 8px;color:#666;">(iter 0 only) <code>eval_before_train</code></td></tr>

<tr><td style="border:1px solid #888;padding:8px;">
<strong>rollout_manager.generate(rollout_id)</strong> &nbsp;<em>[unchanged]</em><br/>
&nbsp;&nbsp;SG serves generations via Ray RPC<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>████</code> &nbsp;&nbsp; MT=<code>░░</code> (frozen)
</td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>rollout_manager.offload()</strong> &nbsp;<strong>[REPLACED]</strong><br/>
&nbsp;&nbsp;<code>SG.pause_generation</code><br/>
&nbsp;&nbsp;<code>SG.flush_cache</code> · KV contents zeroed (pool size unchanged)<br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <strong>C/R FREEZE SG</strong> · whole-proc freeze; all pages reclaimed<br/>
&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>release_memory_occupation</code>, NO graph teardown<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>░░</code> (frozen) &nbsp;&nbsp; MT=<code>░░</code> (frozen)
</td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>actor_model.async_train(rollout_id, rollout_data_ref)</strong> &nbsp;<strong>[PARTIAL]</strong><br/>
&nbsp;&nbsp;<code>actor.train()</code> →<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>wake_up()</code>:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <strong>C/R THAW MT</strong> · pages restored bit-identical<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; · NCCL communicators alive<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; · captured graphs alive<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> realloc optimizer GPU tensors + restore from pinned CPU<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>torch_memory_saver.resume</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>reload_process_groups</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>train(...)</code> &nbsp;<em>[unchanged]</em><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<code>weights_backuper.backup("actor")</code> &nbsp;<em>[unchanged]</em><br/>
&nbsp;&nbsp;<strong>STATE</strong> during train: SG=<code>░░</code> (frozen) &nbsp;&nbsp; MT=<code>████</code>
</td></tr>

<tr><td style="padding:6px 8px;color:#666;"><code>save(rollout_id)</code> &nbsp;<em>[unchanged]</em></td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>offload_train(rollout_id)</strong> → <code>actor.sleep()</code> &nbsp;<strong>[REPLACED]</strong><br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> optimizer state → pinned CPU<br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <code>del</code> optimizer GPU tensors<br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <code>model.zero_grad(set_to_none=True)</code><br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <code>torch.cuda.empty_cache()</code><br/>
&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>torch_memory_saver.pause</code><br/>
&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>destroy_process_groups</code> (NCCL bufs stay alive)<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>░░</code> (frozen) &nbsp;&nbsp; MT=<code>▒▒░░</code> (alive, mostly empty)<br/>
&nbsp;&nbsp;<em>MT weights live in pinned CPU; NCCL communicators stay resident</em>
</td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>rollout_manager.onload_weights()</strong> &nbsp;<strong>[REPLACED]</strong><br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <strong>C/R THAW SG</strong> · whole-proc thaw; full state restored<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; · model + KV pool (empty) + CG + NCCL all<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; bit-identical to pre-freeze<br/>
&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>resume_memory_occupation</code>; SG sees no change from its perspective<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>████</code> (KV empty) &nbsp;&nbsp; MT=<code>▒▒░░</code>
</td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>actor_model.update_weights()</strong> &nbsp;<em>[body unchanged]</em><br/>
&nbsp;&nbsp;<code>weight_updater.update_weights()</code> →<br/>
&nbsp;&nbsp;&nbsp;&nbsp;for bucket in buckets:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MT bucket pinned-CPU → GPU<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;intra-MT NCCL collectives <strong>(NO REBUILD — comms already alive)</strong><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>convert_to_hf</code>, serialize, gloo gather, Ray RPC, <code>load_weights</code><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;del bucket; <code>ipc_collect</code><br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> at end: <strong>C/R FREEZE MT</strong> · reclaims NCCL bufs + scratch + allocator slack<br/>
&nbsp;&nbsp;<strong>STATE</strong> during sync: SG=<code>████</code> (KV empty) &nbsp;&nbsp; MT=<code>▒▒██</code> (1 bucket on GPU)<br/>
&nbsp;&nbsp;<strong>STATE</strong> after sync: SG=<code>████</code> &nbsp;&nbsp; MT=<code>░░</code> (frozen)
</td></tr>

<tr><td style="border:1px solid #4a6dbf;background:#e0eaff;padding:8px;">
<strong>rollout_manager.onload_kv()</strong> &nbsp;<strong>[REPLACED]</strong><br/>
&nbsp;&nbsp;<span style="color:#a31515;font-weight:bold;">✖</span> NO <code>resume_memory_occupation(KV)</code> — KV pool already restored by thaw<br/>
&nbsp;&nbsp;<span style="color:#0a7d2c;font-weight:bold;">✚</span> <code>SG.continue_generation</code> · resume request serving<br/>
&nbsp;&nbsp;<strong>STATE</strong>: SG=<code>████</code> (KV will fill on rollout) &nbsp;&nbsp; MT=<code>░░</code>
</td></tr>

<tr><td style="padding:6px 8px;color:#666;">periodic eval</td></tr>
</table>

<blockquote style="border-left:4px solid #4a6dbf;padding:8px 12px;background:#f0f4ff;margin-top:12px;">
<strong>Per cycle in new design:</strong>
<ul style="margin:4px 0 0 0;">
<li>0× NCCL teardown/rebuild — saves bootstrap latency + buffer churn</li>
<li>0× cudagraph teardown / re-capture — saves capture cost on SG resume</li>
<li>0× <code>torch_memory_saver</code> pause/resume — mechanism removed in both processes</li>
<li>2× C/R freeze + 2× C/R thaw — cheap if your VMM-level reclaim is</li>
<li>still need: <code>TensorBackuper</code> for weights + new optimizer-CPU offload code (replaces saver auto-mirror)</li>
</ul>
</blockquote>

---

## Per-Step Diff

<table>
<thead>
<tr>
  <th align="left">train.py step</th>
  <th align="left">Today</th>
  <th align="left">With C/R</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>generate</code></td>
  <td>SG generates</td>
  <td>unchanged</td>
</tr>
<tr>
  <td><code>rollout_manager.offload()</code></td>
  <td><code>release_memory_occupation</code> (saver pause inside SG, graph teardown)</td>
  <td><code>flush_cache</code> + <code>pause_generation</code> + <strong>C/R FREEZE SG</strong></td>
</tr>
<tr>
  <td><code>actor_model.async_train()</code> (wake)</td>
  <td><code>saver.resume</code> + <code>reload_process_groups</code></td>
  <td><strong>C/R THAW MT</strong> + restore optimizer from pinned CPU</td>
</tr>
<tr>
  <td><code>actor_model.async_train()</code> (body + backup)</td>
  <td>train + <code>weights_backuper.backup</code></td>
  <td>unchanged</td>
</tr>
<tr>
  <td><code>offload_train()</code></td>
  <td><code>clear_memory</code> + <code>destroy_process_groups</code> + <code>saver.pause</code></td>
  <td><strong>app-level optimizer offload</strong> to pinned CPU; no saver, no destroy_pg</td>
</tr>
<tr>
  <td><code>rollout_manager.onload_weights()</code></td>
  <td><code>resume_memory_occupation(WEIGHTS)</code></td>
  <td><strong>C/R THAW SG</strong> (brings back model + empty KV + CG + NCCL bufs at once)</td>
</tr>
<tr>
  <td><code>actor_model.update_weights()</code></td>
  <td>bucket loop</td>
  <td>bucket loop + <strong>C/R FREEZE MT at end</strong></td>
</tr>
<tr>
  <td><code>rollout_manager.onload_kv()</code></td>
  <td><code>resume_memory_occupation(KV_CACHE)</code></td>
  <td>removed; replaced by <code>continue_generation</code></td>
</tr>
</tbody>
</table>

The `train.py` loop body itself doesn't change shape. What changes is what each
Ray-actor method does internally, and one operation (KV onload) becomes a no-op
replaced by a `continue_generation` call.

---

## Implementation Notes

### What still has to live in application code

- **`TensorBackuper` for weights** — still needed. Sync reads from a pinned-CPU
  source so MT can be lightly resident during the bucket loop without holding
  full-model GPU residency.
- **Optimizer state ↔ pinned CPU** — new code. Replaces what
  `torch_memory_saver.pause()`/`resume()` did implicitly via its internal CPU
  mirror. Touches optimizer construction in
  [slime/backends/megatron_utils/actor.py](../slime/backends/megatron_utils/actor.py).
- **Bucket sync loop** — unchanged
  ([update_weight_from_tensor.py](../slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py)).

### What can be deleted

- Slime's `ReloadableProcessGroup` Megatron patch
  ([docker/amd_patch/sglv0.5.0rc0/megatron.patch:211-228](../docker/amd_patch/sglv0.5.0rc0/megatron.patch)
  and equivalents under `docker/patch/`).
- `destroy_process_groups()` / `reload_process_groups()` calls in
  [actor.py:170](../slime/backends/megatron_utils/actor.py#L170) and
  [actor.py:184](../slime/backends/megatron_utils/actor.py#L184).
- All `torch_memory_saver` calls in MT
  ([actor.py:172](../slime/backends/megatron_utils/actor.py#L172),
  [actor.py:181](../slime/backends/megatron_utils/actor.py#L181)) and the
  saver-CPU-backup fallback in
  [common.py:133-139](../slime/backends/megatron_utils/update_weight/common.py#L133-L139).
- SGLang `release_memory_occupation` / `resume_memory_occupation` calls in
  [rollout.py:186](../slime/ray/rollout.py#L186),
  [rollout.py:196](../slime/ray/rollout.py#L196), and
  [rollout.py:308](../slime/ray/rollout.py#L308).

### Sizing constraints to set at startup (not at runtime)

- SGLang's `--mem-fraction-static` and KV pool size: chosen for the worst-case
  coexistence step (sync window: `SG_full + bucket`). This is the only sizing
  knob — no runtime KV resize.
- `--update_weight_buffer_size`: bucket size; trades sync iteration count
  against required headroom.

### Operational risks to verify

- **Ray actor heartbeats**: a frozen process won't respond. Push out
  `RAY_health_check_*` timeouts or proxy heartbeats from a side thread.
- **Quiescence at freeze**: ensure no in-flight CUDA work and no open
  cross-process IPC exports (`ipc_collect` before freezing MT).
- **Driver allocation accounting**: a frozen process may still be ledgered as
  holding VRAM. Either size the surviving process at startup, or confirm your
  C/R fully unmaps physical pages from the driver.
- **NCCL liveness across freeze**: intra-MT collectives must be quiescent at
  the freeze point. Slime is friendly here — the only MT↔SG channel is gloo
  (CPU) for IPC handle gather, plus Ray RPC. Intra-MT NCCL groups are entirely
  within the frozen set.
