import os
from time import time

os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import ray

from gcr import device_mem_get_info, _gpu_processes
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


def log_phase_memory(label: str, num_devices: int):
    G = 2**30
    lines = [f"[phase-mem] {label}"]
    for dev in range(num_devices):
        free, total = device_mem_get_info(dev)
        lines.append(f"  gpu {dev}: free={free / G:.2f}G total={total / G:.2f}G")
    for proc in _gpu_processes():
        lines.append(f"  {proc}")
    print("\n".join(lines), flush=True)


MT_MODEL_TAG= 3
MT_OPTIM_TAG= 4
SGL_KV_CACHE_TAG = 1
SGL_WEIGHTS_TAG = 2

def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # Offload KV cache before Megatron allocates its param buffers.
    if args.colocate:
        ray.get(rollout_manager.gcr_offload_tag.remote([SGL_KV_CACHE_TAG]))

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    # Both MT and SG are alive here; update_weights establishes the MT↔SG NCCL group.
    if not args.critic_train_only:
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))
        # actor_model.dump_segments("before initial gcr_offload_tag(MT_MODEL_OPTIM)")
        actor_model.gcr_offload_tag([MT_MODEL_TAG]);
        actor_model.update_weights()

    # Enter Phase C (rollout): freeze MT, SG stays alive.
    if args.colocate:
        actor_model.log_memory("before initial gcr_suspend")
        actor_model.gcr_suspend()
        if critic_model is not None:
            critic_model.gcr_suspend()
        ray.get(rollout_manager.gcr_restore_tag.remote([SGL_KV_CACHE_TAG]))
        ray.get(rollout_manager.flush_engines_cache.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        # Phase C: generate (MT frozen, SG alive)
        print(f"[driver] entering Phase C: generate(rollout_id={rollout_id})", flush=True)
        if args.colocate:
            log_phase_memory(f"start_of_rollout iter={rollout_id}", args.rollout_num_gpus)
        t0 = time()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        actor_model.add_timer("rollout_generate", time() - t0)

        # Phase C → A: freeze SG, thaw MT
        if args.colocate:
            t0 = time()
            print(f"[driver] phase C→A: triggering rollout gcr_suspend", flush=True)
            ray.get(rollout_manager.gcr_suspend.remote())
            print(f"[driver] phase C→A: rollout gcr_suspend done ({time() - t0:.1f}s), restoring actor", flush=True)
            actor_model.gcr_resume()
            if critic_model is not None:
                critic_model.gcr_resume()
            if rollout_id == 0:
                actor_model.init_optimizer_states()
            actor_model.add_timer("phase_transition_to_train", time() - t0)

        # Phase A: train (MT alive, SG frozen)
        if args.colocate:
            log_phase_memory(f"start_of_train iter={rollout_id}", args.rollout_num_gpus)
            actor_model.log_memory(f"iter {rollout_id} before train")
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        # Phase A → B: offload MT (model + optimizer) data, thaw SG (KV cache restore after weight sync)
        if args.colocate:
            actor_model.log_memory(f"iter {rollout_id} after train")
            t0 = time()
            actor_model.gcr_offload_tag([MT_MODEL_TAG, MT_OPTIM_TAG])
            ray.get(rollout_manager.gcr_resume.remote([SGL_KV_CACHE_TAG]))
            actor_model.add_timer("phase_transition_to_sync", time() - t0)

        # Phase B: update weights (both alive, KV cache still offloaded)
        if args.colocate:
            actor_model.log_memory(f"iter {rollout_id} before update_weights (phase B)")
        if not args.critic_train_only:
            print(f"[driver] Phase B: update_weights start (rollout_id={rollout_id})", flush=True)
            actor_model.update_weights()
            print(f"[driver] Phase B: update_weights done (rollout_id={rollout_id})", flush=True)
        # Phase B → C: freeze MT before restoring KV cache (KV cache is ~80% of GPU)
        if args.colocate:
            print(f"[driver] Phase B→C: gcr_suspend start (rollout_id={rollout_id})", flush=True)
            t0 = time()
            actor_model.gcr_suspend()
            if critic_model is not None:
                critic_model.gcr_suspend()
            actor_model.add_timer("phase_transition_to_rollout", time() - t0)
            print(f"[driver] Phase B→C: gcr_suspend done (rollout_id={rollout_id})", flush=True)
            ray.get(rollout_manager.gcr_restore_tag.remote([SGL_KV_CACHE_TAG]))
            ray.get(rollout_manager.flush_engines_cache.remote())
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

        # Eval (SG alive, MT frozen)
        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            t0 = time()
            ray.get(rollout_manager.eval.remote(rollout_id))
            actor_model.add_timer("eval", time() - t0)

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
