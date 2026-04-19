from time import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


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

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    # Both MT and SG are alive here; update_weights establishes the MT↔SG NCCL group.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    # Enter Phase C (rollout): freeze MT, SG stays alive.
    if args.colocate:
        actor_model.gcr_suspend()
        if critic_model is not None:
            critic_model.gcr_suspend()

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
        t0 = time()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        actor_model.add_timer("rollout_generate", time() - t0)

        # Phase C → A: freeze SG, thaw MT
        if args.colocate:
            t0 = time()
            ray.get(rollout_manager.gcr_suspend.remote())
            actor_model.gcr_resume()
            if critic_model is not None:
                critic_model.gcr_resume()
            actor_model.add_timer("phase_transition_to_train", time() - t0)

        # Phase A: train (MT alive, SG frozen)
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        # Phase A → B: thaw SG (MT stays alive)
        if args.colocate:
            t0 = time()
            ray.get(rollout_manager.gcr_resume.remote())
            actor_model.add_timer("phase_transition_to_sync", time() - t0)

        # Phase B: update weights (both alive)
        if not args.critic_train_only:
            actor_model.update_weights()

        # Phase B → C: freeze MT
        if args.colocate:
            t0 = time()
            actor_model.gcr_suspend()
            if critic_model is not None:
                critic_model.gcr_suspend()
            actor_model.add_timer("phase_transition_to_rollout", time() - t0)
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
