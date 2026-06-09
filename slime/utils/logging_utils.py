import logging
import os
import warnings

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


def _suppress_external_noise():
    """Set env vars and warning filters to silence noisy external library output."""
    # Gloo distributed backend connection spam
    os.environ.setdefault("GLOO_LOG_LEVEL", "ERROR")
    # PyTorch C++ warnings (NCCL unbatched P2P, ProcessGroupNCCL, etc.)
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

    # Suppress Python warnings in spawned subprocesses (e.g. SGLang engine)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")

    # Python warnings in the current process
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*MimoModelConfig is experimental.*")
    warnings.filterwarnings("ignore", message=".*ORJSONResponse is deprecated.*")

    # Silence noisy sglang loggers (model import errors, MoE kernel config)
    for name in [
        "sglang.srt.models.registry",
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    _suppress_external_noise()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
