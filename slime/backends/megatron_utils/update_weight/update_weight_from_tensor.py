import os
import threading
import uuid
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from sglang.srt.weight_sync.vmm_ipc import (
    alloc_vmm_buffer,
    free_vmm_buffer,
    open_sidecar_listener,
    send_fd,
    wrap_as_torch_uint8,
)

from slime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU tensors → VMM buffer → fd over UDS → gather_object(Gloo, metadata only) → Ray HTTP to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, vmm_resources = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            _cleanup_vmm_resources(vmm_resources)
            del hf_named_tensors

        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, vmm_resources = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, vmm_resources


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    device = torch.cuda.current_device()

    # 1. Flatten all tensors to a single uint8 buffer with per-tensor metadata.
    metadata = []
    flat_parts = []
    total_bytes = 0
    for name, tensor in hf_named_tensors:
        flat = tensor.flatten().view(torch.uint8)
        metadata.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "start_idx": total_bytes,
            "end_idx": total_bytes + flat.numel(),
        })
        flat_parts.append(flat)
        total_bytes += flat.numel()

    # 2. Allocate VMM buffer and copy flattened data.
    alloc = alloc_vmm_buffer(total_bytes, device)
    buf = wrap_as_torch_uint8(alloc)
    torch.cat(flat_parts, out=buf[:total_bytes])
    torch.cuda.synchronize(device)
    del flat_parts

    # 3. UDS listener -- serves the VMM fd to the SGLang consumer.
    uds_path = f"/tmp/slime-vmm-{uuid.uuid4()}.sock"
    listener = open_sidecar_listener(uds_path)

    def _serve():
        conn, _ = listener.accept()
        send_fd(conn, alloc.fd)
        conn.close()
        listener.close()

    fd_thread = threading.Thread(target=_serve, daemon=True)
    fd_thread.start()

    # 4. Gather lightweight metadata to src rank (not serialized tensors).
    device_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid!s}"
    my_info = {"uds_path": uds_path, "device_uuid": device_uuid, "buffer_size": alloc.size}
    gather_list = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(my_info, object_gather_list=gather_list, dst=ipc_gather_src, group=ipc_gather_group)

    # 5. Src rank calls the VMM endpoint on the engine.
    refs = []
    if dist.get_rank() == ipc_gather_src:
        refs.append(
            ipc_engine.update_weights_from_tensor_vmm.remote(
                uds_paths={g["device_uuid"]: g["uds_path"] for g in gather_list},
                buffer_sizes={g["device_uuid"]: g["buffer_size"] for g in gather_list},
                tensor_metadata=metadata,
                weight_version=str(weight_version),
            )
        )

    return refs, (fd_thread, alloc, buf, uds_path)


def _cleanup_vmm_resources(vmm_resources) -> None:
    if vmm_resources is None:
        return
    fd_thread, alloc, buf, uds_path = vmm_resources
    # Thread blocks on accept() until consumer connects; join waits for fd send.
    fd_thread.join(timeout=30)
    del buf
    free_vmm_buffer(alloc, close_fd=True)
    try:
        os.unlink(uds_path)
    except FileNotFoundError:
        pass
