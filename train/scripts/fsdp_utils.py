"""
FSDP (Fully Sharded Data Parallel) 相关的工具函数
"""
import argparse
from functools import partial
from typing import Optional, Tuple, Callable

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from prismatic_vlms.prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from train.vlm_train.models import PrismaticVLM


def get_fsdp_wrapping_policy(vlm: PrismaticVLM) -> Callable:

    vision_fsdp_wrapping_policy = vlm.vision_backbone.get_fsdp_wrapping_policy()
    llm_fsdp_wrapping_policy = vlm.llm_backbone.get_fsdp_wrapping_policy()

    # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
    prismatic_fsdp_wrapping_policy = partial(
        _module_wrap_policy,
        module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
    )

    # Return union (_or_) over constituent policies
    #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
    #            automatically be folded into the root VLM FSDP instance.
    return partial(
        _or_policy,
        policies=[
            vision_fsdp_wrapping_policy,
            llm_fsdp_wrapping_policy,
            prismatic_fsdp_wrapping_policy,
        ],
    )


def wrap_vlm_with_fsdp(
    vlm: PrismaticVLM,
    args: argparse.Namespace,
) -> Tuple[PrismaticVLM, Optional[FullStateDictConfig]]:
    """使用 FSDP 包装 VLM 模型"""
    if not getattr(args, "use_fsdp", False):
        return vlm, None

    if args.fsdp_sharding == "fsdp-shard-grad-op":
        sharding = ShardingStrategy._HYBRID_SHARD_ZERO2
    elif args.fsdp_sharding == "fsdp-full-shard":
        sharding = ShardingStrategy.FULL_SHARD
    else:
        raise ValueError(f"未知的 FSDP sharding 策略: {args.fsdp_sharding}")

    reduce_dtype = torch.float32 if args.fsdp_reduce_in_fp32 else torch.bfloat16
    if args.disable_fsdp_mixed_precision:
        mixed_precision = MixedPrecision(
            param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
    else:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype, buffer_dtype=reduce_dtype
        )

    wrapping_policy = get_fsdp_wrapping_policy(vlm)

    # debug
    # print(">>> type(vlm) =", type(vlm))
    # print(">>> is nn.Module:", isinstance(vlm, nn.Module))
    # print(">>> is FSDP:", isinstance(vlm, FSDP))
    # for name, module in vlm.vision_backbone.named_modules():
    #     if isinstance(module, FSDP):
    #         print("[DEBUG] Already FSDP wrapped module:", name, type(module))

    fsdp_model = FSDP(
        vlm,
        # 先不用，试试看，后面再加上auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    if not args.disable_fsdp_activation_checkpoint:
        transformer_layer_cls = vlm.llm_backbone.transformer_layer_cls
        if transformer_layer_cls is not None:
            non_reentrant_wrapper = partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )

            def _check_fn(module: torch.nn.Module) -> bool:
                return isinstance(module, transformer_layer_cls)

            apply_activation_checkpointing(
                fsdp_model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=_check_fn,
            )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    return fsdp_model, state_cfg
