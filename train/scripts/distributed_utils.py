"""
分布式训练相关的工具函数
"""
import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


def setup_distributed(args: argparse.Namespace) -> None:
    """设置分布式训练环境"""
    if not getattr(args, "use_fsdp", False):
        args.rank = 0
        args.world_size = 1
        if getattr(args, "local_rank", -1) == -1:
            args.local_rank = 0
        return

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP 模式需要可用的 CUDA 设备")

    env_rank = int(os.environ.get("RANK", 0))
    env_world_size = int(os.environ.get("WORLD_SIZE", 1))
    env_local_rank = int(os.environ.get(
        "LOCAL_RANK", getattr(args, "local_rank", 0)))

    args.rank = env_rank
    args.world_size = env_world_size
    args.local_rank = env_local_rank

    torch.cuda.set_device(args.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed(args: argparse.Namespace) -> None:
    """清理分布式训练环境"""
    if getattr(args, "use_fsdp", False) and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    """判断是否是主进程"""
    return (not getattr(args, "use_fsdp", False)) or getattr(args, "rank", 0) == 0


def broadcast_reference_state(
    reference_state: Dict[int, Dict[str, torch.Tensor]], args: argparse.Namespace
) -> Dict[int, Dict[str, torch.Tensor]]:
    """在分布式环境中广播参考状态"""
    if not getattr(args, "use_fsdp", False) or not dist.is_available() or not dist.is_initialized():
        return reference_state

    obj_list: List[Optional[Dict[int, Dict[str, torch.Tensor]]]] = [
        reference_state]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0] or {}


def distributed_reduce(values: List[float], device: torch.device, use_fsdp: bool) -> List[float]:
    """在分布式环境中对所有进程的值进行求和"""
    tensor = torch.tensor(values, device=device, dtype=torch.float32)
    if use_fsdp and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()
