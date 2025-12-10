"""
模型相关的工具函数
"""
from typing import Dict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from prismatic_vlms import ModelConfig, ModelRegistry
from train.vlm_train.models import PrismaticVLM


def load_config(model_id: str) -> ModelConfig:
    """根据模型 ID 加载配置"""
    for variant in ModelRegistry:
        if variant.model_id == model_id:
            return variant.value()
    raise ValueError(f"Unknown model id: {model_id}")


def unwrap_prismatic(vlm: PrismaticVLM) -> PrismaticVLM:
    """从 FSDP 包装中提取原始 PrismaticVLM 模型"""
    if isinstance(vlm, FSDP):
        return vlm._fsdp_wrapped_module  # type: ignore[attr-defined]
    return vlm


def extract_actor_state_from_full_state(
    model_state: Dict[str, torch.Tensor],
    vlm: PrismaticVLM,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """从完整模型状态字典中提取 Actor 状态"""
    actor_state: Dict[str, Dict[str, torch.Tensor]] = {}
    core_vlm = unwrap_prismatic(vlm)
    for layer_id in core_vlm.llm_backbone.pruning_actors.actors.keys():
        prefix = f"llm_backbone.pruning_actors.actors.{layer_id}."
        layer_state = {
            key[len(prefix):]: tensor
            for key, tensor in model_state.items()
            if key.startswith(prefix)
        }
        if layer_state:
            actor_state[str(layer_id)] = layer_state
    return actor_state
