"""
utils.py

Utility functions for training.
"""

from pathlib import Path
from typing import Dict, Iterable, List
import torch
from train.vlm_train.models import PrismaticVLM


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """批次整理函数（支持固定长度的批次）"""
    # 如果所有样本长度相同，直接堆叠
    if len(set(item["input_ids"].size(0) for item in batch)) == 1:
        out: Dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            # 跳过非 Tensor 字段（如 question_prompt 等字符串）
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([item[key] for item in batch], dim=0)
        return out

    # 否则进行填充（用于 AI2D 数据集）
    max_len = max(item["input_ids"].size(0) for item in batch)
    has_image_token_mask = "image_token_mask" in batch[0]
    batch_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    if has_image_token_mask:
        batch_dict["image_token_mask"] = []

    # 处理 pixel_values（可能是 tensor 或 dict）
    if isinstance(batch[0]["pixel_values"], torch.Tensor):
        batch_dict["pixel_values"] = torch.stack(
            [item["pixel_values"] for item in batch])
    else:
        batch_dict["pixel_values"] = {
            k: torch.stack([item["pixel_values"][k] for item in batch])
            for k in batch[0]["pixel_values"].keys()
        }

    # 填充序列
    pad_token_id = 0
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        batch_dict["input_ids"].append(
            torch.cat([item["input_ids"], torch.full(
                (pad_len,), pad_token_id, dtype=torch.long)])
        )
        batch_dict["attention_mask"].append(
            torch.cat([item["attention_mask"], torch.zeros(
                pad_len, dtype=torch.long)])
        )
        batch_dict["labels"].append(
            torch.cat([item["labels"], torch.full(
                (pad_len,), -100, dtype=torch.long)])
        )
        if has_image_token_mask:
            image_mask = item["image_token_mask"]
            batch_dict["image_token_mask"].append(
                torch.cat(
                    [
                        image_mask,
                        torch.zeros(pad_len, dtype=image_mask.dtype)
                    ]
                )
            )

        # answer_label 不需要填充
        if "answer_label" in item:
            if "answer_label" not in batch_dict:
                batch_dict["answer_label"] = []
            batch_dict["answer_label"].append(item["answer_label"])

    # 堆叠张量
    batch_dict["input_ids"] = torch.stack(batch_dict["input_ids"])
    batch_dict["attention_mask"] = torch.stack(batch_dict["attention_mask"])
    batch_dict["labels"] = torch.stack(batch_dict["labels"])
    if has_image_token_mask:
        batch_dict["image_token_mask"] = torch.stack(
            batch_dict["image_token_mask"])
    if "answer_label" in batch_dict:
        batch_dict["answer_label"] = torch.stack(batch_dict["answer_label"])

    return batch_dict


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """将批次数据移动到指定设备"""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {sub_k: sub_v.to(device) for sub_k, sub_v in v.items()}
        else:
            result[k] = v
    return result


def collect_actor_parameters(vlm: PrismaticVLM) -> Iterable[torch.nn.Parameter]:
    """收集所有 pruning actor 的参数"""
    for actor in vlm.llm_backbone.pruning_actors.values():
        yield from actor.parameters()


def load_reference_state(checkpoint_path: Path, vlm: PrismaticVLM) -> Dict[int, Dict[str, torch.Tensor]]:
    """加载参考检查点的 actor 状态"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        vlm.load_state_dict(ckpt["model_state_dict"], strict=False)

    actor_state: Dict[int, Dict[str, torch.Tensor]] = {}
    raw_actor_state = ckpt.get("actor_state_dict")
    if isinstance(raw_actor_state, dict):
        for layer_key, layer_state in raw_actor_state.items():
            try:
                layer_id = int(layer_key)
            except (TypeError, ValueError):
                continue
            if isinstance(layer_state, dict):
                actor_state[layer_id] = layer_state
        if actor_state:
            return actor_state

    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", {}))
    for layer_id in vlm.llm_backbone.pruning_actors.actors.keys():
        prefix = f"pruning_actors.actors.{layer_id}."
        sub = {
            name[len(prefix):]: tensor
            for name, tensor in state_dict.items()
            if name.startswith(prefix)
        }
        if sub:
            actor_state[layer_id] = sub
    return actor_state
