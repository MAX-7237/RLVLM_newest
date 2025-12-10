"""
训练循环相关的工具函数
"""
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from train.vlm_train.models import PrismaticVLM

from train.scripts.distributed_utils import distributed_reduce
from train.scripts.utils import move_to_device
from train.vlm_train.vqav2 import VQAIndex, compute_accuracy
# 尝试导入 overwatch
try:
    from vlm_train.overwatch import initialize_overwatch
    overwatch = initialize_overwatch(__name__)
except ImportError:
    overwatch = None


def train(
    vlm: PrismaticVLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    train_questions: Path,
    train_annotations: Path,
    clip_eps: float,
    advantage_alpha: float,
    reward_alpha1: float,
    reward_alpha2: float,
    gumbel_tau: float,
    log_file: Optional[Path] = None,
    sampler: Optional[DistributedSampler] = None,
    use_fsdp: bool = False,
    is_rank_zero: bool = True,
) -> None:
    """
    训练 RL Actor（使用多选题准确率作为奖励）
    """
    # ===== Debug 开关 =====
    DEBUG = True                 # 总开关
    DEBUG_INIT = True            # 是否在训练开始时打印一次 backbone 配置
    DEBUG_MAX_STEPS = 5          # 只在第 0 个 epoch 的前 DEBUG_MAX_STEPS 个 step 打详细 log

    backbone = vlm.llm_backbone

    # 一次性打印一些轻量级配置，不打印 state_dict 这种巨大的东西
    if DEBUG and DEBUG_INIT and is_rank_zero:
        print("====[RLVLM Debug] Backbone RL config ====")
        print("pruning_actor_layers:",
              getattr(backbone, "pruning_actor_layers", None))
        print("pruning_actor_hidden_dim:",
              getattr(backbone, "pruning_actor_hidden_dim", None))
        print("pruning_actor_num_samples:",
              getattr(backbone, "pruning_actor_num_samples", None))
        print("reward_alpha1:", getattr(backbone, "reward_alpha1", None))
        print("reward_alpha2:", getattr(backbone, "reward_alpha2", None))
        print("gumbel_tau:", getattr(backbone, "gumbel_tau", None))
        print("advantage_alpha:", getattr(backbone, "advantage_alpha", None))
        print("=========================================")

    def log_info(msg: str, ctx_level: Optional[int] = None):
        if not is_rank_zero:
            return
        if overwatch:
            overwatch.info(msg, ctx_level=ctx_level or 0)
        else:
            print(msg)

    vqa_index = VQAIndex(train_questions, train_annotations)

    for epoch in range(epochs):
        cumulative_correct = 0.0
        cumulative_samples = 0

        log_info(f"开始训练轮次 {epoch+1}/{epochs}")
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            batch_qids = batch["question_id"].tolist()
            batch_pred = vlm.model.generate_batch(
                pixel_values=batch["pixel_values"],
                texts=batch["question_prompt"],
                max_new_tokens=8,
                do_sample=False,
            )
            batch_accuracy = compute_accuracy(
                batch_qids, batch_pred, vqa_index)
            R = torch.tensor(batch_accuracy, device=device)

            # ---- 只在前几个 step 打印一小部分 debug 信息 ----
            if (
                DEBUG
                and is_rank_zero
                and epoch == 0
                and step <= DEBUG_MAX_STEPS
            ):
                print("====[RLVLM Debug] Batch preds & qids ====")
                print("len(batch_pred):", len(batch_pred))
                print("batch_pred[:3]:", batch_pred[:3])
                print("len(batch_qids):", len(batch_qids))
                print("batch_qids[:3]:", batch_qids[:3])
                print("batch_accuracy:", batch_accuracy)
                print("ground truth:", batch["answer"][:3])
                print("=========================================")

            # ---- RL 统计信息 ----
            stats = backbone.get_rl_statistics()
            if not stats["log_prob"]:
                raise RuntimeError(
                    "RL statistics are empty; ensure pruning actors are enabled and rl_training_mode is active."
                )

            total_loss = torch.tensor(0.0, device=device)

            for layer_id, log_prob in stats["log_prob"].items():
                ref_log_prob = stats["ref_log_prob"][layer_id]
                # adv1: GRPO 组间优势函数（已经在 _rl_select_pruning_mask 中计算）
                # 现在形状是 [num_samples, batch]
                adv1 = stats["sampling_advantage"][layer_id]
                prune_ratio = stats["prune_ratio"][layer_id]  # [batch]
                prune_mask = stats["prune_mask"][layer_id].to(
                    dtype=log_prob.dtype, device=log_prob.device
                )

                num_samples = adv1.shape[0]

                # adv2: 每一层的剪枝率 * R（答案正确与否）
                # prune_ratio: [batch] -> [num_samples, batch]
                adv2 = (prune_ratio * R).unsqueeze(0).expand(num_samples, -1)

                # 最终优势函数：adv = alpha*adv1 + (1-alpha)*adv2
                advantage = (
                    advantage_alpha * adv1 + (1 - advantage_alpha) * adv2
                )  # [num_samples, batch]

                # 每个样本图像 token 数（做平均用）
                token_count = stats["image_mask"].sum(
                    dim=-1).to(dtype=log_prob.dtype)  # [batch]

                delta_log_prob = log_prob - ref_log_prob
                ratio = torch.exp(delta_log_prob)  # [num_samples, batch]
                clipped_ratio = torch.clamp(
                    ratio, 1 - clip_eps, 1 + clip_eps
                )  # [num_samples, batch]

                # 每一组的策略比值乘以本组的优势函数
                policy_loss_per_sample = -torch.min(
                    ratio * advantage, clipped_ratio * advantage
                )  # [num_samples, batch]
                # 对所有组求和（而不是平均）
                layer_loss = policy_loss_per_sample.sum()  # 标量
                total_loss = total_loss + layer_loss

                # ---- 只在前几个 step 对每一层打详细统计 ----
                if (
                    DEBUG
                    and is_rank_zero
                    and epoch == 0
                    and step <= DEBUG_MAX_STEPS
                ):
                    print(f"====[RLVLM Debug] Layer {layer_id} ====")
                    print("log_prob.shape:", log_prob.shape)
                    print("ref_log_prob.shape:", ref_log_prob.shape)
                    print(
                        "adv1.shape:",
                        adv1.shape,
                        "mean:",
                        adv1.mean().item(),
                        "min:",
                        adv1.min().item(),
                        "max:",
                        adv1.max().item(),
                    )
                    print(
                        "prune_ratio.shape:",
                        prune_ratio.shape,
                        "mean:",
                        prune_ratio.mean().item(),
                    )
                    print(
                        "初始的图像token矩阵的形状:",
                        stats["image_mask"].shape,
                        "每张图像的token数目:",
                        token_count[0].item(),
                        "生成的剪枝掩码矩阵的形状:",
                        stats["prune_mask"][layer_id].shape,
                    )
                    print(
                        "新旧策略的比值的形状:",
                        ratio.shape,
                        "新旧策略的比值的平均值:",
                        ratio.mean().item(),
                    )
                    print(
                        "advantage.shape:",
                        advantage.shape,
                        "mean:",
                        advantage.mean().item(),
                    )
                    print("layer_loss:", layer_loss.item())
                    print("=========================================")

            if (
                DEBUG
                and is_rank_zero
                and epoch == 0
                and step <= DEBUG_MAX_STEPS
            ):
                print("====[RLVLM Debug] total_loss ====")
                print("total_loss:", total_loss.item())
                print("total_loss.requires_grad:", total_loss.requires_grad)
                # 再看一下某一层的 log_prob
                for layer_id, log_prob in stats["log_prob"].items():
                    print(f"layer {layer_id} log_prob.requires_grad:",
                          log_prob.requires_grad)
                    print(f"layer {layer_id} ref_log_prob.requires_grad:",
                          stats["ref_log_prob"][layer_id].requires_grad)
                    print(f"layer {layer_id} advantage.requires_grad:",
                          stats["sampling_advantage"][layer_id].requires_grad)
                    break  # 打印一个 layer 就够
                print("=========================================")

            total_loss.backward()
            optimizer.step()

            # 清理 CUDA 缓存以释放内存
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ====== 剪枝率与 token 统计（summary 级别日志） ======
            if step % 10 == 0:
                image_masks = stats["image_mask"]
                if image_masks is not None:
                    base_mask = image_masks.to(dtype=torch.float32)
                    original_token_counts = (
                        base_mask.sum(dim=-1)
                        .clamp_min(1)
                        .to(dtype=torch.float32)
                    )
                    cumulative_mask = torch.ones_like(
                        base_mask,
                        dtype=torch.float32,
                        device=base_mask.device,
                    )
                    for layer_id in sorted(
                        backbone.latest_pruning_masks.keys()
                    ):
                        layer_mask = backbone.latest_pruning_masks[
                            layer_id
                        ].to(
                            dtype=torch.float32,
                            device=base_mask.device,
                        )
                        cumulative_mask = cumulative_mask * layer_mask
                    remaining_tokens = (cumulative_mask * base_mask).sum(
                        dim=-1
                    )
                    total_prune_ratio = 1.0 - (
                        remaining_tokens / original_token_counts
                    )

                    metric_values = [
                        original_token_counts.sum().item(),
                        float(original_token_counts.numel()),
                        remaining_tokens.sum().item(),
                        float(remaining_tokens.numel()),
                        total_prune_ratio.sum().item(),
                        float(total_prune_ratio.numel()),
                    ]
                else:
                    metric_values = [0.0] * 6

                (
                    orig_sum,
                    orig_count,
                    remain_sum,
                    remain_count,
                    prune_sum,
                    prune_count,
                ) = distributed_reduce(metric_values, device, use_fsdp)

                avg_original_tokens = (
                    orig_sum / orig_count if orig_count > 0 else 0.0
                )
                avg_remaining_tokens = (
                    remain_sum / remain_count if remain_count > 0 else 0.0
                )
                avg_total_prune_ratio = (
                    prune_sum / prune_count if prune_count > 0 else 0.0
                )

                # 额外打印一下当前 batch 的平均剪枝率（本地），方便对比
                if DEBUG and is_rank_zero and image_masks is not None:
                    print(
                        f"[Debug] Local batch prune_ratio.mean = "
                        f"{total_prune_ratio.mean().item():.4f}"
                    )

                log_msg = (
                    f"[RLVLM][Epoch {epoch+1}] Step {step} "
                    f"Accuracy={batch_accuracy:.4f}  "
                    f"PruneRatio={avg_total_prune_ratio:.4f}  "
                    f"OrigTokens={avg_original_tokens:.2f}  "
                    f"RemainTokens={avg_remaining_tokens:.2f}  "
                    f"Loss={total_loss.item():.4f}"
                )
                log_info(log_msg, ctx_level=1)

                if log_file:
                    layer_ratios = ", ".join(
                        f"L{layer_id}:{stats['prune_ratio'][layer_id].mean().item():.4f}"
                        for layer_id in stats["prune_ratio"]
                    )
                    with log_file.open("a") as f:
                        f.write(
                            f"{epoch+1},{step},{batch_accuracy:.6f},"
                            f"{avg_total_prune_ratio:.6f},"
                            f"{total_loss.item():.6f},"
                            f"{avg_original_tokens:.2f},"
                            f"{avg_remaining_tokens:.2f},"
                            f"{layer_ratios}\n"
                        )
