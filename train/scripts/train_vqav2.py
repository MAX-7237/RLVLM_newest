import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from train.scripts.utils import (
    collect_actor_parameters,
    load_reference_state,
)

from distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    broadcast_reference_state,
)
from fsdp_utils import wrap_vlm_with_fsdp
from model_utils import load_config, extract_actor_state_from_full_state
from train_utils import train
from train.vlm_train.models import load_vlm
from train.vlm_train.vqav2 import VQAv2MapDataset

# 尝试导入 overwatch
try:
    from train.vlm_train.overwatch import initialize_overwatch
    overwatch = initialize_overwatch(__name__)
except ImportError:
    overwatch = None

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Training pruning actors (GRPO-style) with VQA-v2 dataset."
    )
    parser.add_argument("--model-id", default="reproduction-llava-v15+7b")

    # 数据集相关参数
    parser.add_argument("--dataset-family", type=str, default="vqa-v2")
    parser.add_argument("--data-root", type=Path,
                        default="/home/ubuntu/vlm_prune/RLVLM/datasets_train")
    parser.add_argument("--index-file", type=str,
                        default="index_files/vqa-v2/metadata.json")
    parser.add_argument("--train-questions", type=Path,
                        default="/home/ubuntu/vlm_prune/RLVLM/datasets_train/index_files/vqa-v2/questions-vqa-v2-full.json")
    parser.add_argument("--train-annotations", type=Path,
                        default="/home/ubuntu/vlm_prune/RLVLM/datasets_train/index_files/vqa-v2/annotations-vqa-v2-full.json")

    # 训练相关参数
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--actor-lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=1)

    # 奖励相关参数
    parser.add_argument("--pruning-actor-layers", type=tuple, default=[3, 6],
                        help="Pruning actor layers")
    parser.add_argument("--pruning-actor-hidden-dim", type=int, default=128,
                        help="Pruning actor hidden dim")
    parser.add_argument("--pruning-actor-num-samples", type=int, default=5,
                        help="Pruning actor num samples")
    parser.add_argument("--clip-eps", type=float,
                        default=0.1, help="GRPO clip epsilon")
    parser.add_argument("--advantage-alpha", type=float,
                        default=0.9, help="alpha*adv1 + (1-alpha)*adv2")
    parser.add_argument("--reward-alpha1", type=float,
                        default=10, help="Cross attention weight")
    parser.add_argument("--reward-alpha2", type=float,
                        default=10, help="Self attention weight")
    parser.add_argument("--gumbel-tau", type=float,
                        default=2.0, help="Gumbel temperature")

    # 输出文件路径
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/ubuntu/vlm_prune/RLVLM/train/checkpoints"), help="Output directory for checkpoint")

    # FSDP/分布式相关参数
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument(
        "--fsdp-sharding", choices=["fsdp-shard-grad-op", "fsdp-full-shard"], default="fsdp-full-shard")
    parser.add_argument("--fsdp-reduce-in-fp32", action="store_true")
    parser.add_argument("--disable-fsdp-mixed-precision",
                        action="store_true", help="关闭 FSDP BF16 混合精度，使用 FP32 训练")
    parser.add_argument("--disable-fsdp-activation-checkpoint",
                        action="store_true", help="关闭 FSDP 激活检查点")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="torchrun 自动注入的 local rank 参数")

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    setup_distributed(args)
    rank_zero = is_main_process(args)

    # 定义日志函数
    def log_info(msg: str, ctx_level: Optional[int] = None):
        if not rank_zero:
            return
        if overwatch:
            overwatch.info(msg, ctx_level=ctx_level or 0)
        else:
            print(msg)

    # 定义日志文件路径
    log_info(f"开始 RL 训练，模型: {args.model_id}，使用 VQA-v2 训练数据集")
    log_file_path = Path(__file__).with_name("vqa_v2_training.log")
    if rank_zero:
        log_file_path.write_text(
            "epoch,step,accuracy,total_prune_ratio,loss,orig_image_tokens,remaining_image_tokens,layer_prune_ratios\n"
        )
    else:
        log_file_path = None

    # 加载模型配置
    cfg = load_config(args.model_id)
    if getattr(cfg, "pruning_actor_layers", None) in (None, ()):
        cfg.pruning_actor_layers = (3)
        log_info(f"使用默认 pruning actor 层: {cfg.pruning_actor_layers}")

    # 设置训练设备
    if args.use_fsdp:
        train_device = torch.device(f"cuda:{args.local_rank}")
        model_device = torch.device("cpu")
    else:
        train_device = torch.device("cuda:7")
        model_device = train_device
    log_info(f"使用设备: {train_device}")

    # 加载模型（需要额外的组件）
    vlm = load_vlm(cfg.model_family, cfg.model_id, cfg.run_dir,
                   hf_token=hf_token)
    image_transform = vlm.vision_backbone.image_transform

    if args.reference_checkpoint:
        log_info(f"加载初始检查点: {args.reference_checkpoint}", ctx_level=1)
        reference_state = load_reference_state(args.reference_checkpoint, vlm)
    else:
        log_info("未提供初始检查点，即将使用正交初始化参数 RL Actor 作为训练起点", ctx_level=1)
        reference_state = {}
    reference_state = broadcast_reference_state(reference_state, args)

    # 设置layer、hidden_dim、num_samples、reward和gumbel参数
    vlm.llm_backbone.pruning_actor_layers = args.pruning_actor_layers
    vlm.llm_backbone.pruning_actor_hidden_dim = args.pruning_actor_hidden_dim
    vlm.llm_backbone.pruning_actor_num_samples = args.pruning_actor_num_samples
    vlm.llm_backbone.reward_alpha1 = args.reward_alpha1
    vlm.llm_backbone.reward_alpha2 = args.reward_alpha2
    vlm.llm_backbone.gumbel_tau = args.gumbel_tau
    vlm.llm_backbone.advantage_alpha = args.advantage_alpha
    # 启用 pruning RL剪枝
    vlm.llm_backbone.enable_pruning_rl(reference_state)
    # 如果reference_state不为空，则使用reference_state初始化pruning_actors
    # 如果reference_state为空，则在enable_pruning_rl函数中使用正交初始化参数 RL Actor 作为训练起点
    fsdp_state_cfg: Optional[FullStateDictConfig] = None
    if args.use_fsdp:
        vlm, fsdp_state_cfg = wrap_vlm_with_fsdp(vlm, args)
        vlm.train()
    else:
        vlm.to(train_device)

    # image_processor 就是 image_transform
    image_processor = image_transform
    # 获取 prompt_fn
    prompt_fn = vlm.get_prompt_fn(args.dataset_family)

    log_info(f"加载训练数据: {args.index_file}", ctx_level=1)
    dataset = VQAv2MapDataset(
        root_dir=args.data_root,
        index_file=Path(args.index_file),
        prompt_fn=prompt_fn,
        image_processor=image_processor
    )
    log_info(f"数据集大小: {len(dataset)} 个样本")

    if args.use_fsdp and dist.is_available() and dist.is_initialized():
        sampler: Optional[DistributedSampler] = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_batch,
            sampler=sampler,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_batch,
        )
    tokenizer = vlm.llm_backbone.tokenizer
    max_txt_len = getattr(vlm.llm_backbone, "llm_max_length", 256)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_txt_len
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    actor_parameters = list(collect_actor_parameters(vlm))
    if not actor_parameters:
        raise RuntimeError("No pruning actors available for RL fine-tuning.")
    log_info(f"找到 {len(actor_parameters)} 个 Actor 参数用于训练")

    optimizer = torch.optim.AdamW(actor_parameters, lr=args.actor_lr)
    log_info(f"优化器: AdamW, 学习率={args.actor_lr}")

    # 开始训练
    log_info(f"开始训练 {args.epochs} 轮，批次大小={args.batch_size}")
    train(
        vlm,
        dataloader,
        optimizer,
        train_device,
        args.epochs,
        args.train_questions,
        args.train_annotations,
        clip_eps=args.clip_eps,
        advantage_alpha=args.advantage_alpha,
        reward_alpha1=args.reward_alpha1,
        reward_alpha2=args.reward_alpha2,
        gumbel_tau=args.gumbel_tau,
        log_file=log_file_path,
        sampler=sampler,
        use_fsdp=args.use_fsdp,
        is_rank_zero=rank_zero,
    )

    if args.use_fsdp and dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank_zero:
        log_info("训练完成，正在保存检查点...", ctx_level=1)
        if args.use_fsdp and fsdp_state_cfg is not None:
            with FSDP.state_dict_type(vlm, StateDictType.FULL_STATE_DICT, fsdp_state_cfg):
                full_state = {name: tensor.cpu()
                              for name, tensor in vlm.state_dict().items()}
        else:
            full_state = {name: tensor.cpu()
                          for name, tensor in vlm.state_dict().items()}

        actor_state = extract_actor_state_from_full_state(full_state, vlm)
        checkpoint = {
            "model_state_dict": full_state,
            "actor_state_dict": actor_state,
            "config_id": args.model_id,
        }

        output_path = args.output_dir / \
            f"vqa_v2_rl_actor_{args.model_id.replace('+', '_')}.pt"
        torch.save(checkpoint, output_path)
        log_info(f"检查点已保存至: {output_path}")

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
