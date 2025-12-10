import json
import os
import re
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from train.vlm_train.overwatch import initialize_overwatch
from train.vlm_train.models.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)

# === Map/Iterable Dataset Declarations ===


class VQAv2MapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor
    ) -> None:
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, str]:
        """Return (question_id: int, question_prompt: str, pixel_values: torch.Tensor, question: str, answer: str)."""
        ex = self.examples[idx]
        question_prompt = self.prompt_fn(ex["question"])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(
                self.root_dir / ex["img_path"]).convert("RGB"))

        else:
            # Assume `image_transform` is a HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return ex["question_id"], question_prompt, pixel_values, ex["question"], ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)

    def collate_batch(self,
                      batch: List[Tuple[int, str, torch.Tensor, str, str]]
                      ) -> Dict[str, Any]:
        """
        针对 VQAv2MapDataset 的 collate 函数。

        输入：batch = [
            (question_id, question_prompt, pixel_values, question, answer),
            ...
        ]
        输出：{
            "question_id": LongTensor[B],
            "question_prompt": List[str] 长度 B,
            "pixel_values": Tensor[B, C, H, W],
            "question": List[str] 长度 B,
            "answer": List[str] 长度 B,
        }
        """
        # 拆开 5 个位置
        question_ids = [ex[0] for ex in batch]
        question_prompts = [ex[1] for ex in batch]
        pixel_values_list = [ex[2] for ex in batch]
        questions = [ex[3] for ex in batch]
        answers = [ex[4] for ex in batch]

        # 整理成 batch 形式
        question_ids = torch.tensor(question_ids, dtype=torch.long)
        pixel_values = torch.stack(pixel_values_list, dim=0)  # [B, C, H, W]

        return {
            "question_id": question_ids,
            "question_prompt": question_prompts,
            "pixel_values": pixel_values,
            "question": questions,
            "answer": answers,
        }
