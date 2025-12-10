"""
logger.py

Lightweight utilities for persisting Stage1/Stage2 training metrics to structured
files while keeping compatibility with the existing Prismatic training scripts.

Usage example (inside a training loop):

    from prismatic.training.logger import TrainingLogger

    logger = TrainingLogger(run_name="stage1_phi2", log_dir="logs")
    ...
    logger.log_step(
        stage="stage1",
        epoch=epoch,
        step=step,
        ce_loss=ce_loss.item(),
        pruning_reward=pruning_reward.item(),
        objective=loss.item(),
    )
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StepRecord:
    timestamp: float
    stage: str
    epoch: int
    global_step: int
    ce_loss: float
    pruning_reward: float
    objective: float
    extra: Dict[str, float]


class TrainingLogger:
    """
    Logs per-step metrics to both CSV and JSONL files so that training curves
    can be inspected after long runs.  Files are flushed incrementally to avoid
    losing data on crash.
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str | Path = "logs",
        flush_every: int = 10,
    ) -> None:
        self.run_name = run_name
        self.flush_every = max(flush_every, 1)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"{self.run_name}.csv"
        self.jsonl_path = self.log_dir / f"{self.run_name}.jsonl"

        self._csv_file = self.csv_path.open("a", newline="")
        self._csv_writer = None
        self._jsonl_file = self.jsonl_path.open("a")
        self._buffer: List[StepRecord] = []
        self._global_step = 0

    def __del__(self) -> None:
        self.close()

    def log_step(
        self,
        stage: str,
        epoch: int,
        step: int,
        ce_loss: float,
        pruning_reward: float,
        objective: float,
        **extra: float,
    ) -> None:
        """
        Record a single optimization step.
        """
        self._global_step += 1
        sanitized_extra: Dict[str, float] = {}
        for k, v in extra.items():
            try:
                sanitized_extra[k] = float(v)
            except (TypeError, ValueError):
                continue

        record = StepRecord(
            timestamp=time.time(),
            stage=stage,
            epoch=epoch,
            global_step=self._global_step,
            ce_loss=float(ce_loss),
            pruning_reward=float(pruning_reward),
            objective=float(objective),
            extra=sanitized_extra,
        )
        self._buffer.append(record)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        for record in self._buffer:
            self._write_csv(record)
            self._write_jsonl(record)
        self._buffer.clear()
        self._csv_file.flush()
        self._jsonl_file.flush()

    def close(self) -> None:
        self.flush()
        if not self._csv_file.closed:
            self._csv_file.close()
        if not self._jsonl_file.closed:
            self._jsonl_file.close()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _write_csv(self, record: StepRecord) -> None:
        row_dict = self._record_to_flat_dict(record)
        if self._csv_writer is None:
            fieldnames = list(row_dict.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            if self.csv_path.stat().st_size == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(row_dict)

    def _write_jsonl(self, record: StepRecord) -> None:
        payload = asdict(record)
        payload["extra"] = record.extra  # ensure nested dict is preserved
        self._jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _record_to_flat_dict(record: StepRecord) -> Dict[str, float]:
        row = {
            "timestamp": record.timestamp,
            "stage": record.stage,
            "epoch": record.epoch,
            "global_step": record.global_step,
            "ce_loss": record.ce_loss,
            "pruning_reward": record.pruning_reward,
            "objective": record.objective,
        }
        for key, value in record.extra.items():
            row[key] = value
        return row


__all__ = ["TrainingLogger", "StepRecord"]

