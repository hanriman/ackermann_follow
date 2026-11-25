"""Shared stdout tee logger, CSV metric writer, and TensorBoard wrapper."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from torch.utils.tensorboard import SummaryWriter


class DualLogger:
    """Tee logger that writes both to stdout and to a persistent log file."""

    def __init__(self, filepath: str | Path, terminal_stream: TextIO) -> None:
        self.file = open(filepath, "w", buffering=1, encoding="utf-8")
        self.terminal = terminal_stream

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)
        self.terminal.flush()
        self.file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class MetricsLogger:
    """Unified TensorBoard + CSV logger for all AckBot algorithms."""

    CSV_HEADER = [
        "step",
        "iteration",
        "mean_ep_return",
        "curriculum_stage",
        "unlocked_stage",
        "loss_pi",
        "loss_v",
        "entropy",
        "sps",
        "elapsed_sec",
        "eval_reward",
        "eval_dist",
        "collision_rate",
        "target_in_fov_rate",
        "s1_rew",
        "s2_rew",
        "s3_rew",
        "s4_rew",
        "s5_rew",
        "s6_rew",
        "s7_rew",
    ]

    def __init__(self, csv_path: str | Path, tb_dir: str | Path) -> None:
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.CSV_HEADER)
        self.csv_file.flush()
        self.tb = SummaryWriter(log_dir=str(tb_dir))

    def log_row(self, row: list) -> None:
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self) -> None:
        self.tb.close()
        self.csv_file.close()
