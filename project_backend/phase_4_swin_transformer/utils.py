"""
Utility functions for Phase 4 Swin Transformer.

Handles reproducibility seeding, hardware device resolution, experiment directory
creation, and training-only class weight estimation.
"""

import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .enums import Modality
from .config import get_outputs_dir


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve PyTorch compute device (CUDA -> MPS -> CPU).
    """
    clean = device_str.lower().strip()
    if clean == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(clean)

    logging.info(f"Using Compute Device: {dev} (CUDA available: {torch.cuda.is_available()})")
    return dev


def create_experiment_dir(
    modality: Union[str, Modality],
    base_outputs_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create an isolated, auto-incrementing experiment directory:
    outputs/{modality}/experiment_001/
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    base_dir = Path(base_outputs_dir).resolve() if base_outputs_dir else get_outputs_dir(mod_enum)
    base_dir.mkdir(parents=True, exist_ok=True)

    existing_exp_nums = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("experiment_"):
            try:
                num = int(d.name.split("_")[1])
                existing_exp_nums.append(num)
            except (IndexError, ValueError):
                pass

    next_num = max(existing_exp_nums, default=0) + 1
    exp_dir = base_dir / f"experiment_{next_num:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def compute_class_weights(
    train_labels: Union[List[int], np.ndarray],
    num_classes: int,
) -> Optional[torch.Tensor]:
    """
    Compute balanced class weights computed strictly from training split data.
    """
    labels = np.array(train_labels)
    if len(labels) == 0:
        return None

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < num_classes:
        return None

    total_samples = len(labels)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor
