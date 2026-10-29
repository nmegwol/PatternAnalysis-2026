"""Reproducibility, device selection, logging and JSON files."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_PROMPT_TEMPLATE = (
    "Simplify this radiology report for a layperson:\n{report}"
)


def set_seed(seed, *, deterministic=False):
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def select_device(preference="auto"):
    """Select an available CPU, CUDA, or Apple MPS device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def configure_logger(name="radiology", *, output_dir=None, level=logging.INFO):
    """Configure console logging and optional file logging without duplicates."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handlers = [logging.StreamHandler()]
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(directory / f"{name}.log", encoding="utf-8"))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(data, path):
    """Save experiment configuration, metric history or results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path):
    """Load previously saved JSON configuration or metric history."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
