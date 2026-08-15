"""FIXED config loader."""

from __future__ import annotations

from pathlib import Path

import yaml


def load(path: str | Path = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_task(path: str | Path = "configs/task.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    """Resolve ``auto | cpu | cuda`` to the device used by every model."""
    mode = str(cfg.get("device", "auto")).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if mode == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        if mode == "cuda":
            raise RuntimeError("device=cuda requires PyTorch with CUDA support") from exc
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if mode == "cuda" and not cuda_available:
        raise RuntimeError("device=cuda was requested, but CUDA is not available")
    return "cuda" if cuda_available else "cpu"
