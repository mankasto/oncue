"""Helpers for picking and recording a torch device, and for repeatable GPU inference."""

from __future__ import annotations

import os

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"  # cuBLAS deterministic mode's workspace config


def resolve_device(name: str) -> str:
    """Return ``name`` (``cpu``, ``cuda``, ``cuda:N`` or ``mps``); ValueError if unavailable."""
    import torch

    device = torch.device(name)
    if device.type == "cuda":
        usable = torch.cuda.is_available() and (device.index or 0) < torch.cuda.device_count()
    elif device.type == "mps":
        usable = torch.backends.mps.is_available()
    else:
        usable = device.type == "cpu"
    if not usable:
        raise ValueError(f"device {name!r}: not available here (cpu, cuda, cuda:N or mps)")
    return str(device)


def device_record(name: str) -> dict:
    """Return the device an output records; for CUDA also the GPU model."""
    import torch

    if is_cuda(name):
        return {"name": name, "gpu": torch.cuda.get_device_name(torch.device(name))}
    return {"name": name}


def is_cuda(name: str) -> bool:
    return name.startswith("cuda")


def make_reproducible(name: str) -> None:
    """Make CUDA inference repeat exactly on the same GPU model; no-op elsewhere.

    Call before the first CUDA use: cuBLAS reads its workspace setting at startup.
    """
    if not is_cuda(name):
        return
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_DETERMINISTIC_WORKSPACE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # a non-deterministic kernel warns
