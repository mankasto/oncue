"""Helpers of the training loop, MTR's `tools/train_utils/train_utils.py`: the device and the
processes, the TensorBoard writer, the finite and stop guards, the step schedule's learning rate,
a checkpoint's training configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .config import RouteTrainConfig, TrainConfig, train_config

if TYPE_CHECKING:  # annotations only
    from torch.utils.tensorboard import SummaryWriter


def device() -> torch.device:
    """The device to compute on: CUDA, else Apple's MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensorboard_writer(path: str | Path) -> SummaryWriter | None:
    """A TensorBoard writer, or None when tensorboard is not installed (the JSON log remains)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed: JSON log only", flush=True)
        return None
    return SummaryWriter(str(path))


def setup() -> tuple[int, int, torch.device]:
    """(rank, world size, device); initializes torch.distributed under torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, device()
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():  # CPU processes (a local test of the distributed path)
        dist.init_process_group("gloo")
        return dist.get_rank(), world, torch.device("cpu")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), world, torch.device("cuda", local_rank)


def check_finite(model: torch.nn.Module, what: str) -> None:
    """Every parameter and floating buffer finite, or an error: a checkpoint is
    never saved with NaN weights or BatchNorm statistics."""
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"{what}: {name} is not finite")


def stop_if_bad(bad_streak: int, train_cfg: TrainConfig, step: int) -> None:
    """Stop the training after train_cfg.max_bad_batches consecutive skipped batches (non-finite
    loss or gradient): a run that no longer learns must not go on as if it did. Every process
    decides alike (the loss check is reduced over them, the gradient norm is of averaged
    gradients)."""
    if bad_streak >= train_cfg.max_bad_batches:
        raise RuntimeError(f"{bad_streak} consecutive non-finite batches at step {step}: stopped")


# MTR's step schedule: of its 30 epochs, the learning rate decays from epoch 22 on, every 2 epochs
MTR_EPOCHS, MTR_FIRST_DECAY_EPOCH, DECAY_EVERY_EPOCHS = 30, 22, 2


def decay_epochs(train_cfg: TrainConfig) -> range:
    """The step schedule's decay epochs (0-based): from round(epochs x 22 / 30) every 2 (30 epochs:
    22, 24, 26, 28); `.start` is the first, also when the run is too short for any."""
    first = round(train_cfg.epochs * MTR_FIRST_DECAY_EPOCH / MTR_EPOCHS)
    return range(first, train_cfg.epochs, DECAY_EVERY_EPOCHS)


def step_lr(train_cfg: TrainConfig, epoch: int) -> float:
    """MTR's step schedule: train_cfg.lr, times train_cfg.lr_decay at each of the `decay_epochs`,
    not below train_cfg.lr_min."""
    passed = sum(1 for decay_epoch in decay_epochs(train_cfg) if epoch >= decay_epoch)
    return max(train_cfg.lr * train_cfg.lr_decay**passed, train_cfg.lr_min)


def checkpoint_config(path: str | Path) -> RouteTrainConfig:
    """The RouteTrainConfig a checkpoint was trained with (a resumed run starts from it)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored = checkpoint["route_train"]
    return RouteTrainConfig(**{**stored, "train": train_config(stored["train"])})


def flat(nested: dict, prefix: str = "") -> dict:
    """A nested configuration on one level: its keys joined by dots, its tuples as lists."""
    out = {}
    for key, value in nested.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = list(value) if isinstance(value, tuple) else value
    return out
