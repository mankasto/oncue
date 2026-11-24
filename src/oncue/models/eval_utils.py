"""`validation` and `recalibrate_bn` validate the trajectory generator and recompute its BatchNorm
statistics, every epoch of the training and for a saved checkpoint (`validate_checkpoint`,
`recalibrate_checkpoint`); MTR's `tools/eval_utils/eval_utils.py`."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import RouteTrainConfig, TrainConfig
from .datasets.dataset import SceneStore, draws, scene_inputs, split_files
from .train_utils import check_finite, checkpoint_config, device
from .utils.loss_utils import step_loss

if TYPE_CHECKING:  # annotations only
    from .datasets.waymo.waymo_dataset import FeatureConfig
    from .modeling.model import RouteModel

METRICS = ("minADE6", "minFDE6", "MR6", "ADE1", "FDE1", "wp_err1")
TOP_K = 6  # minADE6, minFDE6, MR6: the best of the 6 best-scored trajectories
MISS_M = 2.0  # MR6: a miss ends farther than this from the logged end
VALIDATION_SEED = 12345  # the fixed validation samples and their drawn waypoints
VALIDATION_PARTS = ("reg", "cls", "wp")  # the loss parts validated (the training's but dense)
PART_COUNT = len(VALIDATION_PARTS)
CONDITIONS = ("cond", "free")  # each sample with its drawn waypoints, and with none


def validate_checkpoint(
    path: str | Path,
    feat_dir: str | Path,
    samples: int = RouteTrainConfig.val_samples,
    batch: int = TrainConfig.batch,
    bn_batch_stats: bool = False,
) -> dict:
    """The training validation (`validation`, one process) of a checkpoint on the validation
    shards of `feat_dir`, `samples` of them (the same first ones as in training for 4096);
    bn_batch_stats: BatchNorm on each batch's statistics instead of the stored running ones."""
    from .modeling.model import load_route_model

    dev = device()
    model = load_route_model(path, dev)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = replace(checkpoint_config(path), val_samples=samples)
    cfg = replace(cfg, train=replace(cfg.train, batch=batch))
    files = split_files(feat_dir, "val")[: cfg.val_files or None]
    timings: dict = {}
    result = validation(model, files, cfg, model.fcfg, dev, 0, 1, {}, timings, bn_batch_stats)
    return {
        "checkpoint": str(path),
        "epoch": checkpoint.get("epoch"),
        "samples": samples,
        "bn_batch_stats": bn_batch_stats,
        "val_files": len(files),
        "val": result,
        "seconds": round(timings["val_batches_s"]),
    }


def validation(
    model: RouteModel, files: list[Path], cfg: RouteTrainConfig, feature_cfg: FeatureConfig,
    dev: torch.device, rank: int, world: int, cache: dict, timings: dict,
    bn_batch_stats: bool = False,
) -> dict | None:  # fmt: skip
    """Mean loss parts and METRICS on a fixed set of validation samples (no input dropout): every
    sample twice, with its fixed drawn waypoints (cond/*, and the loss parts) and with none
    (free/*: no waypoints, the route and signal phases still given: generator along a known
    route, not motion prediction from the past alone; the two are reported apart).
    The validation shards load once (`cache`); every process draws the same waypoints and takes
    every world-th batch (`_validation_sums`), the sums added over processes
    (`_sum_over_processes`; the same samples whatever the world)."""
    if not files:
        return None
    start = time.perf_counter()
    if "data" not in cache:
        cache["data"] = SceneStore(files)
    data = cache["data"]
    timings["val_load_s"] = time.perf_counter() - start
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(VALIDATION_SEED)
    picked = torch.randperm(len(data), generator=generator)[: cfg.val_samples].numpy()
    # the conditions (waypoints, contact frame, window, noise) drawn once per sample, then cut
    # into batches: the same sample keeps its conditions whatever the batch size
    drawn = _without_dropouts(draws(len(picked), cfg, generator))
    model.eval()
    momenta = _batch_statistics(model) if bn_batch_stats else {}
    sums = _validation_sums(model, data, picked, drawn, cfg, feature_cfg, dev, rank, world)
    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train()
    if world > 1:
        sums = _sum_over_processes(*sums, dev)
    timings["val_batches_s"] = time.perf_counter() - start
    return _validation_result(*sums)


def _without_dropouts(drawn: dict) -> dict:
    """The draws with the input dropouts (history, t0 state only, route, signals) cleared, as the
    validation and generation feed the model."""
    for key in ("drop", "state_only", "route_off", "sig_off"):
        drawn[key] = torch.zeros_like(drawn[key])
    return drawn


def _batch_statistics(model: RouteModel) -> dict:
    """Every BatchNorm on each batch's statistics, as in training (a diagnosis), its stored running
    statistics kept; returns the momenta to restore."""
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            momenta[module] = module.momentum
            module.train()
            module.momentum = 0.0  # the stored running statistics stay as they are
    return momenta


def _validation_sums(
    model: RouteModel, data: SceneStore, picked: np.ndarray, drawn: dict, cfg: RouteTrainConfig,
    feature_cfg: FeatureConfig, dev: torch.device, rank: int, world: int,
) -> tuple[torch.Tensor, int, int, torch.Tensor]:  # fmt: skip
    """This process's sums over its batches of the `picked` samples (every world-th): the loss
    parts times their counts, the samples, the given waypoints (the wp part is per waypoint) and
    METRICS' sums and counts per condition (`_metrics`)."""
    loss_sums, samples, waypoints_given = torch.zeros(PART_COUNT), 0, 0
    metric_sums = torch.zeros(len(CONDITIONS), len(METRICS), 2, dtype=torch.float64)  # sum, count
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.bf16):
        for batch_number, first in enumerate(range(0, len(picked), cfg.train.batch)):
            indices = picked[first : first + cfg.train.batch]
            if batch_number % world != rank:
                continue
            batch_draws = {
                key: value[first : first + cfg.train.batch] for key, value in drawn.items()
            }
            inputs, waypoints, waypoint_frames = scene_inputs(
                data.batch(indices), batch_draws, cfg, feature_cfg, dev
            )
            traj, score = model(inputs, waypoints)
            _, parts = step_loss(
                traj, score, inputs, model.ip, waypoints, waypoint_frames, cfg.train.slot_weights
            )
            given_count = int((waypoint_frames >= 0).sum())  # the batch's wp part: mean over these
            loss_sums += parts.cpu().float() * torch.tensor(
                [len(indices), len(indices), given_count]
            )
            samples, waypoints_given = samples + len(indices), waypoints_given + given_count
            metric_sums[0] += _metrics(traj, score, inputs["fut"], waypoints, waypoint_frames)
            # no waypoint at all
            no_waypoints, no_frames = (
                torch.zeros_like(waypoints),
                torch.full_like(waypoint_frames, -1),
            )
            traj, score = model(inputs, no_waypoints)
            metric_sums[1] += _metrics(traj, score, inputs["fut"], no_waypoints, no_frames)
    return loss_sums, samples, waypoints_given, metric_sums


def _sum_over_processes(
    loss_sums: torch.Tensor,
    samples: int,
    waypoints_given: int,
    metric_sums: torch.Tensor,
    dev: torch.device,
) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    """`_validation_sums`' sums added over every process (one all-reduce)."""
    import torch.distributed as dist

    counts = torch.tensor([float(samples), float(waypoints_given)])
    sums = torch.cat([loss_sums.double(), counts, metric_sums.flatten()]).to(dev)
    dist.all_reduce(sums)
    sums = sums.cpu()
    return (
        sums[:PART_COUNT].float(),
        int(sums[PART_COUNT].item()),
        int(sums[PART_COUNT + 1].item()),
        sums[PART_COUNT + 2 :].view(len(CONDITIONS), -1, 2),
    )


def _validation_result(
    loss_sums: torch.Tensor, samples: int, waypoints_given: int, metric_sums: torch.Tensor
) -> dict:
    """The mean loss parts (VALIDATION_PARTS) and per condition the METRICS, from their sums."""
    denominators = torch.tensor(
        [max(samples, 1), max(samples, 1), max(waypoints_given, 1)], dtype=torch.float32
    )
    result = dict(zip(VALIDATION_PARTS, (loss_sums / denominators).tolist(), strict=True))
    for part, rows in zip(CONDITIONS, metric_sums.tolist(), strict=True):
        for key, (total, count) in zip(METRICS, rows, strict=True):
            if count > 0:
                result[f"{part}/{key}"] = total / count
    return result


def _metrics(
    traj: torch.Tensor,
    score: torch.Tensor,
    fut: torch.Tensor,
    waypoints: torch.Tensor,
    waypoint_frames: torch.Tensor,
) -> torch.Tensor:
    """[len(METRICS), 2] sums and counts of one batch (m, over the 8 s future): minADE6 / minFDE6
    over the 6 best-scored trajectories, MR6 (minFDE6 above 2 m), ADE1 / FDE1 of the best-scored
    one, wp_err1 (its mean distance to the given waypoints; samples given any). Simpler than the
    WOMD metrics: no NMS, a fixed 2 m miss threshold, the 8 s horizon only."""
    traj, fut = traj.float(), fut[..., :2].float()
    count, rows = len(traj), torch.arange(len(traj), device=traj.device)
    errors = (traj - fut[:, None]).norm(dim=-1)  # [B, K, F]
    ade, fde = errors.mean(-1), errors[..., -1]
    top = score.float().topk(min(TOP_K, score.shape[1]), dim=1).indices
    ade6, fde6 = ade.gather(1, top).min(1).values, fde.gather(1, top).min(1).values
    best_scored = score.argmax(1)
    path = traj[rows, best_scored]  # [B, F, 2]
    has = waypoint_frames >= 0
    at = torch.gather(path, 1, waypoint_frames.clamp(min=0)[..., None].expand(-1, -1, 2))
    waypoint_error = ((at - waypoints[..., :2].float()).norm(dim=-1) * has).sum(1) / has.sum(
        1
    ).clamp(min=1)
    given = has.any(1)
    metric_rows = [
        (ade6.sum(), count),
        (fde6.sum(), count),
        ((fde6 > MISS_M).float().sum(), count),
        (ade[rows, best_scored].sum(), count),
        (fde[rows, best_scored].sum(), count),
        (waypoint_error[given].sum(), given.sum()),
    ]
    return torch.tensor(
        [[float(total), float(number)] for total, number in metric_rows], dtype=torch.float64
    )


def recalibrate_checkpoint(
    path: str | Path,
    feat_dir: str | Path,
    out: str | Path,
    batches: int = RouteTrainConfig.bn_batches,
    shards: int = RouteTrainConfig.bn_shards,
    clean: bool = False,
) -> dict:
    """Precise BatchNorm (`recalibrate_bn`) for a saved checkpoint, one process: the weights as
    they are, every BatchNorm's running statistics recomputed over `batches` batches across
    `shards` random training shards of `feat_dir`; written to `out` (everything else kept)."""
    from .modeling.model import load_route_model

    dev = device()
    model = load_route_model(path, dev)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = replace(checkpoint_config(path), bn_batches=batches, bn_shards=shards)
    files = split_files(feat_dir, "train")
    record = recalibrate_bn(model, files, cfg, model.fcfg, dev, cfg.seed, clean=clean)
    check_finite(model, f"{path} recalibrated")
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save({**checkpoint, "state": state, "bn_recalibrated": record}, out)
    return {"checkpoint": str(path), "out": str(out), **record}


def recalibrate_bn(
    model: RouteModel,
    files: list[Path],
    cfg: RouteTrainConfig,
    feature_cfg: FeatureConfig,
    dev: torch.device,
    seed: int,
    cache: dict | None = None,
    clean: bool = False,
) -> dict:
    """Precise BatchNorm: every BatchNorm's running statistics reset and recomputed as the plain
    average over cfg.bn_batches batches drawn from cfg.bn_shards random training shards (the
    training's input draws; dropout off, as at inference; no gradient, weights untouched). The
    shards load once per `cache`. `clean`: without the input dropouts (history, t0 state only,
    route, signals), as the validation and generation feed the model, so the statistics come from
    the inputs it sees there. Returns what was used. ValueError for fewer than one batch: the
    statistics would be left reset (mean 0, variance 1) instead of recomputed."""
    if cfg.bn_batches < 1:
        raise ValueError(f"precise BatchNorm over {cfg.bn_batches} batches: at least 1")
    generator = torch.Generator().manual_seed(seed)
    if cache is None or "store" not in cache:
        order = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed))
        chosen = [files[int(index)] for index in order[: cfg.bn_shards]]
        store = SceneStore(chosen)
        if cache is not None:
            cache["store"], cache["files"] = store, chosen
    else:
        store, chosen = cache["store"], cache["files"]
    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    momenta = {module: module.momentum for module in batchnorms}
    was_training = model.training
    model.eval()
    for module in batchnorms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    train_cfg = cfg.train
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.bf16):
        for _ in range(cfg.bn_batches):
            samples = torch.randint(len(store), (train_cfg.batch,), generator=generator).numpy()
            drawn = draws(len(samples), cfg, generator)
            if clean:
                drawn = _without_dropouts(drawn)
            inputs, waypoints, _ = scene_inputs(store.batch(samples), drawn, cfg, feature_cfg, dev)
            model(inputs, waypoints, all_steps=True)
    for module in batchnorms:
        module.momentum = momenta[module]
    model.train(was_training)
    return {
        "batches": cfg.bn_batches,
        "shards": [file.name for file in chosen],
        "seed": seed,
        "clean_inputs": clean,
    }
