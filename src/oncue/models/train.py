"""`train_route` trains the paper's trajectory generator (`model.RouteModel`) on extracted
scene-layout features.

Waypoints drawn from each sample's own future, history dropout, start-from-rest up-weighting
(`config.TrainConfig`), shards streamed `shards_in_memory` at a time; the route and signal inputs
each dropped for part of the samples so that the model also works without them, and a history
dropout that keeps the current state (pose and speed at t0, as an inserted road user placed at a
speed). Every decoder step is supervised, plus MTR's dense future prediction.

Several GPUs: start with `torchrun --nproc_per_node=N` (or tools/train.sh, which also joins
several nodes): the feature shards of each epoch are dealt out to the processes
(`epoch_loader`), every process takes the same number of steps (`common_step_budget`), gradients
are averaged (DistributedDataParallel). The batch is per process. Rank 0 writes the checkpoint
after every epoch (`save_checkpoint`, with the optimizer state, for `resume`) and the validation
loss on a fixed set of validation samples (`eval_utils.validation`).

    oncue model train FEATURES --out run.pt [--epochs 5] [--bf16] [--resume]
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import RouteTrainConfig, TrainConfig
from .datasets.dataset import (
    Prefetch,
    SceneStore,
    excluded,
    group_files,
    scene_batches,
    scene_inputs,
    scene_probe,
    split_files,
)
from .datasets.waymo.data_preprocess import feature_config
from .datasets.waymo.waymo_dataset import FeatureConfig
from .eval_utils import VALIDATION_PARTS, recalibrate_bn, validation
from .modeling.model import RouteModel
from .modeling.motion_decoder.mtr_decoder import attention_logit_max
from .train_utils import (
    check_finite,
    decay_epochs,
    flat,
    setup,
    step_lr,
    stop_if_bad,
    tensorboard_writer,
)
from .utils.loss_utils import PARTS, loss

if TYPE_CHECKING:  # annotations only
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.tensorboard import SummaryWriter

PROBE_SHARDS = 10  # the training shards whose samples' endpoints give the intention points
PROCESS_SEED_STRIDE = 1000  # a process's own random streams: the seed plus this times its rank
PREFETCH_GROUPS = 1  # shard groups loaded ahead: the next loads while one trains
PREFETCH_BATCHES = 8  # batches prepared ahead of the training step
ATTENTION_SAMPLES = 8  # the samples of a batch whose attention logits TensorBoard records
TRACE_EVERY_BOARD_POINTS = 10  # the log's text trace: one row every this many TensorBoard points
RESUME_ONLY = ("optimizer", "scheduler", "generator", "rng")  # not in the kept epochs' weights
KEYS_SHOWN = 3  # a refused --init names this many of its missing and unexpected weights
TIMING_DECIMALS = 3  # an epoch row's timings rounded to this many decimals


def train_route(
    feat_dir: str | Path,
    out: str | Path,
    cfg: RouteTrainConfig | None = None,
    *,
    resume: bool = False,
    reseed: bool = False,
    train_files: int = 0,
    init: str | Path | None = None,
) -> dict:
    """Train (or resume) the route + signal model on `feat_dir`; writes `out` every epoch. `init`:
    a checkpoint whose weights a new run starts from (fine-tuning: its intention points and
    training shards, a new optimizer and schedule; parameters it lacks, as the start
    velocity heads, keep their own initialisation)."""
    check_arguments(out, resume, init)
    init_checkpoint = torch.load(init, map_location="cpu", weights_only=False) if init else None
    cfg = cfg or RouteTrainConfig()
    rank, world, dev = setup()
    if dev.type == "cuda":  # TF32 matrix products
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features_record = Path(feat_dir) / "features.json"
    if not features_record.exists():
        raise SystemExit(f"{feat_dir}: no features.json (not an `oncue model extract` output)")
    feature_cfg = feature_config(json.loads(features_record.read_text()))
    files = select_training_files(feat_dir, out, resume, init_checkpoint, train_files)
    val_files = split_files(feat_dir, "val")[: cfg.val_files]
    sizes = [len(np.load(file)["center"]) for file in files]
    run = TrainingRun(cfg, feature_cfg, rank, world, dev, feat_dir, files, sizes, val_files, out)

    model, points, init_record = build_model(run, init_checkpoint, init)
    optimizer, scheduler, max_steps = optimizer_and_schedule(run, model)
    resumed = resume_from(run, model, optimizer, scheduler, reseed) if resume else None
    forward = training_forward(run, model)
    generator = random_streams(run, resumed)
    progress = resumed.progress if resumed is not None else Progress()
    board = start_logs(run, init_record, max_steps, resumed)
    state = TrainingState(
        model, forward, optimizer, scheduler, generator, progress, board, points, init_record
    )

    epochs = cfg.train.epochs
    start_epoch = resumed.start_epoch if resumed is not None else 0
    upcoming = epoch_loader(run, start_epoch, generator) if start_epoch < epochs else None
    val_cache: dict = {}
    bn_cache: dict = {}  # the shards precise BatchNorm draws from, loaded once
    for epoch in range(start_epoch, epochs):
        loader, budget = upcoming
        outcome = train_epoch(run, state, epoch, loader, budget)
        end_start = time.perf_counter()
        outcome.generator_states, outcome.rng_states = random_states(run, generator)
        outcome.timings["epoch_end_sync_s"] = time.perf_counter() - end_start
        # the next epoch's first shards load during the checkpoint and the validation (its
        # shard order is drawn after the random state was saved: a resume draws the same)
        upcoming = epoch_loader(run, epoch + 1, generator) if epoch + 1 < epochs else None
        outcome.bn_record = precise_batchnorm(run, model, bn_cache, outcome.timings)
        outcome.val_result = validation(
            model, val_files, cfg, feature_cfg, dev, rank, world, val_cache, outcome.timings
        )  # every process: its share of the batches (the same result everywhere)
        track_best(progress, outcome)
        check_finite(model, f"epoch {epoch + 1}")  # every process (the same model after the
        # broadcast), before saving
        if rank == 0:
            save_checkpoint(run, state, outcome)
        mean_loss = epoch_mean_loss(run, outcome)
        if rank == 0:
            log_epoch(run, state, outcome, mean_loss, end_start)
        if stop_early(run, outcome, progress):
            if upcoming is not None:
                upcoming[0].close()
            break
    if board is not None:
        board.close()
    if world > 1:
        dist.destroy_process_group()
    return {"out": str(out), "epochs": epochs}


@dataclass(frozen=True)
class TrainingRun:
    """What the steps of a run read and never change: its configurations, its processes, its
    shards and its output."""

    cfg: RouteTrainConfig
    feature_cfg: FeatureConfig
    rank: int
    world: int  # the number of processes
    dev: torch.device
    feat_dir: str | Path  # the feature directory as given
    files: list[Path]  # the training shards
    sizes: list[int]  # their samples
    val_files: list[Path]
    out: Path  # the checkpoint

    @property
    def train_cfg(self) -> TrainConfig:
        return self.cfg.train


@dataclass
class Progress:
    """What a run carries from step to step and from epoch to epoch."""

    step: int = 0  # the updates made, a resumed run's before it included
    skipped: int = 0  # the batches this run skipped (a non-finite loss or gradient)
    bad_streak: int = 0  # consecutive skipped batches: max_bad_batches of them stop the training
    # the validation loss (reg + cls + wp), its epoch, the epochs since
    best_val: float = float("inf")
    best_epoch: int | None = None
    since_best: int = 0


@dataclass
class Resumed:
    """What a resumed run takes from its checkpoint."""

    checkpoint: dict
    start_epoch: int
    progress: Progress  # its step and its best validation loss so far
    reseeded: bool  # another number of processes: the random streams new from the seed
    changed: dict  # the settings this run changes from the checkpoint's ([was, now])


@dataclass
class TrainingState:
    """What the epochs train and record: the model and what the training step calls (`forward`),
    the optimizer and OneCycle (None: the step schedule), the draws' generator, the progress,
    process 0's TensorBoard writer (None elsewhere); the intention points and the fine-tuning
    record the checkpoint keeps."""

    model: RouteModel
    forward: torch.nn.Module
    optimizer: Optimizer
    scheduler: OneCycleLR | None
    generator: torch.Generator
    progress: Progress
    board: SummaryWriter | None
    points: torch.Tensor
    init_record: dict | None
    started: float = field(default_factory=time.time)  # the state is made as the epochs start:
    # the logs' seconds count from then


@dataclass
class EpochOutcome:
    """One epoch's results, as its checkpoint and its log row record them."""

    epoch: int  # 0-based
    loss_sums: torch.Tensor  # the loss parts summed over this process's updates
    updates: int
    seconds: float
    data_wait_s: float  # the GPU waited for batches this long
    timings: dict
    generator_states: torch.Tensor | list | None = None  # the draws' generator, every process's
    # the global streams (dropout draws from them), every process's
    rng_states: dict | list | None = None
    bn_record: dict | None = None
    val_result: dict | None = None
    val_total: float | None = None
    improved: bool = False


# ---------------------------------------------------------------- the run's start
def check_arguments(out: str | Path, resume: bool, init: str | Path | None) -> None:
    """Refuse a resume without a checkpoint at `out` (before anything loads: a wrong path is no
    new run) and a resume that is also given another checkpoint to start from."""
    if resume and not Path(out).exists():
        raise SystemExit(f"--resume: no checkpoint at {Path(out).resolve()}")
    if resume and init is not None:
        raise SystemExit("--init starts a new run from another checkpoint: not with --resume")


def select_training_files(
    feat_dir: str | Path, out: Path, resume: bool, init_checkpoint: dict | None, train_files: int
) -> list[Path]:
    """The training shards of `feat_dir`: a resumed or fine-tuned run's those its checkpoint
    trained on (refused if one is gone), unless `train_files` is given; else the first
    `train_files` of them, or all."""
    files = split_files(feat_dir, "train")
    if (resume or init_checkpoint is not None) and not train_files:
        source = (
            init_checkpoint
            if init_checkpoint is not None
            else torch.load(out, map_location="cpu", weights_only=False)
        )
        kept_names = source.get("train_files")
        if kept_names is not None:
            files_by_name = {file.name: file for file in files}
            missing = [name for name in kept_names if name not in files_by_name]
            if missing:
                raise SystemExit(f"--resume: {len(missing)} training shards gone ({missing[0]}…)")
            files = [files_by_name[name] for name in kept_names]
    elif train_files:
        files = files[:train_files]
    return files


def build_model(
    run: TrainingRun, init_checkpoint: dict | None, init: str | Path | None
) -> tuple[RouteModel, torch.Tensor, dict | None]:
    """(model, its intention points, the fine-tuning record or None). The intention points from
    the endpoints of the first training shards' samples, the weights from the seed (the same in
    every run with this seed, and on every process); fine-tuning (`init_checkpoint`): its
    intention points (its modes were trained on them) and its weights. Then each process its own
    dropout stream from the seed (a resumed run restores the streams it saved,
    `random_streams`). Some GPU kernels are not bitwise deterministic all the same."""
    points = scene_probe(run.files[:PROBE_SHARDS], run.train_cfg.model.intention_points)
    if init_checkpoint is not None:
        points = init_checkpoint["points"]
    torch.manual_seed(run.cfg.seed)
    model = RouteModel(run.train_cfg.model, points, history=run.feature_cfg.history).to(run.dev)
    init_record = None
    if init_checkpoint is not None:
        init_record = load_init_weights(model, init_checkpoint, init)
        if run.rank == 0:
            print(json.dumps({"init": init_record}), flush=True)
    torch.manual_seed(run.cfg.seed + PROCESS_SEED_STRIDE * run.rank + 1)
    return model, points, init_record


def load_init_weights(model: RouteModel, init_checkpoint: dict, init: str | Path) -> dict:
    """The weights of the checkpoint a fine-tuning starts from (only parameters it does not have
    may be missing, as the start velocity heads); returns its record (path, epoch, the new
    parameters)."""
    loaded = model.load_state_dict(init_checkpoint["state"], strict=False)
    only_new_missing = all(key.startswith("mtr.start.") for key in loaded.missing_keys)
    if loaded.unexpected_keys or not only_new_missing:
        raise SystemExit(
            f"--init {init}: not this model's weights (missing {loaded.missing_keys[:KEYS_SHOWN]}, "
            f"unexpected {loaded.unexpected_keys[:KEYS_SHOWN]})"
        )
    return {"from": str(init), "epoch": init_checkpoint.get("epoch"), "new": loaded.missing_keys}


def optimizer_and_schedule(
    run: TrainingRun, model: RouteModel
) -> tuple[Optimizer, OneCycleLR | None, int]:
    """(AdamW, OneCycle over the updates' upper bound or None for the step schedule (the learning
    rate set per epoch, `step_lr`), that bound). The bound: an epoch takes whole batches of each
    loaded shard group, several processes the fewest any can give, so at most samples / processes
    / batch (whole batches counted per shard can come out under the updates made, and OneCycle
    would then hold its last rate over the rest)."""
    train_cfg = run.train_cfg
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    max_steps = max(train_cfg.epochs * (int(sum(run.sizes)) // run.world // train_cfg.batch), 1)
    scheduler = None
    if train_cfg.schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, train_cfg.lr, total_steps=max_steps
        )
    elif train_cfg.schedule != "step":
        raise ValueError(f"unknown schedule {train_cfg.schedule!r}")
    return optimizer, scheduler, max_steps


# ---------------------------------------------------------------- a resumed run
def resume_from(
    run: TrainingRun,
    model: RouteModel,
    optimizer: Optimizer,
    scheduler: OneCycleLR | None,
    reseed: bool,
) -> Resumed:
    """A resumed run's checkpoint (at run.out): its schedule checked (another one would be handed
    the saved one's state, e.g. OneCycle a step checkpoint's None; it starts from the weights
    with a new optimizer and schedule, --init), its weights, optimizer and scheduler state
    restored, its epoch, step and best validation loss taken; the settings this run changes from
    it said and logged. The random streams are per process: another number of processes is
    refused, unless `reseed` (new streams from the seed; the weights and optimizer state
    restored)."""
    train_cfg = run.train_cfg
    checkpoint = torch.load(run.out, map_location="cpu", weights_only=False)
    saved_schedule = (checkpoint["route_train"].get("train") or {}).get("schedule", "step")
    if saved_schedule != train_cfg.schedule:
        raise SystemExit(
            f"--resume: {run.out} was trained with the {saved_schedule!r} schedule, this run asks "
            f"for {train_cfg.schedule!r}; resume with the same one, or start from its "
            f"weights: --init {run.out}"
        )
    changed = config_changes(run, checkpoint)
    if run.rank == 0:
        print(json.dumps({"resume_changes": changed}), flush=True)
    # an older checkpoint: one random stream per process in a list, a single one without
    saved_generators = checkpoint.get("generator")
    saved_world = checkpoint.get(
        "world", len(saved_generators) if isinstance(saved_generators, list) else 1
    )
    if saved_world != run.world and not reseed:
        raise SystemExit(
            f"--resume: {run.out} was trained with {saved_world} processes, this run has "
            f"{run.world}; resume with the same number, or --reseed (new random streams from the "
            "seed)"
        )
    reseeded = saved_world != run.world
    if reseeded:
        changed["processes"] = [saved_world, run.world]
        if run.rank == 0:
            print(
                f"resume: {saved_world} -> {run.world} processes, random streams from the seed",
                flush=True,
            )
    restore_training_state(run, checkpoint, model, optimizer, scheduler)
    # the updates made, as saved (not the scheduler's count, which a capped OneCycle keeps
    # below them; the log and TensorBoard steps go on from it)
    scheduler_steps = scheduler.last_epoch if scheduler is not None else 0
    start_epoch = checkpoint["epoch"]
    progress = Progress(step=checkpoint.get("step", scheduler_steps))
    if start_epoch:
        progress.best_val = checkpoint.get("best_val", progress.best_val)
        progress.best_epoch = checkpoint.get("best_epoch")
        progress.since_best = checkpoint.get("since_best", 0)
    if train_cfg.schedule == "step":  # the decay epochs follow this run's epochs: said and logged
        changed["schedule"] = {
            "lr_next_epoch": step_lr(train_cfg, start_epoch),
            "decays_after_epochs": list(decay_epochs(train_cfg)),
        }
        if run.rank == 0:
            print(f"resume: {json.dumps(changed['schedule'])}", flush=True)
    return Resumed(checkpoint, start_epoch, progress, reseeded, changed)


def config_changes(run: TrainingRun, checkpoint: dict) -> dict:
    """What this run changes from the checkpoint's own configuration ([was, now]), its training
    shards included (their numbers)."""
    saved_config, run_config = flat(checkpoint["route_train"]), flat(asdict(run.cfg))
    changed = {
        key: [saved_config.get(key), run_config[key]]
        for key in run_config
        if saved_config.get(key) != run_config[key]
    }
    file_names = [file.name for file in run.files]
    if checkpoint.get("train_files") is not None and checkpoint["train_files"] != file_names:
        changed["train_files"] = [len(checkpoint["train_files"]), len(file_names)]
    return changed


def restore_training_state(
    run: TrainingRun,
    checkpoint: dict,
    model: RouteModel,
    optimizer: Optimizer,
    scheduler: OneCycleLR | None,
) -> None:
    """The weights, the optimizer and the scheduler as saved; the run's own settings over the
    restored parameter groups (the learning rate is set per epoch or by the restored scheduler):
    a resumed run with another weight decay uses it, and says so."""
    model.load_state_dict(checkpoint["state"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    weight_decay = run.train_cfg.weight_decay
    for group in optimizer.param_groups:
        if group["weight_decay"] != weight_decay and run.rank == 0:
            print(f"resume: weight decay {group['weight_decay']} -> {weight_decay}", flush=True)
        group["weight_decay"] = weight_decay
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


# ---------------------------------------------------------------- before the first epoch
def training_forward(run: TrainingRun, model: RouteModel) -> torch.nn.Module:
    """What the training step calls: the model, in DistributedDataParallel over several
    processes, compiled with cfg.compile (the wrapped model: DDP's buckets split the compiled
    graph)."""
    forward = model
    if run.world > 1:
        forward = DistributedDataParallel(
            model, device_ids=None if run.dev.type == "cpu" else [run.dev.index]
        )
    if run.cfg.compile:
        forward = torch.compile(forward, mode=run.cfg.compile_mode)
    return forward


def random_streams(run: TrainingRun, resumed: Resumed | None) -> torch.Generator:
    """The generator of the shard groups and the batches' draws: one process, one for everything,
    as the prototype; several, the epoch's shard order shared, the draws per process. A resumed
    run's as it was saved, and so the global streams (dropout draws from them); not when
    reseeded."""
    seed = run.cfg.seed + PROCESS_SEED_STRIDE * run.rank * (run.world > 1)
    generator = torch.Generator().manual_seed(seed)
    if resumed is None or not resumed.start_epoch or resumed.reseeded:
        return generator
    checkpoint = resumed.checkpoint
    if "generator" in checkpoint:
        saved = checkpoint["generator"]
        generator.set_state(saved[run.rank] if run.world > 1 else saved)
    if "rng" in checkpoint:
        saved = checkpoint["rng"]
        saved_rng = saved[run.rank] if run.world > 1 else saved
        torch.set_rng_state(saved_rng["cpu"])
        if saved_rng.get("cuda") is not None and run.dev.type == "cuda":
            torch.cuda.set_rng_state(saved_rng["cuda"], run.dev)
    return generator


def start_logs(
    run: TrainingRun, init_record: dict | None, max_steps: int, resumed: Resumed | None
) -> SummaryWriter | None:
    """Process 0's first log row (samples, shards, processes, the fine-tuning start, the code
    version, the scenes kept out, the update bound, the configuration; a resumed run's changes
    from its checkpoint's settings, [was, now]) and its TensorBoard writer (<out>.tb: losses,
    learning rate, gradient norm, throughput every `log_every` steps, validation every epoch;
    steps continue across a resume), the configuration as its first text; None on the other
    processes."""
    if run.rank != 0:
        return None
    head = {
        "samples": int(sum(run.sizes)),
        "files": len(run.files),
        "world": run.world,
        "init": init_record,
        "excluded": excluded(run.feat_dir),
        "steps": max_steps,
        "config": asdict(run.cfg),
        **({"resume_changes": resumed.changed} if resumed is not None else {}),
    }
    print(json.dumps(head), flush=True)
    append_log(run, head)
    board = tensorboard_writer(run.out.parent / f"{run.out.stem}.tb")
    if board is not None:
        board.add_text("config", json.dumps(asdict(run.cfg)), 0)
    return board


def append_log(run: TrainingRun, row: dict) -> None:
    """A row of the run's JSON log (`<out>.log.jsonl`)."""
    with run.out.with_suffix(".log.jsonl").open("a") as log_file:
        log_file.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- an epoch's training
def epoch_loader(
    run: TrainingRun, epoch: int, generator: torch.Generator
) -> tuple[Prefetch, int | None]:
    """(this epoch's shard groups, loading in the background; the step budget): one process takes
    every shard; several, the shards dealt out in an order drawn per epoch, each process taking
    the same number of steps (`common_step_budget`)."""
    files = run.files
    if run.world == 1:
        my_files = files
    else:
        order_generator = torch.Generator().manual_seed(run.cfg.seed + epoch)
        order = torch.randperm(len(files), generator=order_generator).tolist()
        my_files = [files[j] for j in order[run.rank :: run.world]]
    shard_groups = group_files(my_files, run.train_cfg.shards_in_memory, generator)
    stores = (SceneStore(group) for group in shard_groups)  # loaded one at a time
    budget = common_step_budget(run, shard_groups) if run.world > 1 else None
    return Prefetch(stores, PREFETCH_GROUPS), budget


def common_step_budget(run: TrainingRun, shard_groups: list) -> int:
    """The steps every process takes in an epoch: the fewest any can give (whole batches per
    loaded group, as the batches are drawn; a short group's rest gives none)."""
    samples_of = {file: run.sizes[index] for index, file in enumerate(run.files)}
    batch = run.train_cfg.batch
    own_batches = sum(sum(samples_of[file] for file in group) // batch for group in shard_groups)
    fewest = torch.tensor([own_batches], device=run.dev)
    dist.all_reduce(fewest, op=dist.ReduceOp.MIN)
    return int(fewest.item())


def train_epoch(
    run: TrainingRun, state: TrainingState, epoch: int, loader: Prefetch, budget: int | None
) -> EpochOutcome:
    """One epoch's updates (`update_weights`) over its shard groups (`loader`, at most `budget`
    batches), at the step schedule's learning rate of the epoch (`step_lr`) or OneCycle's: the
    loss parts summed; every `log_every` steps TensorBoard points (`board_step`, the throughput),
    every TRACE_EVERY_BOARD_POINTS times that a row of the log."""
    train_cfg, dev = run.train_cfg, run.dev
    progress, optimizer, board = state.progress, state.optimizer, state.board
    if state.scheduler is None:
        for group in optimizer.param_groups:
            group["lr"] = step_lr(train_cfg, epoch)
    loss_sums, updates = torch.zeros(len(PARTS), dtype=torch.float64), 0
    batches = Prefetch(
        scene_batches(loader, run.cfg, state.generator, budget, dev.type == "cuda"),
        PREFETCH_BATCHES,
    )
    epoch_start = board_mark = time.perf_counter()
    board_waited = input_host_s = 0.0
    for drawn in batches:
        input_start = time.perf_counter()
        batch = scene_inputs(*drawn, run.cfg, run.feature_cfg, dev)
        input_host_s += time.perf_counter() - input_start  # host time only (CUDA work is async)
        updated = update_weights(run, state, batch)
        if updated is None:  # skipped: a non-finite loss or gradient
            progress.skipped += 1
            progress.bad_streak += 1
            stop_if_bad(progress.bad_streak, train_cfg, progress.step)
            continue
        step_parts, grad_norm = updated
        progress.bad_streak = 0
        loss_sums[: len(step_parts)] += step_parts
        updates += 1
        progress.step += 1
        if board is not None and progress.step % train_cfg.log_every == 0:
            board_step(run, state, batch, step_parts, grad_norm)
            board_time = time.perf_counter()
            elapsed, waited = board_time - board_mark, batches.waited - board_waited
            samples = train_cfg.log_every * train_cfg.batch * run.world
            board.add_scalar("samples_per_s", samples / elapsed, progress.step)
            board.add_scalar("data_wait_share", waited / elapsed, progress.step)
            board_mark, board_waited = board_time, batches.waited
        if run.rank == 0 and progress.step % (train_cfg.log_every * TRACE_EVERY_BOARD_POINTS) == 0:
            row = {"step": progress.step, **dict(zip(PARTS, step_parts.tolist(), strict=False))}
            row["lr"] = optimizer.param_groups[0]["lr"]
            row["seconds"] = round(time.time() - state.started)
            append_log(run, row)
    seconds = time.perf_counter() - epoch_start
    if updates == 0:  # every process alike (the skips are decided together)
        raise RuntimeError(
            f"epoch {epoch + 1}: no weight update ({progress.skipped} batches skipped)"
        )
    batches.close()
    loader.close()
    timings = {"train_input_host_s": input_host_s}
    return EpochOutcome(epoch, loss_sums, updates, seconds, batches.waited, timings)


def update_weights(
    run: TrainingRun, state: TrainingState, batch: tuple[dict, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One update on `batch` (inputs, waypoints, waypoint frames): its loss, its gradient
    clipped, the optimizer and schedule steps. Returns (the loss parts, the gradient norm), or
    None when the loss or the gradient is not finite: the update skipped, on every process alike
    (the loss check is reduced over them, the gradient norm is of averaged gradients)."""
    inputs, waypoints, waypoint_frames = batch
    dev, optimizer, scheduler = run.dev, state.optimizer, state.scheduler
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=run.cfg.bf16):
        loss_value, loss_parts = loss(
            state.forward,
            state.model.ip,
            inputs,
            waypoints.to(dev, non_blocking=True),
            waypoint_frames.to(dev, non_blocking=True),
            run.train_cfg.slot_weights,
        )
    optimizer.zero_grad()
    finite = torch.isfinite(loss_value.detach()).float()
    if run.world > 1:  # all processes skip a step together
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() < 1:
        return None
    loss_value.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), run.train_cfg.grad_clip)
    if not torch.isfinite(grad_norm):  # the same on every process (averaged gradients)
        optimizer.zero_grad()  # a non-finite gradient would poison the weights
        return None
    optimizer.step()
    if scheduler is not None and scheduler.last_epoch < scheduler.total_steps - 1:
        scheduler.step()
    return loss_parts.cpu().double(), grad_norm


def board_step(
    run: TrainingRun,
    state: TrainingState,
    batch: tuple[dict, torch.Tensor, torch.Tensor],
    step_parts: torch.Tensor,
    grad_norm: torch.Tensor,
) -> None:
    """A step's TensorBoard points: its loss parts, the attention logits on a few samples
    (QK-norm bounds them), the learning rate, the gradient norm, the batches skipped so far."""
    board, model, step = state.board, state.model, state.progress.step
    inputs, waypoints, _ = batch
    for key, part in zip(PARTS, step_parts.tolist(), strict=False):
        board.add_scalar(f"loss/{key}", part, step)
    model.eval()
    few_inputs = {key: value[:ATTENTION_SAMPLES] for key, value in inputs.items()}
    few_waypoints = waypoints[:ATTENTION_SAMPLES].to(run.dev)
    for key, logit in attention_logit_max(model, few_inputs, few_waypoints).items():
        board.add_scalar(f"attention_logit/{key}", logit, step)
    model.train()
    board.add_scalar("lr", state.optimizer.param_groups[0]["lr"], step)
    board.add_scalar("grad_norm", grad_norm.item(), step)
    board.add_scalar("skipped_batches", state.progress.skipped, step)


# ---------------------------------------------------------------- an epoch's end
def random_states(run: TrainingRun, generator: torch.Generator) -> tuple:
    """(the draws' generator state, the global streams' states: dropout draws from them), with
    several processes every process's in a list (the checkpoint keeps them all)."""
    generator_states = generator.get_state()
    rng_states = {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(run.dev) if run.dev.type == "cuda" else None,
    }
    if run.world > 1:
        gathered = [None] * run.world
        dist.all_gather_object(gathered, generator_states)
        generator_states = gathered
        gathered = [None] * run.world
        dist.all_gather_object(gathered, rng_states)
        rng_states = gathered
    return generator_states, rng_states


def precise_batchnorm(
    run: TrainingRun, model: RouteModel, cache: dict, timings: dict
) -> dict | None:
    """Precise BatchNorm on process 0 (`recalibrate_bn`), then process 0's buffers on every
    process: the last forward updated each process's BatchNorm statistics from its own batch
    (DDP broadcasts buffers at the start of a forward), so the same model, the one the
    checkpoint keeps, is validated and saved everywhere. Returns what was used, or None when
    off."""
    record = None
    if run.cfg.bn_batches and run.rank == 0:
        bn_start = time.perf_counter()
        record = recalibrate_bn(
            model, run.files, run.cfg, run.feature_cfg, run.dev, run.cfg.seed, cache
        )
        timings["bn_recalibrate_s"] = time.perf_counter() - bn_start
    if run.world > 1:
        for buffer in model.buffers():
            dist.broadcast(buffer, 0)
    return record


def track_best(progress: Progress, outcome: EpochOutcome) -> None:
    """The epoch's validation total (reg + cls + wp, as the training loss) and whether it is the
    best so far; the epochs since the best counted."""
    if outcome.val_result is None:
        return
    outcome.val_total = sum(outcome.val_result[key] for key in VALIDATION_PARTS)
    outcome.improved = outcome.val_total < progress.best_val
    if outcome.improved:
        progress.best_val, progress.best_epoch = outcome.val_total, outcome.epoch + 1
        progress.since_best = 0
    else:
        progress.since_best += 1


def save_checkpoint(run: TrainingRun, state: TrainingState, outcome: EpochOutcome) -> None:
    """Process 0's checkpoint of the epoch at run.out, replaced only once completely written (a
    failed write keeps the previous checkpoint); with keep_epochs also the epoch's weights apart
    (`keep_epoch`); the best model so far apart (<out>.best.pt)."""
    save_start = time.perf_counter()
    progress = state.progress
    checkpoint = {
        "state": state.model.state_dict(),
        "points": state.points,
        "train": asdict(run.train_cfg),
        "history": run.feature_cfg.history,
        "epoch": outcome.epoch + 1,
        "use_route": True,
        "use_sig": True,
        "use_gmm": False,
        "use_frenet": False,
        "use_kin": False,
        "route_train": asdict(run.cfg),
        "features": asdict(run.feature_cfg),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict() if state.scheduler is not None else None,
        "step": progress.step,
        "generator": outcome.generator_states,
        "world": run.world,
        "init": state.init_record,
        "train_files": [file.name for file in run.files],
        "rng": outcome.rng_states,
        "val": outcome.val_result,
        "val_total": outcome.val_total,
        "best_val": progress.best_val,
        "best_epoch": progress.best_epoch,
        "since_best": progress.since_best,
        "bn_recalibrated": outcome.bn_record,
    }
    written = temporary(run.out)
    torch.save(checkpoint, written)
    if run.cfg.keep_epochs:
        keep_epoch(run, checkpoint, outcome.epoch, written)
    if outcome.improved:
        copy_replacing(written, run.out.with_suffix(".best.pt"))
    os.replace(written, run.out)
    outcome.timings["checkpoint_save_s"] = time.perf_counter() - save_start


def temporary(path: Path) -> Path:
    """Where `path` is written before it replaces the previous one."""
    return path.with_name(path.name + ".tmp")


def keep_epoch(run: TrainingRun, checkpoint: dict, epoch: int, written: Path) -> None:
    """The epoch's weights kept apart (<out>.epochs/epochNN.pt, without the RESUME_ONLY state);
    the last epoch before the step schedule's first decay also whole (<out>.pre_decay.pt, to
    resume from with more epochs, or to start from with --init under another schedule)."""
    epochs_dir = run.out.with_name(run.out.stem + ".epochs")
    epochs_dir.mkdir(exist_ok=True)
    weights = {key: value for key, value in checkpoint.items() if key not in RESUME_ONLY}
    epoch_file = epochs_dir / f"epoch{epoch + 1:02d}.pt"
    torch.save(weights, temporary(epoch_file))
    os.replace(temporary(epoch_file), epoch_file)
    if run.train_cfg.schedule == "step" and epoch + 1 == decay_epochs(run.train_cfg).start:
        copy_replacing(written, run.out.with_suffix(".pre_decay.pt"))


def copy_replacing(source: Path, target: Path) -> None:
    """`source` copied to `target`, which is replaced only once the copy is complete."""
    shutil.copyfile(source, temporary(target))
    os.replace(temporary(target), target)


def epoch_mean_loss(run: TrainingRun, outcome: EpochOutcome) -> list[float]:
    """The epoch's mean loss parts over every process's updates (each process sums its own; the
    step trace and the board's losses are process 0's batches only)."""
    loss_sums, updates = outcome.loss_sums, outcome.updates
    if run.world > 1:
        count = torch.tensor([float(updates)], dtype=torch.float64)
        sums = torch.cat([loss_sums, count]).to(run.dev)
        dist.all_reduce(sums)
        sums = sums.cpu()
        loss_sums, updates = sums[:-1], int(sums[-1].item())
    return (loss_sums / max(updates, 1)).tolist()


def log_epoch(
    run: TrainingRun,
    state: TrainingState,
    outcome: EpochOutcome,
    mean_loss: list[float],
    end_start: float,
) -> None:
    """Process 0's row of the epoch: the mean training loss parts, the steps, the durations, the
    validation and the best so far, the timings; printed, logged and on TensorBoard."""
    outcome.timings["epoch_end_total_s"] = time.perf_counter() - end_start
    progress = state.progress
    row = {
        "epoch": outcome.epoch + 1,
        **dict(zip(PARTS, mean_loss, strict=False)),
        "steps": outcome.updates,
        "epoch_s": round(outcome.seconds),
        "data_wait_s": round(outcome.data_wait_s),
        "seconds": round(time.time() - state.started),
    }
    row["val"] = outcome.val_result
    row["val_total"], row["best"] = outcome.val_total, outcome.improved
    row["best_epoch"], row["best_val"] = progress.best_epoch, progress.best_val
    row["timings"] = {key: round(value, TIMING_DECIMALS) for key, value in outcome.timings.items()}
    print(json.dumps(row), flush=True)
    append_log(run, row)
    if state.board is not None:
        board_epoch(state.board, row, outcome.epoch)


def board_epoch(board: SummaryWriter, row: dict, epoch: int) -> None:
    """The epoch's TensorBoard points: the training loss parts, the validation and its total, the
    durations and timings."""
    for key in PARTS:
        if key in row:
            board.add_scalar(f"epoch/{key}", row[key], epoch + 1)
    for key, value in (row["val"] or {}).items():
        board.add_scalar(f"val/{key}", value, epoch + 1)
    if row["val_total"] is not None:
        board.add_scalar("val/total", row["val_total"], epoch + 1)
    board.add_scalar("epoch_s", row["epoch_s"], epoch + 1)
    board.add_scalar("data_wait_s", row["data_wait_s"], epoch + 1)
    for key, value in row["timings"].items():
        board.add_scalar(f"timing/{key}", value, epoch + 1)
    board.flush()


def stop_early(run: TrainingRun, outcome: EpochOutcome, progress: Progress) -> bool:
    """cfg.patience epochs without a better validation loss stop the run (every process decides
    alike); said and logged by process 0."""
    patience = run.cfg.patience
    if not patience or progress.since_best < patience:
        return False
    if run.rank == 0:
        stop = {
            "early_stop": outcome.epoch + 1,
            "best_val": progress.best_val,
            "patience": patience,
        }
        print(json.dumps(stop), flush=True)
        append_log(run, stop)
    return True
