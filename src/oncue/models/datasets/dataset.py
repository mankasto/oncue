"""The training data in the order `train.train_route` reads it: the shards (`split_files`), the
intention points (`scene_probe`), the scenes kept out (`excluded`), then per epoch the shard groups
(`group_files`, a `SceneStore` each) loaded in the background (`Prefetch`), their
batches (`scene_batches`, `draws`) and model inputs (`scene_inputs`). MTR's
`mtr/datasets/dataset.py` with its DataLoader."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...utils.geometry import rotate
from ..config import RouteTrainConfig
from ..utils.motion_utils import intention_points
from .waymo.waymo_dataset import (
    FRAME_KEYS,
    FUTURE_FRAMES,
    HIST_ACCELERATION,
    HIST_ELEVATION,
    HIST_SPEED,
    HIST_VALID,
    HIST_VELOCITY,
    LAYOUT,
    pad,
    pad_frames,
)

if TYPE_CHECKING:  # annotations only
    from .waymo.waymo_dataset import FeatureConfig

MAX_WAYPOINTS = 3  # timed waypoints per sample: the model's waypoint input wp [B, 3, 6]
WAYPOINT_SIZE = 6  # a waypoint row: x, y, time, speed, speed given, present
WAYPOINT_TIME_FRAMES = 80.0  # a waypoint's time input: its frames after the anchor over these (8 s)
SPEED_GIVEN_SHARE = 0.5  # a drawn waypoint carries its speed this often
VALID_ABOVE = 0.5  # hist's valid channel (0 or 1) counts as set above this
PUT_RETRY_S = 0.5  # Prefetch's thread, its queue full, checks this often whether to stop


def split_files(data_dir: str | Path, split: str) -> list[Path]:
    """The feature shards of one split (\"train\" or \"val\") in `data_dir`, sorted by name."""
    return sorted(Path(data_dir).glob(f"*.{split}.npz"))


def scene_probe(files: list[Path], points_per_type: int) -> torch.Tensor:
    """The intention points [3, k, 2] (`points_per_type` per agent type) from the 8 s endpoints of
    the samples of `files`."""
    ends, types = [], []
    for file in files:
        arrays = np.load(file)  # each array read (decompressed) once
        agents, sample_scenes = arrays["agents"], arrays["sample_scene"]
        rows = arrays["agent_off"][sample_scenes] + arrays["center"]
        anchor = arrays["a0"][sample_scenes]
        start, end = (
            agents[rows, anchor],
            agents[rows, np.minimum(anchor + FUTURE_FRAMES, agents.shape[1] - 1)],
        )
        ends.append(rotate(end[:, :2] - start[:, :2], -start[:, 2]))
        types.append(arrays["agent_static"][rows, 0].astype(int))
    return intention_points(np.concatenate(ends), np.concatenate(types), points_per_type)


def excluded(feat_dir: str | Path) -> dict | None:
    """The scenes kept out of the features (`excluded.json` of `data_preprocess.extract`), if
    recorded."""
    record = Path(feat_dir) / "excluded.json"
    return json.loads(record.read_text()) if record.exists() else None


# ------------------------------- an epoch: its shard groups loaded, its batches and model inputs
def group_files(files: list[Path], size: int, generator: torch.Generator) -> list[list[Path]]:
    """The epoch's shard groups, `size` shards each in a random order (`size` <= 0: all in one),
    drawn before any is loaded, so the batches they give are known in advance."""
    if size <= 0:
        return [list(files)]
    order = torch.randperm(len(files), generator=generator).tolist()
    return [
        [files[index] for index in order[start : start + size]]
        for start in range(0, len(order), size)
    ]


class SceneStore:
    """Scene-layout feature files loaded together (`data_preprocess.write_scenes`): per-scene
    arrays with offsets, and the sample table (scene, center, route, route_ok, rest)."""

    def __init__(self, files: list[Path]):
        loaded = [np.load(file) for file in files]
        frames = max(part["agents"].shape[1] for part in loaded)

        def cat(key: str) -> np.ndarray:
            arrays = [part[key] for part in loaded]
            if key in FRAME_KEYS:
                arrays = [pad_frames(array, frames) for array in arrays]
            return np.concatenate(arrays)

        self.arrays = {key: cat(key) for key in LAYOUT}
        agent_off, map_off, scene_start, sample_scene = [0], [0], 0, []
        for part in loaded:
            agent_off += list(agent_off[-1] + part["agent_off"][1:])
            map_off += list(map_off[-1] + part["map_off"][1:])
            sample_scene.append(part["sample_scene"] + scene_start)
            scene_start += len(part["a0"])
        self.agent_off, self.map_off = np.array(agent_off), np.array(map_off)
        self.anchor = cat("a0")
        self.sample_scene = np.concatenate(sample_scene)
        self.center, self.route = cat("center"), cat("route")
        self.route_ok, self.rest = cat("route_ok"), cat("rest")

    def __len__(self) -> int:
        return len(self.center)

    def batch(self, samples: np.ndarray) -> dict:
        """Samples `samples` as padded tensors (`waymo_dataset.assemble` inputs plus center, the
        anchor frame as key a0, route)."""
        scenes = self.sample_scene[samples]
        views = []
        for scene in scenes:
            agent_rows = slice(self.agent_off[scene], self.agent_off[scene + 1])
            map_rows = slice(self.map_off[scene], self.map_off[scene + 1])
            views.append(
                {
                    key: value[agent_rows if LAYOUT[key][0] == "agent" else map_rows]
                    for key, value in self.arrays.items()
                }
            )
        out = pad(views)
        out = {key: torch.from_numpy(value) for key, value in out.items()}
        out["center"] = torch.from_numpy(self.center[samples].astype(np.int64))
        out["a0"] = torch.from_numpy(self.anchor[scenes].astype(np.int64))
        out["route"] = torch.from_numpy(self.route[samples].astype(np.float32))
        out["route_ok"] = torch.from_numpy(self.route_ok[samples].astype(np.float32))
        return out


class Prefetch:
    """Iterates `items` in a background thread, up to `depth` ahead (loading the next shard group,
    preparing the next batches on the CPU while the GPU computes). Exceptions are re-raised."""

    _END = object()

    def __init__(self, items: Iterable, depth: int):
        self.queue: queue.Queue = queue.Queue(depth)
        self.error: BaseException | None = None
        self.waited = 0.0  # seconds the consumer waited for an item
        self.stop = threading.Event()

        def put(item: object) -> bool:
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=PUT_RETRY_S)
                    return True
                except queue.Full:
                    pass
            return False

        def run() -> None:
            try:
                for item in items:
                    if not put(item):
                        return
            except BaseException as error:
                self.error = error
            finally:
                put(self._END)

        threading.Thread(target=run, daemon=True).start()

    def close(self) -> None:
        """Stop the thread and drop what it prepared (a process that stopped early at the common
        step budget would otherwise keep a loaded shard group, blocked, every epoch)."""
        self.stop.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def __iter__(self) -> Iterator:
        while True:
            wait_start = time.perf_counter()
            item = self.queue.get()
            self.waited += time.perf_counter() - wait_start
            if item is self._END:
                if self.error is not None:
                    raise self.error
                return
            yield item


def scene_batches(
    groups: Iterable[SceneStore],
    cfg: RouteTrainConfig,
    generator: torch.Generator,
    budget: int | None,
    pin: bool,
) -> Iterator[tuple[dict, dict]]:
    """The epoch's batches: per shard group a start-from-rest weighted resampling, per batch the
    padded scenes of its samples and its random draws. `budget`: at most this many batches
    (several processes take the same number)."""
    train_cfg, done = cfg.train, 0
    for store in groups:
        if budget is not None and done >= budget:
            return
        weights = torch.where(
            torch.from_numpy(store.rest), torch.tensor(train_cfg.start_weight), torch.tensor(1.0)
        )
        order = torch.multinomial(weights, len(store), replacement=True, generator=generator)
        for batch_number in range(len(store) // train_cfg.batch):
            if budget is not None and done >= budget:
                return
            samples = order[
                batch_number * train_cfg.batch : (batch_number + 1) * train_cfg.batch
            ].numpy()
            stored, drawn = store.batch(samples), draws(len(samples), cfg, generator)
            if pin:
                stored = {key: value.pin_memory() for key, value in stored.items()}
            done += 1
            yield stored, drawn
        del store


def draws(count: int, cfg: RouteTrainConfig, generator: torch.Generator) -> dict:
    """The random draws of one batch (input dropouts and waypoints), shape-only, on the CPU."""
    train_cfg = cfg.train
    return {
        "drop": torch.rand(count, generator=generator) < train_cfg.history_dropout,
        "state_only": torch.rand(count, generator=generator) < cfg.p_state_only,
        "route_off": torch.rand(count, generator=generator) < cfg.p_route_off,
        "sig_off": torch.rand(count, generator=generator) < cfg.p_sig_off,
        "count": torch.multinomial(
            torch.tensor(train_cfg.waypoint_counts), count, replacement=True, generator=generator
        ),
        "tc": torch.randint(*train_cfg.contact_frames, (count,), generator=generator),
        "window": torch.randint(*train_cfg.window_frames, (count,), generator=generator),
        "extra": torch.rand(count, generator=generator),
        "noise": torch.randn(count, MAX_WAYPOINTS, 2, generator=generator),
        "given": (
            torch.rand(count, MAX_WAYPOINTS, generator=generator) < SPEED_GIVEN_SHARE
        ).float(),
    }


def scene_inputs(
    stored: dict, drawn: dict, cfg: RouteTrainConfig, feature_cfg: FeatureConfig, dev: torch.device
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """A drawn batch on the device: assembled (`waymo_dataset.assemble`), dropouts and waypoints."""
    from .waymo.waymo_dataset import assemble

    stored = {key: value.to(dev, non_blocking=True) for key, value in stored.items()}
    drawn = {key: value.to(dev, non_blocking=True) for key, value in drawn.items()}
    inputs = assemble(stored, stored["center"], stored["a0"], feature_cfg)
    inputs["route"], inputs["route_ok"] = stored["route"], stored["route_ok"]
    return _apply_draws(inputs, drawn, cfg)


def _apply_draws(
    inputs: dict, drawn: dict, cfg: RouteTrainConfig
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """The draws applied (on the inputs' device): history dropout keeping the current pose
    (speed, velocity and acceleration cleared), the t0-state-only variant, route and signal
    dropout, and the waypoints from the sample's own future. The draw "tc" is the contact
    frame."""
    train_cfg = cfg.train
    hist = inputs["hist"].clone()
    drop = drawn["drop"]
    hist[drop, :-1] = 0
    hist[drop, -1, HIST_SPEED] = 0
    hist[drop, -1, HIST_VELOCITY:HIST_ELEVATION] = 0  # velocity and acceleration
    state_only = drawn["state_only"] & (hist[:, :-1, HIST_VALID].sum(1) > 0)
    hist[state_only, :-1] = 0
    hist[state_only, -1, HIST_ACCELERATION:HIST_ELEVATION] = (
        0  # its acceleration came from the history: none, as generating
    )
    inputs = {**inputs, "hist": hist}
    # the start velocity known unless the dropout cleared it (the t0-state-only
    # variant keeps it, as a generated rolling start has it)
    inputs["v0_ok"] = ((~drop) & (hist[:, -1, HIST_VALID] > VALID_ABOVE)).float()
    inputs["route_ok"] = inputs["route_ok"] * (~drawn["route_off"]).float()
    inputs["msig"] = inputs["msig"] * (~drawn["sig_off"])[:, None, None].float()
    fut = inputs["fut"]
    reveal_frame = (drawn["tc"] - drawn["window"]).clamp(min=0)
    extra_frame = (drawn["extra"] * drawn["tc"].float()).long()
    frames = torch.stack([drawn["tc"], reveal_frame, extra_frame], 1)
    present = torch.arange(MAX_WAYPOINTS, device=fut.device)[None] < drawn["count"][:, None]
    positions = (
        torch.gather(fut[..., :2], 1, frames[..., None].expand(-1, -1, 2))
        + train_cfg.noise_m * drawn["noise"]
    )
    speed = torch.gather(fut[..., 2], 1, frames)
    given = drawn["given"]
    time_share = waypoint_time(frames + 1)[..., None]
    ones = torch.ones_like(given)[..., None]
    waypoints = torch.cat(
        [positions, time_share, (speed * given)[..., None], given[..., None], ones], -1
    )
    return (
        inputs,
        waypoints * present[..., None],
        torch.where(present, frames, torch.full_like(frames, -1)),
    )


def waypoint_time(frames_after_anchor: int | torch.Tensor) -> float | torch.Tensor:
    """A waypoint's time input (wp[..., 2]): its frames after the anchor as a share of the 8 s
    future; training, `compare` and generation build the waypoint rows with it."""
    return frames_after_anchor / WAYPOINT_TIME_FRAMES
