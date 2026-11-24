"""`compare` compares trajectory generators on held-out raw scenes (accepting a retrained one).

Every model reads the same samples through the inputs generation uses (`inputs.track_inputs`,
each with the context sizes it was trained with), so checkpoints compare on equal terms. A
sample (`_samples`) is a road user learned from (`data_preprocess.scene_centers`) of a held-out
shard, taken out of its scene and given back as the track to complete, with its logged route
(as in training). Reported per model, on the agent's 8 s future:

  no_waypoints    minADE6 / minFDE6 of the 6 best-scored candidates, no waypoints; the route
                  (from the logged future: a vehicle's lanes, else its simplified path) and the
                  signal phases are still given, so this is generation along a known route, not
                  motion prediction from the past alone
  waypoints       a contact waypoint at a random frame and a reveal waypoint 0.5-5 s before it
                  (`_waypoints`): minADE6 / minFDE6, and the waypoint error (the larger
                  of the two distances between a candidate at the waypoint's frame and the
                  waypoint) of the best candidate and of the best-scored one, median and p90
each also for the moving samples alone (moved more than 5 m in the 8 s; None where there are none,
the count under "moving"). For the moving samples, per type (and all): `_breakdown` of each
waypoint (reveal, contact) into how close a candidate passes it and how early or late it gets
there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .datasets.dataset import MAX_WAYPOINTS, WAYPOINT_SIZE, waypoint_time
from .datasets.routes import logged_route, scene_lanes
from .datasets.waymo.data_preprocess import scene_centers
from .datasets.waymo.waymo_dataset import FRAMES_PER_S, FUTURE_FRAMES, TYPES, FeatureConfig
from .eval_utils import TOP_K
from .inputs import to_local, track_inputs
from .modeling.model import RouteModel, load_route_model

HIT_M, HIT_S = 0.5, 0.25  # a waypoint hit: passed within 0.5 m, at most 0.25 s early or late
MOVING_M = 5.0  # a moving sample moved more than this in the 8 s
ANCHOR_AND_FUTURE = FUTURE_FRAMES + 1  # a sample's frames: the anchor, then its 8 s future
# a sample's contact waypoint from this frame on, its reveal waypoint 0.5 s to 5 s before (the
# training's draws, TrainConfig.contact_frames and window_frames; the ends exclusive)
CONTACT_FIRST_FRAME = 14
REVEAL_LEAD_MIN_FRAMES, REVEAL_LEAD_END_FRAMES = 5, 51
HIGH_PERCENTILE = 90  # the report's p90
DECIMALS = 3  # the report's values rounded to this many decimals


@dataclass(frozen=True)
class CompareConfig:
    """The samples drawn from the shard, the seed of their draw and their waypoints', and the
    samples per forward."""

    samples: int = 4000
    seed: int = 1
    batch: int = 64


@dataclass(frozen=True)
class SampleSet:
    """What every model is measured on, per sample: its logged 8 s future [N, 80, 2] in its
    frame, its waypoints [N, 3, 6] (`_waypoints`) and their frames [N, 2], whether it moves
    (more than MOVING_M), its type."""

    fut: torch.Tensor
    waypoints: torch.Tensor
    frames: np.ndarray
    moving: torch.Tensor
    types: list


@torch.no_grad()
def compare(
    checkpoints: list[str | Path],
    shard: str | Path,
    cfg: CompareConfig | None = None,
    version: str | None = None,
) -> dict:
    """The report of every checkpoint on the same samples of the raw `shard` (see the module
    docstring): the samples (`_samples`), their inputs per model (`_sample_inputs`), then each
    model's report (`_model_report`)."""
    cfg = cfg or CompareConfig()
    models = {str(checkpoint): load_route_model(checkpoint) for checkpoint in checkpoints}
    refs, samples = _samples(shard, cfg, version)
    inputs, sample_set = _sample_inputs(models, refs, samples, cfg, version)
    moving = sample_set.moving
    report = {"samples": len(moving), "moving": int(moving.sum()), "shard": Path(shard).name}
    for name, model in models.items():
        report[name] = _model_report(model, inputs[name], sample_set, cfg.batch)
    return report


def _samples(
    shard: str | Path, cfg: CompareConfig, version: str | None
) -> tuple[list, list[tuple[int, str]]]:
    """[(record index, track id)] drawn from the learned-from road users of the shard (the ego
    excluded: generation never completes it, and a scene cannot lose it)."""
    from ..data.waymo.scenes import load_record
    from ..data.waymo.tfrecord import index_records

    refs = index_records(Path(shard))
    pool = []
    for record_index, ref in enumerate(refs):
        scene = load_record(ref, dataset_version=version) if version else load_record(ref)
        tracks = [track for track in scene.tracks.values() if track.agent_type in TYPES]
        picked = (tracks[index].track_id for index in scene_centers(scene, FeatureConfig(), tracks))
        # the ego never completed
        pool += [(record_index, track_id) for track_id in picked if track_id != scene.ego_id]
    rng = np.random.default_rng(cfg.seed)
    chosen = sorted(rng.choice(len(pool), min(cfg.samples, len(pool)), replace=False))
    return refs, [pool[index] for index in chosen]


def _sample_inputs(
    models: dict, refs: list, samples: list, cfg: CompareConfig, version: str | None
) -> tuple[dict, SampleSet]:
    """(per model the inputs of every sample, the samples' `SampleSet`): each sample taken out of
    its scene and given back as the track to complete, with its logged route (each scene loaded
    once)."""
    from ..data.waymo.scenes import load_record

    rng = np.random.default_rng(cfg.seed + 1)
    inputs = {name: [] for name in models}
    futs, wps, frames, types = [], [], [], []
    by_record: dict[int, list[str]] = {}
    for record_index, track_id in samples:
        by_record.setdefault(record_index, []).append(track_id)
    for record_index, track_ids in by_record.items():
        scene = (
            load_record(refs[record_index], dataset_version=version)
            if version
            else load_record(refs[record_index])
        )
        anchor, lanes = scene.anchor, scene_lanes(scene)
        for track_id in track_ids:
            track = scene.tracks[track_id]
            others = replace(
                scene, tracks={key: value for key, value in scene.tracks.items() if key != track_id}
            )
            world = track.states[anchor : anchor + ANCHOR_AND_FUTURE, :2]
            route = logged_route(
                track.agent_type, world, lanes, track.states[anchor : anchor + ANCHOR_AND_FUTURE, 2]
            )
            origin, h0 = track.states[anchor, :2], track.states[anchor, 2]
            fut = to_local(world[1:], origin, h0)
            waypoints, waypoint_frames = _waypoints(fut, rng)
            futs.append(fut)
            types.append(track.agent_type)
            wps.append(waypoints)
            frames.append(waypoint_frames)
            for name, model in models.items():
                track_batch, _, _ = track_inputs(others, track, model, route)
                track_batch.pop("fut", None)
                inputs[name].append(track_batch)
    fut = torch.tensor(np.array(futs), dtype=torch.float32)  # [N, 80, 2]
    moving = fut[:, -1].norm(dim=-1) > MOVING_M
    return inputs, SampleSet(fut, torch.stack(wps), np.array(frames), moving, types)


def _waypoints(fut: np.ndarray, rng: np.random.Generator) -> tuple[torch.Tensor, np.ndarray]:
    """[3, 6] contact and reveal waypoints (x, y, `waypoint_time`, 0, 0, present) and their
    frames."""
    contact_frame = int(rng.integers(CONTACT_FIRST_FRAME, len(fut)))
    reveal_frame = max(
        contact_frame - int(rng.integers(REVEAL_LEAD_MIN_FRAMES, REVEAL_LEAD_END_FRAMES)), 0
    )
    waypoints = torch.zeros(MAX_WAYPOINTS, WAYPOINT_SIZE)
    waypoints[0] = torch.tensor([*fut[contact_frame], waypoint_time(contact_frame + 1), 0, 0, 1])
    waypoints[1] = torch.tensor([*fut[reveal_frame], waypoint_time(reveal_frame + 1), 0, 0, 1])
    return waypoints, np.array([contact_frame, reveal_frame])


def _model_report(model: RouteModel, rows: list, sample_set: SampleSet, batch_size: int) -> dict:
    """One model's report: per condition (no waypoints, the waypoints) minADE6 and minFDE6 of all
    samples and of the moving ones; with the waypoints also the waypoint errors
    (`_waypoint_errors`) and, per type, the moving samples' `_breakdown`."""
    fut, all_waypoints = sample_set.fut, sample_set.waypoints
    frames, moving = sample_set.frames, sample_set.moving
    batch = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    out = {}
    for condition, condition_waypoints in (
        ("no_waypoints", torch.zeros_like(all_waypoints)),
        ("waypoints", all_waypoints),
    ):
        traj, score = _predict(model, batch, condition_waypoints, len(fut), batch_size)
        traj = traj[:, :, : fut.shape[1]]
        top = score.topk(TOP_K, dim=1).indices
        errors = (traj[torch.arange(len(top))[:, None], top] - fut[:, None]).norm(dim=-1)
        ade, fde = errors.mean(-1).min(1).values, errors[..., -1].min(1).values
        for sub, keep in (("", slice(None)), ("moving_", moving)):
            out[f"{sub}{condition}_minADE6"] = _stat(ade[keep], torch.mean)
            out[f"{sub}{condition}_minFDE6"] = _stat(fde[keep], torch.mean)
        if condition == "waypoints":
            out.update(_waypoint_errors(traj, score, sample_set))
            kinds = np.array(sample_set.types)
            out["moving_waypoint_breakdown"] = {
                kind: _breakdown(traj[keep], score[keep], all_waypoints[keep], frames[keep])
                for kind in ("all", *TYPES)
                if (keep := moving.numpy() & ((kinds == kind) | (kind == "all"))).any()
            }
    return out


def _predict(
    model: RouteModel, batch: dict, waypoints: torch.Tensor, samples: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(trajectories, scores) of the model on every sample, `batch_size` samples at a time."""
    traj, score = [], []
    for start in range(0, samples, batch_size):
        part = {key: value[start : start + batch_size] for key, value in batch.items()}
        part_traj, part_score = model(part, waypoints[start : start + batch_size])
        traj.append(part_traj)
        score.append(part_score)
    return torch.cat(traj), torch.cat(score)


def _waypoint_errors(traj: torch.Tensor, score: torch.Tensor, sample_set: SampleSet) -> dict:
    """The waypoint error (the larger of a candidate's distances to the contact and the reveal
    waypoint at their frames) of the best candidate and of the best-scored one, median and p90,
    of all samples and of the moving ones."""
    all_waypoints, frames, moving = sample_set.waypoints, sample_set.frames, sample_set.moving
    out = {}
    sample_rows = torch.arange(len(traj))
    at = torch.stack([traj[sample_rows, :, frames[:, row]] for row in range(2)], 1)  # [N, 2, K, 2]
    miss = (at - all_waypoints[:, :2, None, :2]).norm(dim=-1).max(1).values  # [N, K]: the larger
    best = miss.min(1).values.numpy()
    best_scored = miss[torch.arange(len(miss)), score.argmax(1)].numpy()
    for sub, keep in (("", slice(None)), ("moving_", moving.numpy())):
        for label, errors_kept in (
            ("best", best[keep]),
            ("best_scored", best_scored[keep]),
        ):
            key = f"{sub}waypoint_error_{label}"
            out[f"{key}_median"] = _stat(errors_kept, np.median)
            out[f"{key}_p90"] = _stat(
                errors_kept, lambda values: np.percentile(values, HIGH_PERCENTILE)
            )
    return out


def _stat(values: np.ndarray | torch.Tensor, statistic: Callable) -> float | None:
    """statistic(values) rounded; None for an empty subset (not computable: where all samples of
    a shard stand, the moving subset is empty and np.percentile fails on it)."""
    return round(float(statistic(values)), DECIMALS) if len(values) else None


def _breakdown(
    traj: torch.Tensor, score: torch.Tensor, waypoints: torch.Tensor, frames: np.ndarray
) -> dict:
    """Each waypoint (reveal, contact) of these samples: the best-scored candidate's closest pass
    (m) and its time offset there (s, closest frame minus the waypoint's frame; negative: early),
    medians and p90 of the absolute (diagnostics); the shares of samples whose best-scored
    candidate, one of the 6 best-scored, or one of all candidates hits both waypoints: within
    HIT_M of each at some frame within HIT_S of its time (the one candidate; not the
    closest pass's time, which a candidate standing on the waypoint gets wrong)."""
    rows = torch.arange(len(traj))
    best_scored = score.argmax(1)
    top = score.topk(min(TOP_K, score.shape[1]), dim=1).indices
    out, both = {"samples": len(traj)}, None
    for row, label in ((1, "reveal"), (0, "contact")):  # waypoint rows: 0 contact, 1 reveal
        distances = (traj - waypoints[:, None, None, row, :2]).norm(dim=-1)  # [N, K, F]
        closest, closest_frame = distances.min(-1)  # closest pass and its frame, per candidate
        offset = (closest_frame - torch.as_tensor(frames[:, row])[:, None]).float() / FRAMES_PER_S
        frame_numbers = torch.arange(distances.shape[-1])
        window = (frame_numbers[None, :] - torch.as_tensor(frames[:, row])[:, None]).abs() <= round(
            HIT_S * FRAMES_PER_S
        )
        hit = distances.masked_fill(~window[:, None, :], float("inf")).min(-1).values <= HIT_M
        both = hit if both is None else both & hit
        for key, values in (
            ("pass_m", closest[rows, best_scored]),
            ("offset_s", offset[rows, best_scored].abs()),
        ):
            values = values.numpy()
            out[f"{label}_{key}_median"] = round(float(np.median(values)), DECIMALS)
            out[f"{label}_{key}_p90"] = round(
                float(np.percentile(values, HIGH_PERCENTILE)), DECIMALS
            )
        late = offset[rows, best_scored]
        out[f"{label}_late_share"] = round(float((late > HIT_S).float().mean()), DECIMALS)
        out[f"{label}_early_share"] = round(float((late < -HIT_S).float().mean()), DECIMALS)
    out["hit_both_best_scored"] = round(float(both[rows, best_scored].float().mean()), DECIMALS)
    out["hit_both_top6"] = round(float(both[rows[:, None], top].any(1).float().mean()), DECIMALS)
    out["hit_both_any"] = round(float(both.any(1).float().mean()), DECIMALS)
    return out
