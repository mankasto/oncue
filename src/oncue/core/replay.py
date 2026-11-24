"""Replay rule: fill missing states where the log leaves no doubt, leave the rest absent.

Every test runs on the replayed scene; training and log statistics read the raw records. The ego
is never filled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from ..utils.geometry import boxes_overlap, lerp_states
from ..utils.tracks import TIME_TOL_S
from .config import ReplayConfig
from .types import Scene, Track

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

MIN_GAP_FRAMES = 2  # observations this many frames apart or more have frames missing between
BEFORE, AFTER = -1, 1  # frame step from an observation outward
# Boxes whose circumcircles are farther apart than this cannot touch (a speed-up only).
NEAR_MARGIN_M = 0.1


def fill_gaps(scene: Scene, cfg: ReplayConfig) -> Scene:
    """Return ``scene`` with every non-ego track filled per ``cfg``."""
    everyone = _Others.of(list(scene.tracks.values()))  # stack once per scene
    tracks = {}
    for index, (track_id, track) in enumerate(scene.tracks.items()):
        if track_id == scene.ego_id:
            tracks[track_id] = track
            continue
        tracks[track_id] = _filled(track, scene.timestamps, cfg, replace(everyone, skip=index))
    return replace(scene, tracks=tracks)


def fill_track(
    track: Track, timestamps: np.ndarray, cfg: ReplayConfig, others: Sequence[Track] = ()
) -> Track:
    """Return the track with missing frames filled (marked in ``filled``), clear of ``others``."""
    return _filled(track, timestamps, cfg, _Others.of(others))


@dataclass(frozen=True, eq=False)
class _Others:
    """Road users a standing end must stay clear of; ``skip`` is the track being filled."""

    tracks: tuple[Track, ...]
    states: np.ndarray
    valid: np.ndarray
    radii: np.ndarray
    skip: int | None = None

    @classmethod
    def of(cls, tracks: Sequence[Track]) -> _Others:
        if not tracks:
            return cls((), np.zeros((0, 0, 2)), np.zeros((0, 0), bool), np.zeros(0))
        return cls(
            tuple(tracks),
            np.stack([t.states for t in tracks]),
            np.stack([t.valid for t in tracks]),
            np.array([np.hypot(*np.asarray(t.size, float)) / 2 for t in tracks]),
        )

    def touching(self, frames: np.ndarray, state: np.ndarray, size: ArrayLike) -> np.ndarray:
        """Return per frame whether the box ``(state, size)`` touches one of them."""
        touched = np.zeros(len(frames), bool)
        if not len(frames) or not len(self.tracks):
            return touched
        reach = np.hypot(*np.asarray(size, float)) / 2
        gap = np.linalg.norm(self.states[:, frames, :2] - state[:2], axis=-1)
        near = self.valid[:, frames] & (gap <= reach + self.radii[:, None] + NEAR_MARGIN_M)
        if self.skip is not None:
            near[self.skip] = False
        for n in np.flatnonzero(near.any(axis=1)):
            other, close = self.tracks[n], near[n]
            touched[close] |= boxes_overlap(state, size, other.states[frames[close]], other.size)
        return touched


def _filled(track: Track, timestamps: np.ndarray, cfg: ReplayConfig, others: _Others) -> Track:
    """Inner form of ``fill_track`` with the others already stacked."""
    seen = np.flatnonzero(track.valid)
    if not len(seen) or len(seen) == len(track.valid):
        return track
    t = np.asarray(timestamps, float)
    out = _Filling(
        states=track.states.copy(),
        velocity=None if track.velocity is None else track.velocity.copy(),
        z=None if track.z is None else track.z.copy(),
        filled=np.zeros(len(t), bool),
    )
    still = track.states[:, 3] < cfg.still_mps
    _fill_gaps_between(out, track, t, seen, still, cfg)
    _fill_ends(out, track, t, seen, still, cfg, others)
    if not out.filled.any():
        return track
    return replace(
        track,
        states=out.states,
        valid=track.valid | out.filled,
        velocity=out.velocity,
        z=out.z,
        filled=out.filled,
    )


@dataclass
class _Filling:
    """Working copies of a track's states, velocity and z, and the fill mask."""

    states: np.ndarray
    velocity: np.ndarray | None
    z: np.ndarray | None
    filled: np.ndarray


def _fill_gaps_between(
    out: _Filling,
    track: Track,
    t: np.ndarray,
    seen: np.ndarray,
    still: np.ndarray,
    cfg: ReplayConfig,
) -> None:
    """Fill gaps between observations: short ones always, longer ones only standing in place."""
    for i, j in pairwise(seen):
        if j - i < MIN_GAP_FRAMES:
            continue
        apart = np.hypot(*(track.states[j, :2] - track.states[i, :2]))
        if t[j] - t[i] <= cfg.bridge_s + TIME_TOL_S:
            _between(out, track, t, i, j, standing=False)
        elif still[i] and still[j] and apart <= cfg.same_place_m:
            _between(out, track, t, i, j, standing=True)


def _between(out: _Filling, track: Track, t: np.ndarray, i: int, j: int, standing: bool) -> None:
    """Interpolate the frames between observations ``i`` and ``j`` (at rest if ``standing``)."""
    frames = np.arange(i + 1, j)
    fraction = ((t[frames] - t[i]) / (t[j] - t[i]))[:, None]
    out.states[frames] = lerp_states(track.states[i], track.states[j], fraction[:, 0])
    if out.velocity is not None:
        out.velocity[frames] = out.velocity[i] + fraction * (out.velocity[j] - out.velocity[i])
    if out.z is not None:
        out.z[frames] = out.z[i] + fraction[:, 0] * (out.z[j] - out.z[i])
    if standing:
        out.states[frames, 3] = 0.0
        if out.velocity is not None:
            out.velocity[frames] = 0.0
    out.filled[frames] = True


def _fill_ends(
    out: _Filling,
    track: Track,
    t: np.ndarray,
    seen: np.ndarray,
    still: np.ndarray,
    cfg: ReplayConfig,
    others: _Others,
) -> None:
    """Extend a still road user before its first and after its last observation."""
    first, last = seen[0], seen[-1]
    if first > 0 and still[first]:
        _stand(out, track, _clear_end(track, t, first, BEFORE, cfg, others), first)
    if last < len(t) - 1 and still[last]:
        _stand(out, track, _clear_end(track, t, last, AFTER, cfg, others), last)


def _clear_end(
    track: Track, t: np.ndarray, k: int, step: int, cfg: ReplayConfig, others: _Others
) -> np.ndarray:
    """Return frames outward from observation ``k`` within ``cfg.end_fill_s``, up to a touch."""
    stop = -1 if step == BEFORE else len(t)
    frames = np.arange(k + step, stop, step)
    frames = frames[np.abs(t[frames] - t[k]) <= cfg.end_fill_s + TIME_TOL_S]
    touched = others.touching(frames, track.states[k], track.size)
    blocked = np.flatnonzero(touched)
    return frames[: blocked[0]] if len(blocked) else frames


def _stand(out: _Filling, track: Track, frames: np.ndarray, k: int) -> None:
    """Fill ``frames`` with observation ``k``'s state at zero speed and zero velocity."""
    out.states[frames] = track.states[k]
    out.states[frames, 3] = 0.0
    if out.velocity is not None:
        out.velocity[frames] = 0.0
    if out.z is not None:
        out.z[frames] = out.z[k]
    out.filled[frames] = True
