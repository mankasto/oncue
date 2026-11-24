"""Interpolate a track's state at arbitrary times between its logged frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .geometry import lerp_states

if TYPE_CHECKING:
    from typing import Protocol

    from numpy.typing import ArrayLike

    class TrackArrays(Protocol):
        """The shape utils/ needs from a track, without importing ``core.types.Track``."""

        states: np.ndarray  # [T, >= 3]
        valid: np.ndarray  # [T]


FRAME_SNAP = 1e-6  # a query within this many frames of a frame is treated as at it
# how close two times must be to count as equal (covers timestamp rounding and frame-step sums)
TIME_TOL_S = 1e-9


def frame_of(seconds: float, dt: float) -> int:
    """Return the frame index of ``seconds`` at step ``dt``, rounded half to even."""
    return int(np.rint(seconds / dt))


def track_at(
    track: TrackArrays, timestamps: np.ndarray, query: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(states, valid)`` of a track at the ``query`` times (seconds).

    States interpolate between adjacent frames and are invalid where either neighbour is; a query
    within ``FRAME_SNAP`` of a frame snaps to it.
    """
    t0, dt = timestamps[0], timestamps[1] - timestamps[0]
    query = np.asarray(query, float)
    position = (query - t0) / dt
    nearest = np.rint(position)
    position = np.where(np.abs(position - nearest) < FRAME_SNAP, nearest, position)
    left = np.clip(np.floor(position).astype(int), 0, len(timestamps) - 1)
    right = np.minimum(left + 1, len(timestamps) - 1)
    w = np.clip(position - left, 0.0, 1.0)
    exact = w < FRAME_SNAP
    valid = track.valid[left] & (exact | track.valid[right])
    a = track.states[left]
    b = np.where(track.valid[right][:, None], track.states[right], a)  # don't read invalid states
    return lerp_states(a, b, w), valid


def substep_states(
    track: TrackArrays, timestamps: np.ndarray, start: int, end: int, substeps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(times, states, valid)`` sampled ``substeps`` times per frame from ``start`` to
    ``end`` (inclusive)."""
    dt = float(timestamps[1] - timestamps[0])
    times = timestamps[start] + np.arange((end - start) * substeps + 1) * dt / substeps
    states, valid = track_at(track, timestamps, times)
    return times, states, valid
