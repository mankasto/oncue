"""Kinematic measures of a trajectory, rare-motion labels and the physical reject rule.

All measures are Savitzky-Golay derivatives of the positions over a 1 s window, the same ones
for generation, the closed loop and ``oncue stats``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import savgol_filter

from ..utils.geometry import MIN_DIVISOR, ZERO_LENGTH_M

if TYPE_CHECKING:
    from .config import PhysicalLimits

SMOOTH_FRAMES = 11  # 1 s at 10 Hz, centred
SMOOTH_ORDER = 3  # cubic, so jerk is defined
# Curvature counts only above this speed: slower frames turn smoothing error into a tiny radius.
CURVATURE_MIN_MPS = 3.0
LIMIT_ROUNDING = 1e-6  # relative float slack above a limit that still counts as at it
YAW_SMOOTH_STEPS = 5  # moving-average length of `polyline_yaw`
TURNING_MIN_MPS = 1.0  # turning measures count only above this speed
# Upper statistical limits `exceeded` checks, peak speed first (`KinematicLimits` fields).
UPPER_LIMITS = ("peak_speed", "accel", "jerk", "lateral", "yaw_rate")
# Each physical limit (`PhysicalLimits` field) and the `measures` key it bounds.
PHYSICAL_MEASURE = {
    "speed": "peak_speed",
    "accel": "speed_up",
    "brake": "brake",
    "lateral": "lateral",
    "curvature": "curvature",
}


def measures(
    xy: np.ndarray, dt: float, moving_mps: float = TURNING_MIN_MPS, upto: int | None = None
) -> dict[str, float] | None:
    """Return the peak measures of positions ``xy`` sampled every ``dt`` seconds.

    The keys are ``peak_speed``, ``accel``, ``speed_up``, ``brake``, ``jerk``, ``lateral``,
    ``yaw_rate``, ``curvature`` and ``moving_speed`` (the median speed over the moving windows).
    Turning measures count only above ``moving_mps``. ``upto`` reads only windows wholly before
    that point. Returns None when no full window fits: unknown, not a pass.
    """
    xy = np.asarray(xy, float)
    if upto is not None:
        xy = xy[: max(upto, 0)]
    if len(xy) < SMOOTH_FRAMES:
        return None
    vel, acc, jrk, keep = _derivatives(xy, dt)
    v = np.linalg.norm(vel, axis=1)
    tangent = vel / np.maximum(v, MIN_DIVISOR)[:, None]
    along = np.einsum("ti,ti->t", acc, tangent)
    across = tangent[:, 0] * acc[:, 1] - tangent[:, 1] * acc[:, 0]
    jerk = np.einsum("ti,ti->t", jrk, tangent)
    moving = keep & (v > moving_mps)
    turning = moving.any()
    fast = keep & (v > max(moving_mps, CURVATURE_MIN_MPS))
    return {
        "peak_speed": float(v[keep].max()),
        "accel": float(np.abs(along[keep]).max()),
        "speed_up": float(max(along[keep].max(), 0.0)),
        "brake": float(max(-along[keep].min(), 0.0)),
        "jerk": float(np.abs(jerk[keep]).max()),
        "lateral": float(np.abs(across[moving]).max()) if turning else 0.0,
        "yaw_rate": float(np.abs(across[moving] / v[moving]).max()) if turning else 0.0,
        "curvature": float(np.abs(across[fast] / v[fast] ** 2).max()) if fast.any() else 0.0,
        "moving_speed": float(np.median(v[moving])) if turning else 0.0,
    }


def speeds(xy: np.ndarray, dt: float) -> np.ndarray | None:
    """Return ``[T]`` smoothed speeds, NaN outside full windows; None when no window is full."""
    xy = np.asarray(xy, float)
    if len(xy) < SMOOTH_FRAMES:
        return None
    vel, _, _, keep = _derivatives(xy, dt)
    return np.where(keep, np.linalg.norm(vel, axis=1), np.nan)


def _derivatives(xy: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Return velocity, acceleration, jerk and the full-window mask of positions ``xy``."""
    vel, acc, jrk = (
        savgol_filter(xy, SMOOTH_FRAMES, SMOOTH_ORDER, deriv=k, delta=dt, axis=0) for k in (1, 2, 3)
    )
    half = SMOOTH_FRAMES // 2
    keep = np.zeros(len(xy), bool)
    keep[half : len(xy) - half] = True
    return vel, acc, jrk, keep


def polyline_yaw(xy: np.ndarray, still: float | None = None) -> np.ndarray:
    """Return a smoothed heading per point along the polyline ``xy``.

    A zero-length step takes the nearest moving step's heading; a polyline that never moves
    gets ``still`` (None: 0).
    """
    d = np.diff(xy, axis=0, append=xy[-1:] + (xy[-1:] - xy[-2:-1]))
    moving = np.hypot(d[:, 0], d[:, 1]) > ZERO_LENGTH_M
    if not moving.any() and still is not None:
        return np.full(len(xy), float(still))
    if moving.any() and not moving.all():
        last = np.maximum.accumulate(np.where(moving, np.arange(len(d)), -1))
        last = np.where(last < 0, np.flatnonzero(moving)[0], last)
        d = d[last]
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.convolve(
        np.pad(yaw, 2, mode="edge"), np.ones(YAW_SMOOTH_STEPS) / YAW_SMOOTH_STEPS, mode="valid"
    )


def exceeded(
    found: dict | None,
    limits: KinematicLimits,
    factor: float = 1.0,
    required_speed: float | None = None,
) -> list[str]:
    """Return the limits the peak measures ``found`` are beyond (a label, never a reject).

    Upper limits are scaled by ``factor``; a peak speed below the lower bound reads
    ``"min_peak_speed"``. ``required_speed`` moves the peak-speed bounds (``peak_speed_bounds``).
    """
    if found is None:
        return []
    fastest, slowest = peak_speed_bounds(limits, required_speed)
    upper = {k: getattr(limits, k) for k in UPPER_LIMITS} | {"peak_speed": fastest}
    above = [
        k
        for k in UPPER_LIMITS
        if upper[k] is not None and k in found and found[k] > factor * upper[k]
    ]
    too_slow = slowest is not None and "peak_speed" in found and found["peak_speed"] < slowest
    return [*above, "min_peak_speed"] if too_slow else above


def peak_speed_bounds(
    limits: KinematicLimits, required_speed: float | None = None
) -> tuple[float | None, float | None]:
    """Return ``(upper, lower)`` peak-speed bounds; None means not checked.

    With ``required_speed``, the upper bound is at least ``limits.pace_ratio`` times it and the
    lower bound at most it.
    """
    upper, lower = limits.peak_speed, limits.min_peak_speed
    if required_speed is None:
        return upper, lower
    if upper is not None and limits.pace_ratio is not None:
        upper = max(upper, required_speed * limits.pace_ratio)
    if lower is not None:
        lower = min(lower, required_speed)
    return upper, lower


def beyond_physical(found: dict | None, limits: PhysicalLimits) -> list[str]:
    """Return the physical limits the peak measures ``found`` exceed (a reject).

    Also empty when nothing was measured, which callers must not take as a pass.
    """
    if found is None:
        return []
    return [
        name
        for name, measure in PHYSICAL_MEASURE.items()
        if getattr(limits, name) is not None
        and found[measure] > getattr(limits, name) * (1 + LIMIT_ROUNDING)
    ]


@dataclass(frozen=True)
class KinematicLimits:
    """Bounds on the peak measures of a trajectory; None means not checked."""

    accel: float | None  # |longitudinal acceleration|, m/s^2
    jerk: float | None  # |d accel / dt|, m/s^3
    lateral: float | None  # |lateral acceleration| while moving, m/s^2
    yaw_rate: float | None  # |yaw rate| while moving, rad/s
    peak_speed: float | None  # m/s
    moving_mps: float = TURNING_MIN_MPS
    min_peak_speed: float | None = None  # m/s: a slower peak is rare too
    # A peak up to this times a test's required speed is not rare (None: ``peak_speed`` alone).
    pace_ratio: float | None = None
