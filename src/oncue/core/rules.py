"""Traffic rules along the rule-based driver's path: signals, stop signs, and a speed cap.

Stations are arc lengths along ``driver_path``. Logged signals lag the real light, so a stop
line counts as green once the logged ego has crossed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..data.map_geometry import LEVEL_GAP_M, ego_level, lane_continuation, lane_z_at
from ..utils.geometry import (
    MIN_LINE_POINTS,
    STATION_TOL_M,
    cumulative_arc_length,
    line_heading,
    wrap_angle,
)
from .config import DriverConfig
from .road import SAME_WAY_DEG
from .types import FeatureKind, LaneSignal, MapFeature, Scene, Track

if TYPE_CHECKING:
    from ..core.types import SignalTrack

# A stop sign's lane stays within this of the path.
STOP_LANE_MATCH_M = 1.5
LANE_FIT_M = 2.0  # a signal's lane: mean distance from the path at most this
LANE_BEFORE_M, LANE_AFTER_M = 10.0, 20.0  # stretch compared before and after the stop line
LANE_MIN_M = 5.0  # of a lane that short, the whole thing
STOP_POINT_SAME_M = 1.0  # signals with stop points this near share one stop line
LANE_FIT_TIE_M = 0.25  # lanes within this of the best fit all count as equally good
STOP_LINE_MATCH_M = 2.5
SPEED_LIMIT_MATCH_M = 2.0
SPEED_LIMIT_POSE_STEP = 5  # the posted limit is read every fifth logged pose
# Chord length for reading a lane's heading at a point (route headings use their own).
SPEED_LIMIT_CHORD_M = 1.0
SAME_WAY_RAD = np.radians(SAME_WAY_DEG)  # road.SAME_WAY_DEG: same way as the ego or path
STOP_LANE_SHARE = 0.8  # share of LANE_BEFORE_M a stop-sign lane must sit near the path
STOP_LINE_AHEAD_M = 0.5  # a stop line counts only this far along the path or farther
CURVATURE_SMOOTH_POINTS = 5  # moving-mean length (metres) for the curvature cap
MIN_CURVATURE = 1e-4  # 1/m: below this the path counts as straight (only the speed limit caps)
DEFAULT_SPEED_LIMIT = 11.2  # m/s (about 25 mph) when no lane near the path records a limit
MIN_CHORD_M = 1e-6  # a chord shorter than this has no direction
STATION_END_SLACK_M = 1e-9  # keep np.arange at the end of the path


def traffic_rules(scene: Scene, speed_limit: float, cfg: DriverConfig) -> TrafficRules:
    """Return the signals, stop signs and speed cap along the whole driver path."""
    path = driver_path(scene, cfg)
    whole = _with_history(scene, path)
    lanes = {f.feature_id: f for f in scene.map_features if f.kind is FeatureKind.LANE}
    signals, signal_lanes = _signals(scene, path, whole, lanes)
    stop_signs = _stop_signs(scene, path, whole, lanes)
    cap_stations, cap_speeds = _speed_cap(path, speed_limit, cfg)
    return TrafficRules(
        signals=tuple(signals),
        signal_lanes=tuple(signal_lanes),
        stop_signs=stop_signs,
        cap_stations=cap_stations,
        cap_speeds=cap_speeds,
    )


@dataclass(frozen=True, eq=False)
class TrafficRules:
    signals: tuple[tuple[float, np.ndarray], ...]  # (stop-line station, phase per frame)
    signal_lanes: tuple[str, ...]  # per signal, the lane from its stop line the path follows
    stop_signs: tuple[float, ...]  # stop-line stations along the path
    cap_stations: np.ndarray  # stations of the speed cap
    cap_speeds: np.ndarray  # min(speed limit, curvature cap), with braking into it

    def speed_cap(self, station: float) -> float:
        """Return the speed cap at ``station`` (linear interpolation)."""
        return float(np.interp(station, self.cap_stations, self.cap_speeds))


def driver_path(scene: Scene, cfg: DriverConfig) -> shapely.LineString:
    """Return the logged ego path from the anchor, extended by ``cfg.extension_m``.

    The extension follows the lane graph where possible and goes straight otherwise.
    """
    ego = scene.ego
    frames = _logged_frames(scene)
    logged = ego.states[frames, :2]
    heading = float(ego.states[frames[-1], 2])
    tail = lane_continuation(scene, logged[-1], heading, cfg.extension_m)
    if tail is None:
        steps = np.arange(1, cfg.extension_m + 1, 1.0)
        tail = logged[-1] + np.outer(steps, [np.cos(heading), np.sin(heading)])
    return shapely.LineString(np.vstack([logged, tail]))


def _logged_frames(scene: Scene) -> np.ndarray:
    """Return the frames the ego is logged at, from the anchor onward."""
    frames = np.flatnonzero(scene.ego.valid)
    return frames[frames >= scene.anchor]


def _with_history(scene: Scene, path: shapely.LineString) -> shapely.LineString:
    """Return ``path`` with the ego's logged positions before the anchor prepended."""
    ego = scene.ego
    frames = np.flatnonzero(ego.valid)
    before = ego.states[frames[frames < scene.anchor], :2]
    return shapely.LineString(np.vstack([before, path.coords])) if len(before) else path


def _signals(
    scene: Scene, path: shapely.LineString, whole: shapely.LineString, lanes: dict
) -> tuple[list, list[str]]:
    """Return the signals along ``path`` as ``(station, phase per frame)`` and their lanes.

    Per stop line only the best-fitting lanes count; phases turn green from the frame the logged
    ego crossed the line.
    """
    fits = []
    for sig in scene.signals:
        stop = shapely.Point(*sig.stop_point)
        if path.distance(stop) > STOP_LINE_MATCH_M or path.project(stop) <= STOP_LINE_AHEAD_M:
            continue
        fit = _signal_fit(scene.ego, whole, lanes.get(sig.lane_id), stop)
        if fit is not None:
            fits.append((sig, stop, fit))
    frames, front = _logged_front(scene, path)
    signals, signal_lanes = [], []
    for sig, stop, fit in fits:
        line_fits = [g for _, q, g in fits if q.distance(stop) <= STOP_POINT_SAME_M]
        if fit > min(line_fits) + LANE_FIT_TIE_M:
            continue
        station = path.project(stop)
        held = held_phase(sig)
        crossed = np.flatnonzero(front >= station)
        if len(crossed):
            held[int(frames[crossed[0]]) :] = "go"
        signals.append((float(station), held))
        signal_lanes.append(sig.lane_id)
    return signals, signal_lanes


def _signal_fit(
    ego: Track, whole: shapely.LineString, lane: MapFeature | None, stop: shapely.Point
) -> float | None:
    """Return the signal lane's mean distance from the path, or None when it does not fit.

    A mean rather than a share within a band, since the ego cuts a turn's corner.
    """
    off = _lane_offsets(ego, whole, lane, stop, LANE_AFTER_M)
    if off is None or float(np.mean(off)) > LANE_FIT_M:
        return None
    return float(np.mean(off))


def _lane_offsets(
    ego: Track,
    whole: shapely.LineString,
    lane: MapFeature | None,
    stop: shapely.Point,
    after_m: float,
) -> np.ndarray | None:
    """Return the lane's per-metre distance from the path around the stop line.

    None when the lane is too short, runs another way, or is on another level (an overpass).
    """
    if lane is None or len(lane.points) < MIN_LINE_POINTS:
        return None
    line = shapely.LineString(lane.points)
    at = line.project(stop)
    lo, hi = max(at - LANE_BEFORE_M, 0.0), min(at + after_m, line.length)
    if hi - lo < min(LANE_MIN_M, line.length) - STATION_TOL_M or hi - lo < 1.0:
        return None
    near = shapely.line_interpolate_point(line, np.linspace(lo, hi, int(hi - lo) + 2))
    stations = shapely.line_locate_point(whole, near)
    # not past either path end
    beside = (stations > STATION_TOL_M) & (stations < whole.length - STATION_TOL_M)
    near, stations = near[beside], stations[beside]
    if len(near) < MIN_LINE_POINTS:
        return None
    lane_from, lane_to = np.array(near[0].coords[0]), np.array(near[-1].coords[0])
    path_from, path_to = (np.array(whole.interpolate(stations[k]).coords[0]) for k in (0, -1))
    if np.linalg.norm(path_to - path_from) < MIN_CHORD_M:
        return None
    turn = np.arctan2(*(lane_to - lane_from)[::-1]) - np.arctan2(*(path_to - path_from)[::-1])
    if abs(wrap_angle(turn)) > SAME_WAY_RAD:
        return None
    z, own = lane_z_at(lane, stop), ego_level(ego, stop)
    if z is not None and own is not None and abs(z - own) > LEVEL_GAP_M:
        return None
    return shapely.distance(whole, near)


def _logged_front(scene: Scene, path: shapely.LineString) -> tuple[np.ndarray, np.ndarray]:
    """Return the ego's logged frames from the anchor and its front's station along ``path``."""
    ego, frames = scene.ego, _logged_frames(scene)
    front = np.array([path.project(shapely.Point(*ego.states[i, :2])) for i in frames])
    front += ego.size[0] / 2
    return frames, front


def phase_at(phase: np.ndarray, frame: int) -> str:
    """Return a held phase (``held_phase``) at ``frame``; past its end, the last one."""
    return phase[min(frame, len(phase) - 1)]


def held_phase(sig: SignalTrack) -> np.ndarray:
    """Return a lane signal's phase per frame (``go``, ``caution``, ``stop``, ``unknown``).

    An unreported frame holds the last reported phase.
    """
    phase = np.array([LaneSignal(int(v)).phase for v in sig.states], dtype=object)
    held = phase.copy()
    last = "unknown"
    for i, p in enumerate(phase):
        last = p if p != "unknown" else last
        held[i] = last
    return held


def stop_sign_lanes(scene: Scene) -> set[str]:
    """Return the ids of lanes controlled by stop signs; each starts at its stop line."""
    return {
        lane_id
        for f in scene.map_features
        if f.kind is FeatureKind.STOP_SIGN
        for lane_id in f.controlled_lanes
    }


def chain_stop_lines(
    points: np.ndarray,
    owner: list[str],
    chain: list[str],
    stop_lanes: set[str],
    lanes: dict,
) -> list[float]:
    """Return ascending stations along a lane chain where its stop-sign lanes begin.

    ``owner`` names the lane of each of ``points``; lanes entered partway or owning no point are
    skipped.
    """
    arcs = cumulative_arc_length(np.asarray(points, float))
    out = []
    for lane_id in chain:
        if lane_id not in stop_lanes or lane_id not in lanes:
            continue
        first = next((k for k, o in enumerate(owner) if o == lane_id), None)
        if first is None:  # cut away before the points start: the stop line is not on them
            continue
        start = np.asarray(lanes[lane_id].points[0], float)[:2]
        # `map_geometry.lane_chain` drops a lane's first point at a joint: try the point before.
        nearest = min(
            range(max(first - 1, 0), first + 1),
            key=lambda k: np.linalg.norm(np.asarray(points[k], float)[:2] - start),
        )
        if np.linalg.norm(np.asarray(points[nearest], float)[:2] - start) <= STOP_LINE_MATCH_M:
            out.append(float(arcs[nearest]))
    return sorted(out)


def _stop_signs(
    scene: Scene, path: shapely.LineString, whole: shapely.LineString, lanes: dict
) -> tuple[float, ...]:
    """Return ascending stations along ``path`` of the stop-sign lanes it follows."""
    stop_signs = set()
    for lane_id in stop_sign_lanes(scene):
        if lane_id not in lanes or len(lanes[lane_id].points) < MIN_LINE_POINTS:
            continue
        start = shapely.Point(*lanes[lane_id].points[0][:2])
        near = path.distance(start) < STOP_LINE_MATCH_M and path.project(start) > STOP_LINE_AHEAD_M
        if near and _leads_from(scene.ego, whole, lanes[lane_id], start):
            stop_signs.add(round(float(path.project(start)), 1))
    return tuple(sorted(stop_signs))


def _leads_from(
    ego: Track, whole: shapely.LineString, lane: MapFeature, stop: shapely.Point
) -> bool:
    """Return whether a stop-sign lane leads from its stop line along the path.

    ``STOP_LANE_SHARE`` of its first ``LANE_BEFORE_M`` must lie within ``STOP_LANE_MATCH_M``.
    """
    off = _lane_offsets(ego, whole, lane, stop, LANE_BEFORE_M)
    return off is not None and float(np.mean(off < STOP_LANE_MATCH_M)) >= STOP_LANE_SHARE


def _speed_cap(
    path: shapely.LineString, speed_limit: float, cfg: DriverConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(stations, cap)`` every metre along the path.

    The cap is ``min(speed_limit, curvature cap)``, braked into at ``cfg.anticipate_decel``.
    """
    stations = np.arange(0.0, path.length + STATION_END_SLACK_M, 1.0)
    xy = np.array([path.interpolate(station).coords[0] for station in stations])
    kappa = np.zeros(len(stations))
    if len(stations) >= CURVATURE_SMOOTH_POINTS:
        heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
        window = np.ones(CURVATURE_SMOOTH_POINTS) / CURVATURE_SMOOTH_POINTS
        kappa = np.convolve(np.abs(np.gradient(heading, stations)), window, mode="same")
    cap = np.minimum(speed_limit, np.sqrt(cfg.lateral_accel / np.maximum(kappa, MIN_CURVATURE)))
    for k in range(len(cap) - 2, -1, -1):
        reach = np.sqrt(
            cap[k + 1] ** 2 + 2 * cfg.anticipate_decel * (stations[k + 1] - stations[k])
        )
        cap[k] = min(cap[k], reach)
    return stations, cap


def logged_path(scene: Scene) -> shapely.LineString:
    """Return the ego's logged path from the anchor onward."""
    return shapely.LineString(scene.ego.states[_logged_frames(scene), :2])


def lane_speed_limit(scene: Scene) -> float:
    """Return the median posted limit (m/s) of the nearest same-way lanes along the logged path."""
    poses = scene.ego.states[scene.ego.valid][::SPEED_LIMIT_POSE_STEP]
    lanes = [
        (shapely.LineString(f.points), float(f.speed_limit_mps))
        for f in scene.map_features
        if f.kind is FeatureKind.LANE and f.speed_limit_mps and len(f.points) >= MIN_LINE_POINTS
    ]
    if not lanes or not len(poses):
        return DEFAULT_SPEED_LIMIT
    points = shapely.points(poses[:, :2])
    best = [(np.inf, None)] * len(poses)
    for line, limit in lanes:
        d = shapely.distance(line, points)
        for i in np.flatnonzero(d < SPEED_LIMIT_MATCH_M):
            station = line.project(points[i])
            along = line_heading(line, station, back=SPEED_LIMIT_CHORD_M, ahead=SPEED_LIMIT_CHORD_M)
            off = abs(wrap_angle(along - poses[i, 2]))
            if off <= SAME_WAY_RAD and d[i] < best[i][0]:
                best[i] = (d[i], limit)
    limits = [limit for _, limit in best if limit is not None]
    return float(np.median(limits)) if limits else DEFAULT_SPEED_LIMIT
