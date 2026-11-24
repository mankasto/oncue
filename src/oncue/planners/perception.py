"""``Perception`` shared by every closed-loop planner: unoccluded, or masked by visibility.

Masked perception knows a road user only once it is visible from the ego, and keeps it for
``memory_s``, coasted at constant velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from ..core.config import DriverConfig, VisibilityConfig
from ..core.types import STATE_SIZE, Scene, height_of
from ..core.visibility import blocking_arrays, in_range, occluders, visible_fraction
from ..utils.geometry import box_corners
from ..utils.tracks import frame_of


@dataclass(frozen=True, eq=False)
class Beliefs:
    """Known road users at observation ``frame`` with coasted state ``(x, y, heading, speed)``.

    Ordered by first observation, even after a road user was forgotten and seen again.
    """

    frame: int
    states: dict[str, np.ndarray]


class Perception:
    """Perception and short-term tracker for one closed-loop run.

    ``ignore`` names a road user never perceived (the reference ego's own target).
    """

    def __init__(
        self,
        scene: Scene,
        cfg: DriverConfig,
        *,
        masked: bool = False,
        vis_cfg: VisibilityConfig | None = None,
        static: object = None,
        ignore: str | None = None,
    ):
        if masked and vis_cfg is None:
            raise ValueError("Masked perception needs a VisibilityConfig")
        self.cfg, self.masked, self.vis_cfg, self.static = cfg, masked, vis_cfg, static
        self.ignore = ignore
        self.anchor, self.dt = scene.anchor, scene.dt
        self.latency_frames = frame_of(cfg.latency_s, self.dt)
        self.period_frames = max(1, frame_of(cfg.perception_period_s, self.dt))
        self.seen_at: dict[str, int] = {}  # road user -> last frame it was seen
        # Road user -> (observation frame, observed state) of its last observation.
        self.last_observed: dict[str, tuple[int, np.ndarray]] = {}
        self.in_range_at_update: set[str] = set()  # unoccluded: in range at the last update

    def observe(self, frame: int, ego_states: np.ndarray, scene: Scene) -> Beliefs:
        """Return the planner's beliefs at ``frame``.

        Call once per frame from the anchor on; ``ego_states [T, 4]`` is indexed by scene frame.
        """
        obs_frame = max(frame - self.latency_frames, self.anchor)
        if (frame - self.anchor) % self.period_frames == 0:
            self._update(frame, obs_frame, ego_states[obs_frame], scene)
        known = self._known(frame)
        return Beliefs(obs_frame, {r: self._belief(r, obs_frame) for r in known})

    def _update(self, frame: int, obs_frame: int, eye: np.ndarray, scene: Scene) -> None:
        """Observe road users in range at ``obs_frame`` from ``eye`` (masked: visible only)."""
        near = [
            road_user
            for road_user, track in scene.tracks.items()
            if road_user not in (scene.ego_id, self.ignore)
            and track.valid[obs_frame]
            and in_range(eye, track.states[obs_frame], self.cfg.perception_range_m)
        ]
        if self.masked:
            observed = []
            if near:
                values = _visible(scene, eye, obs_frame, near, self.vis_cfg, self.static)
                observed = [
                    road_user
                    for road_user, value in zip(near, values, strict=True)
                    if value >= self.vis_cfg.threshold
                ]
        else:
            observed = near
            self.in_range_at_update = set(near)  # unoccluded knows what is in range then
        for road_user in observed:
            self.seen_at[road_user] = frame
            self.last_observed[road_user] = (
                obs_frame,
                scene.tracks[road_user].states[obs_frame].copy(),
            )

    def _known(self, frame: int) -> list[str]:
        """Return the road users seen within ``memory_s`` (masked) or in range at last update."""
        if not self.masked:
            return [r for r in self.last_observed if r in self.in_range_at_update]
        return [
            road_user
            for road_user, seen_frame in self.seen_at.items()
            if (frame - seen_frame) * self.dt <= self.cfg.memory_s
        ]

    def _belief(self, road_user: str, obs_frame: int) -> np.ndarray:
        """Return the road user's last observation coasted at constant velocity to ``obs_frame``."""
        seen_frame, seen_state = self.last_observed[road_user]
        return _coasted(seen_state, (obs_frame - seen_frame) * self.dt)

    def unseen_vehicles(
        self,
        eye: np.ndarray,
        boxes: np.ndarray,
        size: tuple[float, float],
        height: float,
        beliefs: Beliefs,
        scene: Scene,
    ) -> np.ndarray:
        """Return ``[K]``: would a vehicle at each of ``boxes`` be unseen from ``eye``?

        For phantom vehicles: only known road users (at believed states) and static occluders
        can hide one.
        """
        unseen = ~in_range(eye, boxes, self.cfg.perception_range_m)
        occluded = np.zeros(len(boxes), bool)
        if self.masked and not unseen.all():  # beyond the range it is unseen anyway
            near = np.flatnonzero(~unseen)
            blockers = self._known_occluders(beliefs, scene)
            occluded[near] = _occluded(
                eye, boxes[near], size, height, blockers, self.vis_cfg, self.static
            )
        return unseen | occluded

    def _known_occluders(self, beliefs: Beliefs, scene: Scene):
        """Return ``(states, sizes, heights)`` for known occluding road users at believed states."""
        known = [
            road_user
            for road_user in beliefs.states
            if scene.tracks[road_user].agent_type in self.vis_cfg.occluder_types
        ]
        tracks = [scene.tracks[road_user] for road_user in known]
        # Reshape so an empty set still has shapes [0, 4] and [0, 2].
        states = np.array([beliefs.states[road_user] for road_user in known], float)
        states = states.reshape(len(known), STATE_SIZE)
        sizes = np.array([track.size for track in tracks], float).reshape(len(known), 2)
        heights = np.array([height_of(track) for track in tracks], float)
        return states, sizes, heights


def _visible(
    scene: Scene,
    ego_state: np.ndarray,
    frame: int,
    road_users: list[str],
    vis_cfg: VisibilityConfig,
    static: shapely.Geometry | None,
) -> np.ndarray:
    """Return the visible fraction of each road user at ``frame`` from ``ego_state``."""
    blockers = [b for b in occluders(scene, "", vis_cfg) if b.valid[frame]]  # never the ego
    n_blockers = len(blockers)
    blocker_states = np.array([b.states[frame] for b in blockers], float).reshape(
        n_blockers, STATE_SIZE
    )
    blocker_sizes = np.array([b.size for b in blockers], float).reshape(n_blockers, 2)
    blocker_heights = np.array([height_of(b) for b in blockers], float)
    blocker_ids = np.array([b.track_id for b in blockers], object)
    tracks = [scene.tracks[road_user] for road_user in road_users]
    n_road_users = len(road_users)
    hits = blocking_arrays(
        np.repeat(np.asarray(ego_state, float)[None, :2], n_road_users, axis=0),
        np.array([t.states[frame] for t in tracks], float),
        np.array([t.size for t in tracks], float),
        np.array([height_of(t) for t in tracks], float),
        np.repeat(blocker_states[None], n_road_users, axis=0),
        blocker_sizes,
        np.ones((n_road_users, n_blockers), bool),
        blocker_heights,
        vis_cfg,
        static,
    )
    own = blocker_ids[None, :] == np.array(list(road_users), object)[:, None]
    hits[:, :n_blockers][own] = False  # never occluded by itself
    return visible_fraction(hits)


def _coasted(state: np.ndarray, coast_s: float) -> np.ndarray:
    """Return ``state`` moved on at constant velocity for ``coast_s`` seconds."""
    return np.array(
        [
            state[0] + state[3] * np.cos(state[2]) * coast_s,
            state[1] + state[3] * np.sin(state[2]) * coast_s,
            state[2],
            state[3],
        ]
    )


def _occluded(
    eye: np.ndarray,
    boxes: np.ndarray,
    size: tuple[float, float],
    height: float,
    blockers: tuple[np.ndarray, np.ndarray, np.ndarray],
    vis_cfg: VisibilityConfig,
    static: object,
) -> np.ndarray:
    """Return ``[K]``: is a road user of ``size`` and ``height`` at each of ``boxes`` occluded?"""
    blocker_states, blocker_sizes, blocker_heights = blockers
    n_blockers, n_boxes = len(blocker_states), len(boxes)
    origins = np.repeat(np.asarray(eye, float)[None, :2], n_boxes, axis=0)
    blocked = blocking_arrays(
        origins,
        boxes,
        size,
        height,
        np.repeat(blocker_states.reshape(1, n_blockers, STATE_SIZE), n_boxes, 0),
        blocker_sizes,
        np.ones((n_boxes, n_blockers), bool),
        blocker_heights,
        vis_cfg,
        None,
    ).any(axis=1)  # [K, 5]
    if static is not None and not static.is_empty:
        # Static occluders are the costliest check: test only the rays still open.
        row, ray = np.nonzero(~blocked)
        if len(row):
            ends = np.concatenate((boxes[:, None, :2], box_corners(boxes, size)), axis=1)
            segments = shapely.linestrings(np.stack((origins[row], ends[row, ray]), axis=1))
            blocked[row, ray] = shapely.intersects(segments, static)
    return visible_fraction(blocked[:, None]) < vis_cfg.threshold
