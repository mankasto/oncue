"""The simulation loop shared by the reference ego and the closed loop: `drive` steps a planner
and its perception in a world until the horizon, a target contact or the end of the route."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..core.config import DriverConfig, VisibilityConfig
from ..core.rules import TrafficRules, driver_path
from ..core.types import Scene
from ..utils.geometry import step_start_frames, swept_overlap
from ..utils.tracks import frame_of
from .idm.planner import IDMPlanner
from .perception import Beliefs, Perception

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import ArrayLike


class ReplayWorld:
    """World where everyone but the ego and a moving target replays its logged track.

    The target is moved by `target_motion(frame, ego_state)` (None: logged); `target_id` None
    means a run without a target.
    """

    out_of_road: OutOfRoad | None = None  # replay has no roads to leave, so this stays unjudged

    def __init__(
        self,
        scene: Scene,
        target_id: str | None,
        target_motion: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    ):
        if target_id == scene.ego_id:
            raise ValueError("the ego is no target: a run without a target passes target_id None")
        if target_motion is not None and target_id is None:
            raise ValueError("target_motion moves a target: this run has none")
        self.scene, self.target_id, self.target_motion = scene, target_id, target_motion

    def advance(self, frame: int, ego_state: np.ndarray) -> Scene:
        """Return the scene at `frame`, with the ego at `ego_state`."""
        if self.target_motion is not None:
            moved = self.target_motion(frame, ego_state)
            self.scene = _move_target(self.scene, self.target_id, moved)
        return self.scene


@dataclass(frozen=True)
class OutOfRoad:
    """A world's out-of-road judgement for a run; both None when nothing was found."""

    first_s: float | None = None  # seconds after the anchor the ego first left the road
    lateral_m: float | None = None  # the ego's distance from its route then


def _move_target(scene: Scene, target_id: str, moved: np.ndarray | None) -> Scene:
    """Return the scene with the target's states updated to `moved` (None leaves it alone)."""
    if moved is None:
        return scene
    track = scene.tracks[target_id].with_states(moved)
    return replace(scene, tracks={**scene.tracks, target_id: track})


def drive(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    *,
    masked: bool = False,
    vis_cfg: VisibilityConfig | None = None,
    static: shapely.Geometry | None = None,
    v_des: float,
    rules: TrafficRules | None = None,
    ignore_target: bool = False,
    target_motion: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    world: type[ReplayWorld] = ReplayWorld,
    planner: Callable[..., Any] = IDMPlanner,
    ends_at: shapely.LineString | None = None,
) -> Drive:
    """Simulate the ego from the anchor onward.

    Args:
        target_id: None for a run without a target.
        v_des: desired speed in m/s.
        ignore_target: the target is never perceived and a contact with it does not end the run.
        target_motion: `target_motion(frame, ego_state)` returns the new states of a target that
            reacts to the ego, or None.
        world: called as `world(scene, target_id, target_motion)`.
        planner: called as `planner(scene, target_id, cfg, perception, v_des=, rules=)`; must
            behave like `IDMPlanner`.
        ends_at: a route that ends before the driver path; the run stops at its end.
    """
    ego, anchor, dt = scene.ego, scene.anchor, scene.dt
    scene_world = world(scene, target_id, target_motion)
    perception = Perception(
        scene,
        cfg,
        masked=masked,
        vis_cfg=vis_cfg,
        static=static,
        ignore=target_id if ignore_target else None,
    )
    end = min(len(scene.timestamps), anchor + frame_of(cfg.horizon_s, dt) + 1)
    ego_planner = planner(scene, target_id, cfg, perception, v_des=v_des, rules=rules)
    route = driver_path(scene, cfg)
    end_route = route if ends_at is None else ends_at
    states, valid = ego.states.copy(), ego.valid.copy()
    result = Drive(states, anchor)
    state = ego_planner.state()
    frame = anchor
    for frame in range(anchor, end):
        states[frame], valid[frame] = state, True  # place the ego
        scene = scene_world.advance(frame, states[frame])  # move everyone else
        if not ignore_target and _touches_target(  # check the target contact
            scene, target_id, states, frame, ego.size, cfg.collision_substeps
        ):
            result.target_contact_s = (frame - anchor) * dt
            result.target_contact_speed = float(states[frame, 3])
            break
        beliefs = perception.observe(frame, states, scene)  # perceive
        if target_id in beliefs.states and result.target_known_s is None:
            result.target_known_s = (frame - anchor) * dt
            result.target_seen_s = (beliefs.frame - anchor) * dt
        if cfg.ignore_behind:
            beliefs = ahead_only(beliefs, route, states[beliefs.frame])
        state = ego_planner.step(frame, states, scene, beliefs)  # plan
        if reached_route_end(end_route, state):
            break
    states[frame + 1 :], valid[frame + 1 :] = 0.0, False
    result.stations, result.speeds, result.accels = (
        ego_planner.stations,
        ego_planner.speeds,
        ego_planner.accels,
    )
    result.target_brake_s = ego_planner.target_brake_s
    result.valid_until, result.ego_valid = frame + 1, valid
    result.out_of_road = scene_world.out_of_road
    return result


@dataclass(eq=False)
class Drive:
    states: np.ndarray  # [T, 4] ego states (logged before the anchor)
    valid_until: int  # frames [anchor, valid_until) are simulated
    target_contact_s: float | None = None  # seconds after the anchor
    target_contact_speed: float | None = None  # ego speed at that contact (m/s)
    target_known_s: float | None = None  # first time the target is known
    target_seen_s: float | None = None  # observation it was known from (latency before)
    target_brake_s: float | None = None  # first braking (< -1 m/s^2) the target causes
    stations: list[float] = field(default_factory=list)  # per simulated frame
    speeds: list[float] = field(default_factory=list)
    # IDM's commanded acceleration; the speed change per step for planners that move the ego
    # themselves (``StepRecords``)
    accels: list[float] = field(default_factory=list)
    # the ego's validity to go with `states`, read by the post-run checks
    ego_valid: np.ndarray | None = None
    out_of_road: OutOfRoad | None = None  # None: not judged (the replay world judges none)

    def distance_at(self, seconds: float, dt: float) -> float:
        step = min(frame_of(seconds, dt), len(self.stations) - 1)
        return self.stations[step] if self.stations else 0.0


def ahead_only(beliefs: Beliefs, route: shapely.LineString, ego_state: np.ndarray) -> Beliefs:
    """Drop road users whose centre is behind the ego's along `route`."""
    if not beliefs.states:
        return beliefs
    believed = list(beliefs.states.items())
    centres = shapely.points([state[:2] for _, state in believed])
    stations = shapely.line_locate_point(route, centres)
    own = route.project(shapely.Point(ego_state[0], ego_state[1]))
    ahead = {k: state for (k, state), at in zip(believed, stations, strict=True) if at >= own}
    return Beliefs(beliefs.frame, ahead)


def _touches_target(
    scene: Scene,
    target_id: str | None,
    ego_states: np.ndarray,
    frame: int,
    ego_size: ArrayLike,
    substeps: int,
) -> bool:
    """Return whether the ego touches the target over the step into `frame` (swept check)."""
    target = scene.tracks.get(target_id)
    if target is None or not target.valid[frame]:
        return False
    before = int(step_start_frames(frame, scene.anchor, target.valid))
    return swept_overlap(
        ego_states[before],
        ego_states[frame],
        ego_size,
        target.states[before],
        target.states[frame],
        target.size,
        substeps,
    )


def reached_route_end(route: shapely.LineString, state: np.ndarray) -> bool:
    """Return whether the ego's centre at `state` projects to the end of `route` or past it."""
    return route.project(shapely.Point(state[0], state[1])) >= route.length
