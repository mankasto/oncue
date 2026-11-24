"""Mirror the closed-loop replay world in MetaDrive 0.4.3 so a run can be rendered.

It only renders: the runs and every check read the replay world's arrays. After each step every
position and heading is read back and must agree within ``READ_BACK_*``.
"""

from __future__ import annotations

import atexit
import functools
import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from metadrive.component.traffic_participants.cyclist import CyclistBoundingBox
from metadrive.component.traffic_participants.pedestrian import PedestrianBoundingBox
from metadrive.component.vehicle.vehicle_type import VaryingDynamicsBoundingBoxVehicle
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.scenario_description import ScenarioDescription

from ..core.types import STATE_SIZE, Scene, Track, height_of
from ..data.metadrive.description import closed_loop_description
from ..planners.driver import ReplayWorld
from ..utils.geometry import wrap_angle

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable

    from metadrive.base_class.base_object import BaseObject

# Read-back tolerances; MetaDrive stores state as float32.
READ_BACK_POSITION_M = 1e-3
READ_BACK_HEADING_RAD = 1e-3
# Road user types ScenarioTrafficManager actually replays in after_reset; others are only logged.
REPLAYED_TYPES = frozenset({"VEHICLE", "PEDESTRIAN", "CYCLIST"})


class MetaDriveWorld(ReplayWorld):
    """``planners.driver.ReplayWorld`` mirrored in MetaDrive, with a read-back check."""

    def __init__(
        self,
        scene: Scene,
        target_id: str | None,
        target_motion: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    ):
        super().__init__(scene, target_id, target_motion)
        others = {key: track for key, track in scene.tracks.items() if key != target_id}
        description = closed_loop_description(replace(scene, tracks=others), target_id)
        self.replayed_ids = [
            track_id
            for track_id, track in description["tracks"].items()
            if track["type"] in REPLAYED_TYPES and track_id != scene.ego_id
        ]
        self.env = _environment()
        self.target_manager = self.env.engine.target_manager
        self.target_manager.track = scene.tracks.get(target_id)
        self.env.set_scenario(ScenarioDescription(description))
        self.env.reset()

    def advance(self, frame: int, ego_state: np.ndarray) -> Scene:
        """Return the ReplayWorld scene at ``frame``, keeping MetaDrive in step."""
        scene = super().advance(frame, ego_state)
        target_state = None
        if self.target_manager.track is not None:
            track = scene.tracks[self.target_manager.track.track_id]
            target_state = track.states[frame] if track.valid[frame] else None
        if frame == scene.anchor:
            ego = self.env.agent
            write_state(ego, ego_state)
            self.target_manager.place(target_state)
            ego.navigation.update_localization(ego)  # as the ego's after_step does
        else:
            self.target_manager.next_state = target_state
            self.env.step(ego_state)
        self._check_read_back(scene, frame, ego_state, target_state)
        return scene

    def _check_read_back(
        self, scene: Scene, frame: int, ego_state: np.ndarray, target_state: np.ndarray | None
    ) -> None:
        """Raise ``ValueError`` if MetaDrive disagrees with the scene's arrays."""
        step = frame - scene.anchor
        if self.env.engine.episode_step != step:
            raise ValueError(f"MetaDrive is at step {self.env.engine.episode_step}, not {step}")
        self._check_object("the ego", frame, self.env.agent, ego_state)
        self._check_object("the target", frame, self.target_manager.object, target_state)
        traffic = self.env.engine.traffic_manager
        object_ids = traffic.scenario_id_to_obj_id
        for track_id in self.replayed_ids:
            track = scene.tracks[track_id]
            object_id = object_ids.get(track_id)
            replayed = None if object_id is None else traffic.spawned_objects[object_id]
            state = track.states[frame] if track.valid[frame] else None
            self._check_object(f"road user {track_id}", frame, replayed, state)

    def _check_object(
        self, name: str, frame: int, obj: BaseObject | None, state: np.ndarray | None
    ) -> None:
        """Check one MetaDrive object against the scene state; None means absent or not valid."""
        if (obj is None) != (state is None):
            raise ValueError(
                f"frame {frame}: {name} is {'not ' * (obj is None)}in MetaDrive but "
                f"{'not ' * (state is None)}valid in the scene"
            )
        if obj is None:
            return
        position_error = float(np.hypot(*(np.asarray(obj.position) - state[:2])))
        heading_error = abs(float(wrap_angle(obj.heading_theta - state[2])))
        if position_error > READ_BACK_POSITION_M or heading_error > READ_BACK_HEADING_RAD:
            raise ValueError(
                f"frame {frame}: {name} reads back {position_error:.2e} m, {heading_error:.2e} "
                f"rad off the state written or replayed"
            )


@functools.cache
def _environment() -> ClosedLoopEnv:
    """Return this process's env, created once and closed at exit.

    Without the close, the engine crashes on teardown (exit code -11) and the job reads as failed.
    """
    env = ClosedLoopEnv(ENV_CONFIG)
    atexit.register(env.close)
    return env


class ClosedLoopEnv(ScenarioOnlineEnv):
    """``ScenarioOnlineEnv`` with the target manager; only OnCue's rules end a run."""

    def setup_engine(self) -> None:
        super().setup_engine()
        # Same-priority managers run in registration order: this one after the traffic manager.
        self.engine.register_manager("target_manager", TargetManager())

    def done_function(self, vehicle_id: str) -> tuple[bool, dict]:
        return False, {}


class TargetManager(BaseManager):
    """Manages the inserted target, whose start depends on the tested ego (MetaDrive issue #856).

    Like a replayed road user it exists exactly at its valid frames, written from ``next_state``.
    """

    def __init__(self):
        super().__init__()
        # The target's track (type and size); None without a target.
        self.track: Track | None = None
        self.next_state = None
        # The MetaDrive object while the target is present.
        self.object = None

    def before_reset(self) -> None:
        super().before_reset()  # clears the previous run's target
        self.next_state, self.object = None, None

    def after_step(self, *args, **kwargs) -> dict:
        self.place(self.next_state)
        return {}

    def place(self, state: np.ndarray | None) -> None:
        """Place the target at ``[x, y, heading, speed]`` (spawned if needed); None removes it."""
        if state is None:
            if self.object is not None:
                self.clear_objects([self.object.id])
                self.object = None
            return
        if self.object is None:
            self.object = self._spawn(state)
        write_state(self.object, state)

    def _spawn(self, state: np.ndarray):
        """Spawn the target with its recorded size.

        Mirrors ``ScenarioTrafficManager`` with ``use_bounding_box`` set
        (scenario_traffic_manager.py L201-238, L269-286, L297-313).
        """
        track = self.track
        length, width = (float(value) for value in track.size)
        height = float(height_of(track))
        position, heading = [float(state[0]), float(state[1])], float(state[2])
        if track.agent_type == "vehicle":
            vehicle = VaryingDynamicsBoundingBoxVehicle
            config = ScenarioTrafficManager.get_traffic_v_config()
            config.update(width=width, length=length, height=height)
            config["scale"] = (
                width / vehicle.DEFAULT_WIDTH,
                length / vehicle.DEFAULT_LENGTH,
                height / vehicle.DEFAULT_HEIGHT,
            )
            return self.spawn_object(
                vehicle,
                position=[*position, height / 2],
                heading=heading,
                vehicle_config=config,
                name=track.track_id,
            )
        participant = {"pedestrian": PedestrianBoundingBox, "cyclist": CyclistBoundingBox}
        return self.spawn_object(
            participant[track.agent_type],
            name=track.track_id,
            position=position,
            heading_theta=heading,
            width=width,
            length=length,
            height=height,
            force_spawn=True,
        )


def write_state(obj: BaseObject, state: np.ndarray) -> None:
    """Place a MetaDrive object at ``[x, y, heading, speed]`` and freeze its physics.

    Mirrors ``ReplayTrafficParticipantPolicy.act`` (metadrive/policy/replay_policy.py L62-72).
    """
    x, y, heading, speed = (float(value) for value in state)
    obj.set_position([x, y])
    obj.set_velocity([math.cos(heading), math.sin(heading)], speed)
    obj.set_heading_theta(heading)
    obj.set_static(True)


class StateWritePolicy(BasePolicy):
    """Ego policy that writes whatever state ``env.step`` passes in.

    It runs in ``before_step``, so the ego's navigation, updated in ``after_step``, sees it.
    """

    @classmethod
    def get_input_space(cls) -> gym.spaces.Box:
        """Return the action space: the ego's next ``[x, y, heading, speed]``."""
        return gym.spaces.Box(-np.inf, np.inf, shape=(STATE_SIZE,), dtype=np.float64)

    def act(self, agent_id: str) -> None:
        write_state(self.control_object, self.engine.external_actions[agent_id])
        return None  # no steering or throttle for the vehicle to apply


ENV_CONFIG = dict(
    use_render=False,
    # OnCue does its own perception, so no lidar or images from MetaDrive.
    agent_observation=DummyObservation,
    agent_policy=StateWritePolicy,
    # Replayed road users stay where they are written, as in ReplayTrafficParticipantPolicy.
    set_static=True,
    # Keep every road user so MetaDrive shows what the oracle sees.
    filter_overlapping_car=False,
    # Name replayed road users by track id for lookup.
    force_reuse_object_name=True,
    log_level=logging.ERROR,
)
