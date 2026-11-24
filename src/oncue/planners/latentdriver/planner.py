"""LatentDriver (Xiao et al., ICRA 2025) as the closed-loop ego planner, ``LatentDriverPlanner``.

It sees road users from ``planners.perception`` and the route every planner drives, but no
signals or stop signs. PlanT from the same release runs in it too, as upstream's simulator runs it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ...core.config import DriverConfig
from ...core.rules import driver_path
from ...core.types import Scene
from ...utils.geometry import wrap_angle
from ..perception import Beliefs
from ..step_records import StepRecords
from .model import WEIGHTS, load_model
from .observation import object_slots, observation, scene_boxes, update_slots

CONTEXT_STEPS = 2  # ``configs/method/latentdriver.yaml`` max_len (eval_context_length / max_length)
ACTION_SIZE = 3  # (dx, dy, dyaw)
# The scene's frame step must be STEP_S within this (WOMD's 0.1 s up to rounding).
WAYMAX_STEP_TOLERANCE_S = 1e-6
STEP_S = 0.1  # Waymax's WOMD step length, which the model was trained on
_MODELS: dict[str, torch.nn.Module] = {}  # weights path -> model, one per process


class LatentDriverPlanner(StepRecords):
    """One-drive LatentDriver planner; step records go through ``StepRecords``.

    ``model_loader``: ``model.load_plant`` runs PlanT, which reads only the last observation.
    """

    def __init__(
        self,
        scene: Scene,
        driver: DriverConfig,
        weights: Path = WEIGHTS,
        log_route: bool = False,
        model_loader: Callable[[Path], torch.nn.Module] = load_model,
    ):
        if abs(scene.dt - STEP_S) > WAYMAX_STEP_TOLERANCE_S:
            raise ValueError(f"LatentDriver steps every {STEP_S} s, the scene {scene.dt} s")
        # Record stations along the route every planner drives, which is also the route input.
        super().__init__(driver_path(scene, driver), scene.dt)
        self.model = shared_model(weights, model_loader)
        self.anchor_state = scene.ego.states[scene.anchor].copy()
        # ``log_route``: upstream's route, for the comparison run.
        self.route_boxes, self.road_boxes = scene_boxes(scene, driver, log_route)
        self.slots = object_slots(scene)  # at the anchor; re-chosen every step
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = [np.zeros(ACTION_SIZE, np.float32)]
        self.target_brake_s: float | None = None  # no separate target brake for a learned planner

    def state(self) -> np.ndarray:
        """Return the ego's logged state at the anchor ``(x, y, heading, speed)``."""
        return self.anchor_state.copy()

    def step(
        self, frame: int, ego_states: np.ndarray, scene: Scene, beliefs: Beliefs
    ) -> np.ndarray:
        """Return the ego's state at the next frame, stepping as upstream's ``LTDSimulator.run``."""
        current = ego_states[frame]
        self.slots = update_slots(self.slots, scene, current, beliefs.states)
        self.observations.append(
            observation(
                self.route_boxes, self.road_boxes, self.slots, scene, current, beliefs.states
            )
        )
        steps = len(self.observations)
        states = torch.from_numpy(np.stack(self.observations[-CONTEXT_STEPS:]))[None]
        actions = torch.from_numpy(np.stack(self.actions[-CONTEXT_STEPS:]))[None]
        timesteps = torch.arange(steps)[-CONTEXT_STEPS:][None].long()
        with torch.no_grad():
            action = self.model.get_predictions(states, actions, timesteps, num_envs=1)[0].numpy()
        self.actions.append(action.astype(np.float32))
        nxt = _moved(current, action, self.dt)
        self.record_step(current, nxt)
        return nxt


def shared_model(
    weights: Path = WEIGHTS, model_loader: Callable[[Path], torch.nn.Module] = load_model
) -> torch.nn.Module:
    """Return the model for ``weights``, loading it on first use in this process (CPU, 1 thread)."""
    key = str(weights)
    if key not in _MODELS:
        torch.set_num_threads(1)  # closed-loop runs go one per process, in parallel
        _MODELS[key] = model_loader(weights)
    return _MODELS[key]


def _moved(state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    """Apply Waymax's ``DeltaLocal`` action ``(dx, dy, dyaw)``, given in the ego frame."""
    dx, dy, dyaw = (float(v) for v in action)
    x, y, heading, _ = (float(v) for v in state)
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            x + c * dx - s * dy,
            y + s * dx + c * dy,
            float(wrap_angle(heading + dyaw)),
            math.hypot(dx, dy) / dt,
        ]
    )
