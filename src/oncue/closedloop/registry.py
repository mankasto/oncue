"""Registered closed-loop planners: ``PLANNERS`` maps a name to its drive function and options.

``DEFAULT_PLANNERS`` run when none are named; ``register`` adds a planner, and
``import_planner_modules`` imports modules that register their own (``--planner-module``).
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import TYPE_CHECKING

from ..core.runs import model_info
from ..generation.reference import log_end_route
from ..planners import driver as ego_driver
from ..planners.idm.planner import IDMPlanner
from ..planners.latentdriver import (
    FINETUNED_LOGS_WEIGHTS,
    FINETUNED_WEIGHTS,
    PLANT_FINETUNED_LOGS_WEIGHTS,
    PLANT_FINETUNED_WEIGHTS,
    PLANT_WEIGHTS,
)
from ..planners.latentdriver import WEIGHTS as LATENTDRIVER_WEIGHTS
from ..planners.pdm.planner import OFFSET_RAMP_S, PDMClosedPlanner, PDMConfig
from .execution import executed

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable, Sequence
    from typing import Any

    from ..core.config import DriverConfig
    from ..core.rules import TrafficRules
    from ..core.types import Scene
    from ..planners.latentdriver.planner import LatentDriverPlanner
    from ..planners.perception import Perception

# The three IDM classes differ only in desired speed and in the hidden hazards they assume.
CONSERVATIVE_ALPHA = 0.30  # desired-speed factor used by the conservative IDM
DRIVER_OPTIONS = {
    # Reactive: follows, yields and brakes for what it has already seen; assumes no one hidden.
    "reactive": {},
    # Conservative: desired speed scaled by alpha, slowing to it at comfortable braking.
    "conservative": {"desired_speed_factor": CONSERVATIVE_ALPHA, "comfort_free_brake": True},
    # Occlusion-aware: keeps the margin to stop for phantoms where hazards could hide
    # (pedestrians beside parked vehicles, vehicles on crossing lanes).
    "occlusion_aware": {
        "occlusion_aware": True,
        "b_caution": 5.0,
        "caution_lateral_m": 2.5,
        "phantom_brake": 5.0,
    },
}


def run_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    planner: Callable[..., Any],
    **options: Any,
) -> ego_driver.Drive:
    """Run ``planners.driver.drive`` with ``planner``, ending where the reference ego ends."""
    ends_at = log_end_route(scene, cfg)
    return ego_driver.drive(scene, target_id, cfg, planner=planner, ends_at=ends_at, **options)


def drive_idm(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run the driver with the executed IDM planner; its class is picked by ``cfg``."""
    return run_planner(scene, target_id, cfg, executed(IDMPlanner), **options)


def drive_pdm_closed(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run the driver with PDM-Closed as in the main experiments (``main_pdm_config``)."""
    return run_planner(scene, target_id, cfg, executed(pdm_closed_planner), **options)


def drive_pdm_original(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run the driver with upstream's PDM-Closed (braking up to 3 m/s^2, LQR simulation)."""
    return run_planner(scene, target_id, cfg, executed(pdm_original_planner), **options)


def pdm_closed_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> PDMClosedPlanner:
    """Build PDM-Closed with ``main_pdm_config``; it drives at its lane's limit, not ``v_des``."""
    return PDMClosedPlanner(scene, cfg, rules, main_pdm_config(cfg))


def main_pdm_config(cfg: DriverConfig) -> PDMConfig:
    """Return the main experiments' PDM-Closed config.

    Upstream's IDM proposals, but the emergency brake reaches the vehicle limit at any speed
    and proposals are simulated through the shared executor.
    """
    return PDMConfig(
        emergency_max_speed=None,
        emergency_decel=cfg.b_max,
        simulate_with_executor=True,
        offset_ramp_s=OFFSET_RAMP_S,
    )


def pdm_original_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> PDMClosedPlanner:
    """Build PDM-Closed with upstream's defaults."""
    return PDMClosedPlanner(scene, cfg, rules)


def drive_pdm_side_fault(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Like ``drive_pdm_closed`` but counting active lateral collisions at fault."""
    return run_planner(scene, target_id, cfg, executed(pdm_side_fault_planner), **options)


def pdm_side_fault_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> PDMClosedPlanner:
    """Build the ``pdm_closed_planner`` variant that counts active side impacts at fault."""
    return PDMClosedPlanner(scene, cfg, rules, replace(main_pdm_config(cfg), side_at_fault=True))


def drive_latentdriver(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run the driver with the executed LatentDriver planner."""
    return run_planner(scene, target_id, cfg, executed(latentdriver_planner), **options)


def drive_latentdriver_native(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run LatentDriver by its own displacement (Waymax's DeltaLocal), bypassing the executor."""
    return run_planner(scene, target_id, cfg, latentdriver_planner, **options)


def drive_latentdriver_log_route(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run LatentDriver with upstream's route, the logged ego's whole trajectory.

    That route carries the human's speed, so it tells the model how fast to go.
    """
    return run_planner(scene, target_id, cfg, executed(latentdriver_log_route_planner), **options)


def latentdriver_log_route_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build ``latentdriver_planner`` using ``observation.log_route_points`` for its route."""
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, log_route=True)


def latentdriver_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build LatentDriver, which picks its own speed and does not see signals or stop signs.

    The import is local so importing the package does not load torch.
    """
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg)


def drive_latentdriver_finetuned(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run ``drive_latentdriver`` with the fine-tuned checkpoint (OnCue tests + logged driving)."""
    return run_planner(scene, target_id, cfg, executed(latentdriver_finetuned_planner), **options)


def latentdriver_finetuned_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build ``latentdriver_planner`` with ``FINETUNED_WEIGHTS``."""
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, FINETUNED_WEIGHTS)


def drive_latentdriver_finetuned_logs(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run ``drive_latentdriver`` with the logs-only fine-tuned checkpoint (the control)."""
    return run_planner(
        scene, target_id, cfg, executed(latentdriver_finetuned_logs_planner), **options
    )


def latentdriver_finetuned_logs_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build ``latentdriver_planner`` with ``FINETUNED_LOGS_WEIGHTS``."""
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, FINETUNED_LOGS_WEIGHTS)


def drive_plant(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run the driver with the executed PlanT planner (LatentDriver's release)."""
    return run_planner(scene, target_id, cfg, executed(plant_planner), **options)


def plant_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build PlanT, run as ``latentdriver_planner`` with PlanT's model."""
    from ..planners.latentdriver.model import load_plant
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, PLANT_WEIGHTS, model_loader=load_plant)


def drive_plant_finetuned(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run ``drive_plant`` with the fine-tuned checkpoint (OnCue tests + logged driving)."""
    return run_planner(scene, target_id, cfg, executed(plant_finetuned_planner), **options)


def plant_finetuned_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build ``plant_planner`` with ``PLANT_FINETUNED_WEIGHTS``."""
    from ..planners.latentdriver.model import load_plant
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, PLANT_FINETUNED_WEIGHTS, model_loader=load_plant)


def drive_plant_finetuned_logs(
    scene: Scene, target_id: str | None, cfg: DriverConfig, **options: Any
) -> ego_driver.Drive:
    """Run ``drive_plant`` with the logs-only fine-tuned checkpoint (the control)."""
    return run_planner(scene, target_id, cfg, executed(plant_finetuned_logs_planner), **options)


def plant_finetuned_logs_planner(
    scene: Scene,
    target_id: str | None,
    cfg: DriverConfig,
    perception: Perception,
    *,
    v_des: float,
    rules: TrafficRules,
) -> LatentDriverPlanner:
    """Build ``plant_planner`` with ``PLANT_FINETUNED_LOGS_WEIGHTS``."""
    from ..planners.latentdriver.model import load_plant
    from ..planners.latentdriver.planner import LatentDriverPlanner

    return LatentDriverPlanner(scene, cfg, PLANT_FINETUNED_LOGS_WEIGHTS, model_loader=load_plant)


# Built-in drive functions. Running one without a target produces the nominal distances.
BUILT_IN_DRIVES = (
    drive_idm,
    drive_pdm_closed,
    drive_pdm_original,
    drive_pdm_side_fault,
    drive_latentdriver,
    drive_latentdriver_native,
    drive_latentdriver_log_route,
    drive_latentdriver_finetuned,
    drive_latentdriver_finetuned_logs,
    drive_plant,
    drive_plant_finetuned,
    drive_plant_finetuned_logs,
)

# name -> (drive function, DriverConfig options). External planners register themselves here.
PLANNERS = {name: (drive_idm, options) for name, options in DRIVER_OPTIONS.items()}
PLANNERS["pdm_closed"] = (drive_pdm_closed, {})
PLANNERS["pdm_closed_original"] = (drive_pdm_original, {})  # supplement variant
PLANNERS["pdm_closed_side_fault"] = (drive_pdm_side_fault, {})  # supplement variant
PLANNERS["latentdriver"] = (drive_latentdriver, {})
PLANNERS["latentdriver_native"] = (drive_latentdriver_native, {})  # native-execution comparison
PLANNERS["latentdriver_log_route"] = (drive_latentdriver_log_route, {})  # log-route comparison
PLANNERS["latentdriver_finetuned"] = (drive_latentdriver_finetuned, {})
PLANNERS["latentdriver_finetuned_logs"] = (drive_latentdriver_finetuned_logs, {})
PLANNERS["plant"] = (drive_plant, {})  # PlanT, the second learned planner (supplement)
PLANNERS["plant_finetuned"] = (drive_plant_finetuned, {})
PLANNERS["plant_finetuned_logs"] = (drive_plant_finetuned_logs, {})
# Planners run when none are named; pdm_closed_original and the two fine-tuned LatentDrivers
# run only when named.
DEFAULT_PLANNERS = (
    *DRIVER_OPTIONS,
    "pdm_closed",
    "pdm_closed_side_fault",
    "latentdriver",
    "latentdriver_native",
    "latentdriver_log_route",
)
# Planner -> the weight file it loads; ``planner_weights`` records each by content.
PLANNER_WEIGHTS = {
    "latentdriver": LATENTDRIVER_WEIGHTS,
    "latentdriver_native": LATENTDRIVER_WEIGHTS,
    "latentdriver_log_route": LATENTDRIVER_WEIGHTS,
    "latentdriver_finetuned": FINETUNED_WEIGHTS,
    "latentdriver_finetuned_logs": FINETUNED_LOGS_WEIGHTS,
    "plant": PLANT_WEIGHTS,
    "plant_finetuned": PLANT_FINETUNED_WEIGHTS,
    "plant_finetuned_logs": PLANT_FINETUNED_LOGS_WEIGHTS,
}


def planner_weights(planners: list[str] | tuple[str, ...]) -> dict:
    """Return ``{planner: {path, md5, bytes}}`` for the weight files the named planners load.

    A missing file is recorded with ``md5=None``.
    """
    return {
        name: model_info(path) if path.exists() else {"path": str(path), "md5": None}
        for name, path in PLANNER_WEIGHTS.items()
        if name in planners
    }


def register(name: str, fn: Callable[..., ego_driver.Drive], options: dict | None = None) -> None:
    """Add a planner under ``name``.

    ``fn(scene, target_id, cfg, **options)`` returns a ``planners.driver.Drive``.
    """
    PLANNERS[name] = (fn, options or {})


def import_planner_modules(modules: Sequence[str]) -> None:
    """Import each named module; a module registers its planners with ``register`` when imported.

    The closed loop does this in its own process and in every worker, so a planner outside the
    package runs like a built-in one.
    """
    for module in modules:
        importlib.import_module(module)
