"""Closed-loop replay of generated tests against planners.

``closed_loop_scene`` runs one scene's tests; ``oncue.closedloop.run.run`` runs a whole label
output. ``register`` adds a planner to ``PLANNERS``.
"""

from .registry import DEFAULT_PLANNERS, PLANNERS, register
from .run import TESTS
from .scene import MODES, RunOptions, closed_loop_scene, run_name

__all__ = [
    "DEFAULT_PLANNERS",
    "MODES",
    "PLANNERS",
    "TESTS",
    "RunOptions",
    "closed_loop_scene",
    "register",
    "run_name",
]
