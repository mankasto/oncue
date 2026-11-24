"""Motion helpers, MTR's `mtr/utils/motion_utils.py`: the fixed cubic-spline bases that turn a
mode's NCTRL control points (every 0.5 s, from the current position) into its F future positions,
natural at both ends (`spline_basis`) or, with the model's `start_velocity`, its first derivative
at the origin given (`start_spline_basis`); and the intention points (k-means of endpoints)."""

from __future__ import annotations

import numpy as np
import torch

NCTRL = 16


def spline_basis(future: int) -> torch.Tensor:
    """[future, NCTRL + 1] linear map from control points (origin first) to future positions."""
    from scipy.interpolate import CubicSpline

    knots = np.arange(NCTRL + 1) * 8.0 / NCTRL
    times = np.arange(1, future + 1) / 10.0
    eye = np.eye(NCTRL + 1)
    basis = np.stack(
        [CubicSpline(knots, eye[j], bc_type="natural")(times) for j in range(NCTRL + 1)], 1
    )
    return torch.tensor(basis, dtype=torch.float32)


def start_spline_basis(future: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(basis [future, NCTRL + 1], start [future]): the cubic spline through the control points
    (origin first) with its first derivative at the origin given and a natural end, as positions
    = basis @ control points + start x that derivative (`spline_basis` leaves it free)."""
    from scipy.interpolate import CubicSpline

    knots = np.arange(NCTRL + 1) * 8.0 / NCTRL
    times = np.arange(1, future + 1) / 10.0
    eye = np.eye(NCTRL + 1)
    clamped = ((1, 0.0), (2, 0.0))
    basis = np.stack(
        [CubicSpline(knots, eye[j], bc_type=clamped)(times) for j in range(NCTRL + 1)], 1
    )
    start = CubicSpline(knots, np.zeros(NCTRL + 1), bc_type=((1, 1.0), (2, 0.0)))(times)
    return (
        torch.tensor(basis, dtype=torch.float32),
        torch.tensor(start, dtype=torch.float32),
    )


def kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((k, 2))
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), min(k, len(x)), replace=False)]
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(len(c))])
    return np.resize(c, (k, 2)) if len(c) < k else c


def intention_points(endpoints: np.ndarray, types: np.ndarray, k: int) -> torch.Tensor:
    """[3, k, 2] per agent type."""
    return torch.tensor(
        np.stack([kmeans(endpoints[types == t], k) for t in range(3)]), dtype=torch.float32
    )
