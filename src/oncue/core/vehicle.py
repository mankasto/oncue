"""Execute desired trajectories with nuPlan's two-stage controller, within the vehicle limits.

The oracle's responses, every closed-loop ego and PDM-Closed's proposals all run through
``execute``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.geometry import MIN_DIVISOR, wrap_angle
from .config import VEHICLE_LIMITS, VehicleConfig
from .tracker import (
    ACCEL_TIME_CONSTANT,
    STEERING_ANGLE_TIME_CONSTANT,
    STOPPING_PROPORTIONAL_GAIN,
    STOPPING_VELOCITY,
    TRACKING_HORIZON,
    KinematicBicycle,
    LQRTracker,
    StateIndex,
)
from .types import Track

# Size of the generated vehicle target and of the planners' phantom vehicles.
VEHICLE_SIZE_M = (4.65, 2.08)  # length, width
# Generated vehicle target height (unlogged vehicles use core.types.UNLOGGED_HEIGHT_M).
VEHICLE_HEIGHT_M = 1.71
MAX_SLIP_SINE = 0.99  # clip for sin(slip) so arcsin stays finite
MIN_STEERING_SPEED_MPS = 0.5  # slower than this, the logged yaw rate implies no steering
MIN_TRACKED_POSES = 3  # the tracker's velocity fit needs at least this many poses
INTERVAL_TOL_S = 1e-6  # desired poses are tolerated this far from equal spacing in time
# Horizon-LQR costs for `HorizonTracker`.
LATERAL_ERROR_COST = 1.0  # per m^2
HEADING_ERROR_COST = 1.0  # per rad^2
LATERAL_JERK_COST = 1e-5  # per (m/s^3)^2: the steering rate's cost as the lateral jerk it makes
MIN_STEERING_RATE_COST = 1e-6  # per (rad/s)^2 at standstill
STATION_ERROR_COST = 1.0  # per m^2
SPEED_ERROR_COST = 1.0  # per (m/s)^2
ACCEL_COST = 0.1  # per (m/s^2)^2 of commanded accel off the reference's
# Horizon-LQR state length (a trailing 1 carries the reference's terms).
LATERAL_STATE = 4  # lateral error, heading error, steering angle, 1
LONGITUDINAL_STATE = 4  # station error, speed error, acceleration, 1


def execute(
    desired: np.ndarray,
    times: np.ndarray,
    steering0: float,
    vehicle: VehicleConfig,
    commands: np.ndarray | None = None,
    accel0: float | np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> Execution:
    """Track desired box-centre states and return the executed trajectory.

    Args:
        desired: ``[R, N, 4]`` states (x, y, heading, speed) at equally spaced ``times [N]``.
        commands: ``[R, N]`` direct steering angles; a finite row steers to them instead of
            tracking the path sideways, and its tracking error is NaN.
        accel0: the starting acceleration; None takes the desired speed's first change.
        start: ``[R, 4]`` start states, when not ``desired[:, 0]``.
    """
    desired = np.asarray(desired, float)
    times = np.asarray(times, float)
    rows, poses = desired.shape[:2]
    if poses < MIN_TRACKED_POSES:
        raise ValueError(f"{poses} desired poses; the tracker needs {MIN_TRACKED_POSES}")
    interval = float(times[1] - times[0])
    if np.abs(np.diff(times) - interval).max() > INTERVAL_TOL_S:
        raise ValueError("the desired poses must be equally spaced in time")
    steer = None
    if commands is not None:
        commands = np.asarray(commands, float)
        if commands.shape != desired.shape[:2]:
            raise ValueError(f"commands {commands.shape} for desired {desired.shape}")
        steer = np.isfinite(commands).all(axis=1)
        if (np.isfinite(commands).any(axis=1) != steer).any():
            raise ValueError("a commanded row needs a command at every frame")
    first = desired[:, 0] if start is None else np.asarray(start, float)
    if accel0 is None:
        accel0 = (desired[:, 1, 3] - first[:, 3]) / interval
    lr, wheelbase = vehicle.center_to_rear_axle, vehicle.wheelbase
    tracker = HorizonTracker(wheelbase, interval)
    tracker.update(_shift(desired[..., :3], -lr))
    bicycle = LimitedBicycle(wheelbase, lr)
    states = np.zeros((rows, poses, len(StateIndex)))
    states[:, 0] = _start(first, steering0, accel0, vehicle)
    for i in range(1, poses):
        command = tracker.track(i - 1, states[:, i - 1], speed_only=steer)
        if steer is not None:  # the steering rate that reaches the commanded angle in one step
            angle = np.clip(commands[steer, i - 1], -vehicle.max_steering, vehicle.max_steering)
            reach = angle - states[steer, i - 1, StateIndex.STEERING_ANGLE]
            command[steer, 1] = reach / interval
        states[:, i] = bicycle.propagate(states[:, i - 1], command, interval)
    frames = np.concatenate(
        (
            _shift(states[..., :3], lr),
            states[..., [StateIndex.VELOCITY_X, StateIndex.STEERING_ANGLE]],
        ),
        axis=-1,
    )
    error = np.abs(_lateral_error(frames, desired)).max(axis=1)
    if steer is not None:
        error = np.where(steer, np.nan, error)
    dense_times, dense = _densify(times, frames, vehicle.substeps)
    return Execution(dense_times, dense, error, states)


@dataclass(frozen=True, eq=False)
class Execution:
    times: np.ndarray  # [M] scene clock, takeover first; `substeps` samples per frame
    states: np.ndarray  # [R, M, 5]: box centre x, y, heading, speed, steering
    tracking_error_m: np.ndarray  # [R] largest lateral distance from the desired pose (frames)
    # [R, N, 11] rear-axle bicycle states (`core.tracker.StateIndex`) at the desired times
    bicycle_states: np.ndarray


def _start(
    first: np.ndarray, steering0: float, accel0: float | np.ndarray, vehicle: VehicleConfig
) -> np.ndarray:
    """Return ``[R, 11]`` rear-axle bicycle states (``StateIndex``) at desired box-centre states."""
    states = np.zeros((len(first), len(StateIndex)))
    states[:, :3] = _shift(first[:, :3], -vehicle.center_to_rear_axle)
    states[:, StateIndex.VELOCITY_X] = first[:, 3]
    states[:, StateIndex.ACCELERATION_X] = accel0
    states[:, StateIndex.STEERING_ANGLE] = steering0
    states[:, StateIndex.ANGULAR_VELOCITY] = first[:, 3] * np.tan(steering0) / vehicle.wheelbase
    return states


def _shift(poses: np.ndarray, distance: float) -> np.ndarray:
    """Shift poses ``[..., 3]`` by ``distance`` along their headings (box centre <-> rear axle)."""
    out = np.array(poses, float)
    out[..., 0] += distance * np.cos(out[..., 2])
    out[..., 1] += distance * np.sin(out[..., 2])
    return out


def _lateral_error(executed: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Return ``[R, N]`` the signed cross-heading offset of executed box centres from desired."""
    gap = executed[..., :2] - desired[..., :2]
    heading = desired[..., 2]
    return -gap[..., 0] * np.sin(heading) + gap[..., 1] * np.cos(heading)


def _densify(times: np.ndarray, frames: np.ndarray, substeps: int) -> tuple:
    """Return ``substeps`` samples per interval, linear between frames, with unwrapped headings."""
    dense_times = np.linspace(times[0], times[-1], (len(times) - 1) * substeps + 1)
    unwrapped = frames.copy()
    unwrapped[..., 2] = np.unwrap(frames[..., 2], axis=1)
    dense = np.stack(
        [
            np.stack([np.interp(dense_times, times, row[:, k]) for k in range(row.shape[1])], -1)
            for row in unwrapped
        ]
    )
    dense[..., 2] = wrap_angle(dense[..., 2])
    return dense_times, dense


class HorizonTracker(LQRTracker):
    """nuPlan's LQR tracker with both LQRs solved over the full horizon.

    Each step runs a Riccati recursion (``first_gain``) on the bicycle's discrete model and
    applies the first command, where nuPlan holds one command over the horizon. The stopping
    controller, profile fitting and lateral model are nuPlan's.
    """

    def track(
        self, time_index: int, states: np.ndarray, speed_only: np.ndarray | None = None
    ) -> np.ndarray:
        """Return ``[P, 2]`` acceleration and steering rate for ego states at ``time_index``.

        ``speed_only [P]`` marks rows under direct steering, which track the reference speed only.
        """
        velocity, lateral_state = self._initial_velocity_and_lateral_state(time_index, states)
        reference_velocity, curvature = self._reference_velocity_and_curvature(time_index)
        stop = (reference_velocity <= STOPPING_VELOCITY) & (velocity <= STOPPING_VELOCITY)
        accel = -STOPPING_PROPORTIONAL_GAIN * (velocity - reference_velocity)  # batch_lqr.py:185
        rate = np.zeros(len(states))
        go = ~stop
        speeds = self._reference_speeds(time_index)[go]
        ahead = self._station_error(time_index, states)
        if speed_only is not None:
            ahead = np.where(speed_only, 0.0, ahead)
        ahead = ahead[go]
        accel[go] = self._longitudinal(
            ahead, velocity[go] - speeds[:, 0], states[go, StateIndex.ACCELERATION_X], speeds
        )
        # the speeds the lateral model expects: the reference's, off by the ego's speed error
        expected = np.maximum(speeds[:, :-1] + (velocity[go] - speeds[:, 0])[:, None], 0.0)
        rate[go] = self._lateral_lqr(lateral_state[go], expected, curvature[go])
        return np.stack([accel, rate], axis=-1)

    def _reference_speeds(self, time_index: int) -> np.ndarray:
        """Return ``[P, H + 1]`` reference speeds over the horizon, held past the profile's end."""
        last = self._velocity_profile.shape[1] - 1
        steps = np.minimum(np.arange(time_index, time_index + TRACKING_HORIZON + 1), last)
        return self._velocity_profile[:, steps]

    def _station_error(self, time_index: int, states: np.ndarray) -> np.ndarray:
        """Return ``[P]`` how far each state is ahead of the reference pose, along its heading."""
        reference = self._poses[:, time_index]
        gap = states[:, :2] - reference[:, :2]
        heading = reference[:, 2]
        return gap[:, 0] * np.cos(heading) + gap[:, 1] * np.sin(heading)

    def _longitudinal(
        self, ahead: np.ndarray, speed_error: np.ndarray, accel: np.ndarray, speeds: np.ndarray
    ) -> np.ndarray:
        """Return ``[P]`` acceleration commands from the longitudinal LQR.

        Its model matches ``KinematicBicycle.propagate``, the acceleration lagging the command.
        """
        rows, dt = len(ahead), self._discretization_time
        follow = dt / (dt + ACCEL_TIME_CONSTANT)
        reference_accel = np.diff(speeds, axis=1) / dt  # [P, H]
        moves = np.tile(np.eye(LONGITUDINAL_STATE), (TRACKING_HORIZON, rows, 1, 1))
        moves[:, :, 0, 1] = dt
        moves[:, :, 1, 2] = dt * (1 - follow)
        moves[:, :, 1, 3] = -reference_accel.T * dt
        moves[:, :, 2, 2] = 1 - follow
        command = np.array([0.0, dt * follow, follow, 0.0])
        state_cost = np.diag([STATION_ERROR_COST, SPEED_ERROR_COST, 0.0, 0.0])
        # ACCEL_COST x (command - reference acceleration)^2: its cross term with the constant 1
        cross = np.zeros((TRACKING_HORIZON, rows, LONGITUDINAL_STATE))
        cross[:, :, 3] = -ACCEL_COST * reference_accel.T
        gain = first_gain(moves, command, state_cost, np.full(rows, ACCEL_COST), cross)
        state = np.column_stack((ahead, speed_error, accel, np.ones(rows)))
        return -np.einsum("pj,pj->p", gain, state)

    def _lateral_lqr(
        self,
        initial_lateral_state: np.ndarray,
        velocity_profile: np.ndarray,
        curvature_profile: np.ndarray,
    ) -> np.ndarray:
        """Return ``[P]`` steering rates from the lateral LQR.

        Linearised as in nuPlan; the rate's cost is scaled as the lateral jerk it produces.
        """
        rows, horizon = velocity_profile.shape
        dt, wheelbase = self._discretization_time, self._wheel_base
        moves = np.tile(np.eye(LATERAL_STATE), (horizon, rows, 1, 1))
        moves[:, :, 0, 1] = velocity_profile.T * dt
        moves[:, :, 1, 2] = velocity_profile.T * dt / wheelbase
        moves[:, :, 1, 3] = -(velocity_profile * curvature_profile).T * dt
        command = np.array([0.0, 0.0, dt * dt / (dt + STEERING_ANGLE_TIME_CONSTANT), 0.0])
        state_cost = np.diag([LATERAL_ERROR_COST, HEADING_ERROR_COST, 0.0, 0.0])
        jerk_per_rate = velocity_profile[:, 0] ** 2 / wheelbase
        rate_cost = LATERAL_JERK_COST * jerk_per_rate**2 + MIN_STEERING_RATE_COST
        gain = first_gain(moves, command, state_cost, rate_cost)
        state = np.column_stack((initial_lateral_state, np.ones(rows)))
        return -np.einsum("pj,pj->p", gain, state)


def first_gain(
    moves: np.ndarray,
    command: np.ndarray,
    state_cost: np.ndarray,
    command_cost: np.ndarray,
    cross: np.ndarray | None = None,
) -> np.ndarray:
    """Return the first-step gain ``[P, n]`` of a finite-horizon LQR by Riccati recursion.

    Dynamics: ``z_{k+1} = moves[k] z_k + command u_k`` (``moves [H, P, n, n]``, ``command [n]``).
    Stage cost: ``z' state_cost z + 2 z' cross[k] u + command_cost u^2`` (``state_cost [n, n]``,
    ``cross [H, P, n]`` or ``None``, ``command_cost [P]``). Terminal cost: ``z' state_cost z``.
    The first command is then ``u_0 = -gain . z_0``.
    """
    rows, size = moves.shape[1], moves.shape[2]
    cost_to_go = np.broadcast_to(state_cost, (rows, size, size))
    for k in range(len(moves) - 1, -1, -1):
        move = moves[k]
        weighted = cost_to_go @ command  # P b
        coupling = (weighted[:, None, :] @ move)[:, 0]  # b' P A (+ cross')
        if cross is not None:
            coupling = coupling + cross[k]
        gain = coupling / (command_cost + weighted @ command)[:, None]
        cost_to_go = (
            state_cost
            + move.transpose(0, 2, 1) @ cost_to_go @ move
            - coupling[:, :, None] * gain[:, None, :]
        )
    return gain


class LimitedBicycle(KinematicBicycle):
    """nuPlan's kinematic bicycle whose lagged commands stay within ``VEHICLE_LIMITS``.

    Acceleration, brake-onset jerk, steering angle and steering rate are bounded, and braking
    stops at standstill.
    """

    def __init__(self, wheel_base: float, center_to_rear_axle: float):
        super().__init__(wheel_base)
        self.center_to_rear_axle = center_to_rear_axle
        self.limits = VEHICLE_LIMITS

    def _update_commands(
        self, states: np.ndarray, commands: np.ndarray, sampling_time: float
    ) -> np.ndarray:
        out = super()._update_commands(states, commands, sampling_time)
        limits = self.limits
        speed = np.maximum(states[:, StateIndex.VELOCITY_X], 0.0)
        previous = states[:, StateIndex.ACCELERATION_X]
        lowest = np.maximum(previous - limits.brake_jerk * sampling_time, -limits.brake)
        accel = np.clip(out[:, StateIndex.ACCELERATION_X], lowest, limits.accel)
        accel = np.maximum(accel, -speed / sampling_time)  # stops, never reverses
        out[:, StateIndex.ACCELERATION_X] = accel
        fastest = np.maximum(speed, speed + accel * sampling_time)  # over this step
        steering = states[:, StateIndex.STEERING_ANGLE]
        with np.errstate(divide="ignore"):
            rate_bound = limits.lateral_jerk * self._wheel_base * np.cos(steering) ** 2 / fastest**2
        rate = np.clip(out[:, StateIndex.STEERING_RATE], -rate_bound, rate_bound)
        bound = self.steering_bound(fastest)
        reached = np.clip(steering + rate * sampling_time, -bound, bound)
        out[:, StateIndex.STEERING_RATE] = (reached - steering) / sampling_time
        return out

    def steering_bound(self, speed: np.ndarray) -> np.ndarray:
        """Return the largest steering angle at rear-axle ``speed``.

        Bounded by the curvature limit and, at the box centre,
        ``v^2 k sqrt(1 + (lr k)^2) <= lateral`` for the rear-axle curvature ``k``.
        """
        limits, lr = self.limits, self.center_to_rear_axle
        speed_squared = np.maximum(speed, MIN_DIVISOR) ** 2
        ratio = 2 * lr * limits.lateral / speed_squared
        lateral = limits.lateral / speed_squared * np.sqrt(2 / (1 + np.sqrt(1 + ratio**2)))
        return np.arctan(self._wheel_base * np.minimum(limits.curvature, lateral))


def initial_steering(
    track: Track, start: int, dt: float, vehicle: VehicleConfig, anchor: int
) -> float:
    """Return the steering implied by the track's yaw rate just before ``start``, else 0.

    At the anchor the frame after it is used, since the track may change drivers there.
    """
    first = start + 1 if start == anchor else start
    if first < 1 or first >= len(track.valid) or not track.valid[first - 1 : first + 1].all():
        return 0.0
    prev, cur = track.states[first - 1], track.states[first]
    speed = (prev[3] + cur[3]) / 2
    if speed < MIN_STEERING_SPEED_MPS:
        return 0.0
    ratio = vehicle.center_to_rear_axle * wrap_angle(cur[2] - prev[2]) / dt / speed
    slip = np.arcsin(np.clip(ratio, -MAX_SLIP_SINE, MAX_SLIP_SINE))
    steering = np.arctan(vehicle.wheelbase / vehicle.center_to_rear_axle * np.tan(slip))
    return float(np.clip(steering, -vehicle.max_steering, vehicle.max_steering))
