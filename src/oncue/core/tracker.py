"""nuPlan's two-stage controller: the LQR tracker (`LQRTracker`) drives a kinematic bicycle
(`KinematicBicycle`) along planned poses, as nuPlan's simulation executes a planned trajectory;
`simulate` runs both over a batch of plans. PDM-Closed simulates its proposals with it
(`planners.pdm`).

Ported from tuplan_garage b51d5d04, planning/simulation/planner/pdm_planner/ (copyright 2023 Daniel
Dauner, Marcel Hallgarten, Andreas Geiger and Kashyap Chitta), and nuplan-devkit e9241677 where
named (copyright Motional), both under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0; the notice also in planners/pdm/LICENSE):

| ours | upstream | how |
| --- | --- | --- |
| StateIndex, STATE_SE2, VELOCITY_2D, ACCELERATION_2D | utils/pdm_enums.py | copied |
| simulate | simulation/pdm_simulator.py PDMSimulator.simulate_proposals | copied |
| LQRTracker | simulation/batch_lqr.py BatchLQRTracker | copied |
| profile fitting functions | simulation/batch_lqr_utils.py | copied, two calls replaced |
| KinematicBicycle | simulation/batch_kinematic_bicycle.py BatchKinematicBicycleModel | copied |
| principal_value | nuPlan common/geometry/compute.py | copied |

The profile fitting multiplies batches by np.matmul and inverts its normal matrices (regularized,
so invertible) by np.linalg.inv instead of upstream's einsum and pinv: the same fit to 1e-10, about
15 times faster (it was most of the oracle's labelling time).

`simulate` comes first, then what it calls in that order: the tracker (`LQRTracker`), its profile
fitting, the bicycle (`KinematicBicycle`). The profile fitting keeps batch_lqr_utils.py's order,
its helpers before `velocity_curvature_profiles_from_poses`.

Copied line by line, with upstream's matrices A, B, g, Q, R named in words. The tracker solves
two decoupled subsystems, each a one-step LQR over the tracking horizon (small-angle linearized,
Euler with zero-order hold):
- longitudinal: state velocity, input acceleration;
- lateral: states lateral error, heading error and steering angle, input steering rate.
Below STOPPING_VELOCITY (reference and ego) a P controller stops the ego instead. The reference
velocity and curvature are fitted to the poses by regularized least squares. A state is nuPlan's
EgoState at the rear axle (11 values, `StateIndex`). The bicycle is at the rear axle, with
first-order lags on acceleration and steering; speeds are not clipped, so a stopping ego may roll
back slightly, as in nuPlan.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..utils.geometry import wrap_angle


class StateIndex(IntEnum):
    """Columns of an ego state array (pdm_enums.py:4-17)."""

    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8
    ANGULAR_VELOCITY = 9
    ANGULAR_ACCELERATION = 10


STATE_SE2 = slice(StateIndex.X, StateIndex.HEADING + 1)  # pdm_enums.py:91-95
VELOCITY_2D = slice(StateIndex.VELOCITY_X, StateIndex.VELOCITY_Y + 1)  # pdm_enums.py:97-101
ACCELERATION_2D = slice(StateIndex.ACCELERATION_X, StateIndex.ACCELERATION_Y + 1)  # :103-107

# batch_lqr.py:74-83 (nuPlan's lqr_tracker.yaml)
Q_LONGITUDINAL = 10.0  # velocity tracking cost
R_LONGITUDINAL = 1.0  # acceleration cost
Q_LATERAL = (1.0, 10.0, 0.0)  # lateral error, heading error, steering angle costs
R_LATERAL = 1.0  # steering rate cost
TRACKING_HORIZON = 10  # time steps ahead the LQR objective looks
JERK_PENALTY = 1e-4  # velocity profile fit
CURVATURE_RATE_PENALTY = 1e-2  # curvature profile fit
STOPPING_PROPORTIONAL_GAIN = 0.5
STOPPING_VELOCITY = 0.2  # m/s
INITIAL_CURVATURE_PENALTY = 1e-10  # batch_lqr_utils.py:16, conditions a fit from standstill
# batch_kinematic_bicycle.py:42-44 (nuPlan's KinematicBicycleModel defaults)
MAX_STEERING_ANGLE = np.pi / 3  # rad
ACCEL_TIME_CONSTANT = 0.2  # s, low-pass filter of the acceleration
STEERING_ANGLE_TIME_CONSTANT = 0.05  # s, low-pass filter of the steering angle
MICROSECONDS_PER_S = int(1e6)  # nuPlan's TimePoint unit


def simulate(
    poses: np.ndarray, initial_state: np.ndarray, wheel_base: float, interval: float
) -> np.ndarray:
    """[P, N, 11] every proposal (poses [P, N, 3], `interval` apart) tracked from the ego's state
    [11], one tracker and one bicycle step per pose (pdm_simulator.py:39-90)."""
    tracker = LQRTracker(wheel_base, interval)
    bicycle = KinematicBicycle(wheel_base)
    tracker.update(poses[..., STATE_SE2])
    # nuPlan's bicycle steps by integer microseconds read back as `time_us * 1e-6`
    # (state_representation.py:49, 216), so 0.1 s becomes 0.09999999999999999 s; the tracker
    # uses `interval` itself
    sampling_time = int(interval * MICROSECONDS_PER_S) * 1e-6
    simulated = np.zeros((*poses.shape[:2], len(StateIndex)), dtype=np.float64)
    simulated[:, 0] = initial_state
    for time_index in range(1, poses.shape[1]):
        commands = tracker.track(time_index - 1, simulated[:, time_index - 1])
        simulated[:, time_index] = bicycle.propagate(
            simulated[:, time_index - 1], commands, sampling_time
        )
    return simulated


# ---------------------------------------------------------------- the LQR tracker
class LateralStateIndex(IntEnum):
    """The lateral subsystem's state (batch_lqr.py:27-36)."""

    LATERAL_ERROR = 0  # m, at the rear axle
    HEADING_ERROR = 1  # rad
    STEERING_ANGLE = 2  # rad, of the wheels


class LQRTracker:
    """Tracks a batch of proposals (poses [P, N, 3] every `discretization_time`) from ego states
    [P, 11] (batch_lqr.py:39-523, BatchLQRTracker)."""

    def __init__(self, wheel_base: float, discretization_time: float):
        self._wheel_base = wheel_base
        self._discretization_time = discretization_time
        self._q_lateral = np.diag(Q_LATERAL)
        self._r_lateral = np.diag([R_LATERAL])
        self._poses: np.ndarray | None = None
        self._velocity_profile: np.ndarray | None = None
        self._curvature_profile: np.ndarray | None = None

    def update(self, poses: np.ndarray) -> None:
        """Loads the proposals' poses [P, N, 3] (x, y, heading) and resets their profiles."""
        self._poses = poses
        self._velocity_profile, self._curvature_profile = None, None

    def track(self, time_index: int, states: np.ndarray) -> np.ndarray:
        """[P, 2]: acceleration and steering rate commands for ego states [P, 11] at
        `time_index` of the proposals (batch_lqr.py:155-227, track_trajectory)."""
        batch_size = len(states)
        initial_velocity, initial_lateral_state = self._initial_velocity_and_lateral_state(
            time_index, states
        )
        reference_velocities, curvature_profiles = self._reference_velocity_and_curvature(
            time_index
        )
        accel_cmds = np.zeros(batch_size, dtype=np.float64)
        steering_rate_cmds = np.zeros(batch_size, dtype=np.float64)

        # 1. the stopping controller
        should_stop = np.logical_and(
            reference_velocities <= STOPPING_VELOCITY, initial_velocity <= STOPPING_VELOCITY
        )
        accel_cmds[should_stop] = -STOPPING_PROPORTIONAL_GAIN * (
            initial_velocity[should_stop] - reference_velocities[should_stop]
        )
        steering_rate_cmds[should_stop] = 0.0

        # 2. the LQR controllers
        accel_cmds[~should_stop] = self._longitudinal_lqr(
            initial_velocity[~should_stop], reference_velocities[~should_stop]
        )
        velocity_profiles = profile_from_initial_condition_and_derivatives(
            initial_condition=initial_velocity[~should_stop],
            derivatives=np.repeat(accel_cmds[~should_stop, None], TRACKING_HORIZON, axis=-1),
            discretization_time=self._discretization_time,
        )[:, :TRACKING_HORIZON]
        steering_rate_cmds[~should_stop] = self._lateral_lqr(
            initial_lateral_state[~should_stop],
            velocity_profiles,
            curvature_profiles[~should_stop],
        )
        return np.stack([accel_cmds, steering_rate_cmds], axis=-1)

    def _initial_velocity_and_lateral_state(self, time_index: int, states: np.ndarray):
        """(velocity [P], lateral state [P, 3]): the ego's tracking error in the frame of the
        proposal's pose at `time_index` (batch_lqr.py:229-272)."""
        reference = self._poses[:, time_index]
        x_errors = states[:, StateIndex.X] - reference[:, StateIndex.X]
        y_errors = states[:, StateIndex.Y] - reference[:, StateIndex.Y]
        heading_references = reference[:, StateIndex.HEADING]
        lateral_errors = -x_errors * np.sin(heading_references) + y_errors * np.cos(
            heading_references
        )
        heading_errors = wrap_angle(states[:, StateIndex.HEADING] - heading_references)
        lateral_state = np.stack(
            [lateral_errors, heading_errors, states[:, StateIndex.STEERING_ANGLE]], axis=-1
        )
        return states[:, StateIndex.VELOCITY_X], lateral_state

    def _reference_velocity_and_curvature(self, time_index: int):
        """(velocity [P] TRACKING_HORIZON steps ahead, curvature [P, TRACKING_HORIZON] up to
        there), from profiles fitted once to the poses (batch_lqr.py:274-321)."""
        if self._velocity_profile is None or self._curvature_profile is None:
            self._velocity_profile, _, self._curvature_profile, _ = (
                velocity_curvature_profiles_from_poses(
                    discretization_time=self._discretization_time,
                    poses=self._poses[..., STATE_SE2],
                    jerk_penalty=JERK_PENALTY,
                    curvature_rate_penalty=CURVATURE_RATE_PENALTY,
                )
            )
        batch_size, num_poses = self._velocity_profile.shape
        reference_index = min(time_index + TRACKING_HORIZON, num_poses - 1)
        reference_velocities = self._velocity_profile[:, reference_index]
        reference_curvatures = np.zeros((batch_size, TRACKING_HORIZON), dtype=np.float64)
        reference_length = reference_index - time_index
        reference_curvatures[:, 0:reference_length] = self._curvature_profile[
            :, time_index:reference_index
        ]
        if reference_length < TRACKING_HORIZON:  # past the profile's end: its last value
            reference_curvatures[:, reference_length:] = self._curvature_profile[
                :, reference_index, None
            ]
        return reference_velocities, reference_curvatures

    def _longitudinal_lqr(
        self, initial_velocities: np.ndarray, reference_velocities: np.ndarray
    ) -> np.ndarray:
        """The acceleration, held over the horizon, that best reaches the reference velocity
        there: velocity_N = velocity_0 + (N * dt) * acceleration (batch_lqr.py:339-371, 462-484)."""
        batch_size = len(initial_velocities)
        state_matrix = np.ones(batch_size, dtype=np.float64)
        input_matrix = np.full(batch_size, TRACKING_HORIZON * self._discretization_time)
        affine_term = np.zeros(batch_size, dtype=np.float64)
        state_error_zero_input = (
            state_matrix * initial_velocities + affine_term - reference_velocities
        )
        inverse = -1 / (input_matrix * Q_LONGITUDINAL * input_matrix + R_LONGITUDINAL)
        return inverse * input_matrix * Q_LONGITUDINAL * state_error_zero_input

    def _lateral_lqr(
        self,
        initial_lateral_state: np.ndarray,
        velocity_profile: np.ndarray,
        curvature_profile: np.ndarray,
    ):
        """[P]: the steering rate, held over the horizon, that best zeroes the lateral state
        there, along the linear time-varying dynamics of the velocity and curvature profiles
        (batch_lqr.py:373-460, 486-523)."""
        batch_dim = velocity_profile.shape[0]
        n_lateral_states = len(LateralStateIndex)
        dt = self._discretization_time
        identity = np.eye(n_lateral_states, dtype=np.float64)
        in_matrix = np.zeros((n_lateral_states, 1), np.float64)
        in_matrix[LateralStateIndex.STEERING_ANGLE] = dt
        states_matrix_at_step = np.tile(
            identity[None, None, ...], [TRACKING_HORIZON, batch_dim, 1, 1]
        )
        states_matrix_at_step[
            :, :, LateralStateIndex.LATERAL_ERROR, LateralStateIndex.HEADING_ERROR
        ] = velocity_profile.T * dt
        states_matrix_at_step[
            :, :, LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE
        ] = velocity_profile.T * dt / self._wheel_base
        affine_terms = np.zeros((TRACKING_HORIZON, batch_dim, n_lateral_states), dtype=np.float64)
        affine_terms[:, :, LateralStateIndex.HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * dt
        )
        # lateral state at the horizon = state_matrix @ initial + input_matrix @ rate + affine
        state_matrix = np.tile(identity[None, ...], [batch_dim, 1, 1])
        input_matrix = np.zeros((batch_dim, n_lateral_states, 1), dtype=np.float64)
        affine_term = np.zeros((batch_dim, n_lateral_states), dtype=np.float64)
        for state_matrix_at_step, affine_term_at_step in zip(
            states_matrix_at_step, affine_terms, strict=True
        ):
            state_matrix = np.einsum("bij, bjk -> bik", state_matrix_at_step, state_matrix)
            input_matrix = np.einsum("bij, bjk -> bik", state_matrix_at_step, input_matrix)
            input_matrix = input_matrix + in_matrix
            affine_term = np.einsum("bij, bj  -> bi", state_matrix_at_step, affine_term)
            affine_term = affine_term + affine_term_at_step
        steering_rate = self._solve_one_step_lateral_lqr(
            initial_lateral_state, state_matrix, input_matrix, affine_term
        )
        return np.squeeze(steering_rate, axis=-1)

    def _solve_one_step_lateral_lqr(
        self,
        initial_state: np.ndarray,
        state_matrix: np.ndarray,
        input_matrix: np.ndarray,
        affine_term: np.ndarray,
    ) -> np.ndarray:
        """The one-step LQR input toward a zero lateral state, the angle errors wrapped
        (batch_lqr.py:486-523)."""
        state_cost, input_cost = self._q_lateral, self._r_lateral
        angle_diff_indices = [
            LateralStateIndex.HEADING_ERROR.value,
            LateralStateIndex.STEERING_ANGLE.value,
        ]
        input_matrix_transposed = input_matrix.transpose(0, 2, 1)
        state_error_zero_input = (
            np.einsum("bij, bj -> bi", state_matrix, initial_state) + affine_term
        )
        angle = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = np.arctan2(np.sin(angle), np.cos(angle))
        transposed_input_times_cost = np.einsum(
            "bij, jk -> bik", input_matrix_transposed, state_cost
        )
        inverse = -1 / (
            np.einsum("bij, bji -> bi", transposed_input_times_cost, input_matrix) + input_cost
        )
        tail = np.einsum("bij, bj -> bi", transposed_input_times_cost, state_error_zero_input)
        return inverse * tail


# ---------------------------------------------------------------- the LQR tracker's profile fitting
def _batch_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.matmul(left, right)


def profile_from_initial_condition_and_derivatives(
    initial_condition: np.ndarray, derivatives: np.ndarray, discretization_time: float
) -> np.ndarray:
    """[P, N + 1]: a profile integrated from its initial values [P] and derivatives [P, N]
    (batch_lqr_utils.py:22-40)."""
    cumsum = np.cumsum(derivatives * discretization_time, axis=-1)
    return initial_condition[..., None] + np.pad(cumsum, [(0, 0), (1, 0)], mode="constant")


def _xy_heading_displacements(poses: np.ndarray):
    """(xy displacements [P, N - 1, 2], heading displacements [P, N - 1]) between consecutive
    poses [P, N, 3] (batch_lqr_utils.py:43-64)."""
    pose_differences = np.diff(poses, axis=1)
    return pose_differences[..., :2], wrap_angle(pose_differences[..., 2])


def _banded_difference_matrix(number_rows: int) -> np.ndarray:
    """[R, R + 1]: maps [x_1, ..., x_R+1] to [x_2 - x_1, ..., x_R+1 - x_R]
    (batch_lqr_utils.py:67-78)."""
    banded_matrix = np.zeros((number_rows, number_rows + 1), dtype=np.float64)
    eye = np.eye(number_rows, dtype=np.float64)
    banded_matrix[:, 1:] = eye
    banded_matrix[:, :-1] = -eye
    return banded_matrix


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: np.ndarray,
    heading_profile: np.ndarray,
    discretization_time: float,
    jerk_penalty: float,
):
    """(initial velocity [P], accelerations [P, M - 1]) whose integration best reproduces the
    M displacements along the headings, jerk penalized (batch_lqr_utils.py:81-148)."""
    num_displacements = xy_displacements.shape[1]
    batch_size = heading_profile.shape[0]
    targets = xy_displacements.reshape(batch_size, -1)  # [delta x_0, delta y_0, ...]
    headings = np.array(heading_profile, dtype=np.float64)
    design_column = np.zeros(targets.shape, dtype=np.float64)
    design_column[:, 0::2] = np.cos(headings)
    design_column[:, 1::2] = np.sin(headings)
    design = np.repeat(design_column[..., None] * discretization_time**2, num_displacements, axis=2)
    design[..., 0] = design_column * discretization_time
    upper_triangle_mask = np.triu(np.ones((num_displacements, num_displacements), dtype=bool), k=1)
    upper_triangle_mask = np.repeat(upper_triangle_mask, 2, axis=0)
    design[:, upper_triangle_mask] = 0.0
    # jerk: the differences of the M - 1 accelerations
    banded_matrix = _banded_difference_matrix(num_displacements - 2)
    jerk = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])
    jerk = np.repeat(jerk[None, ...], batch_size, axis=0)
    design_transposed, jerk_transposed = (
        np.transpose(design, (0, 2, 1)),
        np.transpose(jerk, (0, 2, 1)),
    )
    intermediate_solution = _batch_matmul(
        np.linalg.inv(
            _batch_matmul(design_transposed, design)
            + jerk_penalty * _batch_matmul(jerk_transposed, jerk)
        ),
        design_transposed,
    )
    solution = np.einsum("bij, bj -> bi", intermediate_solution, targets)
    return solution[:, 0], solution[:, 1:]


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: np.ndarray,
    velocity_profile: np.ndarray,
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
):
    """(initial curvature [P], curvature rates [P, M - 1]) best reproducing the heading
    displacements at the given velocities, curvature rate penalized (batch_lqr_utils.py:151-206)."""
    targets = heading_displacements
    batch_dim, dim = targets.shape
    design = np.repeat(np.tri(dim, dtype=np.float64)[None, ...], batch_dim, axis=0)
    design[:, :, 0] = velocity_profile * discretization_time
    velocity = velocity_profile * discretization_time**2
    design[:, 1:, 1:] *= velocity[:, None, 1:].transpose(0, 2, 1)
    # a small weight on the initial curvature too: from standstill its column is zero
    regularization = curvature_rate_penalty * np.eye(dim)
    regularization[0, 0] = initial_curvature_penalty
    design_transposed = design.transpose(0, 2, 1)
    intermediate = _batch_matmul(
        np.linalg.inv(_batch_matmul(design_transposed, design) + regularization),
        design_transposed,
    )
    solution = np.einsum("bij,bj->bi", intermediate, targets)
    return solution[:, 0], solution[:, 1:]


def velocity_curvature_profiles_from_poses(
    discretization_time: float,
    poses: np.ndarray,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(velocity [P, N - 1], acceleration [P, N - 2], curvature [P, N - 1], curvature rate
    [P, N - 2]) of N poses [P, N, 3] (batch_lqr_utils.py:209-271)."""
    xy_displacements, heading_displacements = _xy_heading_displacements(poses)
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )
    velocity_profile = profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )
    curvature_profile = profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )
    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile


# ---------------------------------------------------------------- the kinematic bicycle
class KinematicBicycle:
    """A batch of rear-axle bicycles (batch_kinematic_bicycle.py:34-213)."""

    def __init__(self, wheel_base: float):
        self._wheel_base = wheel_base

    def _state_dot(self, states: np.ndarray) -> np.ndarray:
        """The rate of change of each state (batch_kinematic_bicycle.py:58-85)."""
        state_dots = np.zeros(states.shape, dtype=np.float64)
        longitudinal_speeds = states[:, StateIndex.VELOCITY_X]
        state_dots[:, StateIndex.X] = longitudinal_speeds * np.cos(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.Y] = longitudinal_speeds * np.sin(states[:, StateIndex.HEADING])
        state_dots[:, StateIndex.HEADING] = (
            longitudinal_speeds * np.tan(states[:, StateIndex.STEERING_ANGLE]) / self._wheel_base
        )
        state_dots[:, VELOCITY_2D] = states[:, ACCELERATION_2D]
        state_dots[:, ACCELERATION_2D] = 0.0
        state_dots[:, StateIndex.STEERING_ANGLE] = states[:, StateIndex.STEERING_RATE]
        return state_dots

    def _update_commands(
        self, states: np.ndarray, commands: np.ndarray, sampling_time: float
    ) -> np.ndarray:
        """The states with acceleration and steering rate moved toward the commands through the
        first-order lags (batch_kinematic_bicycle.py:87-133)."""
        propagating_state = states.copy()
        accel = states[:, StateIndex.ACCELERATION_X]
        steering_angle = states[:, StateIndex.STEERING_ANGLE]
        ideal_accel_x = commands[:, 0]
        ideal_steering_angle = sampling_time * commands[:, 1] + steering_angle
        updated_accel_x = (
            sampling_time / (sampling_time + ACCEL_TIME_CONSTANT) * (ideal_accel_x - accel) + accel
        )
        updated_steering_angle = (
            sampling_time
            / (sampling_time + STEERING_ANGLE_TIME_CONSTANT)
            * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / sampling_time
        propagating_state[:, StateIndex.ACCELERATION_X] = updated_accel_x
        propagating_state[:, StateIndex.ACCELERATION_Y] = 0.0
        propagating_state[:, StateIndex.STEERING_RATE] = updated_steering_rate
        return propagating_state

    def propagate(
        self, states: np.ndarray, commands: np.ndarray, sampling_time: float
    ) -> np.ndarray:
        """[P, 11]: the states after `sampling_time` under commands [P, 2] (acceleration,
        steering rate) (batch_kinematic_bicycle.py:135-213)."""
        if len(states) != len(commands):
            raise ValueError("Batch size of states and commands does not match")
        propagating_state = self._update_commands(states, commands, sampling_time)
        output_state = states.copy()
        state_dot = self._state_dot(propagating_state)
        output_state[:, StateIndex.X] = (
            states[:, StateIndex.X] + state_dot[:, StateIndex.X] * sampling_time
        )
        output_state[:, StateIndex.Y] = (
            states[:, StateIndex.Y] + state_dot[:, StateIndex.Y] * sampling_time
        )
        output_state[:, StateIndex.HEADING] = principal_value(
            states[:, StateIndex.HEADING] + state_dot[:, StateIndex.HEADING] * sampling_time
        )
        output_state[:, StateIndex.VELOCITY_X] = (
            states[:, StateIndex.VELOCITY_X] + state_dot[:, StateIndex.VELOCITY_X] * sampling_time
        )
        output_state[:, StateIndex.VELOCITY_Y] = 0.0  # none in the kinematic bicycle
        output_state[:, StateIndex.STEERING_ANGLE] = np.clip(
            propagating_state[:, StateIndex.STEERING_ANGLE]
            + state_dot[:, StateIndex.STEERING_ANGLE] * sampling_time,
            -MAX_STEERING_ANGLE,
            MAX_STEERING_ANGLE,
        )
        output_state[:, StateIndex.ANGULAR_VELOCITY] = (
            output_state[:, StateIndex.VELOCITY_X]
            * np.tan(output_state[:, StateIndex.STEERING_ANGLE])
            / self._wheel_base
        )
        output_state[:, ACCELERATION_2D] = state_dot[:, VELOCITY_2D]
        output_state[:, StateIndex.ANGULAR_ACCELERATION] = (
            output_state[:, StateIndex.ANGULAR_VELOCITY] - states[:, StateIndex.ANGULAR_VELOCITY]
        ) / sampling_time
        output_state[:, StateIndex.STEERING_RATE] = state_dot[:, StateIndex.STEERING_ANGLE]
        return output_state


def principal_value(angle: np.ndarray, min_angle: float = -np.pi) -> np.ndarray:
    """An angle wrapped into [min_angle, min_angle + 2 pi) (nuplan-devkit e9241677
    common/geometry/compute.py:92-106). Differs from `wrap_angle` in rounding."""
    if not np.all(np.isfinite(angle)):
        raise ValueError("angle is not finite")
    return (angle - min_angle) % (2 * np.pi) + min_angle
