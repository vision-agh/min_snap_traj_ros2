"""Minimum Snap Trajectory generation."""

# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Hubert Szolc, EVS AGH University of Krakow, Poland
#
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#
#  You may obtain a copy of the License at
#  https://opensource.org/licenses/MIT
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ------------------------------------------------------------------------------

import numpy as np
import cvxpy as cp
from math import factorial

# import matplotlib.pyplot as plt

from typing import List, Tuple

DEG = 7
N_COEFF = DEG + 1
K_R = 4
N_COORDS = 4  # Number of coordinates (x, y, z, yaw)

Point = List[
    float
]  # Type alias for a trajectory point in flat output space (x, y, z, yaw)


def t_vec(t: float, deg: int, deriv: float = 0) -> List[float]:
    """Generate the n-th derivative of time vector for the polynomial coefficients.

    Function generates the n-th derivative of vector of time powers for polynomial coefficients
    [1, t, t^2, ..., t^N_COEFF-1].

    Args:
        t (float): The time at which to evaluate the polynomial coefficients.
        deg (int): The degree of the polynomial.
        deriv (int): The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.
    Raises:
        ValueError: If deriv is not in the range [0, deg].
    Returns:
        List[float]: A list of polynomial coefficients evaluated at time t.
    """
    n_coeff = deg + 1
    if deriv < 0 or deriv >= n_coeff:
        raise ValueError(f"deriv must be in range [0, {deg}]")
    ret_vec = [0.0] * n_coeff
    for i in range(deriv, n_coeff):
        ret_vec[i] = factorial(i) / factorial(i - deriv) * t ** (i - deriv)
    return ret_vec


def snap_cost_entry(i: int, j: int, t1: float, t2: float):
    """Calculate the cost entry for the snap cost matrix.

    This function calculates the integral of the polynomial coefficients. It can be used to generate
    the snap cost matrix for one segment of the trajectory between two time points t1 and t2.

    Args:
        i (int): The index of the first polynomial coefficient.
        j (int): The index of the second polynomial coefficient.
        t1 (float): The start time of the segment.
        t2 (float): The end time of the segment.
    Returns:
        float: The value of the cost entry for the snap cost matrix.
    """
    if i < K_R or j < K_R:
        return 0.0
    coef = factorial(i) / factorial(i - K_R) * factorial(j) / factorial(j - K_R)
    t_pow = i + j - 2 * K_R + 1
    return coef * (t2**t_pow - t1**t_pow) / t_pow


def snap_cost_matrix(t1: float, t2: float) -> np.ndarray:
    """Generate the snap cost matrix.

    This function generates the snap cost matrix for a polynomial segment between two time points t1 and t2.

    Args:
        t1 (float): The start time of the segment.
        t2 (float): The end time of the segment.
    Returns:
        np.ndarray: The snap cost matrix of shape (N_COEFF, N_COEFF).
    """
    Q = np.zeros((N_COEFF, N_COEFF))
    for i in range(N_COEFF):
        for j in range(N_COEFF):
            Q[i, j] = snap_cost_entry(i, j, t1, t2)
    return Q


def snap_cost_block_xyz(t1: float, t2: float) -> np.ndarray:
    """Generate the snap cost block for given coordinates.

    Internally, this function generates snap cost matrices for each coordinate and places them along the diagonal of a
    larger block matrix.

    Args:
        t1 (float): The start time of the segment.
        t2 (float): The end time of the segment.
    Returns:
        np.ndarray: The snap cost block of shape (N_COORDS * N_COEFF, N_COORDS * N_COEFF).
    """
    H = np.zeros((N_COORDS * N_COEFF, N_COORDS * N_COEFF))
    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        H[start_idx:end_idx, start_idx:end_idx] = snap_cost_matrix(t1, t2)

    return H


def constraints_block_xyz(t: float, deriv: int = 0) -> np.ndarray:
    """Generate the constraints block for given coordinates.

    This function facilitates the creation of the constraints matrix. It generates N_COORDS rows, each containing the
    time basis vector for the specified derivative order.

    Args:
        t (float): The time at which to evaluate the polynomial coefficients.
        deriv (int): The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.
    Returns:
        np.ndarray: The constraints block of shape (N_COORDS, N_COORDS * N_COEFF).
    Raises:
        ValueError: If deriv is not in the range [0, DEG].
    """
    A = np.zeros((N_COORDS, N_COORDS * N_COEFF))
    # Get time vectors for x, y, z
    time_vec = t_vec(t, DEG, deriv)

    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        A[i, start_idx:end_idx] = time_vec

    return A


def create_constraints(wps: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """Create the constraints matrix and vector for the optimization problem.

    This function constructs the constraints matrix A and vector b based on the provided waypoints. Internally, it uses
    the `constraints_block_xyz` function to generate the constraints for each coordinate at each waypoint.
    The constraints include initial and final positions, velocities, and continuity conditions for internal waypoints.
    The constraints are structured as follows:
    1. Initial position (N_COORDS)
    2. Initial velocity is zero (N_COORDS)
    3. Final position (N_COORDS)
    4. Final velocity is zero (N_COORDS)
    5. Position at each internal waypoint - the same for 2 polynomials ((n_wps - 2) * 2 * N_COORDS)
    6. Velocity continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    7. Acceleration continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    8. Jerk continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    Total constraints: (4 + (n_wps - 2) * 5) * N_COORDS

    Args:
        wps (List[Point]): A list of waypoints, where each waypoint is a list of flat output coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the constraints matrix A and vector b.
    """
    n_wps = len(wps)
    n_vars = (n_wps - 1) * N_COORDS * N_COEFF
    n_constraints = (4 + (n_wps - 2) * 5) * N_COORDS

    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    m = 0  # Constraint row index

    # 1. Initial position constraints
    A[m : m + N_COORDS, 0 : N_COORDS * N_COEFF] = constraints_block_xyz(wps[0][0], 0)
    b[m : m + N_COORDS] = wps[0][1 : 1 + N_COORDS]
    m += N_COORDS

    # 2. Initial velocity constraints
    A[m : m + N_COORDS, 0 : N_COORDS * N_COEFF] = constraints_block_xyz(wps[0][0], 1)
    b[m : m + N_COORDS] = [0.0] * N_COORDS  # Initial velocity is zero
    m += N_COORDS

    # 3. Final position constraints
    A[m : m + N_COORDS, -N_COORDS * N_COEFF :] = constraints_block_xyz(wps[-1][0], 0)
    b[m : m + N_COORDS] = wps[-1][1 : 1 + N_COORDS]
    m += N_COORDS

    # 4. Final velocity constraints
    A[m : m + N_COORDS, -N_COORDS * N_COEFF :] = constraints_block_xyz(wps[-1][0], 1)
    b[m : m + N_COORDS] = [0.0] * N_COORDS  # Final velocity is zero
    m += N_COORDS

    # 5. - 7. Constraints for internal waypoints
    for i in range(1, n_wps - 1):
        t = wps[i][0]
        start_col_prev_idx = (i - 1) * N_COORDS * N_COEFF
        end_col_prev_idx = start_col_prev_idx + N_COORDS * N_COEFF
        start_col_next_idx = i * N_COORDS * N_COEFF
        end_col_next_idx = start_col_next_idx + N_COORDS * N_COEFF

        # Position for both segments
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = (
            constraints_block_xyz(t, 0)
        )
        b[m : m + N_COORDS] = wps[i][1 : 1 + N_COORDS]
        A[m + N_COORDS : m + 2 * N_COORDS, start_col_next_idx:end_col_next_idx] = (
            constraints_block_xyz(t, 0)
        )
        b[m + N_COORDS : m + 2 * N_COORDS] = wps[i][1 : 1 + N_COORDS]
        m += 2 * N_COORDS

        # Velocity continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = (
            constraints_block_xyz(t, 1)
        )
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = (
            -constraints_block_xyz(t, 1)
        )
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS

        # Acceleration continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = (
            constraints_block_xyz(t, 2)
        )
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = (
            -constraints_block_xyz(t, 2)
        )
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS

        # Jerk continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = (
            constraints_block_xyz(t, 3)
        )
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = (
            -constraints_block_xyz(t, 3)
        )
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS
    return A, b


def get_flat_output(
    c: cp.Variable, t: float, bound_times: List[float], deriv: int = 0
) -> Point:
    """Get the flat output of the trajectory at time t.

    This function evaluates the polynomial coefficients at a given time t and returns the flat output
    coordinates for the trajectory. It uses the coefficient vector c (optimization result) and the bound times
    to determine which polynomial segment to evaluate. It handles also derivatives of the flat output.

    Args:
        c (cp.Variable): The coefficient vector from the optimization problem.
        t (float): The time at which to evaluate the polynomial coefficients.
        bound_times (List[float]): A list of times that define the bounds of the polynomial segments. For k segments,
            there should be k+1 bound times.
        deriv (int): The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.

    Returns:
        Point: A list of flat output coordinates evaluated at time t.

    Raises:
        ValueError: If the number of bound times is less than 2, if the coefficient vector length does not match
            the expected size, or if the time t is out of bounds of the provided waypoints.
    """
    n_bounds = len(bound_times)
    if n_bounds < 2:
        raise ValueError(
            "At least two bound times are required to evaluate the polynomial."
        )
    n_polynomials = n_bounds - 1
    if len(c) != n_polynomials * N_COORDS * N_COEFF:
        raise ValueError("Coefficient vector length does not match the expected size.")
    if t < bound_times[0] or t > bound_times[-1]:
        raise ValueError(f"Time t={t} is out of bounds of the provided waypoints.")

    # Determine which polynomial segment to use
    for i in range(n_bounds - 1):
        if bound_times[i] <= t <= bound_times[i + 1]:
            start_idx = i * N_COORDS * N_COEFF
            end_idx = start_idx + N_COORDS * N_COEFF
            c_segment = c[start_idx:end_idx]
            break
    else:
        raise ValueError(
            f"Time t={t} does not fall within the bounds of the provided waypoints."
        )

    # Calculate the polynomial value at time t
    t_vec_eval = t_vec(t, DEG, deriv)
    flat_outputs = []
    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        flat_outputs.append(np.dot(c_segment[start_idx:end_idx], t_vec_eval))

    return flat_outputs


def get_trajectory_minmax(c: cp.Variable, bound_times: List[float]):
    """Get the position, velocity, and acceleration minimum and maximum values.

    This function evaluates the trajectory defined by the polynomial coefficients c at a set of time points
    between the first and last bound times. It computes the minimum and maximum values of position, velocity,
    and acceleration across the trajectory.

    Args:
        c (cp.Variable): The coefficient vector from the optimization problem.
        bound_times (List[float]): A list of times that define the bounds of the polynomial segments.
    Returns:
        Tuple[float, float, float, float, float, float]: A tuple containing the minimum and maximum values of
            position (x, y, z), velocity (vx, vy, vz), and acceleration (ax, ay, az) across the trajectory.
    """
    t = np.linspace(bound_times[0], bound_times[-1], 1000)
    positions = np.array([get_flat_output(c, ti, bound_times, 0) for ti in t])
    velocities = np.array([get_flat_output(c, ti, bound_times, 1) for ti in t])
    accelerations = np.array([get_flat_output(c, ti, bound_times, 2) for ti in t])
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    vel_min = np.min(velocities, axis=0)
    vel_max = np.max(velocities, axis=0)
    acc_min = np.min(accelerations, axis=0)
    acc_max = np.max(accelerations, axis=0)
    return pos_min, pos_max, vel_min, vel_max, acc_min, acc_max


def main():
    """Run the minimum snap trajectory optimization."""
    # --- Define the waypoints ---
    wps = [
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Initial waypoint (t, x, y, z, yaw)
        [1.0, 1.0, 1.0, 2.0, -np.pi / 2],  # Second waypoint (t, x, y, z, yaw)
        [2.0, 2.0, 0.0, 3.0, -np.pi / 6],  # Third waypoint (t, x, y, z, yaw)
        [3.0, 1.0, -1.0, 1.0, np.pi / 3],  # Fourth waypoint (t, x, y, z, yaw)
        [4.0, -1.0, -1.0, 1.0, 0.0],  # Fifth waypoint (t, x, y, z, yaw)
    ]
    n_wps = len(wps)
    n_vars = (n_wps - 1) * N_COORDS * N_COEFF  # Number of variables

    # --- Decision variables ---
    c = cp.Variable(n_vars)

    # --- Objective function ---
    H = np.zeros((n_vars, n_vars))
    for i in range(n_wps - 1):
        t1 = wps[i][0]
        t2 = wps[i + 1][0]
        H_block = snap_cost_block_xyz(t1, t2)
        start_idx = i * N_COORDS * N_COEFF
        end_idx = start_idx + N_COORDS * N_COEFF
        H[start_idx:end_idx, start_idx:end_idx] = H_block
    H = cp.psd_wrap(H)  # Ensure H is positive semidefinite
    cost = cp.quad_form(c, H)
    objective = cp.Minimize(cost)

    # --- Define the constraints ---
    A, b = create_constraints(wps)
    constraints = [A @ c == b]

    # --- Problem definition ---
    problem = cp.Problem(objective, constraints)

    # --- Solve the problem ---
    problem.solve()

    # --- Output the results ---
    print("Optimal coefficients:", c.value)
    print("Optimal cost:", problem.value)

    # --- Get trajectory min/max values ---
    pos_min, pos_max, vel_min, vel_max, acc_min, acc_max = get_trajectory_minmax(
        c.value, [wp[0] for wp in wps]
    )
    print("Position min:", pos_min)
    print("Position max:", pos_max)
    print("Velocity min:", vel_min)
    print("Velocity max:", vel_max)
    print("Acceleration min:", acc_min)
    print("Acceleration max:", acc_max)

    # --- Plot the trajectory in 3D ---
    # t_values = np.linspace(wps[0][0], wps[-1][0], 1000)
    # trajectory = np.array(
    #     [get_flat_output(c.value, t, [wp[0] for wp in wps]) for t in t_values]
    # )
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory")
    # ax.scatter(*zip(*[wp[1:4] for wp in wps]), color="red", label="Waypoints")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(t_values, trajectory[:, 3], label="Yaw")
    # ax2.plot([wp[0] for wp in wps], [wp[4] for wp in wps], "ro", label="Waypoints Yaw")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Yaw (rad)")
    # ax2.legend()
    # ax2.grid()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
