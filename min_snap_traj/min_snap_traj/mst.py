"""Minimum Snap Trajectory generation."""

# -------------------------------------------------------------------------------
# Copyright (c) 2025 Hubert Szolc, EVS AGH University of Krakow, Poland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -------------------------------------------------------------------------------
import copy
from math import factorial
from typing import List, Tuple

import cvxpy as cp
import numpy as np


# import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

DEG = 7
N_COEFF = DEG + 1
K_R = 4
N_COORDS = 4  # Number of coordinates (x, y, z, yaw)

Point = List[float]  # Type alias for a trajectory point in flat output space (x, y, z, yaw)


def t_vec(t: float, deg: int, deriv: int = 0) -> List[float]:
    """
    Generate the n-th derivative of time vector for the polynomial coefficients.

    Function generates the n-th derivative of vector of time powers for polynomial coefficients \
    [1, t, t^2, ..., t^N_COEFF-1].

    Parameters
    ----------
    t : float
        The time at which to evaluate the polynomial coefficients.
    deg : int
        The degree of the polynomial.
    deriv : int, optional
        The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.

    Returns
    -------
    List[float]
        A list of polynomial coefficients evaluated at time t.

    Raises
    ------
    ValueError
        If deriv is not in the range [0, deg].

    """
    n_coeff = deg + 1
    if deriv < 0 or deriv >= n_coeff:
        raise ValueError(f"deriv must be in range [0, {deg}]")
    ret_vec = [0.0] * n_coeff
    for i in range(deriv, n_coeff):
        ret_vec[i] = factorial(i) / factorial(i - deriv) * t ** (i - deriv)
    return ret_vec


def snap_cost_entry(i: int, j: int, t1: float, t2: float) -> float:
    """
    Calculate the cost entry for the snap cost matrix.

    This function calculates the integral of the polynomial coefficients. It can be used to \
    generate the snap cost matrix for one segment of the trajectory between two time points t1 \
    and t2.

    Parameters
    ----------
    i : int
        The index of the first polynomial coefficient.
    j : int
        The index of the second polynomial coefficient.
    t1 : float
        The start time of the segment.
    t2 : float
        The end time of the segment.

    Returns
    -------
    float
        The cost entry value for the snap cost matrix.

    """
    if i < K_R or j < K_R:
        return 0.0
    coef = factorial(i) / factorial(i - K_R) * factorial(j) / factorial(j - K_R)
    t_pow = i + j - 2 * K_R + 1
    return coef * (t2**t_pow - t1**t_pow) / t_pow


def snap_cost_matrix(t1: float, t2: float) -> np.ndarray:
    """
    Generate the snap cost matrix.

    This function generates the snap cost matrix for a polynomial segment between two time points \
    t1 and t2.

    Parameters
    ----------
    t1 : float
        The start time of the segment.
    t2 : float
        The end time of the segment.

    Returns
    -------
    np.ndarray
        A square matrix of shape (N_COEFF, N_COEFF) representing the snap cost for the polynomial \
        segment.

    """
    Q = np.zeros((N_COEFF, N_COEFF))
    for i in range(N_COEFF):
        for j in range(N_COEFF):
            Q[i, j] = snap_cost_entry(i, j, t1, t2)
    return Q


def snap_cost_block_xyz(t1: float, t2: float) -> np.ndarray:
    """
    Generate the snap cost block for given coordinates.

    Internally, this function generates snap cost matrices for each coordinate and places them \
    along the diagonal of a larger block matrix.

    Parameters
    ----------
    t1 : float
        The start time of the segment.
    t2 : float
        The end time of the segment.

    Returns
    -------
    np.ndarray
        A block matrix of shape (N_COORDS * N_COEFF, N_COORDS * N_COEFF) where each diagonal \
        block corresponds to the snap cost matrix for the a coordinate between times t1 and t2.

    """
    H = np.zeros((N_COORDS * N_COEFF, N_COORDS * N_COEFF))
    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        H[start_idx:end_idx, start_idx:end_idx] = snap_cost_matrix(t1, t2)

    return H


def constraints_block_xyz(t: float, deriv: int = 0) -> np.ndarray:
    """
    Generate the constraints block for given coordinates.

    This function facilitates the creation of the constraints matrix. It generates N_COORDS rows, \
    each containing the time basis vector for the specified derivative order.

    Parameters
    ----------
    t : float
        The time at which to evaluate the polynomial coefficients.
    deriv : int, optional
        The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.

    Returns
    -------
    np.ndarray
        A block matrix of shape (N_COORDS, N_COORDS * N_COEFF) where each row corresponds to the \
        time basis vector for the specified derivative order at time t.

    """
    A = np.zeros((N_COORDS, N_COORDS * N_COEFF))
    # Get time vectors for x, y, z
    time_vec = t_vec(t, DEG, deriv)

    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        A[i, start_idx:end_idx] = time_vec

    return A


def create_equality_constraints(wps: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the constraints matrix and vector for the optimization problem.

    This function constructs the constraints matrix A and vector b based on the provided \
    waypoints. Internally, it uses the `constraints_block_xyz` function to generate the \
    constraints for each coordinate at each waypoint. The constraints include initial and final \
    positions, velocities, and continuity conditions for internal waypoints.

    The equality constraints are structured as follows:
    1. Initial position (N_COORDS)
    2. Initial velocity is zero (N_COORDS)
    3. Final position (N_COORDS)
    4. Final velocity is zero (N_COORDS)
    5. Position at each internal waypoint - the same for 2 polynomials ((n_wps - 2) * 2 * N_COORDS)
    6. Velocity continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    7. Acceleration continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    8. Jerk continuity at each internal waypoint ((n_wps - 2) * N_COORDS)
    Total equality constraints: (4 + (n_wps - 2) * 5) * N_COORDS

    Parameters
    ----------
    wps : List[Point]
        A list of waypoints, where each waypoint is a list containing the time and flat output \
        coordinates (x, y, z, yaw).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the constraints matrix A and vector b.

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
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = constraints_block_xyz(t, 0)
        b[m : m + N_COORDS] = wps[i][1 : 1 + N_COORDS]
        A[m + N_COORDS : m + 2 * N_COORDS, start_col_next_idx:end_col_next_idx] = (
            constraints_block_xyz(t, 0)
        )
        b[m + N_COORDS : m + 2 * N_COORDS] = wps[i][1 : 1 + N_COORDS]
        m += 2 * N_COORDS

        # Velocity continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = constraints_block_xyz(t, 1)
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = -constraints_block_xyz(t, 1)
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS

        # Acceleration continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = constraints_block_xyz(t, 2)
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = -constraints_block_xyz(t, 2)
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS

        # Jerk continuity
        A[m : m + N_COORDS, start_col_prev_idx:end_col_prev_idx] = constraints_block_xyz(t, 3)
        A[m : m + N_COORDS, start_col_next_idx:end_col_next_idx] = -constraints_block_xyz(t, 3)
        b[m : m + N_COORDS] = [0.0] * N_COORDS
        m += N_COORDS
    return A, b


def create_inequality_constraints(
    wps: List[Point], vlim: np.ndarray, alim: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the inequality constraints matrix and vector for the optimization problem.

    The inequality constraints are structured as follows:
    1. Above minimum velocity in each internal waypoint ((n_wps - 2) * N_COORDS)
    2. Below maximum velocity in each internal waypoint ((n_wps - 2) * N_COORDS)
    3. Above minimum acceleration in each internal waypoint ((n_wps - 2) * N_COORDS)
    4. Below maximum acceleration in each internal waypoint ((n_wps - 2) * N_COORDS)
    Total inequality constraints: (n_wps - 2) * 4 * N_COORDS

    Parameters
    ----------
    wps : List[Point]
        A list of waypoints, where each waypoint is a list containing the time and flat output \
        coordinates (x, y, z, yaw).
    vlim : np.ndarray
        A 2D array of shape (N_COORDS, 2) containing the velocity limits for each coordinate. \
        Each row corresponds to a coordinate and contains [min_velocity, max_velocity].
    alim : np.ndarray
        A 2D array of shape (N_COORDS, 2) containing the acceleration limits for each coordinate. \
        Each row corresponds to a coordinate and contains [min_acceleration, max_acceleration].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the inequality constraints matrix A and vector b.

    Raises
    ------
    ValueError
        If the shapes of vlim or alim are not as expected.

    """
    # Check limits dimensions
    if vlim.shape != (N_COORDS, 2):
        raise ValueError(f"vlim must have shape ({N_COORDS}, 2), got {vlim.shape}")
    if alim.shape != (N_COORDS, 2):
        raise ValueError(f"alim must have shape ({N_COORDS}, 2), got {alim.shape}")
    n_wps = len(wps)
    n_vars = (n_wps - 1) * N_COORDS * N_COEFF
    n_constraints = (n_wps - 2) * 4 * N_COORDS
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    m = 0  # Constraint row index

    for i in range(1, n_wps - 1):
        t = wps[i][0]
        start_col_idx = (i - 1) * N_COORDS * N_COEFF
        end_col_idx = start_col_idx + N_COORDS * N_COEFF

        # Velocity constraints
        A[m : m + N_COORDS, start_col_idx:end_col_idx] = -constraints_block_xyz(t, 1)
        b[m : m + N_COORDS] = -vlim[:, 0]
        m += N_COORDS

        A[m : m + N_COORDS, start_col_idx:end_col_idx] = constraints_block_xyz(t, 1)
        b[m : m + N_COORDS] = vlim[:, 1]
        m += N_COORDS

        # Acceleration constraints
        A[m : m + N_COORDS, start_col_idx:end_col_idx] = -constraints_block_xyz(t, 2)
        b[m : m + N_COORDS] = -alim[:, 0]
        m += N_COORDS

        A[m : m + N_COORDS, start_col_idx:end_col_idx] = constraints_block_xyz(t, 2)
        b[m : m + N_COORDS] = alim[:, 1]
        m += N_COORDS

    return A, b


def get_flat_output(
    c: cp.Variable,
    t: float,
    bound_times: List[float],
    deriv: int = 0,
    alpha: float = 1.0,
) -> Point:
    """
    Get the flat output of the trajectory at time t.

    This function evaluates the polynomial coefficients at a given time t and returns the flat \
    output coordinates for the trajectory. It uses the coefficient vector c (optimization result) \
    and the bound times to determine which polynomial segment to evaluate. It handles also \
    derivatives of the flat output.

    Parameters
    ----------
    c : cp.Variable
        The coefficient vector from the optimization problem, which contains the polynomial \
        coefficients for each segment.
    t : float
        The time at which to evaluate the polynomial coefficients.
    bound_times : List[float]
        A list of times that define the bounds of the polynomial segments. For k segments,
        there should be k+1 bound times.
    deriv : int, optional
        The derivative order to evaluate (0 for position, 1 for velocity, etc.). Default is 0.
    alpha : float, optional
        A scaling factor for the time vector. Default is 1.0.

    Returns
    -------
    Point
        A list of flat output coordinates evaluated at time t. The order is [x, y, z, yaw] for \
        position, and [vx, vy, vz, vyaw] for velocity, and so on for higher derivatives.

    Raises
    ------
    ValueError
        If the number of bound times is less than 2, if the coefficient vector length does not \
        match the expected size, or if the time t is out of bounds of the provided waypoints.

    """
    n_bounds = len(bound_times)
    if n_bounds < 2:
        raise ValueError("At least two bound times are required to evaluate the polynomial.")
    n_polynomials = n_bounds - 1
    if len(c.value) != n_polynomials * N_COORDS * N_COEFF:
        raise ValueError("Coefficient vector length does not match the expected size.")
    if t < bound_times[0] or t > bound_times[-1]:
        raise ValueError(f"Time t={t} is out of bounds of the provided waypoints.")

    # Determine which polynomial segment to use
    for i in range(n_bounds - 1):
        if bound_times[i] <= t <= bound_times[i + 1]:
            start_idx = i * N_COORDS * N_COEFF
            end_idx = start_idx + N_COORDS * N_COEFF
            c_segment = c.value[start_idx:end_idx]
            break
    else:
        raise ValueError(f"Time t={t} does not fall within the bounds of the provided waypoints.")

    # Calculate the polynomial value at time t
    t_vec_eval = np.array(t_vec(t / alpha, DEG, deriv)) / alpha**deriv
    flat_outputs = []
    for i in range(N_COORDS):
        start_idx = i * N_COEFF
        end_idx = start_idx + N_COEFF
        flat_outputs.append(np.dot(c_segment[start_idx:end_idx], t_vec_eval))

    return flat_outputs


def get_trajectory_minmax(
    c: cp.Variable, bound_times: List[float], alpha: float = 1.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Get the position, velocity, and acceleration minimum and maximum values.

    This function evaluates the trajectory defined by the polynomial coefficients c at a set of \
    time points between the first and last bound times. It computes the minimum and maximum \
    values of position, velocity, and acceleration across the trajectory.

    Parameters
    ----------
    c : cp.Variable
        The coefficient vector from the optimization problem, which contains the polynomial \
        coefficients for each segment.
    bound_times : List[float]
        A list of times that define the bounds of the polynomial segments. For k segments,
        there should be k+1 bound times.
    alpha : float, optional
        A scaling factor for the time vector. Default is 1.0.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        A tuple containing the minimum and maximum values of position (x, y, z), velocity \
        (vx, vy, vz), and acceleration (ax, ay, az) across the trajectory.

    """
    # Check flat outputs every 1 ms
    t = np.linspace(bound_times[0], bound_times[-1], int(alpha * 1000))
    positions = np.array([get_flat_output(c, ti, bound_times, 0, alpha) for ti in t])
    velocities = np.array([get_flat_output(c, ti, bound_times, 1, alpha) for ti in t])
    accelerations = np.array([get_flat_output(c, ti, bound_times, 2, alpha) for ti in t])
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    vel_min = np.min(velocities, axis=0)
    vel_max = np.max(velocities, axis=0)
    acc_min = np.min(accelerations, axis=0)
    acc_max = np.max(accelerations, axis=0)
    return pos_min, pos_max, vel_min, vel_max, acc_min, acc_max


def compute_trajectory(
    wps: List[float],
    vlim: np.ndarray = None,
    alim: np.ndarray = None,
    alpha: float = 1.0,
    verbose: bool = False,
) -> Tuple[cp.Variable, List[float], float]:
    """
    Compute the minimum snap trajectory given a list of waypoints.

    This function sets up and solves a convex optimization problem to find the polynomial \
    coefficients that minimize the snap cost while satisfying the constraints defined by the \
    waypoints. The waypoints are expected to be in the format [time, x, y, z, yaw]. The function \
    returns the optimized coefficients, the time bounds for the trajectory, and the optimal cost.
    The trajectory is represented as a piecewise polynomial, where each segment corresponds to a \
    polynomial between two consecutive waypoints.

    Parameters
    ----------
    wps : List[float]
        A list of waypoints, where each waypoint is a list containing the time and flat output \
        coordinates ([time, x, y, z, yaw]).
    vlim : np.ndarray, optional
        A 2D array of shape (N_COORDS, 2) representing the velocity limits for each coordinate. \
        If None, no velocity limits are applied. Default is None.
    alim : np.ndarray, optional
        A 2D array of shape (N_COORDS, 2) representing the acceleration limits for each \
        coordinate. If None, no acceleration limits are applied. Default is None.
    alpha : float, optional
        A scaling factor for the time vector to nondimensionalize the problem. Default is 1.0.
    verbose : bool, optional
        If True, prints additional information during the optimization process. Default is False.

    Returns
    -------
    Tuple[cp.Variable, List[float], float]
        A tuple containing:
        - cp.Variable: The optimized polynomial coefficients.
        - List[float]: The time bounds for the trajectory.
        - float: The optimal cost of the trajectory.

    """
    n_wps = len(wps)
    n_vars = (n_wps - 1) * N_COORDS * N_COEFF  # Number of variables

    wps_copy = copy.deepcopy(wps)
    n_solve_iters = 500_000

    # --- Decision variables ---
    c = cp.Variable(n_vars)

    # --- Objective function ---
    H = np.zeros((n_vars, n_vars))
    for i in range(n_wps - 1):
        t1 = wps_copy[i][0]
        t2 = wps_copy[i + 1][0]
        H_block = snap_cost_block_xyz(t1, t2)
        start_idx = i * N_COORDS * N_COEFF
        end_idx = start_idx + N_COORDS * N_COEFF
        H[start_idx:end_idx, start_idx:end_idx] = H_block

    # Scale the cost matrix to account for the time scaling
    H /= alpha ** (2 * K_R - 1)

    # Ensure H is positive semidefinite
    H = cp.psd_wrap(H)

    cost = cp.quad_form(c, H)
    objective = cp.Minimize(cost)

    # --- Define the constraints ---
    A1, b1 = create_equality_constraints(wps_copy)
    constraints = [A1 @ c == b1]
    # if vlim is not None and alim is not None:
    #     A2, b2 = create_inequality_constraints(wps_copy, vlim, alim)
    #     constraints.append(A2 @ c <= b2)

    # --- Problem definition ---
    problem = cp.Problem(objective, constraints)

    # --- Solve the problem ---
    try:
        problem.solve(max_iters=n_solve_iters, solver=cp.SCS, verbose=verbose)
    except cp.SolverError as e:
        if verbose:
            print(f"Solver error: {e}")
        return

    # --- Ensure trajectory limits ---
    alpha_inc_factor = 1.3
    while True:
        bound_times = [wp[0] * alpha for wp in wps_copy]

        if vlim is None:
            vlim = np.concat(
                (np.ones((N_COORDS, 1)) * (-np.inf), np.ones((N_COORDS, 1)) * np.inf),
                axis=1,
            )
        if alim is None:
            alim = np.concat(
                (np.ones((N_COORDS, 1)) * (-np.inf), np.ones((N_COORDS, 1)) * np.inf),
                axis=1,
            )
        pos_min, pos_max, vel_min, vel_max, acc_min, acc_max = get_trajectory_minmax(
            c, bound_times, alpha
        )
        if verbose:
            print(f"--- Trajectory limits (alpha {alpha}) ---")
            print("Position min:", pos_min)
            print("Position max:", pos_max)
            print("Velocity min:", vel_min)
            print("Velocity max:", vel_max)
            print("Acceleration min:", acc_min)
            print("Acceleration max:", acc_max)

        # Check if the trajectory limits are satisfied
        if (
            np.all(vel_min >= vlim[:, 0])
            and np.all(vel_max <= vlim[:, 1])
            and np.all(acc_min >= alim[:, 0])
            and np.all(acc_max <= alim[:, 1])
        ):
            break

        if verbose:
            print("Trajectory limits not satisfied. Adjusting waypoints and retrying...")

        # If limits are not satisfied, increase the time scale
        alpha *= alpha_inc_factor

    return c, alpha, problem.value


def main():
    """Perform a test run of the minimum snap trajectory generation."""
    # --- Define the waypoints ---
    # wps = [
    #     [0.0, 0.0, 0.0, 0.0, 0.0],  # Initial waypoint (t, x, y, z, yaw)
    #     [1.0, 1.0, 1.0, 2.0, -np.pi / 2],  # Second waypoint (t, x, y, z, yaw)
    #     [2.0, 2.0, 0.0, 3.0, -np.pi / 6],  # Third waypoint (t, x, y, z, yaw)
    #     [3.0, 1.0, -1.0, 1.0, np.pi / 3],  # Fourth waypoint (t, x, y, z, yaw)
    #     [4.0, -1.0, -1.0, 1.0, 0.0],  # Fifth waypoint (t, x, y, z, yaw)
    # ]

    wps = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, -5.0, 0.0],
        [3.0, 0.0, -1.0, -5.0, 0.0],
        [4.0, 0.0, -2.0, -5.0, 0.0],
        [5.0, 0.0, -3.0, -5.0, 0.0],
        [6.0, 0.0, -4.0, -5.0, 0.0],
        [7.0, 1.0, -4.0, -5.0, 0.0],
        [8.0, 2.0, -4.0, -5.0, 0.0],
        [9.0, 3.0, -4.0, -5.0, 0.0],
        [10.0, 4.0, -4.0, -5.0, 0.0],
        [11.0, 4.0, -3.0, -5.0, 0.0],
        [12.0, 4.0, -2.0, -5.0, 0.0],
        [13.0, 4.0, -1.0, -5.0, 0.0],
        [14.0, 4.0, 0.0, -5.0, 0.0],
        [15.0, 3.0, 0.0, -5.0, 0.0],
        [16.0, 2.0, 0.0, -5.0, 0.0],
        [17.0, 1.0, 0.0, -5.0, 0.0],
        [18.0, 0.0, 0.0, -5.0, 0.0],
    ]

    vlims = np.array(
        [
            [-1.0, 1.0],  # Velocity limits for x
            [-1.0, 1.0],  # Velocity limits for y
            [-2.0, 2.0],  # Velocity limits for z
            [-np.pi / 4, np.pi / 4],  # Velocity limits for yaw
        ]
    )
    alims = np.array(
        [
            [-0.5, 0.5],  # Acceleration limits for x
            [-0.5, 0.5],  # Acceleration limits for y
            [-1.0, 1.0],  # Acceleration limits for z
            [-np.pi / 8, np.pi / 8],  # Acceleration limits for yaw
        ]
    )

    # --- Create a list of wps with nondimensional time ---
    wps_nondim = copy.deepcopy(wps)
    alpha = wps_nondim[-1][0]  # Scale factor for time
    for i in range(len(wps_nondim)):
        wps_nondim[i][0] /= alpha

    # --- Compute the trajectory ---
    opt_coeffs, opt_end_time, opt_val = compute_trajectory(
        wps_nondim, vlims, alims, alpha, verbose=True
    )

    print("Optimal value:", opt_val)
    print("Opt end time:", opt_end_time)

    # --- Polynomial time bound ---
    time_bounds = [wp[0] * opt_end_time for wp in wps_nondim]
    print("Time bounds:", time_bounds)

    # # --- Plot the trajectory in 3D ---
    # t_values = np.linspace(time_bounds[0], time_bounds[-1], 1000)
    # trajectory = np.array(
    #     [get_flat_output(opt_coeffs, t, time_bounds, alpha=opt_end_time) for t in t_values]
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
    # ax2.plot(time_bounds, [wp[4] for wp in wps], "ro", label="Waypoints Yaw")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Yaw (rad)")
    # ax2.legend()
    # ax2.grid()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
