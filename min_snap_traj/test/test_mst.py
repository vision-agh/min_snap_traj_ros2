"""Tests for the mst module."""

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

import min_snap_traj.mst as mst
import cvxpy as cp
import numpy as np


def test_t_vec_degs():
    """Test t_vec generation for different degrees."""
    for deg in range(0, 12):
        t_vec = mst.t_vec(1.0, deg)
        assert (
            len(t_vec) == deg + 1
        ), f"[deg={deg}] t_vec length for degree is incorrect ({len(t_vec)})."
        assert all(
            isinstance(t, float) for t in t_vec
        ), f"[deg={deg}] All elements in t_vec should be floats."
        assert all(
            t == 1.0 for t in t_vec
        ), f"[[deg={deg}] All elements in t_vec should be equal to 1.0."

    for deg in range(0, 12):
        t_vec = mst.t_vec(2.0, deg)
        assert (
            len(t_vec) == deg + 1
        ), f"[deg={deg}] t_vec length for degree is incorrect ({len(t_vec)})."
        assert all(
            isinstance(t, float) for t in t_vec
        ), f"[deg={deg}] All elements in t_vec should be floats."
        assert all(
            t == 2.0**i for i, t in enumerate(t_vec)
        ), f"[deg={deg}] Elements in t_vec should be powers of 2.0."


def test_compute_trajectory_basic():
    """Test trajectory computation with no limits."""
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

    # Create time nondimensionalized waypoints
    wps_nondim = copy.deepcopy(wps)
    alpha = wps_nondim[-1][0]  # Scale factor for time
    for i in range(len(wps_nondim)):
        wps_nondim[i][0] /= alpha

    # Compute the trajectory
    opt_c, opt_end_time, opt_val = mst.compute_trajectory(wps_nondim, alpha=alpha, verbose=False)

    # Check the optimized coefficients type
    assert isinstance(opt_c, cp.Variable), "Optimized coefficients should be a cvxpy Variable."

    # There should be no time scalling
    assert opt_end_time == alpha, "Optimal end time should match the scale factor."

    # Optimal value should be a float and non-negative
    assert isinstance(opt_val, float), "Optimal value should be a float."
    assert opt_val >= 0, "Optimal value should be non-negative."


def test_compute_trajectory_with_limits():
    """Test trajectory computation with velocity and acceleration limits."""
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

    vlim = np.array(
        [
            [-1.5, 1.5],  # vx lim
            [-1.5, 1.5],  # vy lim
            [-2.5, 2.5],  # vz lim
            [-np.pi / 4, np.pi / 4],  # yaw rate lim
        ]
    )
    alim = np.array(
        [
            [-1.0, 1.0],  # ax lim
            [-1.0, 1.0],  # ay lim
            [-1.5, 1.5],  # az lim
            [-np.pi / 8, np.pi / 8],  # yaw acceleration lim
        ]
    )

    # Create time nondimensionalized waypoints
    wps_nondim = copy.deepcopy(wps)
    alpha = wps_nondim[-1][0]  # Scale factor for time
    for i in range(len(wps_nondim)):
        wps_nondim[i][0] /= alpha

    # Compute the trajectory
    opt_c, opt_end_time, opt_val = mst.compute_trajectory(
        wps_nondim, vlim=vlim, alim=alim, alpha=alpha, verbose=True
    )

    # Check the optimized coefficients type
    assert isinstance(opt_c, cp.Variable), "Optimized coefficients should be a cvxpy Variable."

    # Optimal value should be a float and non-negative
    assert isinstance(opt_val, float), "Optimal value should be a float."
    assert opt_val >= 0, "Optimal value should be non-negative."

    # Check if the velocity and acceleration limits are satisfied in the internal waypoints
    bound_times = [wp[0] * opt_end_time for wp in wps_nondim]
    for i in range(1, len(bound_times) - 1):
        flat_output_t_d1 = mst.get_flat_output(opt_c, bound_times[i], bound_times, 1, opt_end_time)
        flat_output_t_d2 = mst.get_flat_output(opt_c, bound_times[i], bound_times, 2, opt_end_time)

        # Check velocity limits
        limit_tol = 1e-4  # Tolerance for floating-point comparison
        for j, vel in enumerate(flat_output_t_d1):
            assert (
                vlim[j, 0] <= vel + limit_tol and vel - limit_tol <= vlim[j, 1]
            ), f"Velocity limit violated at waypoint {i} for coordinate {j}."

        # Check acceleration limits
        for j, acc in enumerate(flat_output_t_d2):
            assert (
                alim[j, 0] <= acc + limit_tol and acc - limit_tol <= alim[j, 1]
            ), f"Acceleration limit violated at waypoint {i} for coordinate {j}."

    # Check if the velocity and acceleration limits are satisfied in the whole trajectory
    _, _, vel_min, vel_max, acc_min, acc_max = mst.get_trajectory_minmax(
        opt_c, bound_times, opt_end_time
    )
    assert np.all(
        vel_min + limit_tol >= vlim[:, 0]
    ), "Minimum velocity should satisfy lower limit."
    assert np.all(
        vel_max - limit_tol <= vlim[:, 1]
    ), "Maximum velocity should satisfy upper limit."
    assert np.all(
        acc_min + limit_tol >= alim[:, 0]
    ), "Minimum acceleration should satisfy lower limit."
    assert np.all(
        acc_max - limit_tol <= alim[:, 1]
    ), "Maximum acceleration should satisfy upper limit."


def test_get_flat_output():
    """Test the get_flat_output function."""
    wps = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, -5.0, 0.0],
        [3.0, 0.0, -1.0, -5.0, 0.0],
        [4.0, 0.0, -2.0, -5.0, 0.0],
        [5.0, 0.0, -3.0, -5.0, 0.0],
    ]

    # Create time nondimensionalized waypoints
    wps_nondim = copy.deepcopy(wps)
    alpha = wps_nondim[-1][0]  # Scale factor for time
    for i in range(len(wps_nondim)):
        wps_nondim[i][0] /= alpha

    # Compute the trajectory
    opt_c, opt_end_time, _ = mst.compute_trajectory(wps_nondim, alpha=alpha)

    # Scale time bounds according to the optimal end time
    bound_times = [wp[0] * opt_end_time for wp in wps_nondim]

    # Test for different derivatives
    for t in bound_times:
        for deriv in range(4):
            flat_output = mst.get_flat_output(opt_c, t, bound_times, deriv, opt_end_time)
            assert isinstance(
                flat_output, list
            ), f"Flat output should be a list for deriv={deriv}."
            assert (
                len(flat_output) == len(wps_nondim[0]) - 1
            ), f"Flat output length should match number of dimensions for deriv={deriv}."
            assert all(
                isinstance(val, float) for val in flat_output
            ), f"All elements in flat output should be floats for deriv={deriv}."
