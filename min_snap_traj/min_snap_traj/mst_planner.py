"""Implementation of the Minimum Snap Trajectory Planner ROS2 node."""

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

import rclpy
import rclpy.logging
from rclpy.node import Node

from typing import List

import cvxpy as cp

from min_snap_traj.mst import compute_trajectory, get_flat_output

from min_snap_traj_msgs.msg import Waypoint
from min_snap_traj_msgs.srv import GetFlatOutput, SetTrajectory


class MSTPlanner(Node):
    """Node for Minimum Snap Trajectory Planner."""

    def __init__(self):
        super().__init__("mst_planner")

        self.waypoints: List[List[float]] = []
        self.opt_poly_coeffs: cp.Variable | None = None
        self.time_knots: List[float] | None = None

        # Services
        self._get_flat_output_srv = self.create_service(
            GetFlatOutput,
            "mst_planner/get_flat_output",
            self._get_flat_output_callback,
        )

        self._set_trajectory_srv = self.create_service(
            SetTrajectory,
            "mst_planner/set_trajectory",
            self._set_trajectory_callback,
        )

        self.get_logger().info("Minimum Snap Trajectory Planner Node has been started.")

    def _get_flat_output_callback(self, request, response):
        self.get_logger().debug("Received GetFlatOutput request.")
        if self.opt_poly_coeffs is None or self.time_knots is None:
            self.get_logger().error(
                "Received GetFlatOutput request while trajectory is not set. "
                "Returning empty response."
            )
            response.flat_output.timestamp = -1.0
            return response
        if request.time < self.time_knots[0] or request.time > self.time_knots[-1]:
            self.get_logger().error(
                f"Requested timestamp {request.time} is out of bounds "
                f"[{self.time_knots[0]}, {self.time_knots[-1]}]. Returning empty response."
            )
            response.flat_output.timestamp = -1.0
            return response

        flat_output = get_flat_output(
            self.opt_poly_coeffs, request.time, self.time_knots, deriv=0
        )
        flat_output_d1 = get_flat_output(
            self.opt_poly_coeffs, request.time, self.time_knots, deriv=1
        )
        flat_output_d2 = get_flat_output(
            self.opt_poly_coeffs, request.time, self.time_knots, deriv=2
        )

        response.flat_output.timestamp = request.time
        response.flat_output.x = flat_output[0]
        response.flat_output.y = flat_output[1]
        response.flat_output.z = flat_output[2]
        response.flat_output.yaw = flat_output[3]
        response.flat_output.vx = flat_output_d1[0]
        response.flat_output.vy = flat_output_d1[1]
        response.flat_output.vz = flat_output_d1[2]
        response.flat_output.yaw_rate = flat_output_d1[3]
        response.flat_output.ax = flat_output_d2[0]
        response.flat_output.ay = flat_output_d2[1]
        response.flat_output.az = flat_output_d2[2]
        response.flat_output.yaw_accel = flat_output_d2[3]

        self.get_logger().debug(
            f"Flat output at time {request.time}: "
            f"x={flat_output[0]:.2f}, y={flat_output[1]:.2f}, z={flat_output[2]:.2f}, "
            f"yaw={flat_output[3]:.2f}, vx={flat_output_d1[0]:.2f}, "
            f"vy={flat_output_d1[1]:.2f}, vz={flat_output_d1[2]:.2f}, "
            f"yaw_rate={flat_output_d1[3]:.2f}, ax={flat_output_d2[0]:.2f}, "
            f"ay={flat_output_d2[1]:.2f}, az={flat_output_d2[2]:.2f}, "
            f"yaw_accel={flat_output_d2[3]:.2f}"
        )

        return response

    def _set_trajectory_callback(self, request, response):
        self.get_logger().debug("Received SetTrajectory request.")
        if len(self.waypoints) > 0:
            self.get_logger().error(
                "Received SetTrajectory request while waypoints are already set. "
                "Not clearing existing waypoints."
            )
            response.duration = -1.0
            return response
        wp: Waypoint
        for wp in request.waypoints:
            self.waypoints.append([wp.timestamp, wp.x, wp.y, wp.z, wp.yaw])

        self.opt_poly_coeffs, self.time_knots, opt_val = compute_trajectory(
            self.waypoints
        )
        self.get_logger().info(f"Trajectory set with optimal value: {opt_val:.4f}")

        response.duration = 0.0
        return response


def main(args=None):
    """Run the MST Planner node."""
    rclpy.init(args=args)
    rclpy.logging.set_logger_level("mst_planner", rclpy.logging.LoggingSeverity.DEBUG)

    planner = MSTPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info(
            "Keyboard interrupt received, shutting down MST Planner node."
        )
    finally:
        planner.destroy_node()


if __name__ == "__main__":
    main()
