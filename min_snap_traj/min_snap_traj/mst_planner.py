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

# import min_snap_traj.mst as mst
from min_snap_traj_msgs.msg import Waypoint
from min_snap_traj_msgs.srv import SetTrajectory


class MSTPlanner(Node):
    """Node for Minimum Snap Trajectory Planner."""

    def __init__(self):
        super().__init__("mst_planner")

        self.opt_poly_coeffs: cp.Variable | None = None
        self.time_knots: List[float] | None = None

        # Services
        self.set_trajectory_srv = self.create_service(
            SetTrajectory,
            "mst_planner/set_trajectory",
            self._set_trajectory_callback,
        )

        self.get_logger().info("Minimum Snap Trajectory Planner Node has been started.")

    def _set_trajectory_callback(self, request, response):
        self.get_logger().debug("Received SetTrajectory request.")
        waypoints: List[List[float]] = []
        wp: Waypoint
        for wp in request.waypoints:
            waypoints.append([wp.timestamp, wp.x, wp.y, wp.z, wp.yaw])
        self.get_logger().debug(f"Read {len(waypoints)} waypoint(s) from request.")
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
