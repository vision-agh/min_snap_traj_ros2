"""Implementation of the Minimum Snap Trajectory Planner ROS2 node."""

import rclpy
import rclpy.logging
from rclpy.node import Node

from typing import List

import cvxpy as cp

import min_snap_traj.mst as mst
from min_snap_traj_msgs.msg import Waypoint
from min_snap_traj_msgs.srv import SetTrajectory


class MSTPlanner(Node):
    """Node for Minimum Snap Trajectory Planner."""

    def __init__(self):
        super().__init__("mst_planner")
        self.get_logger().info("Minimum Snap Trajectory Planner Node has been started.")

        self.opt_poly_coeffs: cp.Variable | None = None
        self.time_knots: List[float] | None = None

        # Services
        self.set_trajectory_srv = self.create_service(
            SetTrajectory,
            "mst_planner/set_trajectory",
            self._set_trajectory_callback,
        )

    def _set_trajectory_callback(self, request, response):
        self.get_logger().debug("Received SetTrajectory request.")
        waypoints: List[List[float]] = []
        wp: Waypoint
        for wp in request.waypoints:
            waypoints.append([wp.timestamp, wp.x, wp.y, wp.z, wp.yaw])
        self.get_logger().debug(f"Read {len(waypoints)} waypoint(s) from request.")
        return response

def main(args=None):
    """Main function to run the MST Planner node."""
    rclpy.init(args=args)
    rclpy.logging.set_logger_level("mst_planner", rclpy.logging.LoggingSeverity.DEBUG)

    planner = MSTPlanner()
    rclpy.spin(planner)

    planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
