"""Implementation of the Minimum Snap Trajectory Planner ROS2 node."""

import rclpy
import rclpy.logging
from rclpy.node import Node

from typing import List

import min_snap_traj.mst as mst


class MSTPlanner(Node):
    """Node for Minimum Snap Trajectory Planner."""

    def __init__(self):
        super().__init__("mst_planner")
        self.get_logger().info("Minimum Snap Trajectory Planner Node has been started.")

        self.waypoints: List[mst.Point] | None = None
        self.time_knots: List[float] | None = None


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
