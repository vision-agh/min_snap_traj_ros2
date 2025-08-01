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

import numpy as np

from min_snap_traj.mst import compute_trajectory, get_flat_output

from min_snap_traj_msgs.msg import FlatOutput, Waypoint
from min_snap_traj_msgs.srv import GetFlatOutput, SetTrajectory


class MSTPlanner(Node):
    """
    Node for Minimum Snap Trajectory Planner.

    This node serves as a ROS2 interface for computing minimum snap trajectories
    based on a set of waypoints. The basic pipeline is as follows:
    1. Set a trajectory using the `SetTrajectory` service.
    2. Read the flat output published periodically on the `mst_planner/trajectory` topic.
    Additionally, the node provides a service to get the flat output at a specific time
    using the `GetFlatOutput` service.
    The flat output includes position, velocity, acceleration, and yaw information
    at the requested time, computed from the polynomial coefficients of the trajectory, with
    its derivatives up to the second order. It is possible to adjust the parameters via the
    configuration file.

    """

    def __init__(self):
        super().__init__("mst_planner")
        self._load_params()

        self.opt_poly_coeffs: cp.Variable | None = None
        self.time_knots: List[float] | None = None
        self.current_time: float = 0.0
        self.traj_duration: float | None = None

        # Topics
        self._trajectory_publisher = self.create_publisher(
            FlatOutput,
            "mst_planner/trajectory",
            10,
        )

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

        # Timer to publish trajectory periodically
        self._publish_timer = self.create_timer(
            self.dt,
            self.publish_trajectory,
        )

        self.get_logger().info("Minimum Snap Trajectory Planner Node has been started.")

    def _get_flat_output_callback(self, request, response):
        """Return the flat output at the requested time."""
        self.get_logger().debug("Received GetFlatOutput request.")

        # Read the flat output at the requested time
        response.flat_output = self._read_flat_output(request.time)

        if response.flat_output.timestamp >= 0:
            self.get_logger().debug(
                f"Flat output at time {response.flat_output.timestamp}: "
                f"x={response.flat_output.x:.2f}, y={response.flat_output.y:.2f}, "
                f"z={response.flat_output.z:.2f}, yaw={response.flat_output.yaw:.2f}, "
                f"vx={response.flat_output.vx:.2f}, vy={response.flat_output.vy:.2f}, "
                f"vz={response.flat_output.vz:.2f}, yaw_rate={response.flat_output.yaw_rate:.2f}, "
                f"ax={response.flat_output.ax:.2f}, ay={response.flat_output.ay:.2f}, "
                f"az={response.flat_output.az:.2f}, yaw_acc={response.flat_output.yaw_accel:.2f}, "
                f" jx={response.flat_output.jx:.2f}, jy={response.flat_output.jy:.2f}, "
                f"jz={response.flat_output.jz:.2f}, yaw_jerk={response.flat_output.yaw_jerk:.2f}"
            )

        return response

    def _load_params(self):
        """Load parameters from the ROS2 parameter server."""
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("v_max.x", 1.0)
        self.declare_parameter("v_max.y", 1.0)
        self.declare_parameter("v_max.z", 1.0)
        self.declare_parameter("v_max.yaw", 1.0)
        self.declare_parameter("v_min.x", -1.0)
        self.declare_parameter("v_min.y", -1.0)
        self.declare_parameter("v_min.z", -1.0)
        self.declare_parameter("v_min.yaw", -1.0)
        self.declare_parameter("a_max.x", 1.0)
        self.declare_parameter("a_max.y", 1.0)
        self.declare_parameter("a_max.z", 1.0)
        self.declare_parameter("a_max.yaw", 1.0)
        self.declare_parameter("a_min.x", -1.0)
        self.declare_parameter("a_min.y", -1.0)
        self.declare_parameter("a_min.z", -1.0)
        self.declare_parameter("a_min.yaw", -1.0)

        self.get_logger().info("Loading parameters...")

        self.dt = self.get_parameter("dt").get_parameter_value().double_value
        self.get_logger().debug(f"Time step: {self.dt:.2f} seconds")

        self.vlims = np.array(
            [
                [
                    self.get_parameter("v_min.x").get_parameter_value().double_value,
                    self.get_parameter("v_max.x").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("v_min.y").get_parameter_value().double_value,
                    self.get_parameter("v_max.y").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("v_min.z").get_parameter_value().double_value,
                    self.get_parameter("v_max.z").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("v_min.yaw").get_parameter_value().double_value,
                    self.get_parameter("v_max.yaw").get_parameter_value().double_value,
                ],
            ]
        )
        self.get_logger().debug(f"Velocity limits:\n {self.vlims}")

        self.alims = np.array(
            [
                [
                    self.get_parameter("a_min.x").get_parameter_value().double_value,
                    self.get_parameter("a_max.x").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("a_min.y").get_parameter_value().double_value,
                    self.get_parameter("a_max.y").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("a_min.z").get_parameter_value().double_value,
                    self.get_parameter("a_max.z").get_parameter_value().double_value,
                ],
                [
                    self.get_parameter("a_min.yaw").get_parameter_value().double_value,
                    self.get_parameter("a_max.yaw").get_parameter_value().double_value,
                ],
            ]
        )
        self.get_logger().debug(f"Acceleration limits:\n {self.alims}")
        self.get_logger().info("Parameters loaded successfully.")

    def _read_flat_output(self, time: float) -> FlatOutput:
        """Read the flat output at the given time."""
        if self.opt_poly_coeffs is None or self.time_knots is None:
            self.get_logger().warning("No trajectory set. Cannot read flat output.")
            return FlatOutput(timestamp=-1.0)
        if time < self.time_knots[0] or time > self.time_knots[-1]:
            self.get_logger().warning(
                f"Requested time {time} is out of bounds "
                f"[{self.time_knots[0]}, {self.time_knots[-1]}]. Returning empty FlatOutput."
            )
            return FlatOutput(timestamp=-1.0)

        # Compute the flat output at the given time
        flat_output = get_flat_output(
            self.opt_poly_coeffs, time, self.time_knots, deriv=0, alpha=self.duration
        )
        flat_output_d1 = get_flat_output(
            self.opt_poly_coeffs, time, self.time_knots, deriv=1, alpha=self.duration
        )
        flat_output_d2 = get_flat_output(
            self.opt_poly_coeffs, time, self.time_knots, deriv=2, alpha=self.duration
        )
        flat_output_d3 = get_flat_output(
            self.opt_poly_coeffs, time, self.time_knots, deriv=3, alpha=self.duration
        )

        # Create FlatOutput message
        flat_output_msg = FlatOutput()
        flat_output_msg.timestamp = time
        flat_output_msg.x = flat_output[0]
        flat_output_msg.y = flat_output[1]
        flat_output_msg.z = flat_output[2]
        flat_output_msg.yaw = flat_output[3]
        flat_output_msg.vx = flat_output_d1[0]
        flat_output_msg.vy = flat_output_d1[1]
        flat_output_msg.vz = flat_output_d1[2]
        flat_output_msg.yaw_rate = flat_output_d1[3]
        flat_output_msg.ax = flat_output_d2[0]
        flat_output_msg.ay = flat_output_d2[1]
        flat_output_msg.az = flat_output_d2[2]
        flat_output_msg.yaw_accel = flat_output_d2[3]
        flat_output_msg.jx = flat_output_d3[0]
        flat_output_msg.jy = flat_output_d3[1]
        flat_output_msg.jz = flat_output_d3[2]
        flat_output_msg.yaw_jerk = flat_output_d3[3]

        return flat_output_msg

    def _set_trajectory_callback(self, request, response):
        """Set the trajectory based on the provided waypoints."""
        self.get_logger().debug("Received SetTrajectory request.")
        if self.time_knots is not None and self.current_time < self.time_knots[-1]:
            self.get_logger().error(
                "Received SetTrajectory request while trajectory is executed. "
                "Not clearing existing trajectory."
            )
            response.duration = -1.0
            return response

        waypoints: List[List[float]] = []
        wp: Waypoint
        for wp in request.waypoints:
            waypoints.append([wp.timestamp, wp.x, wp.y, wp.z, wp.yaw])

        try:
            nondim_waypoints = []
            for wp in waypoints:
                nondim_waypoints.append([wp[0] / waypoints[-1][0], wp[1], wp[2], wp[3], wp[4]])
            self.opt_poly_coeffs, self.duration, opt_val = compute_trajectory(
                nondim_waypoints,
                self.vlims,
                self.alims,
                waypoints[-1][0],
                verbose=True,
            )
        except cp.SolverError as e:
            self.get_logger().error(
                f"SolverError while computing trajectory: {e}. " "Returning empty trajectory."
            )
            response.duration = -1.0
            return response
        self.time_knots = [wp[0] * self.duration for wp in nondim_waypoints]
        self.get_logger().info(f"Trajectory set with optimal value: {opt_val:.4f}")
        self.get_logger().debug(f"Time knots: {self.time_knots}")
        response.duration = self.duration

        self.current_time = 0.0

        return response

    def publish_trajectory(self):
        """
        Publish the trajectory point as a FlatOutput message.

        This method should be called periodically to publish the current trajectory state.
        It checks if the trajectory has been set and publishes the FlatOutput message with the
        current time and flat output coordinates. If the trajectory is not set, it logs a warning.

        """
        # If called before trajectory is set, do nothing
        if self.opt_poly_coeffs is None or self.time_knots is None:
            self.get_logger().warning("No trajectory set. Cannot publish FlatOutput.")
            return

        # Read the flat output at the current time
        flat_output_msg = self._read_flat_output(self.current_time)
        if flat_output_msg.timestamp < 0:
            self.get_logger().warning(
                f"Flat output at time {self.current_time} is invalid. " "Skipping publishing."
            )
            return

        # Publish the FlatOutput message
        self._trajectory_publisher.publish(flat_output_msg)

        # Increment the current time only if it will not exceed the last time knot

        if self.current_time + self.dt <= self.time_knots[-1]:
            self.current_time += self.dt


def main(args=None):
    """Run the MST Planner node."""
    rclpy.init(args=args)
    rclpy.logging.set_logger_level("mst_planner", rclpy.logging.LoggingSeverity.DEBUG)

    planner = MSTPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info("Keyboard interrupt received, shutting down MST Planner node.")
    finally:
        planner.destroy_node()


if __name__ == "__main__":
    main()
