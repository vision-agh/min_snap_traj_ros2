"""Integration test for the MSTPlanner class."""

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
import unittest

import launch
import launch_ros
import launch_testing.actions
import launch_testing.asserts
import rclpy

import numpy as np

from min_snap_traj_msgs.msg import Waypoint
from min_snap_traj_msgs.srv import GetFlatOutput, SetTrajectory


class TestMSTPlanner(unittest.TestCase):
    """Test the MSTPlanner node functionality."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.test_node = rclpy.create_node("test_mst_planner")

    def tearDown(self):
        self.test_node.destroy_node()

    def test_logs_node_init(self, proc_output):
        """Test if the node logs its initialization."""
        proc_output.assertWaitFor(
            "Minimum Snap Trajectory Planner Node has been started.",
            timeout=5,
            stream="stderr",
        )

    def test_set_trajectory_available(self):
        """Test if the SetTrajectory service is available."""
        service = self.test_node.create_client(
            SetTrajectory, "mst_planner/set_trajectory"
        )
        self.assertTrue(
            service.wait_for_service(timeout_sec=5), "Service not available"
        )

    def test_get_flat_output_available(self):
        """Test if the get_flat_output_callback service is available."""
        service = self.test_node.create_client(
            GetFlatOutput, "mst_planner/get_flat_output"
        )
        self.assertTrue(
            service.wait_for_service(timeout_sec=5), "Service not available"
        )

    def test_trajectory_pipeline(self):
        """Test the Trajectory pipeline."""
        set_trajectory_service = self.test_node.create_client(
            SetTrajectory, "mst_planner/set_trajectory"
        )
        self.assertTrue(
            set_trajectory_service.wait_for_service(timeout_sec=5),
            "Service not available",
        )

        # Create a request
        request = SetTrajectory.Request()
        request.waypoints = [
            Waypoint(timestamp=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0),
            Waypoint(timestamp=1.0, x=1.0, y=1.0, z=2.0, yaw=-np.pi / 2),
            Waypoint(timestamp=2.0, x=2.0, y=0.0, z=3.0, yaw=-np.pi / 6),
            Waypoint(timestamp=3.0, x=1.0, y=-1.0, z=1.0, yaw=np.pi / 3),
            Waypoint(timestamp=4.0, x=-1.0, y=-1.0, z=1.0, yaw=0.0),
        ]

        # Call the service
        future = set_trajectory_service.call_async(request)
        rclpy.spin_until_future_complete(self.test_node, future)

        # Check the response
        self.assertGreaterEqual(future.result().duration, 0.0)

        # Another call to the service should not clear existing waypoints
        request.waypoints = [
            Waypoint(timestamp=0.3, x=3.0, y=3.0, z=0.0, yaw=0.0),
        ]
        future = set_trajectory_service.call_async(request)
        rclpy.spin_until_future_complete(self.test_node, future)

        # Check the response
        self.assertEqual(future.result().duration, -1.0)

        # Call the get_flat_output_callback service
        get_flat_output_service = self.test_node.create_client(
            GetFlatOutput, "mst_planner/get_flat_output"
        )
        self.assertTrue(
            get_flat_output_service.wait_for_service(timeout_sec=5),
            "Service not available",
        )

        # Check bad request handling
        get_flat_output_request = GetFlatOutput.Request()
        get_flat_output_request.time = -1.0
        future = get_flat_output_service.call_async(get_flat_output_request)
        rclpy.spin_until_future_complete(self.test_node, future)

        self.assertEqual(
            future.result().flat_output.timestamp,
            -1.0,
            "Service should not succeed with negative time",
        )

        # Check valid request handling
        get_flat_output_request.time = 1.0
        future = get_flat_output_service.call_async(get_flat_output_request)
        rclpy.spin_until_future_complete(self.test_node, future)
        self.assertGreaterEqual(
            future.result().flat_output.timestamp,
            1.0,
            "Service should succeed with valid time",
        )
        self.assertIsNotNone(
            future.result().flat_output.x,
            "Flat trajectory x should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.y,
            "Flat trajectory y should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.z,
            "Flat trajectory z should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.yaw,
            "Flat trajectory yaw should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.vx,
            "Flat trajectory vx should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.vy,
            "Flat trajectory vy should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.vz,
            "Flat trajectory vz should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.yaw_rate,
            "Flat trajectory yaw_rate should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.ax,
            "Flat trajectory ax should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.ay,
            "Flat trajectory ay should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.az,
            "Flat trajectory az should not be None",
        )
        self.assertIsNotNone(
            future.result().flat_output.yaw_accel,
            "Flat trajectory yaw_accel should not be None",
        )


@launch_testing.post_shutdown_test()
class TestMSTPlannerShutdown(unittest.TestCase):
    """Test that the MSTPlanner node shuts down cleanly."""

    def test_exit_codes(self, proc_info):
        """Test if the node exits with code 0."""
        launch_testing.asserts.assertExitCodes(proc_info)


def generate_test_description():
    """Generate the test description for the MSTPlanner integration test."""
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="min_snap_traj",
                executable="mst_planner",
                name="mst_planner",
                output="screen",
            ),
            launch.actions.TimerAction(
                period=0.5, actions=[launch_testing.actions.ReadyToTest()]
            ),
        ],
    )
