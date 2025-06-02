"""Integration test for the MSTPlanner class."""

import unittest

import launch
import launch_ros
import launch_testing.actions
import launch_testing.asserts
import rclpy

from min_snap_traj_msgs.msg import Waypoint
from min_snap_traj_msgs.srv import SetTrajectory


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

    def test_service_available(self):
        """Test if the SetTrajectory service is available."""
        service_name = "/mst_planner/set_trajectory"
        service = self.test_node.create_client(SetTrajectory, service_name)
        self.assertTrue(service.wait_for_service(timeout_sec=5))

    def test_set_trajectory_service(self):
        """Test the SetTrajectory service functionality."""
        service_name = "/mst_planner/set_trajectory"
        service = self.test_node.create_client(SetTrajectory, service_name)

        # Create a request
        request = SetTrajectory.Request()
        request.waypoints = [
            Waypoint(timestamp=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0),
            Waypoint(timestamp=0.1, x=1.0, y=1.0, z=0.0, yaw=0.0),
            Waypoint(timestamp=0.2, x=2.0, y=2.0, z=0.0, yaw=0.0),
        ]

        # Call the service
        future = service.call_async(request)
        rclpy.spin_until_future_complete(self.test_node, future)

        # Check the response
        self.assertGreaterEqual(future.result().duration, 0.0)


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
