# min_snap_traj_ros2

ROS 2 packages for generating **Minimum Snap Trajectories (MST)** for quadrotors based on the paper *Minimum snap trajectory generation and control for quadrotors* by Daniel Mellinger and Vijay Kumar (DOI: [10.1109/ICRA.2011.5980409](https://doi.org/10.1109/ICRA.2011.5980409)).
It was built and tested on Ubuntu 22.04 with ROS 2 Humble.

## How to use it

Clone repository into your ROS 2 workspace and build as usual.
To validate the package, you can invoke set of tests with the `colcon test` command (see the official [ROS 2 documentation](https://docs.ros.org/en/humble/Tutorials/Intermediate/Testing/CLI.html) for more details).

### min_snap_traj

This is the main package, which includes [mst_planner](./min_snap_traj/min_snap_traj/mst_planner.py) node.
After initialisation, the node waits to receive trajectory waypoints via the `mst_planner/set_trajectory` service.
Once the correct request has been received, the node uses the [cvxpy](https://www.cvxpy.org/) Python library to compute the minimum snap trajectory.
Immediately after obtaining the trajectory, the node starts periodically publishing the trajectory setpoint on the `mst_planner/trajectory` topic at a predetrmined time interval.
To obtain setpoints for different time values, use the `mst_planner/get_flat_output` service.

The node has some configurable parameters, which are stored in the [mst.yaml](./min_snap_traj/config/mst.yaml) file:
- `dt` - time interval of setpoint publishing,
- `v_max` - maximum allowed velocity in each direction (x, y, z, yaw),
- `v_min` - minimum llowed velocity in each direction (x, y, z, yaw),
- `a_max` - maximum allowed acceleration in each direction (x, y, z, yaw),
- `a_min` - minimum allowed acceleration in each direction (x, y, z, yaw).

### min_snap_traj_msgs

Just definitions of messages and services that are used in the main package.

### min_snap_traj_integration_test

This package uses the approach [described](https://docs.ros.org/en/humble/Tutorials/Intermediate/Testing/Integration.html) in the official ROS 2 documentation to provide integration tests for the main package.
