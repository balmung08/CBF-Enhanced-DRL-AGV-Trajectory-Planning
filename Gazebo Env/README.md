# AGV 4WIS Gazebo Open Environment

Standalone ROS 2 Humble/Gazebo Classic package containing the AGV simulation
environment, low-level four-wheel independent steering control, and sensors.

This public package intentionally contains no MPC, trajectory tracker, CasADi
solver, optimized trajectory, or comparison result data.

## Included

- Detailed AGV visual model and collision model.
- Eight independently commanded joints: four steering and four drive joints.
- `/cmd_vel` double-Ackermann kinematic solver.
- Optional gamepad control.
- Four 16-channel 3D lidars.
- Merged `PointCloud2` and projected 2D `LaserScan`.
- Odometry, initial-pose `world` frame, TF, and vehicle footprint marker.
- Empty/base world plus Env1 and Env2 obstacle worlds.
- RViz2 configuration.

## Build

Place this package in a ROS 2 workspace `src` directory:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select agv_4wis_gazebo_open --symlink-install
source install/setup.bash
```

## Launch

```bash
# Base environment
ros2 launch agv_4wis_gazebo_open agv_4wis.launch.py

# Obstacle environments
ros2 launch agv_4wis_gazebo_open env1.launch.py
ros2 launch agv_4wis_gazebo_open env2.launch.py
```

Headless launch:

```bash
ros2 launch agv_4wis_gazebo_open env1.launch.py gui:=false rviz:=false
```

## Low-level control

The default controller subscribes to:

```text
/cmd_vel  geometry_msgs/msg/Twist
```

`linear.x` is body forward speed in m/s and `angular.z` is yaw rate in rad/s.
The controller evaluates the rigid-body velocity at every wheel center and
independently computes each wheel's steering angle and rolling speed.

Example:

```bash
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0}, angular: {z: 0.15}}"
```

The controller stops the vehicle if commands time out. Geometry, limits, drive
signs, and timeout are in `config/double_ackermann_controller.yaml`.

Raw joint command topics remain available under:

```text
/agv_4wis/frame_wheel10/command
/agv_4wis/frame_wheel20/command
/agv_4wis/frame_wheel30/command
/agv_4wis/frame_wheel40/command
/agv_4wis/frame_wheel1/command
/agv_4wis/frame_wheel2/command
/agv_4wis/frame_wheel3/command
/agv_4wis/frame_wheel4/command
```

Do not enable `cmd_vel_control` and `gamepad` simultaneously because both
publish to the same low-level joint topics.

## Sensors and state

```text
/agv_4wis/odom
/agv_4wis/odom_local
/joint_states
/agv_4wis/lidar/<corner>/points
/agv_4wis/lidar/points_merged
/agv_4wis/scan
/agv_4wis/vehicle_footprint
```

The dynamic TF is `world -> frame`, where `world` is initialized at the AGV
starting pose. Lidar static transforms are children of `frame`.

## License

Apache-2.0. Check the licenses of third-party mesh assets before redistributing
them outside the terms under which they were originally obtained.
