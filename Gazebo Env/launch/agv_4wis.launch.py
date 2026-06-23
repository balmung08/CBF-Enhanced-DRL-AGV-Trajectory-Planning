import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PACKAGE_NAME = "agv_4wis_gazebo_open"


def append_path(path, current):
    return path if not current else f"{path}:{current}"


def generate_launch_description():
    package_share = get_package_share_directory(PACKAGE_NAME)
    package_prefix = get_package_prefix(PACKAGE_NAME)
    gazebo_share = get_package_share_directory("gazebo_ros")
    default_world = os.path.join(package_share, "worlds", "agv_4wis.world")

    actions = [
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("verbose", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("cmd_vel_control", default_value="true"),
        DeclareLaunchArgument("gamepad", default_value="false"),
        DeclareLaunchArgument("filter_center_x", default_value="0.04"),
        DeclareLaunchArgument("filter_center_y", default_value="-0.01"),
        DeclareLaunchArgument("filter_size_x", default_value="6.8"),
        DeclareLaunchArgument("filter_size_y", default_value="3.0"),
        DeclareLaunchArgument("filter_min_z", default_value="0.05"),
        DeclareLaunchArgument("filter_max_z", default_value="3.0"),
        DeclareLaunchArgument("scan_angle_increment", default_value="0.0174532925199433"),
        DeclareLaunchArgument("scan_z", default_value="1.13"),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            append_path(
                os.path.join(package_share, "models"),
                os.environ.get("GAZEBO_MODEL_PATH", ""),
            ),
        ),
        SetEnvironmentVariable(
            "GAZEBO_PLUGIN_PATH",
            append_path(
                os.path.join(package_prefix, "lib"),
                os.environ.get("GAZEBO_PLUGIN_PATH", ""),
            ),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")
            ),
            launch_arguments={
                "world": LaunchConfiguration("world"),
                "gui": LaunchConfiguration("gui"),
                "pause": LaunchConfiguration("paused"),
                "verbose": LaunchConfiguration("verbose"),
            }.items(),
        ),
        Node(
            package=PACKAGE_NAME,
            executable="double_ackermann_controller",
            name="double_ackermann_controller",
            output="screen",
            condition=IfCondition(LaunchConfiguration("cmd_vel_control")),
            parameters=[
                os.path.join(
                    package_share, "config", "double_ackermann_controller.yaml"
                ),
                {"use_sim_time": True},
            ],
        ),
        Node(
            package="joy",
            executable="game_controller_node",
            name="game_controller",
            output="screen",
            condition=IfCondition(LaunchConfiguration("gamepad")),
        ),
        Node(
            package=PACKAGE_NAME,
            executable="gamepad_teleop",
            name="gamepad_teleop",
            output="screen",
            condition=IfCondition(LaunchConfiguration("gamepad")),
            parameters=[os.path.join(package_share, "config", "gamepad.yaml")],
        ),
        Node(
            package=PACKAGE_NAME,
            executable="pointcloud_merger",
            name="pointcloud_merger",
            output="screen",
            parameters=[
                os.path.join(package_share, "config", "pointcloud_merger.yaml"),
                {
                    "use_sim_time": True,
                    "filter_center_x": ParameterValue(
                        LaunchConfiguration("filter_center_x"), value_type=float
                    ),
                    "filter_center_y": ParameterValue(
                        LaunchConfiguration("filter_center_y"), value_type=float
                    ),
                    "filter_size_x": ParameterValue(
                        LaunchConfiguration("filter_size_x"), value_type=float
                    ),
                    "filter_size_y": ParameterValue(
                        LaunchConfiguration("filter_size_y"), value_type=float
                    ),
                    "filter_min_z": ParameterValue(
                        LaunchConfiguration("filter_min_z"), value_type=float
                    ),
                    "filter_max_z": ParameterValue(
                        LaunchConfiguration("filter_max_z"), value_type=float
                    ),
                },
            ],
        ),
        Node(
            package=PACKAGE_NAME,
            executable="pointcloud_to_scan",
            name="pointcloud_to_scan",
            output="screen",
            parameters=[
                os.path.join(package_share, "config", "pointcloud_to_scan.yaml"),
                {
                    "use_sim_time": True,
                    "angle_increment": ParameterValue(
                        LaunchConfiguration("scan_angle_increment"), value_type=float
                    ),
                    "scan_z": ParameterValue(
                        LaunchConfiguration("scan_z"), value_type=float
                    ),
                },
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", os.path.join(package_share, "rviz", "agv_4wis.rviz")],
            parameters=[{"use_sim_time": True}],
        ),
    ]

    for child, x, y in (
        ("lidar_rear_right", -2.65652943, -1.17497706),
        ("lidar_rear_left", -2.65652823, 1.15019238),
        ("lidar_front_right", 2.74429417, -1.17498243),
        ("lidar_front_left", 2.74429154, 1.15019083),
    ):
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{child}_tf",
                arguments=[
                    "--x", str(x),
                    "--y", str(y),
                    "--z", "1.13",
                    "--roll", "0",
                    "--pitch", "0",
                    "--yaw", "0",
                    "--frame-id", "frame",
                    "--child-frame-id", f"{child}_frame",
                ],
            )
        )

    return LaunchDescription(actions)
