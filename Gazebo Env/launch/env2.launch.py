import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory("agv_4wis_gazebo_open")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(share, "launch", "agv_4wis.launch.py")
                ),
                launch_arguments={
                    "world": os.path.join(
                        share, "worlds", "comparison_env2.world"
                    )
                }.items(),
            )
        ]
    )
