from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mistnet',
            executable='mistnode',
            name='mistnode',
            output='screen',
            parameters=[
                {"engine_file": "/home/rayz/code/engine.trt"},
                {"lidar_model": "m2w"},
                {"input_channel": "udp://0.0.0.0:2368"},
                {"output_channel": "ws://0.0.0.0:12369"},
                {"output_type": "mark"},
                {"debug": True},
                {"protocol": 6}

            ]
        )
    ])
