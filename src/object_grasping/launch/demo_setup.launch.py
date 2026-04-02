from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Paths to existing launch files
    ur3_control_launch = os.path.join(
        get_package_share_directory('ur3_control'),
        'launch',
        'ur3_with_gripper.launch.py'
    )

    ur_moveit_launch = os.path.join(
        get_package_share_directory('ur_moveit_config'),
        'launch',
        'ur_moveit.launch.py'
    )

    home = os.path.expanduser("~")
    calibration_file = os.path.join(home, "my_robot_calibration.yaml")

    return LaunchDescription([

        # UR3 control (driver + gripper)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ur3_control_launch),
            launch_arguments={
                'robot_ip': '192.168.0.113',
                'launch_rviz': 'false',
                'use_tool_communication': 'false',
            }.items()
        ),

        # MoveIt
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ur_moveit_launch),
            launch_arguments={
                'ur_type': 'ur3e',
                'launch_rviz': 'true',
                'kinematics_params_file': calibration_file,
            }.items()
        ),

        # RealSense camera node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            output='screen'
        ),
    ])