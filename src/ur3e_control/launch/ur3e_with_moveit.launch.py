from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, FindPackageShare

def generate_launch_description():
    # Path to the MoveIt2 launch file in your MoveIt config package
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ur3e_moveit_config"),  # your MoveIt config package
                "launch",
                "move_group.launch.py"
            ])
        ])
    )

    # Your C++ node that uses MoveGroupInterface
    controller_node = Node(
        package="ur3e_control",              # your package name
        executable="ur3e_controller",        # your node executable
        output="screen",
        parameters=[{"use_sim_time": False}]  # optional
    )

    return LaunchDescription([
        moveit_launch,
        controller_node
    ])