## original universal robots driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.0.113 kinematics_params_file:="/home/parses/my_robot_calibration.yaml" launch_rviz:=false

## driver with robotiq gripper
ros2 launch ur3_control ur3_with_gripper.launch.py robot_ip:=192.168.0.113 launch_rviz:=false use_tool_communication:=false

## then start external_control on tablet

## then start moveit
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_rviz:=true kinematics_params_file:="{$HOME}/my_robot_calibration.yaml"
