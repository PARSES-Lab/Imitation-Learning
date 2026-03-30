## original universal robots driver
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.0.113 kinematics_params_file:="/home/parses/my_robot_calibration.yaml" launch_rviz:=false

## driver with robotiq gripper
ros2 launch ur3_control ur3_with_gripper.launch.py robot_ip:=192.168.0.113 launch_rviz:=false use_tool_communication:=false

## then start external_control on tablet

## then start moveit
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_rviz:=true kinematics_params_file:="{$HOME}/my_robot_calibration.yaml"


## For simulated robot:
ros2 launch ur_robot_driver ur_control.launch.py     ur_type:=ur3e     robot_ip:=192.168.0.113     use_mock_hardware:=true     launch_rviz:=false

ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_rviz:=true


## To enable freedrive mode:
ros2 control set_controller_state scaled_joint_trajectory_controller inactive
ros2 control set_controller_state freedrive_mode_controller active
ros2 topic pub --rate 2 /freedrive_mode_controller/enable_freedrive_mode std_msgs/msg/Bool "{data: true}"


## For camera:
realsense-viewer
ros2 run realsense2_camera realsense2_camera_node 

## For recording arm+gripper:
ros2 run object_grasping record_joints_node --output my_demo.json
ros2 run object_grasping replay_joints_node --input my_demo.json

