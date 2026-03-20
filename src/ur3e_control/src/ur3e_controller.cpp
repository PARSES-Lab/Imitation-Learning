#include <moveit/move_group_interface/move_group_interface.h>
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  // Create node with moveit parameters enabled
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto move_group_node = rclcpp::Node::make_shared("ur3e_control_node", node_options);

  // Sping the node in background
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_node);
  std::thread spinner([&executor]() { executor.spin(); });

  static const std::string PLANNING_GROUP = "ur_manipulator";
  moveit::planning_interface::MoveGroupInterface move_group(move_group_node, PLANNING_GROUP);

  // Defining target pose as relative to current pose (need to add path constraints)
  // geometry_msgs::msg::Pose current_pose = move_group.getCurrentPose().pose;
  // geometry_msgs::msg::Pose target_pose = current_pose;
  // target_pose.position.x += 0.03;

  // Defining absolute target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = 0.3;
  target_pose.position.y = 0.1;
  target_pose.position.z = 0.2;
  target_pose.orientation.w = 1.0;  // neutral orientation

  move_group.setStartStateToCurrentState();
  move_group.setPoseTarget(target_pose);

  move_group.setMaxVelocityScalingFactor(0.05);
  move_group.setMaxAccelerationScalingFactor(0.05);


  // Plan
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;

  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  RCLCPP_INFO(move_group_node->get_logger(), "Visualizing plan 1 (pose goal) %s", success ? "" : "FAILED");

  move_group.execute(my_plan);
  rclcpp::shutdown();
  spinner.join();
  return 0;
}