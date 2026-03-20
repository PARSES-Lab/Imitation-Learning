#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "geometry_msgs/msg/pose.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Node must be created with parameters enabled for MoveIt
  auto node = rclcpp::Node::make_shared(
    "ur3e_moveit_pose_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  // Spin node in background (MoveIt needs this)
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  // Create MoveGroup interface
  static const std::string PLANNING_GROUP = "ur_manipulator";
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP);

  // Optional but useful
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.2);
  move_group.setMaxAccelerationScalingFactor(0.2);

  // Define target pose
  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = 0.3;
  target_pose.position.y = 0.1;
  target_pose.position.z = 0.2;

  target_pose.orientation.w = 1.0;  // neutral orientation

  move_group.setPoseTarget(target_pose);

  // Plan
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (success)
  {
    RCLCPP_INFO(node->get_logger(), "Planning successful, executing...");
    move_group.execute(plan);
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Planning failed!");
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}


// find_package(moveit_ros_planning_interface REQUIRED)

// add_executable(ur3e_moveit_pose src/ur3e_moveit_pose.cpp)

// ament_target_dependencies(
//   ur3e_moveit_pose
//   rclcpp
//   moveit_ros_planning_interface
// )

// install(TARGETS ur3e_moveit_pose
//   DESTINATION lib/${PROJECT_NAME})



// <depend>moveit_ros_planning_interface</depend>