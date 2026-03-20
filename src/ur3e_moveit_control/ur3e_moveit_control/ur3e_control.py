#!/usr/bin/env python3

import rclpy
from rclpy.logging import get_logger

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from ament_index_python.packages import get_package_share_directory
import os

def plan_and_execute(
        robot,
        planning_component,
        logger,
        single_plan_parameters=None
):
    """Helper function to plan and execute a motion."""

    if single_plan_parameters is not None:
        plan_result = planning_component.plan(single_plan_parameters=single_plan_parameters)
    else:
        plan_result = planning_component.plan()

    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")


def main():
    rclpy.init()
    logger = get_logger("moveit_py.pose_goal")

    ur_moveit_dir = get_package_share_directory("ur_moveit_config")

    ur = MoveItPy(node_name="ur3e_moveit_py", 
                  launch_params_filepaths=[
                      os.path.join(ur_moveit_dir, "config", "moveit_cpp.yaml")],
                      provide_planning_service=False)
    ur_planner = ur.get_planning_component("ur_planner")
    logger.info("MoveItPy instance created")

    # get the current robot state
    ur_planner.set_start_state_to_current_state()

    # get the robot model and set to a random goal configuration
    robot_model = ur.get_robot_model()
    robot_state = RobotState(robot_model)
    robot_state.set_to_random_positions()
    ur_planner.set_goal_state(robot_state=robot_state)

    # plan so we can visualize in rviz
    plan_result = ur_planner.plan()
    logger.info("Planning succeeded")
    

if __name__ == "__main__":
    main()