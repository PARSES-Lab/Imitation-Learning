#!/usr/bin/env python3
"""
UR3 Robot Controller — ROS2 Humble + MoveIt2 + Robotiq 85
Single-threaded, no separate executor conflict with pymoveit2.

Each task step is a dict with:
  label    – log message (required)
  act      – "arm", "gripper", or "both"  (required)
  x/y/z    – TCP position OFFSET in metres (required if act in {"arm","both"})
  qx/qy/qz/qw – orientation quaternion override, optional (defaults to current)
  gripper  – Gripper.OPEN / Gripper.CLOSE (required if act in {"gripper","both"})

"both" moves the arm first, then the gripper.

Planning strategy:
  1. Pilz LIN  — straight-line Cartesian, deterministic, fast
  2. OMPL RRTConnect fallback — collision-aware joint-space planning
     (used automatically if Pilz fails)

Launch stack:
  ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 \
    robot_ip:=192.168.0.1 use_fake_hardware:=true launch_rviz:=false \
    initial_joint_controller:=joint_trajectory_controller
  ros2 launch ur_moveit_config ur_moveit.launch.py \
    ur_type:=ur3 launch_rviz:=true use_fake_hardware:=true

Gripper: Trigger services /gripper/open and /gripper/close
"""

import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from pymoveit2 import MoveIt2
from std_srvs.srv import Trigger
from pymoveit2.robots import ur as robot

HOME_JOINTS = [
    0.3219981789588928,   # shoulder_pan_joint
   -1.5710002384581507,   # shoulder_lift_joint
   -0.9818522334098816,   # elbow_joint
   -2.1245953045287074,   # wrist_1_joint
    1.5656001567840576,   # wrist_2_joint
    0.5031509399414062,   # wrist_3_joint
]

class Gripper(Enum):
    OPEN  = auto()
    CLOSE = auto()

PICK_AND_PLACE = [
    {"label": "Approach object",
     "act": "arm",
     "x": -0.0, "y": 0.4, "z": 0.40,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},

    {"label": "Lower to pick",
     "act": "arm",
     "x": -0.3, "y": 0.0, "z": 0.22,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},

    {"label": "Lift up",
     "act": "arm",
     "x": -0.3, "y": 0.0, "z": 0.40,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},

    {"label": "Move to goal",
     "act": "arm",
     "x": -0.3, "y": 0.2, "z": 0.40,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},

    {"label": "Lower to place",
     "act": "arm",
     "x": -0.3, "y": 0.2, "z": 0.22,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},

    {"label": "Retract arm",
     "act": "arm",
     "x": -0.3, "y": 0.2, "z": 0.40,
     "qx": 0.768, "qy": 0.641, "qz": -0.015, "qw": -0.009},
]

# Goal positions defined as offsets relative to current EEF position.
# Omit qx/qy/qz/qw to keep the current orientation.
PICK = [
    {"label": "Move forward 10 cm in Z",
     "act": "arm",
     "x": 0.0, "y": 0.0, "z": 0.1},
]


class UR3Robot(Node):

    PLANNING_TIME       = 5.0
    OMPL_PLANNING_TIME  = 10.0
    MAX_RETRIES         = 3
    TOLERANCE           = 0.01
    GRIPPER_TIMEOUT_SEC = 5.0

    PILZ_MAX_VEL   = 0.3
    PILZ_MAX_ACC   = 0.3

    OMPL_MAX_VEL   = 0.3
    OMPL_MAX_ACC   = 0.3

    def __init__(self):
        super().__init__("ur3_robot")

        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
        )
        self._moveit2.cartesian_avoid_collisions = True


        self._gripper_open  = self.create_client(Trigger, "/gripper/open")
        self._gripper_close = self.create_client(Trigger, "/gripper/close")

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.get_logger().info("UR3Robot ready.")

    def _use_pilz(self):
        self._moveit2.planner_id        = "LIN"
        self._moveit2.planning_pipeline = "pilz_industrial_motion_planner"
        self._moveit2.planning_time     = self.PLANNING_TIME
        self._moveit2.max_velocity      = self.PILZ_MAX_VEL
        self._moveit2.max_acceleration  = self.PILZ_MAX_ACC

    def _use_ompl(self):
        self._moveit2.planner_id        = "ompl/RRTConnectkConfigDefault"
        self._moveit2.planning_pipeline = "ompl"
        self._moveit2.planning_time     = self.OMPL_PLANNING_TIME
        self._moveit2.max_velocity      = self.OMPL_MAX_VEL
        self._moveit2.max_acceleration  = self.OMPL_MAX_ACC

    def move_to_home(self) -> bool:
        self.get_logger().info("Moving to home position...")
        self._use_ompl()
        self._moveit2.move_to_configuration(HOME_JOINTS, joint_names=robot.joint_names())
        self._moveit2.wait_until_executed()
        self.get_logger().info("Home reached.")
        return True

    def run(self, task: list):
        for step in task:
            label = step["label"]
            act   = step["act"]
            self.get_logger().info(f"[{act.upper()}] {label}")

            if act == "arm":
                if not self.move_to(step):
                    self.get_logger().error(f"Arm failed at '{label}'. Aborting.")
                    return

            elif act == "gripper":
                if not self.move_gripper(step["gripper"]):
                    self.get_logger().error(f"Gripper failed at '{label}'. Aborting.")
                    return

            elif act == "both":
                if not self.move_to(step):
                    self.get_logger().error(f"Arm failed at '{label}'. Aborting.")
                    return
                if not self.move_gripper(step["gripper"]):
                    self.get_logger().error(f"Gripper failed at '{label}'. Aborting.")
                    return

            else:
                self.get_logger().error(f"Unknown act='{act}' in step '{label}'. Aborting.")
                return

            time.sleep(1.0)

        self.get_logger().info("Sequence complete.")

    def move_to(self, step: dict) -> bool:
        pos  = [step["x"], step["y"], step["z"]]
        quat = [step["qx"], step["qy"], step["qz"], step["qw"]]

        self._use_ompl()

        # --- 1. OMPL Cartesian ---
        self.get_logger().info("  Planning OMPL Cartesian...")
        trajectory = self._moveit2.plan(
            position=pos,
            quat_xyzw=quat,
            cartesian=True,
        )

        if trajectory is not None:
            self.get_logger().info("  Cartesian plan succeeded, executing...")
            self._moveit2.execute(trajectory)
            self._moveit2.wait_until_executed()

            dist = self._distance_to(pos)
            self.get_logger().info(f"  [OMPL Cartesian] Distance to goal: {dist:.4f} m")
            if dist < self.TOLERANCE:
                self.get_logger().info("  [OMPL Cartesian] Success.")
                return True
            self.get_logger().warn("  [OMPL Cartesian] Executed but did not reach goal.")
        else:
            self.get_logger().warn("  Cartesian plan failed, skipping execution.")

        # --- 2. OMPL joint-space fallback ---
        self.get_logger().warn("  Falling back to OMPL joint-space...")
        for attempt in range(1, self.MAX_RETRIES + 1):
            self.get_logger().info(f"  [OMPL Joint] Attempt {attempt}/{self.MAX_RETRIES}")
            trajectory = self._moveit2.plan(
                position=pos,
                quat_xyzw=quat,
                cartesian=False,
            )

            if trajectory is None:
                self.get_logger().warn(f"  [OMPL Joint] Planning failed on attempt {attempt}.")
                continue

            self.get_logger().info(f"  [OMPL Joint] Plan succeeded, executing...")
            self._moveit2.execute(trajectory)
            self._moveit2.wait_until_executed()

            dist = self._distance_to(pos)
            self.get_logger().info(f"  [OMPL Joint] Distance to goal: {dist:.4f} m")
            if dist < self.TOLERANCE:
                self.get_logger().info("  [OMPL Joint] Success.")
                return True
            self.get_logger().warn(f"  [OMPL Joint] Executed but did not reach goal.")

        self.get_logger().error("  All planning strategies failed.")
        return False

    def move_gripper(self, action: Gripper) -> bool:
        cli   = self._gripper_open if action is Gripper.OPEN else self._gripper_close
        label = action.name
        self.get_logger().info(f"  Gripper → {label}")
        return self._call_trigger(cli)

    def _call_trigger(self, cli) -> bool:
        future   = cli.call_async(Trigger.Request())
        deadline = time.time() + self.GRIPPER_TIMEOUT_SEC
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() > deadline:
                self.get_logger().error("  Gripper service call timed out.")
                return False
        resp = future.result()
        if resp is None:
            self.get_logger().error("  Gripper service returned None.")
            return False
        self.get_logger().info(f"  Gripper result: {resp.message}")
        return resp.success

    def add_obstacle(self, name: str, size: list, position: list):
        self._moveit2.add_collision_box(id=name, size=size, position=position,
                                        quat_xyzw=[0.0, 0.0, 0.0, 1.0])
        self.get_logger().info(f"Added obstacle '{name}' at {position}")

    def _distance_to(self, target: list) -> float:
        try:
            tf = self._tf_buffer.lookup_transform(
                robot.base_link_name(),
                robot.end_effector_name(),
                rclpy.time.Time(),
            )
            t = tf.transform.translation
            return math.sqrt((t.x-target[0])**2 + (t.y-target[1])**2 + (t.z-target[2])**2)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return float("inf")


def main():
    rclpy.init()
    node = UR3Robot()

    time.sleep(2)

    node.add_obstacle("table", size=[2.0, 2.0, 0.02], position=[0.0, 0.0, -0.05])
    node.add_obstacle("wall",  size=[0.02, 2.00, 2.0], position=[0.15, 0.0, 0.0])

    node.move_to_home()
    node.run(PICK_AND_PLACE)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()