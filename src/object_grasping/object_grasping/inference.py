## uses some code from ur3_control

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur as robot
from std_srvs.srv import Trigger
from enum import Enum, auto
import math
import time
import numpy as np
from sensor_msgs.msg import Image
from .model.model import PolicyNetwork
import torch
import torchvision.transforms.v2 as v2
from dataclasses import dataclass
import threading
from rclpy.callback_groups import ReentrantCallbackGroup

MODEL_WEIGHTS_PATH = (
    "/home/parses/ros2_ws/src/object_grasping/object_grasping/Graspingv6.pth"
)
# MODEL_WEIGHTS_PATH = '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/Graspingv1.pth'
HIDDEN_DIM = 200

# Gripper threshold — above this the network predicts close
GRIPPER_THRESHOLD = 0.8


class Gripper(Enum):
    OPEN = auto()
    CLOSE = auto()


@dataclass
class HistoryElement:
    position: list[float]
    orientation: list[float]  # xyzw quaternion
    gripper_state: Gripper


class UR3Inference(Node):
    PLANNING_TIME = 5.0
    OMPL_PLANNING_TIME = 10.0
    MAX_RETRIES = 3
    TOLERANCE = 0.001
    GRIPPER_TIMEOUT_SEC = 5.0

    OMPL_MAX_VEL = 0.1
    OMPL_MAX_ACC = 0.1

    def __init__(self):
        super().__init__("UR3_Inference_Node")

        self.moveit2_callback_group = ReentrantCallbackGroup()

        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=self.moveit2_callback_group,
        )

        self._moveit2.planner_id = "ompl/RRTConnectkConfigDefault"
        self._moveit2.planning_pipeline = "ompl"
        self._moveit2.planning_time = self.OMPL_PLANNING_TIME
        self._moveit2.max_velocity = self.OMPL_MAX_VEL
        self._moveit2.max_acceleration = self.OMPL_MAX_ACC
        self._moveit2.cartesian_avoid_collisions = True
        self._moveit2.cartesian_jump_threshold = 0.0
        self.cartesian_fraction_threshold = 1.0
        self.cartesian_max_step = 0.0025

        self.time_after_action = None
        self.latest_image_time = None

        self.shared_state_lock = threading.Lock()

        self._gripper_open = self.create_client(Trigger, "/gripper/open")
        self._gripper_close = self.create_client(Trigger, "/gripper/close")
        self.current_gripper_state = Gripper.OPEN

        self.image_subscriber = self.create_subscription(
            Image, "/camera/realsense_camera/color/image_raw", self.image_callback, 10
        )
        self.latest_image = None

        # Image preprocessing to match training pipeline
        self.preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.model = PolicyNetwork(hidden_dim=HIDDEN_DIM)
        self.model.eval()
        state_dict = torch.load(
            MODEL_WEIGHTS_PATH, weights_only=True, map_location=torch.device("cpu")
        )
        self.model.load_state_dict(state_dict)

        self.get_logger().info("UR3Inference ready.")


    def _get_latest_state(self):
        joint_state = self._moveit2.joint_state
        if joint_state is None:
            return None

        arm_names = robot.joint_names()
        name_to_pos = dict(zip(joint_state.name, joint_state.position))
        positions = [name_to_pos[n] for n in arm_names]
        fk_pose = self._moveit2.compute_fk(positions)

        gripper = 1.0 if self.current_gripper_state == Gripper.CLOSE else 0.0
        translation = [
            fk_pose.pose.position.x,
            fk_pose.pose.position.y,
            fk_pose.pose.position.z,
        ]
        rotation = [
            fk_pose.pose.orientation.x,
            fk_pose.pose.orientation.y,
            fk_pose.pose.orientation.z,
            fk_pose.pose.orientation.w,
        ]

        return translation, rotation, gripper


    def image_callback(self, msg: Image):
        np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        np_img = np_img[:, :, :3]

        with self.shared_state_lock:
            self.latest_image = np_img
            self.latest_image_time = self.get_clock().now()

    # ── Inference ─────────────────────────────────────────────────────────────

    def run_task(self):
        """
        Main inference loop. Runs until the task completes (gripper closes
        and the arm stops moving) or a move fails.
        """

        self.get_logger().info("Starting inference loop.")

        while True:
            with self.shared_state_lock:
                image = (
                    self.latest_image.copy() if self.latest_image is not None else None
                )
                image_time = self.latest_image_time
                time_after_action = self.time_after_action

            if image is None or image_time is None:
                time.sleep(0.01)
                continue

            if time_after_action is not None:
                if image_time < time_after_action:
                    time.sleep(0.01)
                    continue

            image = self.preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

            current_state = self._get_latest_state()
            if current_state is None:
                self.get_logger().warn("Could not get latest state, trying again.")
                continue

            translation, rotation, gripper = current_state

            with torch.no_grad():
                pred = self.model(image, torch.tensor(translation, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

            delta_position = pred["delta_position"][0].numpy() / 100  # (3,)
            gripper_logit = pred["gripper_state"][0].item()
            target_gripper = (
                Gripper.CLOSE
                if torch.sigmoid(torch.tensor(gripper_logit)) > GRIPPER_THRESHOLD
                else Gripper.OPEN
            )
            
            translation = np.array(translation)

            target_position = translation + delta_position

            # Move arm
            move_success = self.move(
                position=target_position.tolist(),
                orientation=rotation,
            )
            if not move_success:
                self.get_logger().error("Move failed, aborting.")
                return False

            # Move gripper
            gripper_success = self.move_gripper(target_gripper)
            if not gripper_success:
                self.get_logger().error("Gripper action failed, aborting.")
                return False

            with self.shared_state_lock:
                self.time_after_action = self.get_clock().now()

            # Check for task completion: gripper closed and delta is near zero
            # position_delta_norm = np.linalg.norm(delta_position)
            # if (target_gripper == Gripper.CLOSE
            #         and position_delta_norm < self.TOLERANCE):
            #     self.get_logger().info('Task complete.')
            #     return True


    def move_gripper(self, action: Gripper) -> bool:
        if action == self.current_gripper_state:
            return True
        cli = self._gripper_open if action is Gripper.OPEN else self._gripper_close
        label = action.name
        self.get_logger().info(f"Gripper -> {label}")
        success = self._call_trigger(cli)
        if success:
            self.current_gripper_state = action
        return success

    def _call_trigger(self, cli) -> bool:
        future = cli.call_async(Trigger.Request())
        deadline = time.time() + self.GRIPPER_TIMEOUT_SEC
        while not future.done():
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error("Gripper service call timed out.")
                return False
        resp = future.result()
        if resp is None:
            self.get_logger().error("Gripper service returned None.")
            return False
        self.get_logger().info(f"Gripper result: {resp.message}")
        return resp.success

    def move(self, position, orientation) -> bool:
        self.get_logger().info("Planning OMPL Cartesian...")
        trajectory = self._moveit2.plan(
            position=position,
            quat_xyzw=orientation,
            cartesian=True,
            cartesian_fraction_threshold=self.cartesian_fraction_threshold,
            max_step=self.cartesian_max_step,
        )

        if trajectory is not None:
            self.get_logger().info("Cartesian plan succeeded, executing...")
            self._moveit2.execute(trajectory)
            self._moveit2.wait_until_executed()
            dist = self._distance_to(position)
            self.get_logger().info(f"Distance to goal: {dist:.4f} m")
            if dist < self.TOLERANCE:
                self.get_logger().info("Move succeeded.")
                return True
            self.get_logger().error("Executed but did not reach goal.")
            return False
        else:
            self.get_logger().error("Cartesian plan failed.")
            return False

    def add_obstacle(self, name: str, size: list, position: list):
        self._moveit2.add_collision_box(
            id=name, size=size, position=position, quat_xyzw=[0.0, 0.0, 0.0, 1.0]
        )
        self.get_logger().info(f"Added obstacle '{name}' at {position}")

    def _distance_to(self, target: list) -> float:
        joint_state = self._moveit2.joint_state
        arm_names = robot.joint_names()
        name_to_pos = dict(zip(joint_state.name, joint_state.position))
        positions = [name_to_pos[n] for n in arm_names]
        fk_pose = self._moveit2.compute_fk(positions)

        translation = [
            fk_pose.pose.position.x,
            fk_pose.pose.position.y,
            fk_pose.pose.position.z,
        ]
        return math.sqrt(
            (translation[0] - target[0]) ** 2
            + (translation[1] - target[1]) ** 2
            + (translation[2] - target[2]) ** 2
        )

    def move_to_home(self):
        home_position = [
            0.1464331,
            -1.1904891,
            -1.4117968,
            -2.05215813,
            1.5758578,
            0.21868976,
        ]
        self.get_logger().info("Moving to home position...")
        self._moveit2.move_to_configuration(
            home_position, joint_names=robot.joint_names()
        )
        self._moveit2.wait_until_executed()

        joint_state = self._moveit2.joint_state
        arm_names = robot.joint_names()
        name_to_pos = dict(zip(joint_state.name, joint_state.position))
        current = np.array([name_to_pos[n] for n in arm_names])
        home = np.array(home_position)

        distance = np.linalg.norm(current - home)

        if distance > 0.01:
            self.get_logger().error("Move to home failed, aborting.")
            return False

        self.get_logger().info("Home reached.")
        return True


def main():
    rclpy.init()
    node = UR3Inference()

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    node.add_obstacle("table", size=[2.0, 2.0, 0.02], position=[0.0, 0.0, -0.05])
    node.add_obstacle("wall", size=[0.02, 2.00, 2.0], position=[0.15, 0.0, 0.0])
    node.add_obstacle("bar", size=[0.05, 0.05, 2.5], position=[0.1, -0.1, 0.0])

    # home_success = node.move_to_home()
    # if not home_success:
    #     rclpy.shutdown()
    #     return

    node.run_task()
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
