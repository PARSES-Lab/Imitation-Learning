import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur as robot
from std_srvs.srv import Trigger
from tf2_ros import Buffer, Duration, TransformListener
from enum import Enum, auto
import math
import time

class Gripper(Enum):
    OPEN = auto()
    CLOSE = auto()

class UR3Inference(Node):
    PLANNING_TIME       = 5.0
    OMPL_PLANNING_TIME  = 10.0
    MAX_RETRIES         = 3
    TOLERANCE           = 0.01
    GRIPPER_TIMEOUT_SEC = 5.0

    OMPL_MAX_VEL   = 0.1
    OMPL_MAX_ACC   = 0.1

    def __init__(self):
        super().__init__("UR3_Inference_Node")
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM
        )

        self._moveit2.planner_id        = "ompl/RRTConnectkConfigDefault"
        self._moveit2.planning_pipeline = "ompl"
        self._moveit2.planning_time     = self.OMPL_PLANNING_TIME
        self._moveit2.max_velocity      = self.OMPL_MAX_VEL
        self._moveit2.max_acceleration  = self.OMPL_MAX_ACC

        self._gripper_open = self.create_client(Trigger, "/gripper/open")
        self._gripper_close = self.create_client(Trigger, "/gripper/close")

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.get_logger().info("UR3Robot ready.")


    def move(self):
        pos = [-0.19685978353575456,0.08411729105998772,0.41083662882897576]
        quat = [0.7318706791974158,0.6808237485839617,-0.025284886806180202,-0.014318058331861576]
        

        self.get_logger().info("  Planning OMPL Cartesian...")
        trajectory = self._moveit2.plan(
            position=pos,
            quat_xyzw=quat,
            cartesian=True,
        )

        if trajectory is not None:
            self.get_logger().info("Cartesian plan succeeded, executing...")
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
            return False
        
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
    
    def move_joint(self):
        self.get_logger().info("Moving to home position...")
        self._moveit2.move_to_configuration([0.14636051654815674,-1.1905584794333954,-1.4117673635482788,-2.051380773583883,1.5758728981018066,0.2187575399875641], joint_names=robot.joint_names())
        self._moveit2.wait_until_executed()
        self.get_logger().info("Home reached.")
        return True


def main():
    rclpy.init()

    node = UR3Inference()

    time.sleep(5)

    node.add_obstacle("table", size=[2.0, 2.0, 0.02], position=[0.0, 0.0, -0.05])
    node.add_obstacle("wall",  size=[0.02, 2.00, 2.0], position=[0.15, 0.0, 0.0])

    # node.move_joint()
    node.move()

    rclpy.shutdown()


if __name__ == "__main__":
    main()