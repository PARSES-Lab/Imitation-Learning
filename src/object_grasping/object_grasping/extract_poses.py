import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pymoveit2.robots import ur as robot
import csv
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from threading import Thread


TEMP_FILE = 'tmp_joint_states.csv'
OUTPUT_FILE = 'poses.csv'


class JointStateCollector(Node):
    def __init__(self):
        super().__init__('joint_state_collector')
        self.sub = self.create_subscription(JointState, '/joint_states', self.callback, 10)
        self.csv_file = open(TEMP_FILE, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp'] + robot.joint_names())
        self.count = 0

    def callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        if not all(n in name_to_pos for n in robot.joint_names()):
            return
        positions = [name_to_pos[n] for n in robot.joint_names()]
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.writer.writerow([f'{ts:.6f}'] + positions)
        self.count += 1

    def close(self):
        self.csv_file.close()
        self.get_logger().info(f"Saved {self.count} joint states to {TEMP_FILE}")


class FKProcessor(Node):
    def __init__(self):
        super().__init__('fk_processor')

        cb_group = ReentrantCallbackGroup()
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=cb_group,
        )

        executor = rclpy.executors.MultiThreadedExecutor(12)
        executor.add_node(self)
        self._executor_thread = Thread(target=executor.spin, daemon=True)
        self._executor_thread.start()

    def process(self):
        with open(TEMP_FILE, 'r') as infile, open(OUTPUT_FILE, 'w', newline='') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

            rows = list(reader)
            self.get_logger().info(f"Processing {len(rows)} joint states...")

            for i, row in enumerate(rows):
                positions = [float(row[n]) for n in robot.joint_names()]
                fk_pose = self._moveit2.compute_fk(positions)
                if fk_pose is not None:
                    t = fk_pose.pose.position
                    r = fk_pose.pose.orientation
                    writer.writerow([row['timestamp'], t.x, t.y, t.z, r.x, r.y, r.z, r.w])
                else:
                    self.get_logger().warn(f"FK returned None for row {i}")
                
                if i % 100 == 0:
                    self.get_logger().info(f"Wrote to row {i}")

            self.get_logger().info(f"Wrote poses to {OUTPUT_FILE}")


def collect():
    rclpy.init()
    node = JointStateCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


def process():
    rclpy.init()
    node = FKProcessor()
    node.process()
    node.destroy_node()
    rclpy.shutdown()