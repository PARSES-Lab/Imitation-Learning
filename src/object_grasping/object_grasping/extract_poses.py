import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur as robot
from rclpy.callback_groups import ReentrantCallbackGroup

class PoseExtractor(Node):
    def __init__(self):
        super().__init__('pose_extractor')

        self.moveit2_callback_group = ReentrantCallbackGroup()
    
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=self.moveit2_callback_group
        )
        self.sub = self.create_subscription(JointState, '/joint_states', self.callback, 10)
        self.csv_file = open('poses.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    def callback(self, msg: JointState):
        fk_pose = self._moveit2.compute_fk(msg.position)

        t = fk_pose.pose.position
        r = fk_pose.pose.orientation
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.writer.writerow([f'{ts:.6f}', t.x, t.y, t.z, r.x, r.y, r.z, r.w])

    
def main():
    rclpy.init()
    node = PoseExtractor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()