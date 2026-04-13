import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import csv
import argparse

class PoseExtractor(Node):
    def __init__(self):
        super().__init__('pose_extractor')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(JointState, '/joint_states', self.callback, 10)
        self.csv_file = open('poses.csv', 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    def callback(self, msg: JointState):
        try:
            tf = self.tf_buffer.lookup_transform(
                'base', 'ee_link',
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        t = tf.transform.translation
        r = tf.transform.rotation
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.writer.writerow([f'{ts:.6f}', t.x, t.y, t.z, r.x, r.y, r.z, r.w])

    
def main():
    rclpy.init()
    node = PoseExtractor()
    rclpy.shutdown()

if __name__ == '__main__':
    main()