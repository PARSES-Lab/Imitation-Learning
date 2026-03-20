#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class PublisherNode(Node):
    def __init__(self):
        super().__init__("publisher_node")
        self.get_logger().info("Hello")
        self.pub = self.create_publisher(Int32, '/some_topic', 10)
        self.timer = self.create_timer(0.5, self.publisher_callback)

    def publisher_callback(self):
        msg = Int32()
        msg.data = 5
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    publisher_node = PublisherNode()
    rclpy.spin(publisher_node)

    rclpy.shutdown()

if __name__ == "__main__":
    main()