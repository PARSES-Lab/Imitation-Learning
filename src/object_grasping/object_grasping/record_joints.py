#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import yaml
import argparse
import signal
import sys
from datetime import datetime

from rosidl_runtime_py import message_to_yaml


UR_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

GRIPPER_CLOSED_THRESHOLD = 3


class JointRecorder(Node):
    def __init__(self, output_file):
        super().__init__('joint_recorder')

        self.output_file = output_file
        self.start_time = None
        self.message_count = 0
        self.downsample_rate = 1

        self.trajectory = JointTrajectory()
        self.trajectory.joint_names = UR_JOINTS

        self.gripper_events = []
        self.gripper_was_closed = False

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.gripper_subscription = self.create_subscription(
            Int32MultiArray,
            '/gripper/status',
            self.gripper_status_callback,
            10
        )

        self.get_logger().info('Recording... Ctrl-C to stop.')

    def joint_state_callback(self, msg):
        self.message_count += 1
        if self.message_count % self.downsample_rate != 0:
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds

        elapsed = (self.get_clock().now().nanoseconds - self.start_time) / 1e9

        name_to_pos = dict(zip(msg.name, msg.position))

        if any(j not in name_to_pos for j in UR_JOINTS):
            return

        point = JointTrajectoryPoint()
        point.positions = [name_to_pos[j] for j in UR_JOINTS]

        total_ns = int(elapsed * 1e9)
        point.time_from_start = Duration(
            sec=total_ns // 1_000_000_000,
            nanosec=total_ns % 1_000_000_000
        )

        self.trajectory.points.append(point)

    def gripper_status_callback(self, msg):
        if len(msg.data) < 2 or self.start_time is None:
            return

        is_closed = msg.data[1] > GRIPPER_CLOSED_THRESHOLD
        elapsed = (self.get_clock().now().nanoseconds - self.start_time) / 1e9

        if is_closed and not self.gripper_was_closed:
            self.gripper_events.append({'time': elapsed, 'action': 'close'})
            self.get_logger().info(f'Gripper closed at {elapsed:.3f}s')

        elif not is_closed and self.gripper_was_closed:
            self.gripper_events.append({'time': elapsed, 'action': 'open'})
            self.get_logger().info(f'Gripper opened at {elapsed:.3f}s')

        self.gripper_was_closed = is_closed

    def save(self):
        if not self.trajectory.points:
            print('No data recorded.')
            return

        data = {
            'recorded_at': datetime.now().isoformat(),
            'trajectory': yaml.safe_load(message_to_yaml(self.trajectory)),
            'gripper_events': self.gripper_events
        }

        with open(self.output_file, 'w') as f:
            yaml.dump(data, f)

        print(
            f'Saved {len(self.trajectory.points)} points '
            f'and {len(self.gripper_events)} gripper events '
            f'to {self.output_file}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='demo.yaml')
    args = parser.parse_args()

    rclpy.init()
    recorder = JointRecorder(args.output)

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print('Ctrl-C received, saving...')
    finally:
        recorder.save()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()