#!/usr/bin/env python3
"""
Record joint positions from /joint_states until Ctrl-C.
Saves to a JSON file with timestamps.

Usage:
    python3 record_joints.py
    python3 record_joints.py --output my_demo.json
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import json
import argparse
import signal
import sys
from datetime import datetime


class JointRecorder(Node):
    def __init__(self, output_file):
        super().__init__('joint_recorder')
        self.output_file = output_file
        self.recording = []
        self.start_time = None

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.message_count = 0
        self.downsample_rate = 50  # record every Nth message

        self.get_logger().info('Recording joint states... Press Ctrl-C to stop.')

    def joint_state_callback(self, msg):
        self.message_count += 1
        if self.message_count % self.downsample_rate != 0:
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds

        # Time in seconds relative to start of recording
        elapsed = (self.get_clock().now().nanoseconds - self.start_time) / 1e9

        self.recording.append({
            'time': elapsed,
            'joint_names': list(msg.name),
            'positions': list(msg.position),
            'velocities': list(msg.velocity) if msg.velocity else [],
        })

    def save(self):
        if not self.recording:
            self.get_logger().warn('No data recorded.')
            return

        data = {
            'recorded_at': datetime.now().isoformat(),
            'num_points': len(self.recording),
            'duration': self.recording[-1]['time'],
            'joint_names': self.recording[0]['joint_names'],
            'trajectory': self.recording,
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(
            f'Saved {len(self.recording)} points '
            f'({self.recording[-1]["time"]:.2f}s) to {self.output_file}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o',
        default=f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        help='Output JSON file path'
    )
    args = parser.parse_args()

    rclpy.init()
    recorder = JointRecorder(args.output)

    # Handle Ctrl-C gracefully
    def shutdown(sig, frame):
        recorder.get_logger().info('Stopping recording...')
        recorder.save()
        recorder.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    rclpy.spin(recorder)


if __name__ == '__main__':
    main()