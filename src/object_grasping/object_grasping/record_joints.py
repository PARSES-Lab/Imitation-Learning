#!/usr/bin/env python3
"""
Record joint positions from /joint_states until Ctrl-C.
Also monitors /gripper/status and records open/close transition timestamps.
Gripper is considered closed when data[1] > 3.

Usage:
    python3 record_joints.py
    python3 record_joints.py --output my_demo.json
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
import json
import argparse
import signal
import sys
from datetime import datetime


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
        self.recording = []
        self.gripper_events = []
        self.start_time = None
        self.message_count = 0
        self.downsample_rate = 100  # ~5Hz at 500Hz joint state publish rate
        self.gripper_was_closed = False  # track previous state to detect transitions

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

        self.get_logger().info('Recording joint states and gripper status...')
        self.get_logger().info('Press Ctrl-C to stop.')

    def joint_state_callback(self, msg):
        self.message_count += 1
        if self.message_count % self.downsample_rate != 0:
            return

        # Use message timestamp instead of system clock
        msg_time_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        if self.start_time is None:
            self.start_time = msg_time_ns

        elapsed = (msg_time_ns - self.start_time) / 1e9

        filtered = {
            name: pos
            for name, pos in zip(msg.name, msg.position)
            if name in UR_JOINTS
        }

        self.recording.append({
            'time': elapsed,
            'joint_names': UR_JOINTS,
            'positions': [filtered[j] for j in UR_JOINTS],
        })

    def gripper_status_callback(self, msg):
        if len(msg.data) < 2:
            return

        if self.start_time is None:
            return  # don't record gripper events before joint recording starts

        is_closed = msg.data[1] > GRIPPER_CLOSED_THRESHOLD
        elapsed = (self.get_clock().now().nanoseconds - self.start_time) / 1e9

        # Only record on state transitions, not every message
        if is_closed and not self.gripper_was_closed:
            self.gripper_events.append({'time': elapsed, 'action': 'close'})
            self.get_logger().info(f'Gripper closed at t={elapsed:.3f}s')

        elif not is_closed and self.gripper_was_closed:
            self.gripper_events.append({'time': elapsed, 'action': 'open'})
            self.get_logger().info(f'Gripper opened at t={elapsed:.3f}s')

        self.gripper_was_closed = is_closed

    def save(self):
        if not self.recording:
            self.get_logger().warn('No data recorded.')
            return
        
        # Sort by time and remove any duplicate timestamps
        sorted_traj = sorted(self.recording, key=lambda p: p['time'])
        deduped_traj = [sorted_traj[0]]
        for point in sorted_traj[1:]:
            if point['time'] > deduped_traj[-1]['time']:
                deduped_traj.append(point)

        data = {
            'recorded_at': datetime.now().isoformat(),
            'num_points': len(self.recording),
            'duration': self.recording[-1]['time'],
            'joint_names': UR_JOINTS,
            'gripper_events': self.gripper_events,
            'trajectory': self.recording,
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(
            f'Saved {len(self.recording)} points '
            f'({self.recording[-1]["time"]:.2f}s), '
            f'{len(self.gripper_events)} gripper event(s) '
            f'to {self.output_file}'
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