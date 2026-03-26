#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

import yaml
import argparse

from rosidl_runtime_py import set_message_fields

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
POSITION_THRESHOLD = 0.05  # still used if you want position monitoring


class JointReplayer(Node):
    def __init__(self, input_file):
        super().__init__('joint_replayer')

        self.input_file = input_file

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_SERVER
        )

        self.gripper_close_client = self.create_client(Trigger, '/gripper/close')
        self.gripper_open_client = self.create_client(Trigger, '/gripper/open')

        self.start_time = None
        self.pending_triggers = []

    def load(self):
        with open(self.input_file, 'r') as f:
            return yaml.safe_load(f)

    def call_gripper(self, action):
        client = self.gripper_close_client if action == 'close' else self.gripper_open_client

        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'/gripper/{action} not available')
            return

        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: self.get_logger().info(f'Gripper {action} done'))

    def gripper_timer_callback(self):
        if not self.pending_triggers or self.start_time is None:
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Trigger all events whose time has passed
        while self.pending_triggers and elapsed >= self.pending_triggers[0]['time']:
            event = self.pending_triggers.pop(0)
            self.call_gripper(event['action'])

    def replay(self):
        data = self.load()

        traj = JointTrajectory()
        set_message_fields(traj, data['trajectory'])

        # Convert gripper_events into a sorted list by time
        self.pending_triggers = sorted(data.get('gripper_events', []), key=lambda e: e['time'])

        # Wait for action server
        self._action_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

        # Start timer to trigger gripper events by time
        self.start_time = self.get_clock().now()
        self.create_timer(0.01, self.gripper_timer_callback)  # check every 10ms

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected')
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: rclpy.shutdown())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    args = parser.parse_args()

    rclpy.init()
    node = JointReplayer(args.input)
    node.replay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()