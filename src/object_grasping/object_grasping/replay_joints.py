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
POSITION_THRESHOLD = 0.05


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

        self.pending_triggers = []
        self.joint_monitor_sub = None
        self.joint_names = []

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

    def joint_state_callback(self, msg):
        if not self.pending_triggers:
            return

        current = dict(zip(msg.name, msg.position))
        trigger = self.pending_triggers[0]

        errors = [
            abs(current[n] - trigger['positions'][n])
            for n in self.joint_names
            if n in current
        ]

        if errors and max(errors) < POSITION_THRESHOLD:
            self.call_gripper(trigger['action'])
            self.pending_triggers.pop(0)

            if not self.pending_triggers:
                self.destroy_subscription(self.joint_monitor_sub)

    def start_joint_monitor(self):
        self.joint_monitor_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def replay(self):
        data = self.load()

        traj = JointTrajectory()
        set_message_fields(traj, data['trajectory'])

        self.joint_names = traj.joint_names

        # Build triggers from closest trajectory point
        for event in data.get('gripper_events', []):
            idx = min(
                range(len(traj.points)),
                key=lambda i: abs(
                    traj.points[i].time_from_start.sec +
                    traj.points[i].time_from_start.nanosec * 1e-9
                    - event['time']
                )
            )

            positions = dict(zip(
                self.joint_names,
                traj.points[idx].positions
            ))

            self.pending_triggers.append({
                'positions': positions,
                'action': event['action']
            })

        self._action_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected')
            rclpy.shutdown()
            return

        if self.pending_triggers:
            self.start_joint_monitor()

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