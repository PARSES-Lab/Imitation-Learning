#!/usr/bin/env python3
"""
Replay joint positions recorded by record_joints.py.
Publishes a JointTrajectory to the UR3e scaled_joint_trajectory_controller.

Usage:
    python3 replay_joints.py --input demo.json
    python3 replay_joints.py --input demo.json --speed 0.5   # half speed
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import argparse
import sys


# Action server name for UR3e default controller
ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'


class JointReplayer(Node):
    def __init__(self, input_file, speed_factor):
        super().__init__('joint_replayer')
        self.input_file = input_file
        self.speed_factor = speed_factor

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_SERVER
        )

    def load(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)

        self.get_logger().info(
            f'Loaded {data["num_points"]} points '
            f'({data["duration"]:.2f}s) from {self.input_file}'
        )
        return data

    def build_trajectory(self, data):
        traj = JointTrajectory()
        traj.joint_names = data['joint_names']

        for point_data in data['trajectory']:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']

            # Scale time by speed factor (lower = faster, higher = slower)
            scaled_time = point_data['time'] / self.speed_factor
            secs = int(scaled_time)
            nanosecs = int((scaled_time - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nanosecs)

            traj.points.append(point)

        return traj

    def replay(self):
        data = self.load()
        traj = self.build_trajectory(data)

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f'Sending trajectory ({len(traj.points)} points, '
            f'{traj.points[-1].time_from_start.sec}s at {self.speed_factor}x speed)...'
        )

        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Trajectory accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Replay complete.')
        else:
            self.get_logger().error(f'Trajectory failed with error code: {result.error_code}')

        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSON file from record_joints.py'
    )
    parser.add_argument(
        '--speed', '-s',
        type=float,
        default=1.0,
        help='Speed factor (1.0 = same speed, 0.5 = half speed, 2.0 = double speed)'
    )
    args = parser.parse_args()

    rclpy.init()
    replayer = JointReplayer(args.input, args.speed)
    replayer.replay()
    rclpy.spin(replayer)


if __name__ == '__main__':
    main()