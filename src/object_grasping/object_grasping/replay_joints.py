#!/usr/bin/env python3
"""
Replay joint positions recorded by record_joints.py.
Also replays gripper open/close events at the correct timestamps.

Usage:
    python3 replay_joints.py --input demo.json
    python3 replay_joints.py --input demo.json --speed 2.0   # double speed
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
import json
import argparse
import threading
import time


ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'


class JointReplayer(Node):
    def __init__(self, input_file, speed_factor):
        super().__init__('joint_replayer')
        self.input_file = input_file
        self.speed_factor = speed_factor  # 1.0 = same speed, 2.0 = twice as fast

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_SERVER
        )

        self.gripper_close_client = self.create_client(Trigger, '/gripper/close')
        self.gripper_open_client = self.create_client(Trigger, '/gripper/open')

        # Event that fires when trajectory is accepted and robot starts moving
        self.trajectory_started = threading.Event()

    def load(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)

        self.get_logger().info(
            f'Loaded {data["num_points"]} points '
            f'({data["duration"]:.2f}s), '
            f'{len(data.get("gripper_events", []))} gripper event(s) '
            f'from {self.input_file}'
        )
        return data

    def build_trajectory(self, data):
        traj = JointTrajectory()
        traj.joint_names = data['joint_names']

        last_ns = -1

        for point_data in data['trajectory']:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']

            # speed_factor > 1 = faster, < 1 = slower
            scaled_time = point_data['time'] / self.speed_factor
            total_ns = int(scaled_time * 1e9)

            # Enforce strictly increasing timestamps
            if total_ns <= last_ns:
                total_ns = last_ns + 1_000_000  # bump by 1ms
            last_ns = total_ns

            secs = total_ns // 1_000_000_000
            nanosecs = total_ns % 1_000_000_000
            point.time_from_start = Duration(sec=secs, nanosec=nanosecs)

            traj.points.append(point)

        return traj

    def call_gripper(self, action):
        client = self.gripper_close_client if action == 'close' else self.gripper_open_client
        service = f'/gripper/{action}'

        if client.wait_for_service(timeout_sec=1.0):
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() and future.result().success:
                self.get_logger().info(f'Gripper {action} successful')
            else:
                self.get_logger().warn(f'Gripper {action} failed')
        else:
            self.get_logger().error(f'{service} not available')

    def replay_gripper_events(self, gripper_events):
        """Wait for trajectory to be accepted, then replay gripper events."""
        if not gripper_events:
            return None

        def run():
            # Wait until trajectory is accepted and robot starts moving
            self.get_logger().info('Gripper thread waiting for trajectory to start...')
            self.trajectory_started.wait()
            start = time.monotonic()
            self.get_logger().info('Gripper thread started.')

            for event in gripper_events:
                target_time = event['time'] / self.speed_factor

                while True:
                    elapsed = time.monotonic() - start
                    if elapsed >= target_time:
                        break
                    time.sleep(0.005)

                self.get_logger().info(
                    f'Replaying gripper {event["action"]} at t={elapsed:.3f}s '
                    f'(recorded at t={event["time"]:.3f}s)')
                self.call_gripper(event['action'])

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def replay(self):
        data = self.load()
        traj = self.build_trajectory(data)
        gripper_events = data.get('gripper_events', [])

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f'Sending trajectory ({len(traj.points)} points, '
            f'duration={traj.points[-1].time_from_start.sec}s '
            f'at {self.speed_factor}x speed)...'
        )

        # Start gripper thread now — it will block until trajectory is accepted
        self.replay_gripper_events(gripper_events)

        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory rejected by controller.')
            rclpy.shutdown()
            return

        # Signal gripper thread to start timing now
        self.trajectory_started.set()
        self.get_logger().info('Trajectory accepted, executing...')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Replay complete.')
        else:
            self.get_logger().error(
                f'Trajectory failed with error code: {result.error_code}')

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
        help='Speed factor (1.0 = same speed, 2.0 = twice as fast, 0.5 = half speed)'
    )
    args = parser.parse_args()

    rclpy.init()
    replayer = JointReplayer(args.input, args.speed)
    replayer.replay()
    rclpy.spin(replayer)


if __name__ == '__main__':
    main()