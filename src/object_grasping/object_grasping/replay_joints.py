#!/usr/bin/env python3
"""
Replay joint positions recorded by record_joints.py.
Triggers gripper open/close based on joint positions rather than wall clock time.

Usage:
    python3 replay_joints.py --input demo.json
    python3 replay_joints.py --input demo.json --speed 2.0   # double speed
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
import json
import argparse


ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'

# How close the arm needs to be to the trigger position to fire the gripper (radians)
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

        # List of pending gripper triggers: each is a dict with
        # 'positions' (trigger positions) and 'action' ('open' or 'close')
        self.pending_triggers = []
        self.joint_monitor_sub = None

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

    def find_trigger_positions(self, trajectory, event_time):
        """Find the joint positions in the trajectory closest to the event time."""
        idx = min(
            range(len(trajectory)),
            key=lambda i: abs(trajectory[i]['time'] - event_time)
        )
        self.get_logger().info(
            f'Gripper {event_time:.3f}s maps to trajectory point {idx} '
            f'at t={trajectory[idx]["time"]:.3f}s'
        )
        return trajectory[idx]['positions']

    def build_trajectory(self, data):
        traj = JointTrajectory()
        traj.joint_names = data['joint_names']

        last_ns = -1

        for point_data in data['trajectory']:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']

            total_ns = int(point_data['time'] / 1e9)

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

    def joint_state_callback(self, msg):
        """Monitor joint states and fire gripper when trigger positions are reached."""
        if not self.pending_triggers:
            return

        current_positions = list(msg.position[:6])
        trigger = self.pending_triggers[0]

        # Check if arm is within threshold of trigger positions
        error = max(
            abs(a - b)
            for a, b in zip(current_positions, trigger['positions'])
        )

        if error < POSITION_THRESHOLD:
            self.get_logger().info(
                f'Trigger position reached (max error={error:.4f} rad), '
                f'gripper {trigger["action"]}'
            )
            self.call_gripper(trigger['action'])
            self.pending_triggers.pop(0)  # remove fired trigger

            # If no more triggers, unsubscribe
            if not self.pending_triggers:
                self.get_logger().info('All gripper triggers fired, stopping joint monitor.')
                self.destroy_subscription(self.joint_monitor_sub)
                self.joint_monitor_sub = None

    def start_joint_monitor(self):
        self.joint_monitor_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def replay(self):
        data = self.load()
        traj = self.build_trajectory(data)
        gripper_events = data.get('gripper_events', [])

        # Build list of position-based triggers from gripper events
        if gripper_events:
            for event in gripper_events:
                trigger_positions = self.find_trigger_positions(
                    data['trajectory'], event['time'])
                self.pending_triggers.append({
                    'positions': trigger_positions,
                    'action': event['action']
                })
                self.get_logger().info(
                    f'Gripper {event["action"]} will trigger when arm reaches '
                    f'positions: {[round(p, 3) for p in trigger_positions]}'
                )

            self.start_joint_monitor()

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f'Sending trajectory ({len(traj.points)} points, '
            f'duration={traj.points[-1].time_from_start.sec}s '
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
    args = parser.parse_args()

    rclpy.init()
    replayer = JointReplayer(args.input)
    replayer.replay()
    rclpy.spin(replayer)


if __name__ == '__main__':
    main()