#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import yaml
import argparse
from rosidl_runtime_py import set_message_fields
from builtin_interfaces.msg import Duration

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
POSITION_THRESHOLD = 0.05
WAIT_SECONDS = 5.0


class JointReplayer(Node):
    def __init__(self, input_file, gripper_time_shift):
        super().__init__('joint_replayer')
        self.input_file = input_file
        self.gripper_time_shift = gripper_time_shift
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_SERVER
        )
        self.gripper_close_client = self.create_client(Trigger, '/gripper/close')
        self.gripper_open_client = self.create_client(Trigger, '/gripper/open')
        self.start_time = None
        self.pending_triggers = []
        self.full_traj = None  # stored for use after the wait

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
        while self.pending_triggers and elapsed >= self.pending_triggers[0]['time'] + self.gripper_time_shift:
            event = self.pending_triggers.pop(0)
            self.call_gripper(event['action'])

    def replay(self):
        data = self.load()

        self.full_traj = JointTrajectory()
        set_message_fields(self.full_traj, data['trajectory'])

        self.pending_triggers = sorted(data.get('gripper_events', []), key=lambda e: e['time'])

        self._action_client.wait_for_server()

        # --- Build a single-point trajectory to the first position ---
        first_point = self.full_traj.points[0]
        move_to_start_traj = JointTrajectory()
        move_to_start_traj.joint_names = self.full_traj.joint_names

        start_point = JointTrajectoryPoint()
        start_point.positions = first_point.positions
        start_point.velocities = [0.0] * len(first_point.positions)  # arrive at rest
        start_point.time_from_start = Duration(sec=5, nanosec=0)  # allow 5s to reach it

        move_to_start_traj.points = [start_point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = move_to_start_traj

        self.get_logger().info('Moving to start position...')
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.start_position_goal_response_callback)

    # --- Callback: goal accepted for move-to-start ---
    def start_position_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Move-to-start trajectory rejected')
            rclpy.shutdown()
            return
        self.get_logger().info('Moving to start position accepted, waiting for completion...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.start_position_reached_callback)

    # --- Callback: arm has reached the start position, now wait 5s ---
    def start_position_reached_callback(self, future):
        self.get_logger().info(f'Start position reached. Waiting {WAIT_SECONDS}s before replaying...')
        self.wait_timer = self.create_timer(WAIT_SECONDS, self.execute_full_trajectory)

    def execute_full_trajectory(self):
        self.wait_timer.cancel()  # stop it from firing again
        self.get_logger().info('Wait complete. Executing full trajectory...')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self.full_traj

        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

        self.start_time = self.get_clock().now()
        self.create_timer(0.01, self.gripper_timer_callback)

    # --- Original callback for the full trajectory ---
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
    parser.add_argument('--gripper_time_shift', '-g', type=float, default=0.0)
    args = parser.parse_args()

    rclpy.init()
    node = JointReplayer(args.input, float(args.gripper_time_shift))
    node.replay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()