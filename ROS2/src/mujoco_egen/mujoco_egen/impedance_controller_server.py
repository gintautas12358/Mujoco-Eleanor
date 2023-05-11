#
#BSD 3-Clause License
#
#
#
#Copyright 2022 fortiss, Neuromorphic Computing group
#
#
#All rights reserved.
#
#
#
#Redistribution and use in source and binary forms, with or without
#
#modification, are permitted provided that the following conditions are met:
#
#
#
#* Redistributions of source code must retain the above copyright notice, this
#
#  list of conditions and the following disclaimer.
#
#
#
#* Redistributions in binary form must reproduce the above copyright notice,
#
#  this list of conditions and the following disclaimer in the documentation
#
#  and/or other materials provided with the distribution.
#
#
#
#* Neither the name of the copyright holder nor the names of its
#
#  contributors may be used to endorse or promote products derived from
#
#  this software without specific prior written permission.
#
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#

from matplotlib import table
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from controller_interface.action import DesiredPoseName, Saccades2, DesiredPose
from camera_event_data_interface.msg import CameraEvents
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge, CvBridgeError

from .mujoco_env.im_mujoco_env import EsimMujoco
from .utils.read_cfg import read_cfg

import numpy as np
import time


class ImControllerActionServer(Node):

    def __init__(self):
        super().__init__('impedance_controller_server_node')

        # DesiredPoseName action server is created
        self._desired_pose_action_server = ActionServer(
            self,
            DesiredPose,
            'desired_pose_topic',
            self.desired_pose_callback)

        # Saccades action server is created
        self._action_server = ActionServer(
            self,
            Saccades2,
            'saccades_topic',
            self.saccades_callback)

        # Camera events Publisher is created
        self.events_publisher = self.create_publisher(CameraEvents, 'camera_events_topic', 10)
        timer_period = 1.0/60  # seconds
        self.timer = self.create_timer(timer_period, self.composed_callback)
        
        # Camera events Publisher is created
        self.transform_publisher = TransformBroadcaster(self)

        # Camera event image Publisher is created
        self.event_img_publisher = self.create_publisher(Image, 'camera_event_img_topic', 10)

        # Create MuJoCo environment
        init_pose = read_cfg()["saccade"]["start_pose"]
        init_pose_rad = np.array([np.deg2rad(x) for x in init_pose])
        err_limit = 0.1
        self.mj = EsimMujoco(init_pose_rad, err_limit)

        self.capture_events_enable = read_cfg()["operations"]["capture_events_enable"]
        self.save_events = read_cfg()["operations"]["save_events"]
        self.capture_frames_enable = read_cfg()["operations"]["capture_frames_enable"]
        self.save_frames = read_cfg()["operations"]["save_frames"]
        self.save_pose = read_cfg()["operations"]["save_pose"]
        self.save_path = read_cfg()["operations"]["save_path"]

        self.is_saccading = False

        self.get_logger().info('Initialized')

    def desired_pose_callback(self, goal_handle):
        self.get_logger().info('Executing goal: set_desired_pose...')

        des_pose = goal_handle.request.des_pose
        self.mj.set_des_pose(des_pose)

        # create action messages
        feedback_msg = DesiredPose.Feedback()
        result_msg = DesiredPose.Result()

        # run mj loop until the robot reaches the goal pose
        while not self.mj.is_position_reached():
            # run one mj loop and publish
            self.composed_callback()

            # publish feedback
            feedback_msg.feedback_pose_error = self.mj.pose_err().astype(np.float32)

            goal_handle.publish_feedback(feedback_msg)

        # robot reached the goal pose 
        goal_handle.succeed()
        result_msg.pose_error = feedback_msg.feedback_pose_error

        self.get_logger().info('Action finished! (set_desired_pose)')

        return result_msg

    def saccades_callback(self, goal_handle):
        saccade_type = read_cfg()["saccade"]["type"]
        if saccade_type == "circle":
            return self.saccades_callback_body(goal_handle, self.circular_saccades)
        elif saccade_type == "random_circle":
            return self.saccades_callback_body(goal_handle, self.random_circular_saccades)

    def circular_saccades(self, t, start_pose):
        if not self.is_saccading:
            self.is_saccading = True
        des_pos, des_vel = self.mj.circular_pose(t, start_pose)
        self.mj.set_des_pose(des_pos, des_vel)
    
    def random_circular_saccades(self, t, start_pose):
        if not self.is_saccading:
            self.is_saccading = True

        sample_frequency = read_cfg()["saccade"]["sample_frequency"]
        if t % (1.0/sample_frequency) < 0.005:
            self.mj.set_des_pose(self.mj.random_circular_pose(t, start_pose))

    def saccades_callback_body(self, goal_handle, saccade_func):
        self.get_logger().info('Executing goal: saccades...')

        # duration = goal_handle.request.duration
        duration = read_cfg()["saccade"]["duration"]

        # create action messages
        feedback_msg = Saccades2.Feedback()
        result_msg = Saccades2.Result()

        start_pose = self.mj.controller.fk()

        t_0 = self.mj.data.time
        t = 0
        # run mj loop until the robot reaches the goal pose
        while t < duration:
            # set goal pose
            saccade_func(t, start_pose)

            # run one mj loop and publish
            self.composed_callback()

            # publish feedback
            feedback_msg.time_left = t_0 + duration - self.mj.data.time
            goal_handle.publish_feedback(feedback_msg)

            t = self.mj.data.time - t_0

        # back to the start pose
        self.mj.set_des_pose(start_pose)
        while not self.mj.is_position_reached():
            # run one mj loop and publish
            self.composed_callback()

        # robot reached the goal pose 
        goal_handle.succeed()
        result_msg.time_spent = self.mj.data.time - t_0

        self.is_saccading = False
        self.get_logger().info('Action finished! (saccades)')
        return result_msg

    def composed_callback(self):
        self.timer_callback()
        self.transform_callback()

    def timer_callback(self):
        if self.is_saccading:
            raw_img, events_img, events = self.mj.loop(             
                    capture_events_enable=self.capture_events_enable,    
                    save_events=self.save_events,                        
                    capture_frames_enable=self.capture_frames_enable,    
                    save_frames=self.save_frames,   
                    save_pose=self.save_pose,                      
                    save_path=self.save_path                             
                    )
        else: 
            raw_img, events_img, events = self.mj.loop(             
                    capture_events_enable=False,    
                    save_events=False,                        
                    capture_frames_enable=False,    
                    save_frames=False,   
                    save_pose=False,                      
                    save_path=self.save_path                             
                    )

        events_msg = CameraEvents()
        if self.fill_msg_with_events(events_msg, events):
            self.events_publisher.publish(events_msg)

        event_img_msg = Image()
        if self.fill_msg_with_event_im(event_img_msg, events_img):
            self.event_img_publisher.publish(event_img_msg)

    def fill_msg_with_event_im(self, msg, event_img):
        if event_img is not None:
            H, W, _ = event_img.shape
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'mounted_camera'
            msg.height = H
            msg.width = W
            msg.encoding = "rgb8"
            msg.step = 3 * W
            event_img = np.where(event_img == 0, 255, event_img)
            msg.data = event_img.tobytes()
            
            return True
        else:
            return False
        

    def transform_callback(self):
        pos, quat = self.mj.get_camera_pose()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'mounted_camera'
        t.child_frame_id = 'world'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]
        t.transform.rotation.w = quat[0]

        self.transform_publisher.sendTransform(t)

    def fill_msg_with_events(self, msg, events):
        log_start = "Publishing: "
        log_amount = "amount of events-"
        if events is not None:
            # if there are some events
            log_input = f'{events["x"].shape[0]}'
            msg.x = events["x"].tolist()
            msg.y = events["y"].tolist()
            msg.t = events["t"].tolist()
            msg.p = events["p"].tolist()
            return True
        else:
            # if no events
            log_input = f'{0}'
            return False

def main(args=None):
    rclpy.init(args=args)

    controller_action_server = ImControllerActionServer()

    rclpy.spin(controller_action_server)


if __name__ == '__main__':
    main()