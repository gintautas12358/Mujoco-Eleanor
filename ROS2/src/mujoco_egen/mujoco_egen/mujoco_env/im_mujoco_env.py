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


import mujoco
import mujoco_viewer
import os, sys
import time
import numpy as np

from ..utils.read_cfg import read_cfg

import sys
import os
sys.path.append(os.getcwd() + "/..")

from kuka.utils.quaternion import identity_quat, subQuat, quatAdd, mat2Quat, quat2Vel, quat2Mat, quat2eul
from kuka.controllers.NullSpaceViscoElasticCartesianPDController import NullSpaceViscoElasticCartesianPDController

class EsimMujoco:

    def __init__(self, init_pose, err_limit) -> None:

        self.camera_id = 1 # 1 - for mounted camera, 0 - for floating camera
        self.overlay_on = False 

        # self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(read_cfg()["mujoco_model_xml"])
        self.data = mujoco.MjData(self.model)

        stiffness = read_cfg()["motion_controller"]["stiffness"]
        damping = read_cfg()["motion_controller"]["damping"]
        null_space_damping = read_cfg()["motion_controller"]["null_space_damping"]
        null_space_stiffness = read_cfg()["motion_controller"]["null_space_stiffness"]
        self.controller = NullSpaceViscoElasticCartesianPDController(self.model, self.data)

        # init first position
        print("init_pose", init_pose)
        self.data.qpos = init_pose
        mujoco.mj_forward(self.model, self.data)

        self.err_limit = err_limit # error before accepting the pose

        pose = self.controller.fk()
        self.des_pose = pose

        self.controller.set_action(self.des_pose)


        print("## Init pose", self.des_pose)


        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=False, render_every_frame=True, running_events=False)
        cp = read_cfg()["esim"]["Cp"]
        cn = read_cfg()["esim"]["Cn"]
        rp = read_cfg()["esim"]["refractory_period"]
        self.viewer.init_esim(contrast_threshold_negative=cp, contrast_threshold_positive=cn, refractory_period_ns=rp)

    def loop(self, capture_events_enable=False, save_events=False, capture_frames_enable=False, save_frames=False, save_pose=False, save_path="/temp"):
        self.viewer.render(overlay_on=self.overlay_on)

        # mounted view
        self.viewer.change_camera(self.camera_id)

        # first output
        raw_img = None
        if capture_frames_enable:
            raw_img = self.viewer.capture_frame(self.camera_id, self.data.time, save_it=save_frames, path=save_path)

        # second output
        # generate events
        out = None
        if capture_events_enable:
            timestamp = self.data.time         
            out = self.viewer.capture_event(self.camera_id, timestamp, save_it=save_events, path=save_path)

            
            
        if out is not None:
            events_img, events, num_events = out
            # save current camera pose
            if save_pose:
                image_idx = self.viewer._image_idx
                self.write_camera_pose(image_idx, save_path=save_path)
        else:
            events_img, events = None, None
 
        
        
        # set goal pose
        self.controller.set_action(self.des_pose)

        torque = self.controller.get_torque()
        # self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        return raw_img, events_img, events, 

    def set_des_pose(self, des_pose, des_vel=np.array([0,0,0,0,0,0])):
        self.des_pose = des_pose
        self.des_vel = des_vel
        self.controller.set_action(self.des_pose)

    def position_err(self):
        return np.linalg.norm(self.controller.pose_error()[:3])
    
    def pose_err(self):
        return self.controller.pose_error() 

    def is_position_reached(self):
        if self.err_limit < self.position_err():
            return False
        else:
            return True

    def get_camera_pose(self):
        # get camera offset
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "mounted_camera")
        pos_offset = self.model.cam_pos0[cam_id]
        mat_offset = self.model.cam_mat0[cam_id]
        quat_offset = mat2Quat(np.array(mat_offset))

        # get end-effector pose
        pos, quat = self.controller._fk()

        # get camera pose
        cam_pos = pos + pos_offset
        cam_quat = quatAdd(quat, quat2Vel(quat_offset))

        return cam_pos, cam_quat

    def write_camera_pose(self, image_idx, save_path="/temp"):
        pose = self.get_camera_pose()
        
        with open(save_path + "/positions.txt", "a") as f:
            f.write(str(image_idx) +  "   " + np.array2string(pose[0]) + np.array2string(pose[1]) + "\n")

        
    def circular_pose(self, t, start_pose):
        r = read_cfg()["saccade"]["radius"]
        w = read_cfg()["saccade"]["circular_speed"]

        offset = np.zeros(3)
        offset[0] = r * np.sin(w*t)
        offset[1] = r * np.cos(w*t)
        offset[2] = 0

        speed = np.zeros(6)
        speed[0] = w * r * np.cos(w*t)
        speed[1] = - w * r * np.sin(w*t)
        speed[2] = 0

        return self.offset_pose(start_pose, offset), speed
        
    def random_circular_pose(self, t, start_pose):
        rng = np.random.default_rng(int(time.time()))
        tt = rng.random() * 2 * np.pi 
        r = read_cfg()["saccade"]["radius"]
        w = read_cfg()["saccade"]["circular_speed"]

        offset = np.zeros(3)
        offset[0] = r * np.sin(w*tt)
        offset[1] = r * np.cos(w*tt)
        offset[2] = 0

        return self.offset_pose(start_pose, offset)

    def offset_pose(self, start_pose, offset):
        pose = start_pose.copy()
        pose[:3] += offset
        
        return pose
