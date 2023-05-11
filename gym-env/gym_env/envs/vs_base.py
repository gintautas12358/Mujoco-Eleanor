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
import os
import time
import numpy as np
import gym
import time

import sys
import os
sys.path.append(os.getcwd())

from kuka.controllers.NullSpaceViscoElasticCartesianPDController import NullSpaceViscoElasticCartesianPDController

class VSBase(gym.Env):

    def __init__(self, headless=False, render_every_frame=True, running_events=True):
        
        self.target_object = "target"
        xml_file_name = "sim_vga.xml"
        cwd = os.getcwd()
        xml_path = os.path.join(cwd, "kuka", "envs", "assets", xml_file_name)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # init first position
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, headless=headless, render_every_frame=render_every_frame, running_events=running_events)
        self.controller = NullSpaceViscoElasticCartesianPDController(self.model, self.data) 

        # inti esim
        self.viewer.init_esim(contrast_threshold_negative=0.9, contrast_threshold_positive=0.9, refractory_period_ns=100)

        # in radians
        self.init_pose = np.array([-1.63042613, -0.95663209,  0.08660753,  1.55094155, -0.11856101, -0.64075765, -1.48511576])


        self.data.qpos = self.init_pose

        self.init_qvel = self.data.qvel.copy()
        self.init_act = self.data.act.copy()
        self.init_qacc_warmstart = self.data.qacc_warmstart.copy()

        self.init_ctrl = self.data.ctrl.copy()
        self.init_qfrc_applied = self.data.qfrc_applied.copy()
        self.init_xfrc_applied = self.data.xfrc_applied.copy()
        self.init_qacc = self.data.qacc.copy()
        self.init_act_dot = self.data.act_dot.copy()

        self.init_time = self.data.time


        self.err = None


        # init hole position
        self.init_hole_pose = self.get_hole_pose()
        # self.max_rand_offset = np.array([0.16, 0.1, 0])
        self.max_rand_offset = 0.06

        np.random.seed(int(time.time()))

        self.target_off = self.randomize_hole_position()

        self.current_pose = np.array([0.0, 0.58, 0.05, 3.14, 0, 0])

        self.current_step = 0
        self.old_a = 0

        # self.ac_position_scale = self.max_rand_offset[:2] + np.array([0.01, 0.01])
        self.ac_position_scale = 0.3

        self.ac_orientation_scale = 0.1
        
    def get_pose(self, action):
        pass

    def observe(self): 
        pass

    def get_reward(self):
        pass

    def env_reset(self):    
        pass

    def observe_0(self): 
        pass


    def step(self, action):

        # print("hole pos", self.get_hole_pose())
        
        # to check the cartesian position of the initialised joint position
        # print(self.controller.fk())

        # init step gym
        reward = 0
        done = False
        self.current_step += 1

        # ======== apply action ==========
        action = self.change_to_shape(action)

        pose = self.get_pose(action)
        self.controller.set_action(pose)

        for i in range(5):
            self.viewer.make_current()
            self.viewer.render(overlay_on=False)
            torque = self.controller.get_torque()
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
            self.observe_0()

        # ======== observation ==========
        observation = self.observe()

        # ======== reward ==========
        reward, err = self.get_reward()

        # ======== done condition ==========
        off = 0.1
        conditions = np.array([
                    self.controller.fk()[0] < self.current_pose[0] + self.target_off[0] - off,
                    self.controller.fk()[0] > self.current_pose[0] + self.target_off[0] + off,
                    self.controller.fk()[1] < self.current_pose[1] + self.target_off[1] - off,
                    self.controller.fk()[1] > self.current_pose[1] + self.target_off[1] + off,
                    ])
        if np.any(conditions):
            # print(conditions, self.controller.fk()[:3])
            reward = -1000
            done = True

        self.err = err
        # print(err)
        if err < 20:
            reward = 1000
            print("reached")
            done = True

        info = {}
        return observation, reward, done, False, info

    def reset(self):

        self.current_step = 0
        self.old_a = 0

        self.data.qpos = self.init_pose.copy()
        self.data.qvel = self.init_qvel.copy()
        self.data.act = self.init_act.copy()
        self.data.qacc_warmstart = self.init_qacc_warmstart.copy()

        self.data.ctrl = self.init_ctrl.copy()
        self.data.qfrc_applied = self.init_qfrc_applied.copy()
        self.data.xfrc_applied = self.init_xfrc_applied.copy()
        self.data.qacc = self.init_qacc.copy()
        self.data.act_dot = self.init_act_dot.copy()

        self.data.time = self.init_time


        self.viewer.make_current()
        self.viewer.render(overlay_on=False)

        # inti esim
        self.viewer.init_esim(contrast_threshold_negative=0.9, contrast_threshold_positive=0.9, refractory_period_ns=100)

        self.err = None

        self.env_reset()

        pose = self.current_pose
        self.controller.set_action(pose)

        self.target_off = self.randomize_hole_position()

        # ======== observation ==========

        observation = self.observe()

        return observation, {}

    def randomize_hole_position(self):
        offset_pos = 2*(np.random.rand(3) - 0.5) * self.max_rand_offset
        offset_pos[2] = 0.0                     # no offset in z
        # offset_pos = np.array([self.max_rand_offset, self.max_rand_offset, 0])
        self.set_hole_pose(offset_pos)
        return offset_pos

    def close(self):
        self.viewer.close()

    def change_to_shape(self, a):
        return a.flatten()

    def render_frame(self):
        return self.viewer.capture_frame(1, self.data.time)

    def get_hole_pose(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_object)
        return self.model.body_pos[body_id].copy()

    def set_hole_pose(self, offset_pos):
        obj_names = [self.target_object]

        # get body offset
        for name in obj_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.model.body_pos[body_id][:3] =  self.init_hole_pose[:3] + offset_pos         


