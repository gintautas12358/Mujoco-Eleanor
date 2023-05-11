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

import numpy as np
import gym
import cv2
import skimage.measure

from .vs_base import VSBase

class VSActivity(VSBase):

    def __init__(self, headless=False, render_every_frame=True, running_events=True, render_mode=None):
        super().__init__(headless, render_every_frame, running_events)

        y, x = self.viewer.winShape()
        self.ws = x, y
        self.winCenter = self.ws[0]/2, self.ws[1]/2 

        self.env_reset()

        # gym spaces
        position_ac = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(position_ac,), dtype=np.float32)

        position_ob = 3
        img_err = 1
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        

        self.goal_coord = self.winCenter

    def get_pose(self, action):
        pose = self.current_pose.copy()
        ac_position = action
        pose[:2] +=  ac_position * self.ac_position_scale 

        self.action = action
        return pose

    def observe_0(self): 
        # events
        fixedcamid = 1
        timestamp = self.data.time  
        out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        if out is not None:
            e_img, e, num_e = out
            self.img  = self.preprocessing(e_img)

    def observe(self): 
        # # events
        # fixedcamid = 1
        # timestamp = self.data.time  
        # out = self.viewer.capture_event(fixedcamid, timestamp, save_it=False, path=".")

        # if out is not None:
        #     e_img, e, num_e = out
        #     self.img  = self.preprocessing(e_img)

        err = self.dist_metric(self.img)

        pose = self.controller.fk()

        # observation = (pose[0] - self.current_pose[0]) / (self.ac_position_scale ), \
        #               (pose[1] - self.current_pose[1]) / (self.ac_position_scale ), \
        #               (pose[2] - self.current_pose[2]) / (self.ac_position_scale ), \
        #               err / 45.0
        
        observation = ((self.activity_coord[0] - self.winCenter[0]) / self.winCenter[0]), \
                      ((self.activity_coord[1] - self.winCenter[1]) / self.winCenter[1])
                        # err / 45.0

        return observation

    def get_reward(self):
        dx = np.linalg.norm(self.action - self.old_a)
        self.old_a = self.action.copy()
        err = self.dist_metric(self.img)

        # reward = -1 - dx
        reward = -1/0.04 + 1/(0.01*err+0.04) - 1*dx
        
        # if err < 3:
        #     reward = 1

        return reward, err


    def env_reset(self):    
        self.img = np.ones(self.ws) * 127
        self.activity_coord = self.ws[0]-1, self.ws[0]-1

    def change_to_shape(self, a):
        return a.flatten()

    def preprocessing(self, img):

        # size crop
        img = np.where(img == 50, 255, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # max_value = np.max(img)
        img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

        # maxpool
        # for i in range(1):
        #     img = skimage.measure.block_reduce(img, (2,2), np.max)

        # gray background
        img = np.where(img == 0, 127, img)
        img = np.where(img == 129, 0, img)

        # observe result. (debug camera view) 
        # cv2.imshow("resized image", img)
        # cv2.waitKey(0)

        return img

    def dist_metric(self, img):
        out = self.get_activity_coord(img)
        
        x, y = self.activity_coord
        if out is not None:
            x, y = out
            self.activity_coord = x, y 


        latent_img = np.ones(self.ws) * 0
        latent_img = self.box((int(x), int(y)), 20, latent_img)

        # cv2.imshow("coord image", latent_img.astype(np.uint8))
        # cv2.waitKey(0)

        v = np.array((x, y))

        g_out = self.goal_coord 
        g_v = np.array(g_out)

        err = np.linalg.norm(g_v - v)
        
        return err
    
    def box(self, c, r, img):
        val = 255
        x, y = c
        shape = img.shape
        py = shape[1]-1 if y + r > shape[1]-1 else y + r 
        ny = 0 if y - r < 0 else y - r
        px = shape[0]-1 if x + r > shape[0]-1 else  x + r
        nx = 0 if x - r < 0 else x - r


        img[nx, ny:py] = val
        img[px, ny:py] = val
        img[nx:px, ny] = val
        img[nx:px, py] = val


        img[x, y] = val

        return img

    def get_activity_coord(self, img):
        px, py = np.where(img == 0)
        nx, ny = np.where(img == 255)
        x = np.append(nx, px)
        y = np.append(ny, py)

        if x.size == 0:
            return None

        x_mean, y_mean, x_var, y_var = x.mean(), y.mean(), x.var(), y.var()

        return x_mean, y_mean
