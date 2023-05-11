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

import sys
import os
sys.path.append(os.getcwd())

import mujoco
import mujoco_viewer
import os
import numpy as np
# from controllers.NullSpaceViscoElasticCartesianPDController import NullSpaceViscoElasticCartesianPDController
from kuka.utils.read_cfg import get_mjc_xml


def test_run(controllerClass):

    model = mujoco.MjModel.from_xml_path(get_mjc_xml())
    data = mujoco.MjData(model)


    viewer = mujoco_viewer.MujocoViewer(model, data)
    controller = controllerClass(model, data)

    # init first position
    data.qpos = np.array([-1.63042613, -0.95663209,  0.08660753,  1.55094155, -0.11856101, -0.64075765, -1.48511576])

    center_pose = np.array([0.0, 0.58, 0.05, 3.14, 0, 0])

    rand = np.array([0,0])
    count = 0
    while (True):
        viewer.render()

        # if t%0.3 < 0.004:
        if count%10 == 0:
            rand = (np.random.rand(2) - 0.5) / 3
            print(rand)

        # pose = center_pose + np.array([rand[0], rand[1]+0.1, 0, 0, 0, 0])
        # off = 0.05
        off = 0.15
        # pose = center_pose + np.array([0, -off, 0, 0, 0, 0])
        pose = center_pose 

        # print(controller.sim_data.qpos)
        controller.set_action(pose)
        torque = controller.get_torque()


        mujoco.mj_step(model, data)
        t = data.time

        count += 1
