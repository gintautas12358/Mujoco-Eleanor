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
import time
import numpy as np
# from controllers.NullSpaceViscoElasticCartesianPDController import NullSpaceViscoElasticCartesianPDController
from kuka.utils.read_cfg import get_mjc_xml, get_jposes, get_cposes, get_cerr_lim, get_jerr_lim
from kuka.utils.kinematics import current_ee_position
from kuka.controllers.PathController import PathController
from kuka.pathPlanner.CubicPathPlanner import CubicPolyPathGenerator


def test_run(viapoints, controllerClass):

    model = mujoco.MjModel.from_xml_path(get_mjc_xml())
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    pathController = PathController(controllerClass(model, data), CubicPolyPathGenerator(6))


    # init first position
    jposes = get_jposes()
    cposes = get_cposes()

    data.qpos = jposes["HOME_Q"]


    viapoint = viapoints.pop()
    poses = {}
    poses.update(jposes)
    poses.update(cposes)

    pathController.initStartPose(poses[viapoint])

    t = data.time
    err_limit = get_cerr_lim()

    while (True):
        viewer.render()

        pathController.set_time(t)
        torque = pathController.get_torque()
        
        mujoco.mj_step(model, data)
        t = data.time

        # viapoint change when the last viapoint is reached
        error = (pathController.error())
        err_norm = np.linalg.norm(error)
        if err_limit > err_norm  and viapoints:
            viapoint = viapoints.pop()
            tt = data.time
            pathController.newPathTo(poses[viapoint], tt)


        print("norm error", err_norm)
        # print("error", error)
        # print("Current viapoint", viapoint)
        # print("Joint Values:", data.qpos)
