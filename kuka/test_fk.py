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
import time
import numpy as np
import yaml
from yaml.loader import SafeLoader
import sys,os
from utils.read_cfg import get_mjc_xml
from utils.quaternion import mat2Quat
from kinematics.fk1  import fk1 
from kinematics.fk2  import fk2 
from kinematics.fk3  import fk3 
from kinematics.fk4  import fk4 
from kinematics.fk5  import fk5 
from kinematics.fk6  import fk6 
from kinematics.fk7  import fk7 
from kinematics.G import G



model = mujoco.MjModel.from_xml_path(get_mjc_xml())
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

print(data.qpos)

t = data.time

ids = []
for i in range(7):
    ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fk{i+1}"))

while (True):
    viewer.render()

    for i, f in zip(range(7), [fk1 , fk2 , fk3 , fk4 , fk5 , fk6 , fk7 ]):
        pos, mat = f(data.qpos)
        model.body_pos[ids[i]], model.body_quat[ids[i]] = pos, mat2Quat(mat)
    
    x=100*np.sin(t)
    torque=np.ones(7)*x
    data.ctrl[:] = np.clip(torque, -300, 300)

    mujoco.mj_step(model, data)

    t = data.time