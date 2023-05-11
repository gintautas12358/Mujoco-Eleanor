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


import yaml
from yaml.loader import SafeLoader
import numpy as np


def get_jposes(cfg):
    jposes_raw = cfg["poses"]["joint"]
    jposes = {}
    for pose_k, pose_val in jposes_raw.items():
        jposes[pose_k] = np.array([np.deg2rad(x) for x in pose_val])

    return jposes

def get_cposes(cfg):
    cposes_raw = cfg["poses"]["cartesian"]
    cposes = {}
    for pose_k, pose_val in cposes_raw.items():
        cposes[pose_k] = np.array(pose_val[:3] + [np.deg2rad(x) for x in pose_val[3:]])

    return cposes


def read_cfg():
    yaml_path = "src/mujoco_egen/mujoco_egen/cfg/cfg.yaml"
    cfg = None
    with open(yaml_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)

    return cfg