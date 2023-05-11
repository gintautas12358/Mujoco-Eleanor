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


import os
import esim_py

import esim_torch

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from os.path import join

output_folder = "/home/palinauskas/Documents/mujoco-eleanor/img/event_img/"
image_folder = "/home/palinauskas/Documents/mujoco-eleanor/img/upsampled/seq0/imgs/"
timestamps_file = "/home/palinauskas/Documents/mujoco-eleanor/img/upsampled/seq0/timestamps.txt"



def viz_events(events, resolution):
    pos_events = events[events[:,-1]==1]
    neg_events = events[events[:,-1]==-1]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), pos_events[:,-1]**2)
    np.add.at(image_neg, (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), neg_events[:,-1]**2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution), 
            image_neg.reshape(resolution), 
            np.zeros(resolution, dtype="uint8") 
        ], -1
    ) * 50

    return image_rgb    


Cp, Cn = 0.5, 0.1
refractory_period = 1e-4
log_eps = 1e-3
use_log = True
H, W = 704, 1280



esim = esim_py.EventSimulator(Cp, 
                              Cn, 
                              refractory_period, 
                              log_eps, 
                              use_log)

num_events_plot = 30000
esim.setParameters(Cp, Cn, refractory_period, log_eps, use_log)

images = [image_folder + x for x in os.listdir(image_folder)]
timestamps = []
with open(timestamps_file, "r") as f:
    timestamps = f.read().split("\n")


count = 0

events = esim.generateFromFolder(image_folder, timestamps_file)

print(events.shape)
print(events[:3000].shape)
size, _ = events.shape
amount = 3000

for i, ii in enumerate(range(0, size, amount)):
    print(i, ii)
    image_rgb = viz_events(events[ii:ii+amount], [H, W])
    plt.imsave(output_folder + "%08d.png" % i, image_rgb)


