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

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import os

# reletive folder path for ploting events
path = "test_esim_output/event_frames_and_events/events"

keys = ["x", "y", "t", "p"]
total_data = {k: np.array([]) for k in keys}

files = os.listdir(path)
files.sort()

for f in files:
    
    if f.endswith(".npz"):
        print(f)
        data = np.load(os.path.join(path, f))
        for k in keys:
            total_data[k] = np.concatenate((total_data[k], data[k]))

min_time, max_time = min(total_data["t"]), max(total_data["t"])
min_x, max_x = min(total_data["x"]), max(total_data["x"])
min_y, max_y = min(total_data["y"]), max(total_data["y"])

xs = (max_x - min_x) * 1e-0
ys = (max_y - min_y) * 1e-0
ts =  (max_time - min_time) * 1e-5
print(ts, xs, ys)

# Creating figure
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_box_aspect((ts, xs, ys))

e_pos = {key:total_data[key][np.where(total_data['p'] == 1)[0]] for key in total_data}
e_neg = {key:total_data[key][np.where(total_data['p'] == -1)[0]] for key in total_data}

# Creating plot
marker_size = 0.01
scatter1 = ax.scatter3D(e_pos["t"], e_pos["x"], e_pos["y"], s=marker_size, color = "red")
scatter2 = ax.scatter3D(e_neg["t"], e_neg["x"], e_neg["y"], s=marker_size, color = "green")
plt.title("Space-time event plot")
ax.set_xlabel('t', fontweight ='bold')
ax.set_ylabel('x', fontweight ='bold')
ax.set_zlabel('y', fontweight ='bold')
pos_patch = mpatches.Patch(color='red', label='neg events')
neg_patch = mpatches.Patch(color='green', label='pos events')
ax.legend(handles=[pos_patch, neg_patch])
ax.grid(True)
 
# show plot
plt.show()