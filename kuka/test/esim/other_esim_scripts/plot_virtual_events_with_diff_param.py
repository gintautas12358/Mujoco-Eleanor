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


import esim_py

import matplotlib.pyplot as plt
import numpy as np


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
    ) * 255

    return image_rgb    


Cp, Cn = 0.1, 0.1
refractory_period = 100
log_eps = 1e-3
use_log = True
H, W = 704, 1280

image_folder = "test_esim_output/raw_images/raw_imgs"
timestamps_file = "test_esim_output/raw_images/timestamps.txt"


esim = esim_py.EventSimulator(Cp, 
                              Cn, 
                              refractory_period, 
                              log_eps, 
                              use_log)

fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(6,6))

contrast_thresholds_pos = [0.1, 0.4, 0.9, 1.2, 1.5]
contrast_thresholds_neg = [0.1, 0.4, 0.9, 1.2, 1.5]
refractory_periods = [1e-2, 1e-1, 1, 10, 100]

num_events_plot = 100

for i, Cp in enumerate(contrast_thresholds_pos):
    for j, Cn in enumerate(contrast_thresholds_neg):
        esim.setParameters(Cp, Cn, refractory_period, log_eps, use_log)
        events = esim.generateFromFolder(image_folder, timestamps_file)

        image_rgb = viz_events(events[:num_events_plot], [H, W])
        
        offset = 30
        mH = H // 2 - 150
        mW = W // 2
        print(image_rgb.shape)
        ax[i,j].imshow(image_rgb[mH-offset:mH+offset, mW-offset:mW+offset, :])
        ax[i,j].axis('off')
        ax[i,j].set_title("Cp=%s Cn=%s" % (Cp, Cn))

plt.show()
