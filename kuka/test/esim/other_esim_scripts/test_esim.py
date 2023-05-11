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
import glfw
import sys
from threading import Lock
import numpy as np
import time
import imageio
import esim_torch
import torch
import cv2

save_path = "/home/palinauskas/Documents/mujoco-eleanor/img"
save_path_original = save_path + "/original/seq0/imgs"
save_path_subtracted = save_path + "/subtracted/seq0/imgs"
save_path_events = save_path + "/events2/seq0/imgs"

esim = esim_torch.ESIM(0.1, 0.5, 0.0)


def viz_events(events, resolution):

    

    pos_events = events[events[:,-1]==1]
    neg_events = events[events[:,-1]==-1]

    limit = resolution[0]

    # pos_events = events[events[:,1]<limit]
    # neg_events = events[events[:,1]<limit]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    

    # print(pos_events)
    # print(pos_events[:,0])
    # print(pos_events[:,1])
    # print(resolution[1])
    # print(pos_events[:,1]*resolution[1])
    # print(pos_events[:,0]+pos_events[:,1]*resolution[1])
    # print(pos_events[:,-1]**2)
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

def t2e(tensor_dic):
    if not tensor_dic and not tensor_dic.values():
        raise ValueError

    np_dic = {k: v.numpy() for k, v in tensor_dic.items() }
    # print(np_dic)

    size = list(np_dic.values())[0].size
    # print(size)

    lis = []
    for i in range(size):
        event = []
        for v in np_dic.values():
            event.append(v[i])
        lis.append(event)

    # print(lis)

    return np.array(lis) 

def func(img, t):
    log_image = np.log(img.astype("float32") / 255 + 1e-5)
    log_image = torch.from_numpy(log_image).cuda()


    # timestamps = np.genfromtxt(indir, dtype="float64")
    timestamps = np.array([t])
    timestamps_ns = (timestamps * 1e9).astype("int64")
    timestamps_ns = torch.from_numpy(timestamps_ns).cuda()

    print("log_image", log_image.shape)
    sub_events = esim.forward(log_image, timestamps_ns[0])

    # for the first image, no events are generated, so this needs to be skipped
    if sub_events is None:
        return None

    sub_events = {k: v.cpu() for k, v in sub_events.items()}    

    # do something with the events
    H, W = img.shape
    e = t2e(sub_events)
    im = viz_events(e, [H, W])
    if im is None:
        return None

   

    return im
    

def main(i, t):

    path = save_path_events
    img_original = cv2.imread(save_path_original + "/%08d.png" % i, cv2.IMREAD_GRAYSCALE)

    im = func(img_original, t)

    if im is None:
        return None

    path += "/%08d.png" % i
    imageio.imwrite(path, im)

    return 1

ts = [x / 100.0 for x in range(0,1000,2)]

for i in range(10):
    a = main(i, ts[i])
    if not a:
        print("out")