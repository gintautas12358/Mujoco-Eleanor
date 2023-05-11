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

from tqdm import tqdm
import cv2

import sys
import os
sys.path.append(os.getcwd())

from RL.popsan.util import simEpisode
from test_scripts import get_action, setup


env_name = "vs_activity"
param_path = "params/pop_sac_vs_activity_0/model_e50.pt"
rb_param_path = "params/pop_sac_vs_activity_0/replay_buffer_e50.p"

num_test_episodes = 1
max_ep_len = 30

env, ac, replay_buffer, device =  setup(env_name, param_path, rb_param_path)

def recordedAction(env, video_recorder, o):
    frame = env.render_frame()
    video_recorder.write(cv2.flip(frame, 0))
    # Take deterministic actions at test time 
    return get_action(replay_buffer.normalize_obs(o), ac, device, True)

def render_agent(env):
        ###
        # compuate the return mean test reward
        ###
        try:
            os.mkdir("./clip")
            print("Directory clip Created")
        except FileExistsError:
            print("Directory clip already exists")

        print("rendering env...")

        # video_recorder = VideoRecorder(env, f"clip/{env_name}_clip.mp4", enabled=True)
        video_recorder = cv2.VideoWriter(f"clip/{env_name}_clip.avi", 0, 60, (64,64))

        test_reward_sum = 0
        for j in tqdm( range(num_test_episodes) ):
            ep_ret, _ = simEpisode(env, max_ep_len, action_func=lambda o: recordedAction(env, video_recorder, o))
            test_reward_sum += ep_ret

        video_recorder.release()

        print("done rendering env")
        return test_reward_sum / num_test_episodes

render_agent(env)
