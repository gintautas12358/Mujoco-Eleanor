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

import sys
import os
sys.path.append(os.getcwd())

from RL.popsan.util import simEpisode
from test_scripts import get_action, setup


env_name = "InvertedPendulum-v4"
param_path = "params/pop_sac_InvertedPendulum-v4_0/model_e5.pt"
rb_param_path = "params/pop_sac_InvertedPendulum-v4_0/replay_buffer_e5.p"

num_test_episodes = 10
max_ep_len = 200

env, ac, replay_buffer, device =  setup(env_name, param_path, rb_param_path)

def test_agent(env):
        ###
        # compuate the return mean test reward
        ###
        print("testing env...")
        test_reward_sum = 0
        for j in tqdm( range(num_test_episodes) ):
            ep_ret, _ = simEpisode(env, max_ep_len, action_func=lambda o: get_action(replay_buffer.normalize_obs(o), ac, device, True))
            test_reward_sum += ep_ret

        print("done testing env")
        average_reward = test_reward_sum / num_test_episodes
        print("Average reward:", average_reward)
        return average_reward

test_agent(env)
