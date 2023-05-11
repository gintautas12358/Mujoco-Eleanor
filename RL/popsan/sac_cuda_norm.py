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

# from copy import deepcopy
# import itertools
import numpy as np
import torch
# import torch.nn as nn
# from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import gym_env
from tqdm import tqdm
# import pickle

import sys
import os
sys.path.append(os.getcwd())

from RL.popsan.replay_buffer_norm import ReplayBuffer
from RL.popsan.util import simEpisode
from RL.popsan.PopsanTrainer import PopsanTrainer


class SpikeSAC():
    def __init__(self, env, trainer, replay_buffer, 
                 max_ep_len=15, start_steps=100, steps_per_epoch=100, epochs=10, batch_size=100, num_test_episodes=10, save_freq=2,
                path=".", tb_comment="") -> None:  

        self.env = env
        self.trainer = trainer
        self.replay_buffer = replay_buffer        
        self.max_ep_len = max_ep_len
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_test_episodes = num_test_episodes
        self.save_freq = save_freq
        self.path = path
        self.tb_comment = tb_comment

        self.writer = SummaryWriter(comment="_" + tb_comment)

        # Save parameters
        with open(path + "/parameter_log_ss.txt", "w") as f:
            f.write("env " + str(env) + "\n")
            f.write("trainer " + str(trainer) + "\n")
            f.write("replay_buffer " + str(replay_buffer) + "\n")
            f.write("max_ep_len " + str(max_ep_len) + "\n")
            f.write("start_steps " + str(start_steps) + "\n")
            f.write("steps_per_epoch " + str(steps_per_epoch) + "\n")
            f.write("epochs " + str(epochs) + "\n")
            f.write("batch_size " + str(batch_size) + "\n")
            f.write("num_test_episodes " + str(num_test_episodes) + "\n")
            f.write("save_freq " + str(save_freq) + "\n")
            f.write("path " + str(path) + "\n")
            f.write("tb_comment " + str(tb_comment) + "\n")
        
    def test_agent(self):
        test_reward_sum = 0
        for j in tqdm( range(self.num_test_episodes) ):
            ep_ret, _ = simEpisode(self.env, self.max_ep_len, action_func=lambda o: self.trainer.get_action(self.replay_buffer.normalize_obs(o), True))
            test_reward_sum += ep_ret
        return test_reward_sum / self.num_test_episodes
    
    def exploration(self):
        print("Exploration in progress...")
        steps = 0
        while (steps < self.start_steps):
            _, ep_len = simEpisode(self.env, self.max_ep_len, action_func=lambda o: self.env.action_space.sample(), enableStore=True, replay_buffer=self.replay_buffer)
            steps += ep_len
        print("Exploration done")

    def train(self):
        for epoch in tqdm( range(self.epochs), desc ="Training progress" ):
            steps = 0
            while (steps < self.steps_per_epoch):
                ep_ret, ep_len = simEpisode(self.env, self.max_ep_len, action_func=lambda o: self.trainer.get_action(self.replay_buffer.normalize_obs(o)), enableStore=True, replay_buffer=self.replay_buffer)
                steps += ep_len

                self.writer.add_scalar(self.tb_comment + '/Train-Reward', ep_ret, epoch*self.steps_per_epoch + steps)
            
            self.popsanUpdateHandling()

            test_mean_reward = self.test_agent()
            self.writer.add_scalar(self.tb_comment + '/Test-Mean-Reward', test_mean_reward, epoch + 1)
            print("Model: ", self.path, " Epochs: ", epoch + 1, " Mean Reward: ", test_mean_reward)

            if self.isSaveRequired(epoch+1):
                self.saveModels(epoch+1)
    


    def popsanUpdateHandling(self):
        for j in range(self.steps_per_epoch):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.trainer.update(data=batch)
    
    def isSaveRequired(self, epoch):
        return (epoch % self.save_freq == 0) or (epoch == self.epochs)
    
    def saveModels(self, epoch):
        self.trainer.save(epoch)
        self.replay_buffer.save(epoch)

        # self.trainer.printStatistics()
        print("Weights saved in ", self.path)


    def run(self):
        self.exploration()
        self.train()
        self.saveModels(self.epochs)


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='InvertedPendulum-v4')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--start_model_idx', type=int, default=10)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--server', type=bool, default=False)


    args = parser.parse_args()

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=torch.device('cuda'))
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = num * 10
        torch.manual_seed(seed)
        np.random.seed(seed)

        replay_size = int(1e6)
        norm_clip_limit = 3
        steps_per_epoch = 1000
        norm_update = steps_per_epoch

        if args.server:
            env = gym.make(args.env, headless=True)
        else:
            env = gym.make(args.env)

        obs_dim, act_dim = env.observation_space, env.action_space

        model_name = "pop_sac_" + args.env + "_" + str(num)
        path = "./params/" + model_name
        try:
            os.makedirs(path)
            print("Directory ", path, " Created")
        except FileExistsError:
            print("Directory ", path, " already exists")

        trainer = PopsanTrainer(obs_dim, act_dim, device, ac_kwargs=AC_KWARGS, path=path)

        replay_buffer = ReplayBuffer(obs_dim.shape, act_dim.shape[0], device, size=replay_size,
                                    clip_limit=norm_clip_limit, norm_update_every=steps_per_epoch, path=path)
        
        COMMMENT = ""
        ss = SpikeSAC(env, trainer, replay_buffer, steps_per_epoch=steps_per_epoch, max_ep_len=30, epochs=args.epochs, num_test_episodes=300, tb_comment=model_name+COMMMENT, save_freq=25, path=path)

        ss.run()

