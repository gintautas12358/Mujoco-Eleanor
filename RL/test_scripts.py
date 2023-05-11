
import torch
import gym
import gym_env
import math
import pickle
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import sys
import os
sys.path.append(os.getcwd())

from RL.popsan.SpikeActorDeepCritic import SpikeActorDeepCritic


def setup(env_name, param_path, rb_param_path):

    ac_kwargs = dict(hidden_sizes=[256, 256],
                        encoder_pop_dim=10,
                        decoder_pop_dim=10,
                        mean_range=(-3, 3),
                        std=math.sqrt(0.15),
                        spike_ts=5,
                        device=torch.device('cuda'))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name, render_mode="rgb_array")

    ac = SpikeActorDeepCritic(env.observation_space, env.action_space, **ac_kwargs)
    ac.popsan.load_state_dict(torch.load(param_path))
    ac.to(device)

    f = open(rb_param_path, "rb")
    replay_buffer = pickle.load(f)        

    return env, ac, replay_buffer, device


def get_action(o, ac, device, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1,
                      deterministic)