import numpy as np
import torch
import pickle

import sys
import os
sys.path.append(os.getcwd())

import RL.popsan.core_cuda as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.

    with Running Mean and Var from hill-a/stable-baselines
    """

    def __init__(self, obs_dim, act_dim, device, size, clip_limit, norm_update_every=1000, path="."):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param size: buffer sizes
        :param clip_limit: limit for clip value
        :param norm_update_every: update freq
        """

        self.path = path
        self.device = device
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # Running z-score normalization parameters
        self.clip_limit = clip_limit
        self.norm_update_every = norm_update_every
        self.norm_update_batch = np.zeros(core.combined_shape(norm_update_every, obs_dim), dtype=np.float32)
        self.norm_update_count = 0
        self.norm_total_count = np.finfo(np.float32).eps.item()
        self.mean, self.var = np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32)

        # Save parameters
        with open(path + "/parameter_log_replay_buffer.txt", "w") as f:
            f.write("obs_dim " + str(obs_dim) + "\n")
            f.write("act_dim " + str(act_dim) + "\n")
            f.write("device " + str(device) + "\n")
            f.write("size " + str(size) + "\n")
            f.write("clip_limit " + str(clip_limit) + "\n")
            f.write("norm_update_every " + str(norm_update_every) + "\n")

    def store(self, obs, act, rew, next_obs, done):
        """
        Insert entry into memory
        :param obs: observation
        :param act: action
        :param rew: reward
        :param next_obs: observation after action
        :param done: if true then episode done
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # Update Mean and Variance
        # Have to at least update mean and variance once before training starts
        self.norm_update_batch[self.norm_update_count] = obs
        self.norm_update_count += 1
        if self.norm_update_count == self.norm_update_every:
            self.norm_update_count = 0
            batch_mean, batch_var = self.norm_update_batch.mean(axis=0), self.norm_update_batch.var(axis=0)
            tmp_total_count = self.norm_total_count + self.norm_update_every
            delta_mean = batch_mean - self.mean
            self.mean += delta_mean * (self.norm_update_every / tmp_total_count)
            m_a = self.var * self.norm_total_count
            m_b = batch_var * self.norm_update_every
            m_2 = m_a + m_b + np.square(delta_mean) * self.norm_total_count * self.norm_update_every / tmp_total_count
            self.var = m_2 / tmp_total_count
            self.norm_total_count = tmp_total_count

    def sample_batch(self, batch_size=32):
        """
        Sample batch from memory
        :param device: pytorch device
        :param batch_size: batch size
        :return: batch
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.normalize_obs(self.obs_buf[idxs]),
                     obs2=self.normalize_obs(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}

    def normalize_obs(self, obs):
        """
        Do z-score normalization on observation
        :param obs: observation
        :return: norm_obs
        """
        eps = np.finfo(np.float32).eps.item()
        norm_obs = np.clip((obs - self.mean) / np.sqrt(self.var + eps),
                           -self.clip_limit, self.clip_limit)
        return norm_obs
    
    def save(self, epoch):
        rb_path = self.path + '/' + "replay_buffer_e" + str(epoch) + ".p"
        pickle.dump(self, open(rb_path, "wb")) 
