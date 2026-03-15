"""
Experience buffer implementation
References:
- rl_games/common/experience.py
- rl_games/common/datasets.py
- penspin/algo/ppo/experience.py

Notes:
- horizon_length = number of transitions to collect per env
    - not the same as max episode length
"""

import numpy as np

import gym
from gym.spaces import Box, Dict

from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Dict as GymnasiumDict

import torch
from torch.utils.data import Dataset

numpy_to_torch_dtype_dict = {
    np.dtype('bool')       : torch.bool,
    np.dtype('uint8')      : torch.uint8,
    np.dtype('int8')       : torch.int8,
    np.dtype('int16')      : torch.int16,
    np.dtype('int32')      : torch.int32,
    np.dtype('int64')      : torch.int64,
    np.dtype('float16')    : torch.float16,
    np.dtype('float64')    : torch.float32,
    np.dtype('float32')    : torch.float32,
    #np.dtype('float64')    : torch.float64,
    np.dtype('complex64')  : torch.complex64,
    np.dtype('complex128') : torch.complex128,
}

def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class ExperienceBuffer(Dataset):
    def __init__(self, num_envs, horizon_length, batch_size, minibatch_size, is_rnn, seq_length, action_space, state_space, observation_space, device, normalize_advantage=True):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.act_dim = action_space.shape[0]
        self.action_space = action_space
        self.state_space = state_space
        self.observation_space = observation_space
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.normalize_advantage = normalize_advantage

        self.obs_base_shape = (self.transitions_per_env, self.num_envs)

        if self.is_rnn:
            if self.transitions_per_env % self.seq_length != 0:
                raise ValueError("horizon_length must be divisible by seq_length for RNNs.")
            if self.minibatch_size % self.seq_length != 0:
                raise ValueError("minibatch_size must be divisible by seq_length for RNNs.")
            self.seqs_per_env = self.transitions_per_env // self.seq_length
            self.num_games_total = (self.num_envs * self.seqs_per_env)
            self.num_games_batch = self.minibatch_size // self.seq_length

        self.buffer_dict = {
            'states': self._create_tensor_from_space(self.state_space, self.obs_base_shape),
            'obses': self._create_tensor_from_space(self.observation_space, self.obs_base_shape),

            'rewards': self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(1,)), self.obs_base_shape),
            'values': self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(1,)), self.obs_base_shape),
            'neglogpacs': self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(1,)), self.obs_base_shape),
            'dones': self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(1,)), self.obs_base_shape),

            'actions': self._create_tensor_from_space(self.action_space, self.obs_base_shape),
            'mus': self._create_tensor_from_space(self.action_space, self.obs_base_shape),
            'sigmas': self._create_tensor_from_space(self.action_space, self.obs_base_shape),
            'returns': self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(1,)), self.obs_base_shape),
        }

    def __len__(self):
        return self.length

    # this is called only after prepare_training(), which sets self.data_dict
    def __getitem__(self, idx):
        if self.is_rnn:
            gstart = idx * self.num_games_batch
            gend = (idx + 1) * self.num_games_batch
            start = gstart * self.seq_length
            end = gend * self.seq_length
            self.last_range = (start, end)
            input_dict = {}
            for k, v in self.data_dict.items():
                if k == 'rnn_states':
                    continue
                if isinstance(v, dict):
                    input_dict[k] = {kk: vv[start:end] for kk, vv in v.items()}
                else:
                    input_dict[k] = v[start:end]
            input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in self.data_dict['rnn_states']]
            return input_dict
        else:
            start = idx * self.minibatch_size
            end = (idx + 1) * self.minibatch_size
            self.last_range = (start, end)
            input_dict = {}
            for k, v in self.data_dict.items():
                if isinstance(v, dict):
                    input_dict[k] = {kk: vv[start:end] for kk, vv in v.items()}
                else:
                    input_dict[k] = v[start:end]

        return input_dict

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    # index is timestep
    def update_data(self, name, index, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.buffer_dict[name][k][index, :] = v
        else:
            self.buffer_dict[name][index, :] = value

    def compute_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.buffer_dict['rewards'])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.buffer_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.buffer_dict['dones'].float()[t] # this value is 1 if the episode is not done
            # Value bootstrapping. We use the value of the next state to bootstrap the value of the current state.
            delta = self.buffer_dict['rewards'][t] + gamma * next_values * next_nonterminal - self.buffer_dict['values'][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.buffer_dict['returns'][t, :] = mb_advs[t] + self.buffer_dict['values'][t]

    # Same this as Line 816 in rl_games/common/a2c_common.py
    # transform_op flattens the first two axes, which are the number of transitions per env and the number of envs
    # into one axis, which is the number of transitions
    def prepare_training(self):
        self.data_dict = {}
        for k, v in self.buffer_dict.items():
            if isinstance(v, dict):
                self.data_dict[k] = {kk: transform_op(v[kk]) for kk in v}
            else:
                self.data_dict[k] = transform_op(v)
        advantages = self.data_dict['returns'] - self.data_dict['values']
        if self.normalize_advantage:
            # Normalize advantages
            self.data_dict['advantages'] = ((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        else:
            self.data_dict['advantages'] = advantages
        return self.data_dict

    # Converting gym.spaces to a dict of tensors
    def _create_tensor_from_space(self, space, base_shape):   
        if isinstance(space, Box) or isinstance(space, GymnasiumBox):
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)

        if isinstance(space, Dict) or isinstance(space, GymnasiumDict):
            t_dict = {}
            for k, v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict