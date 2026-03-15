"""
Asymmetric Actor-Critic Model

- actor_obs_config: dict of obs configs for the actor
- critic_obs_config: dict of obs configs for the critic
- [ ] we could decouple critic_obs_config into critic_obs_config and critic_priv_obs_config

"""


import torch
import torch.nn as nn

import numpy as np

from rl.models.network_generator import ActorNetwork, CriticNetwork
from rl.models.running_mean_std import RunningMeanStd, RunningMeanStdObs

class ActorCritic(nn.Module):
    def __init__(self, actor_obs_config, critic_obs_config):
        super(ActorCritic, self).__init__()
        self.actor_obs_config = actor_obs_config
        self.critic_obs_config = critic_obs_config

        self.freeze_critic = critic_obs_config.get('freeze_critic', False)

        self.actor_network = ActorNetwork(actor_obs_config)
        self.critic_network = CriticNetwork(critic_obs_config)

        self.states = None
        self.is_rnn = True if ('rnn' in actor_obs_config) or ('rnn' in critic_obs_config) else False

    def _actor_critic(self, obs_dict):
        # NOTE: We provide the raw obs_dict and let the actor and critic decide which obs to use.
        # Actor and critic has its own teacher attribute
        
        # Split RNN states for actor and critic if present
        rnn_states_combined = obs_dict.get('rnn_states', None)
        if rnn_states_combined is not None and isinstance(rnn_states_combined, list) and len(rnn_states_combined) == 2:
            actor_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[0]}
            critic_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[1]}
        else:
            actor_obs_dict = obs_dict
            critic_obs_dict = obs_dict
        
        mu, logstd, actor_states = self.actor_network(actor_obs_dict)

        if self.freeze_critic:
            with torch.no_grad():
                value, critic_states = self.critic_network(critic_obs_dict)
        else:
            value, critic_states = self.critic_network(critic_obs_dict)

        rnn_states = [actor_states, critic_states]

        return mu, logstd, value, rnn_states
    
    def _actor_critic_with_features(self, obs_dict):
        rnn_states_combined = obs_dict.get('rnn_states', None)
        if rnn_states_combined is not None and isinstance(rnn_states_combined, list) and len(rnn_states_combined) == 2:
            actor_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[0]}
            critic_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[1]}
        else:
            actor_obs_dict = obs_dict
            critic_obs_dict = obs_dict
        
        mu, logstd, actor_states, actor_features = self.actor_network.forward_with_features(actor_obs_dict)
        value, critic_states = self.critic_network.forward(critic_obs_dict)

        rnn_states = [actor_states, critic_states]

        return mu, logstd, value, rnn_states, actor_features

    def _actor(self, obs_dict):
        
        # Split RNN states for actor and critic if present
        rnn_states_combined = obs_dict.get('rnn_states', None)
        if rnn_states_combined is not None and isinstance(rnn_states_combined, list) and len(rnn_states_combined) == 2:
            actor_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[0]}
            critic_obs_dict = {**obs_dict, 'rnn_states': rnn_states_combined[1]}
        else:
            actor_obs_dict = obs_dict
            critic_obs_dict = obs_dict
        
        mu, logstd, actor_states = self.actor_network(actor_obs_dict)

        critic_states = None

        rnn_states = [actor_states, critic_states]

        return mu, logstd, rnn_states


    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, rnn_states = self._actor_critic(input_dict)

        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        entropy = distr.entropy().sum(dim=-1)
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(1)
            prev_neglogp = torch.squeeze(prev_neglogp)
        else:
            prev_neglogp = None
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'rnn_states': rnn_states,
        }
        return result

    def forward_with_features(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, rnn_states, actor_features = self._actor_critic_with_features(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        entropy = distr.entropy().sum(dim=-1)
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(1)
            prev_neglogp = torch.squeeze(prev_neglogp)
        else:
            prev_neglogp = None
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'rnn_states': rnn_states,
            'actor_features': actor_features,
        }
        return result
    # Used in play_steps() in ppo.py
    @torch.no_grad()
    def act(self, obs_dict):
        self.actor_network.eval()
        self.critic_network.eval()
        mu, logstd, value, rnn_states = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1).unsqueeze(1),
            'values': value,
            'actions': selected_action, # NOTE: This has to match the negative log probability of selected_action not clamped_actions
            'mus': mu,
            'sigmas': sigma,
            'rnn_states': rnn_states,
        }
        return result
    
    @torch.no_grad()
    def act_inference(self, obs_dict):
        mu, logstd, value, rnn_states = self._actor_critic(obs_dict)
        return mu, logstd, value, rnn_states
    
    @torch.no_grad()
    def act_only(self, obs_dict):
        self.actor_network.eval()
        mu, logstd, rnn_states = self._actor(obs_dict)
        return mu, logstd, rnn_states