"""
PPO algorithm implementation

References:
- rl_games/common/a2c_common.py
- rl_games/common/a2c_continuous.py: calc_gradients()
- penspin/algo/ppo/ppo.py

Key features:
- Handles separate observations, privileged info, critic info, etc.
- Have to define what goes in as obs to actor and critic
- This introduces the flexibility for building asymmetric actor-critic networks
"""

import os
import time
import datetime

import copy
import torch
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from rl.utils.kl_util import policy_kl
from rl.utils.obs_util import _preproc_obs
from rl.utils.misc import AverageScalarMeter
from rl.utils.reformat import omegaconf_to_dict
from rl.utils.scheduler_util import AdaptiveScheduler

from rl.models.actor_critic import ActorCritic
from rl.algo.ppo.experience import ExperienceBuffer
from rl.models.running_mean_std import RunningMeanStd, RunningMeanStdObs

class PPO(object):
    def __init__(self, env, cfg, output_dir):
        # Configs
        self.cfg = cfg
        self.device = cfg.rl_device
        self.ppo_config = cfg.train.ppo

        # Environment
        self.env = env
        self.num_actors = self.ppo_config.num_actors # same as num_envs
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.shape[0]
        self.action_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.action_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.state_space = self.env.state_space
        self.observation_space = self.env.observation_space

        # Model
        self.actor_network_config = cfg.train.actor
        self.critic_network_config = cfg.train.critic
        self.freeze_critic = self.critic_network_config.get('freeze_critic', False)
        
        # Set num_seqs for RNN state initialization (should be num_actors for rollout phase)
        if 'rnn' in self.actor_network_config:
            OmegaConf.set_struct(self.actor_network_config, False)
            self.actor_network_config['num_seqs'] = self.num_actors
            OmegaConf.set_struct(self.actor_network_config, True)
        if 'rnn' in self.critic_network_config:
            OmegaConf.set_struct(self.critic_network_config, False)
            self.critic_network_config['num_seqs'] = self.num_actors
            OmegaConf.set_struct(self.critic_network_config, True)
        
        self.model = ActorCritic(actor_obs_config=self.actor_network_config, critic_obs_config=self.critic_network_config)
        self.model.to(self.device)

        # Running mean std
        # Get per-channel observation lists from config
        per_channel_obs = self.ppo_config.get('per_channel_observations', None)
        per_channel_states = self.ppo_config.get('per_channel_states', None)
        
        self.running_mean_std_obs = RunningMeanStdObs(
            omegaconf_to_dict(self.actor_network_config.input_shape),
            per_channel_obs_list=per_channel_obs
        ).to(self.device)
        self.running_mean_std_states = RunningMeanStdObs(
            omegaconf_to_dict(self.critic_network_config.input_shape),
            per_channel_obs_list=per_channel_states
        ).to(self.device)
        self.running_mean_std_value = RunningMeanStd((1,)).to(self.device)

        # Optimizers
        self.actor_lr = float(self.ppo_config.learning_rate)
        self.critic_lr = float(self.ppo_config.learning_rate)
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)

        # Separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.model.actor_network.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_network.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)
        
        # PPO training parameters
        self.e_clip = self.ppo_config.e_clip
        self.clip_value = self.ppo_config.clip_value
        self.entropy_coef = self.ppo_config.entropy_coef
        self.critic_coef = self.ppo_config.critic_coef
        self.bounds_loss_coef = self.ppo_config.bounds_loss_coef
        self.gamma = self.ppo_config.gamma
        self.tau = self.ppo_config.tau
        self.truncate_grads = self.ppo_config.truncate_grads
        self.grad_norm = self.ppo_config.grad_norm
        self.value_bootstrap = self.ppo_config.value_bootstrap
        self.normalize_advantage = self.ppo_config.normalize_advantage
        self.normalize_value = self.ppo_config.normalize_value
        self.normalize_input = self.ppo_config.normalize_input
        self.seq_length = self.ppo_config.seq_length

        self.horizon_length = self.ppo_config.horizon_length
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config.minibatch_size
        self.mini_epochs_actor = self.ppo_config.get('mini_epochs_actor', self.ppo_config.get('mini_epochs', 4))
        self.mini_epochs_critic = self.ppo_config.get('mini_epochs_critic', self.ppo_config.get('mini_epochs', 4))
        self.kl_threshold_actor = self.ppo_config.kl_threshold_actor
        self.kl_threshold_critic = self.ppo_config.kl_threshold_critic
        self.max_lr = self.ppo_config.get('max_lr', 1e-2)
        self.scheduler_actor = AdaptiveScheduler(self.kl_threshold_actor, self.max_lr)
        self.scheduler_critic = AdaptiveScheduler(self.kl_threshold_critic, self.max_lr)
        self.lr_schedule_actor = self.ppo_config.lr_schedule_actor
        self.lr_schedule_critic = self.ppo_config.lr_schedule_critic

        self.save_freq = self.ppo_config.save_frequency
        self.save_best_after = self.ppo_config.save_best_after

        # RNN configs
        self.is_rnn = self.model.actor_network.has_rnn and self.model.critic_network.has_rnn
        if self.is_rnn:
            self.zero_rnn_on_done = True
            # Initialize states for both actor and critic
            actor_states = self.model.actor_network.get_default_rnn_state()
            critic_states = self.model.critic_network.get_default_rnn_state()
            # each state is a tuple and we need to move to device
            actor_states = [s.to(self.device) for s in actor_states]
            critic_states = [s.to(self.device) for s in critic_states]
            self.rnn_states = [actor_states, critic_states]
            # Create minibatch RNN states storage (flattened for all states)
            num_seqs = self.horizon_length // self.seq_length
            all_states = actor_states + critic_states
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], self.num_actors, s.size()[2]), dtype=torch.float32, device=self.device) for s in all_states]
        else:
            self.rnn_states = None
            self.zero_rnn_on_done = False


        # Experience buffer
        self.episode_rewards = AverageScalarMeter(20000) # this smoothens out rewards
        self.episode_lengths = AverageScalarMeter(20000)
        self.obs = None
        self.epoch_num = 0
        self.experience_buffer = ExperienceBuffer(
            num_envs=self.num_actors,
            horizon_length=self.horizon_length,
            batch_size=self.batch_size,
            minibatch_size=self.ppo_config.minibatch_size,
            is_rnn=self.is_rnn,
            seq_length=self.seq_length,
            action_space=self.action_space,
            state_space=self.state_space,
            observation_space=self.observation_space,
            device=self.device,
            normalize_advantage=self.normalize_advantage,
        )

        # Rollout related
        # NOTE: Initialized as all ones due to reset at the beginning of each episode for computing GAE correctly
        self.dones = torch.ones((self.num_actors,), dtype=torch.uint8, device=self.device)

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config.max_agent_steps
        self.best_rewards = -float('inf')
        self.best_per_epoch_rewards = -float('inf')
        self.best_sr = -float('inf')

        # Output dir
        self.datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f"{output_dir}_{self.datetime_str}"
        self.nn_dir = os.path.join(self.output_dir, "nn")
        self.tb_dir = os.path.join(self.output_dir, "tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        config_path = os.path.join(self.output_dir, 'config.yaml')
        print(f"Saving config to: {config_path}")
        with open(config_path, 'w') as f:
            OmegaConf.save(self.cfg, f)

        # Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.total_time = 0

        writer = SummaryWriter(self.tb_dir)
        self.writer = writer


        cprint(f"freeze_critic: {self.freeze_critic}", "green")

    def train(self):
        start_t = time.time()
        last_t = time.time()

        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas = self.train_epoch()
            
            # Clear experience buffer and force garbage collection
            self.experience_buffer.data_dict = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()

            for k, v in self.extra_info.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            all_fps = self.agent_steps / (time.time() - start_t)
            last_fps = self.batch_size / (time.time() - last_t)
            last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Current Best: {self.best_rewards:.2f} | ' \
                          f'Current Best Per Epoch: {self.best_per_epoch_rewards:.2f} | ' \
                          f'Mean Rewards: {self.episode_rewards.get_mean():.2f} | ' \
                          f'SR: {self.extra_info["successes"]:.2f}'
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas)
            
            # Clear loss tracking lists to free memory
            del a_losses, c_losses, b_losses, entropies, kls, grad_norms
            
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save_model(os.path.join(self.nn_dir, checkpoint_name + '.pth'))
                    self.save_model(os.path.join(self.nn_dir, f'last.pth'))

            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save_model(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}.pth'))

            if self.extra_info["successes"] > self.best_sr and self.agent_steps >= self.save_best_after:
                print(f'save current best sr: {self.extra_info["successes"]:.2f}')
                prev_best_sr_ckpt = os.path.join(self.nn_dir, f'best_sr_{self.extra_info["successes"]:.2f}.pth')
                if os.path.exists(prev_best_sr_ckpt):
                    os.remove(prev_best_sr_ckpt)
                self.best_sr = self.extra_info["successes"]
                self.save_model(os.path.join(self.nn_dir, f'best_sr_{self.extra_info["successes"]:.2f}.pth'))

        print('max steps achieved')

    def train_epoch(self):
        # Reset best_per_epoch_rewards at the start of each epoch
        self.best_per_epoch_rewards = -float('inf')
        
        # Rollout episodes to fill the replay buffer
        start_t = time.time()
        self.set_eval()
        if not self.is_rnn:
            self.play_steps() # this fills the self.experience_buffer
        else:
            self.play_steps_rnn()
        self.data_collect_time += time.time() - start_t

        # Train the actor-critic networks
        start_t = time.time()
        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls, grad_norms = [], [], [], [], [], []
        mus, sigmas = [], []

        max_mini_epochs = max(self.mini_epochs_actor, self.mini_epochs_critic)
        for epoch_idx in range(0, max_mini_epochs):
            ep_kls = []
            train_actor = epoch_idx < self.mini_epochs_actor
            train_critic = epoch_idx < self.mini_epochs_critic
            
            for i in range(len(self.experience_buffer)):
                # calc_gradients() in rl_games/common/a2c_continuous.py
                input_dict = self.experience_buffer[i] # this is from data_dict filled in play_steps()
                value_preds = input_dict['values'] # NOTE: This is normalized
                old_action_log_probs = input_dict['neglogpacs'] # from self.model.forward_actor_critic()
                advantage = input_dict['advantages']
                old_mu = input_dict['mus']
                old_sigma = input_dict['sigmas']
                returns = input_dict['returns'] # NOTE: This is normalized
                actions = input_dict['actions']
                obs = input_dict['obses']
                states = input_dict['states']

                obs = _preproc_obs(obs) # NOTE: converting uint8 to float32 in images

                filtered_obs = {k: v for k, v in obs.items() if k in self.actor_network_config.input_shape}
                filtered_states = {k: v for k, v in states.items() if k in self.critic_network_config.input_shape}

                if self.is_rnn:
                    rnn_states_minibatch = input_dict['rnn_states']
                    rnn_states_minibatch = [
                        (rnn_states_minibatch[0], rnn_states_minibatch[1]),
                        (rnn_states_minibatch[2], rnn_states_minibatch[3]),
                    ]
                else:
                    rnn_states_minibatch = None

                # inputs to the actor-critic
                batch_dict = {
                    'prev_actions': actions,
                    'obs': self.running_mean_std_obs(filtered_obs) if self.normalize_input else filtered_obs,
                    'states': self.running_mean_std_states(filtered_states) if self.normalize_input else filtered_states,
                    'rnn_states': rnn_states_minibatch,
                    'seq_length': self.seq_length if self.is_rnn else 1,   # ensure correct sequence batching for RNN
                    'dones': input_dict['dones'] if self.zero_rnn_on_done else None,    # pass episode boundaries for masking
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']  # NOTE: Already normalized if normalize_value=True
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                
                # CHECK THIS MAYBE THIS WAS THE DEAL BREAKER
                # Ensure consistent shapes: squeeze trailing singleton dims
                # Old rollout log-probs/values/returns/advantages are (B,1); model outputs are often (B,)
                old_action_log_probs = old_action_log_probs.squeeze(-1)
                value_preds = value_preds.squeeze(-1)
                advantage = advantage.squeeze(-1)
                returns = returns.squeeze(-1)
                values = values.squeeze(-1)
                
                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.ppo_config.e_clip, 1.0 + self.ppo_config.e_clip)
                a_loss = torch.max(-surr1, -surr2)

                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_config.e_clip, self.ppo_config.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)

                # bound loss
                if self.ppo_config.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(mu)
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                # Separate optimization for actor and critic
                actor_loss = a_loss - entropy * self.ppo_config.entropy_coef + b_loss * self.ppo_config.bounds_loss_coef
                critic_loss = 0.5 * c_loss * self.ppo_config.critic_coef

                # Optimize actor (only if within mini_epochs_actor)
                if train_actor:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=train_critic)  # retain_graph if critic also needs to train

                # Optimize critic (only if within mini_epochs_critic)
                if train_critic and not self.freeze_critic:
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()

                with torch.no_grad():
                    # Compute gradient norm across all parameters
                    all_grads = []
                    if train_actor:
                        for p in self.model.actor_network.parameters():
                            if p.grad is not None:
                                all_grads.append(p.grad.reshape(-1))
                    if train_critic and not self.freeze_critic:
                        for p in self.model.critic_network.parameters():
                            if p.grad is not None:
                                all_grads.append(p.grad.reshape(-1))
                    if all_grads:
                        grad_norms.append(torch.norm(torch.cat(all_grads)))

                if self.ppo_config.truncate_grads:
                    # Clip gradients for both networks
                    if train_actor:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_network.parameters(), self.grad_norm)
                    if train_critic and not self.freeze_critic:
                        torch.nn.utils.clip_grad_norm_(self.model.critic_network.parameters(), self.grad_norm)

                if train_actor:
                    self.actor_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    
                if train_critic and not self.freeze_critic:
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                
                # Only track losses for networks that are being trained
                if train_actor:
                    a_losses.append(a_loss.detach())
                    ep_kls.append(kl.detach())
                    entropies.append(entropy.detach())
                    if self.ppo_config.bounds_loss_coef is not None:
                        b_losses.append(b_loss.detach())
                    mus.append(mu.detach())
                    sigmas.append(sigma.detach())
                
                if train_critic:
                    c_losses.append(c_loss.detach())
                
                # Always update mu/sigma for experience buffer
                self.experience_buffer.update_mu_sigma(mu.detach(), sigma.detach())           

            # don't need to update statstics more than one miniepoch
            if self.normalize_input:
                self.running_mean_std_obs.eval() 
                self.running_mean_std_states.eval()

            if len(ep_kls) > 0:
                mean_kls = torch.mean(torch.stack(ep_kls))
                kls.append(mean_kls.detach())
            else:
                # Handle case when no minibatches were processed
                mean_kls = torch.tensor(0.0, device=self.device)
                kls.append(mean_kls)

            # Only update learning rates when training the respective networks
            if train_actor and self.lr_schedule_actor == 'adaptive' and len(ep_kls) > 0:
                self.actor_lr = self.scheduler_actor.update(self.actor_lr, mean_kls.item())
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_lr
            if train_critic and not self.freeze_critic and self.lr_schedule_critic == 'adaptive' and len(ep_kls) > 0:
                self.critic_lr = self.scheduler_critic.update(self.critic_lr, mean_kls.item())
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_lr

        self.rl_train_time += time.time() - start_t
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas

    def play_steps(self):
        for n in range(self.horizon_length):
            
            with torch.no_grad():
                res_dict = self.model_act(self.obs) # reset is called in train()

            # Detach tensors before storing to prevent gradient accumulation
            self.experience_buffer.update_data('obses', n, {k: v.detach() for k, v in self.obs['obs'].items()})
            self.experience_buffer.update_data('states', n, {k: v.detach() for k, v in self.obs['states'].items()})
            
            # self.experience_buffer.update_data('dones', n, self.dones)

            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.experience_buffer.update_data(k, n, res_dict[k].detach())
            
            clamped_actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            actions = clamped_actions

            # record_frame = False
            # if self.gif_frame_counter >= self.gif_save_every_n and self.gif_frame_counter % self.gif_save_every_n < self.gif_save_length:
            #     record_frame = True
            # record_frame = record_frame and int(os.getenv('LOCAL_RANK', 0)) == 0
            # self.env.enable_camera_sensors = record_frame
            # self.gif_frame_counter += 1

            # obs_and_states_dict, rew_buf, reset_buf, info_buf from VecTask.step()
            self.obs, rewards, self.dones, infos = self.env.step(actions)

            rewards = rewards.unsqueeze(1)
            reward_scale = 1.0
            shaped_rewards = reward_scale * rewards.detach()  # Use detach instead of clone to avoid gradient accumulation

            # Handling timeouts for value bootstrapping (when episodes reach max episode length)
            # Otherwise, it will drop the last bootstrapped value and will consider the end state with much lower value
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'].detach() * infos['time_outs'].unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards.detach())
            self.experience_buffer.update_data('dones', n, self.dones.unsqueeze(1).detach())

            self.current_rewards += rewards.detach()
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices].detach())
            self.episode_lengths.update(self.current_lengths[done_indices].detach())
            
            # Track best completed episode reward in this epoch
            if len(done_indices) > 0:
                max_completed_reward = torch.max(self.current_rewards[done_indices]).item()
                self.best_per_epoch_rewards = max(self.best_per_epoch_rewards, max_completed_reward)

            assert isinstance(infos, dict), 'Info Should be a Dict'
            # Detach any tensors in infos to prevent gradient accumulation
            self.extra_info = {}
            for k, v in infos.items():
                if isinstance(v, torch.Tensor):
                    self.extra_info[k] = v.detach()
                else:
                    self.extra_info[k] = v

            not_dones = (1.0 - self.dones.float()).to(self.device)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = self.agent_steps + self.batch_size
        self.experience_buffer.compute_return(last_values.detach(), self.gamma, self.tau)
        self.experience_buffer.prepare_training()

        values = self.experience_buffer.data_dict['values']
        returns = self.experience_buffer.data_dict['returns']
        if self.normalize_value:
            if self.model.freeze_critic:
                self.running_mean_std_value.eval()
            else:
                self.running_mean_std_value.train()
            values = self.running_mean_std_value(values)
            returns = self.running_mean_std_value(returns)
            self.running_mean_std_value.eval()

        self.experience_buffer.data_dict['values'] = values
        self.experience_buffer.data_dict['returns'] = returns

    def play_steps_rnn(self):
        mb_rnn_states = self.mb_rnn_states

        for n in range(self.horizon_length):
            if n % self.seq_length == 0:
                # Flatten rnn_states for storage: [actor_states, critic_states] -> flat list of tensors
                flat_states = []
                for states_tuple in self.rnn_states:  # actor_states, critic_states
                    for s in states_tuple:  # h, c for LSTM
                        flat_states.append(s)
                # NOTE: 
                # mb_rnn_states[0].size() = (horizon_length // seq_length, num_layers, num_actors, hidden_size)
                # rnn_states[0].size() = (num_layers, num_actors, hidden_size)
                # We are updating the rnn_states every seq_length steps
                for s, mb_s in zip(flat_states, mb_rnn_states):
                    mb_s[n // self.seq_length, :, :, :] = s

            with torch.no_grad():
                res_dict = self.model_act(self.obs)

            self.rnn_states = res_dict['rnn_states']

            self.experience_buffer.update_data('obses', n, {k: v.detach() for k, v in self.obs['obs'].items()})
            self.experience_buffer.update_data('states', n, {k: v.detach() for k, v in self.obs['states'].items()})
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.experience_buffer.update_data(k, n, res_dict[k].detach())

            clamped_actions = torch.clamp(res_dict['actions'], -1.0, 1.0)

            self.obs, rewards, self.dones, infos = self.env.step(clamped_actions)

            rewards = rewards.unsqueeze(1)
            reward_scale = 1.0
            shaped_rewards = reward_scale * rewards.detach()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'].detach() * infos['time_outs'].unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards.detach())
            self.experience_buffer.update_data('dones', n, self.dones.unsqueeze(1).detach())

            self.current_rewards += rewards.detach()
            self.current_lengths += 1

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices # NOTE: [::1]
            if len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    # Zero out states for both actor and critic
                    # self.rnn_states[0].size() = (num_layers, num_actors, hidden_size)
                    for states_tuple in self.rnn_states:  # actor_states, critic_states
                        for s in states_tuple:  # h, c for LSTM
                            s[:, all_done_indices, :] = 0.0 * s[:, all_done_indices, :]

            self.episode_rewards.update(self.current_rewards[env_done_indices])
            self.episode_lengths.update(self.current_lengths[env_done_indices])
            
            # Track best completed episode reward in this epoch
            if len(env_done_indices) > 0:
                max_completed_reward = torch.max(self.current_rewards[env_done_indices]).item()
                self.best_per_epoch_rewards = max(self.best_per_epoch_rewards, max_completed_reward)

            self.extra_info = {}
            for k, v in infos.items():
                if isinstance(v, torch.Tensor):
                    self.extra_info[k] = v.detach()
                else:
                    self.extra_info[k] = v

            not_dones = (1.0 - self.dones.float()).to(self.device)
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        with torch.no_grad():
            last_values = self.model_act(self.obs)['values']

        self.agent_steps = self.agent_steps + self.batch_size
        self.experience_buffer.compute_return(last_values.detach(), self.gamma, self.tau)
        self.experience_buffer.prepare_training()

        values = self.experience_buffer.data_dict['values']
        returns = self.experience_buffer.data_dict['returns']
        if self.normalize_value:
            if self.model.freeze_critic:
                self.running_mean_std_value.eval()
            else:
                self.running_mean_std_value.train()
            values = self.running_mean_std_value(values)
            returns = self.running_mean_std_value(returns)
            self.running_mean_std_value.eval()
        self.experience_buffer.data_dict['values'] = values
        self.experience_buffer.data_dict['returns'] = returns

        # self.mb_rnn_states[0].size() = (horizon_length // seq_length, num_layers, num_actors, hidden_size)
        # mb_s.permute(1, 2, 0, 3) = (num_layers, num_actors, horizon_length // seq_length, hidden_size)
        # after reshape -> (num_layers, horizon_length // seq_length * num_actors, hidden_size)
        # when we __getitem__, we get states[0][:, 0:num_games_batch, :]
        states = []
        for mb_s in self.mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2] # horizon_length // seq_length * num_actors
            h_size = mb_s.size()[3] # hidden_size
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
        self.experience_buffer.data_dict['rnn_states'] = states

    def model_act(self, obs_dict):
        # NOTE: obsDims and stateDims are specified to tell the environment what observations and states to provide.
        # actor.input_shape and critic.input_shape define the inputs to the models
        filtered_obs_dict = {k: v for k, v in obs_dict['obs'].items() if k in self.actor_network_config.input_shape}
        filtered_obs_dict = _preproc_obs(filtered_obs_dict)
        filtered_states_dict = {k: v for k, v in obs_dict['states'].items() if k in self.critic_network_config.input_shape}
        filtered_states_dict = _preproc_obs(filtered_states_dict)

        input_dict = {
            'obs': self.running_mean_std_obs(filtered_obs_dict) if self.normalize_input else filtered_obs_dict,
            'states': self.running_mean_std_states(filtered_states_dict) if self.normalize_input else filtered_states_dict,
            'rnn_states': self.rnn_states if self.is_rnn else None,
        }
        res_dict = self.model.act(input_dict)
        if self.normalize_value:
            res_dict['values'] = self.running_mean_std_value(res_dict['values'], denorm=True)

        return res_dict

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()

        while True:
            filtered_obs_dict = {k: v for k, v in obs_dict['obs'].items() if k in self.actor_network_config.input_shape}
            filtered_states_dict = {k: v for k, v in obs_dict['states'].items() if k in self.critic_network_config.input_shape}
            # print(filtered_obs_dict['eef_to_socket_quat'][0, ...].cpu().numpy())

            # NOTE: testing the effect of normalization on the tactile force field
            # np.linalg.norm(filtered_obs_dict['tactile_force_field_left'][0, ...].cpu().numpy(), axis=-1).min()
            # new_dict = self.running_mean_std_obs(filtered_obs_dict)
            # np.linalg.norm(new_dict['tactile_force_field_left'][0, ...].cpu().numpy(), axis=-1).min()
            
            # self.running_mean_std_obs.running_mean_std['tactile_force_field_left'].state_dict()['running_mean']
            # np.linalg.norm(self.running_mean_std_obs.running_mean_std['tactile_force_field_left'].state_dict()['running_mean'].cpu().numpy(), axis=-1).min()
            # np.unravel_index(np.argmin(np.linalg.norm(self.running_mean_std_obs.running_mean_std['tactile_force_field_left'].state_dict()['running_mean'].cpu().numpy(), axis=-1)), self.running_mean_std_obs.running_mean_std['tactile_force_field_left'].state_dict()['running_mean'].shape))
            # Save filtered_obs_dict['tactile_force_field_left'] and filtered_obs_dict['tactile_force_field_right']
            # np.save('filtered_obs_dict_tactile_force_field_left.npy', filtered_obs_dict['tactile_force_field_left'][0, ...].cpu().numpy())
            # np.save('filtered_obs_dict_tactile_force_field_right.npy', filtered_obs_dict['tactile_force_field_right'][0, ...].cpu().numpy())

            input_dict = {
                'obs': self.running_mean_std_obs(filtered_obs_dict) if self.normalize_input else filtered_obs_dict,
                'states': self.running_mean_std_states(filtered_states_dict) if self.normalize_input else filtered_states_dict,
                'rnn_states': self.rnn_states if self.is_rnn else None,
            }
            mu, _, _, rnn_states = self.model.act_inference(input_dict)
            self.rnn_states = rnn_states

            # Clamp actions like in play_steps
            clamped_actions = torch.clamp(mu, -1.0, 1.0)
            actions = clamped_actions

            obs_dict, rewards, done, info = self.env.step(actions)

            # print(info.get('successes', 0))


    def test_rollout_and_value_plot(self):
        import matplotlib.pyplot as plt

        self.set_eval()
        obs_dict = self.env.reset()

        values_list = []
        keypoint_rewards_list = []
        action_penalties_list = []
        action_grad_penalties_list = []
        contact_penalties_list = []
        contact_force_table_list = []
        episodic_return_list = []

        step = 0
        max_steps = 256
        scale = 1

        # Setup live plot with multiple lines
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))
        line_value, = ax.plot([], [], linewidth=2, label='Value')
        line_keypoint, = ax.plot([], [], linewidth=2, label='Keypoint Reward')
        line_action, = ax.plot([], [], linewidth=2, label='Action Penalty')
        line_action_grad, = ax.plot([], [], linewidth=2, label='Action Grad Penalty')
        line_contact, = ax.plot([], [], linewidth=2, label='Contact Penalty')
        line_contact_force, = ax.plot([], [], linewidth=2, label='Contact Force Table')
        line_episodic_return, = ax.plot([], [], linewidth=2, label='Episodic Return')
        
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.set_title('Value Function and Reward Components (Live Update)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.show()

        episodic_return = 0.0

        while True:

            info = {}
            filtered_obs_dict = {k: v for k, v in obs_dict['obs'].items() if k in self.actor_network_config.input_shape}
            filtered_states_dict = {k: v for k, v in obs_dict['states'].items() if k in self.critic_network_config.input_shape}
            input_dict = {
                'obs': self.running_mean_std_obs(filtered_obs_dict) if self.normalize_input else filtered_obs_dict,
                'states': self.running_mean_std_states(filtered_states_dict) if self.normalize_input else filtered_states_dict,
                'rnn_states': self.rnn_states if self.is_rnn else None,
            }
            mu, logstd, value, rnn_states = self.model.act_inference(input_dict)

            self.rnn_states = rnn_states

            # denormalize value
            # value = self.running_mean_std_value(value, denorm=True)

            # Clamp actions like in play_steps
            clamped_actions = torch.clamp(mu, -1.0, 1.0)
            actions = clamped_actions

            # print(actions * 0.01, obs_dict["states"]['socket_pos'], obs_dict["states"]['socket_quat'])

            obs_dict, rewards, done, info = self.env.step(actions)

            episodic_return += rewards.item()
            normalized_return = self.running_mean_std_value(episodic_return)

            values_list.append(value.cpu().item() * scale)
            keypoint_rewards_list.append(info['keypoint_reward'].cpu().item() * scale + info['keypoint_reward_exp'].cpu().item() * scale)
            action_penalties_list.append(info['action_penalty'].cpu().item() * scale)
            action_grad_penalties_list.append(info['action_grad_penalty'].cpu().item() * scale)
            contact_penalties_list.append(info['contact_penalty'].cpu().item() * scale)
            contact_force_table_list.append(info['contact_force_table'].cpu().item() * scale)
            episodic_return_list.append(normalized_return.cpu().item())

            # Update all lines
            timesteps = range(len(values_list))
            line_value.set_data(timesteps, values_list)
            line_keypoint.set_data(timesteps, keypoint_rewards_list)
            line_action.set_data(timesteps, action_penalties_list)
            line_action_grad.set_data(timesteps, action_grad_penalties_list)
            line_contact.set_data(timesteps, contact_penalties_list)
            line_contact_force.set_data(timesteps, contact_force_table_list)
            line_episodic_return.set_data(timesteps, episodic_return_list)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # print(info)
            print(f"Step {step} | Reward: {rewards.item():.4f} | Value: {value.item():.4f} | Success: {info.get('successes', 0)}")
            step += 1

            if done:
                values_list = []
                keypoint_rewards_list = []
                action_penalties_list = []
                action_grad_penalties_list = []
                contact_penalties_list = []
                contact_force_table_list = []
                episodic_return_list = []
                step = 0
                episodic_return = 0.0
                obs_dict = self.env.reset()


    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/actor_lr', self.actor_lr, self.agent_steps)
        self.writer.add_scalar('info/critic_lr', self.critic_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar('info/grad_norms', torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)
        self.writer.add_scalar('info/mus', torch.mean(torch.stack(mus)).item(), self.agent_steps)
        self.writer.add_scalar('info/sigmas', torch.mean(torch.stack(sigmas)).item(), self.agent_steps)

        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def save_model(self, ckpt_path):
        # save separate actor and critic models
        weights = {'actor_network': self.model.actor_network.state_dict(), 'critic_network': self.model.critic_network.state_dict()}
        weights['actor_optimizer'] = self.actor_optimizer.state_dict()
        weights['critic_optimizer'] = self.critic_optimizer.state_dict()
        weights['critic_lr'] = self.critic_lr
        weights['actor_lr'] = self.actor_lr
        if self.normalize_input:
            weights['running_mean_std_obs'] = self.running_mean_std_obs.state_dict()
            weights['running_mean_std_states'] = self.running_mean_std_states.state_dict()
        if self.normalize_value:
            weights['running_mean_std_value'] = self.running_mean_std_value.state_dict()
        torch.save(weights, ckpt_path)

    def load_model(self, ckpt_path, load_optimizer=True):
        checkpoint = torch.load(ckpt_path)
        self.model.actor_network.load_state_dict(checkpoint['actor_network'])
        self.model.critic_network.load_state_dict(checkpoint['critic_network'])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.actor_lr = checkpoint.get('actor_lr', self.actor_lr)  # fallback to default if not saved
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.critic_lr = checkpoint.get('critic_lr', self.critic_lr)  # fallback to default if not saved
        if self.normalize_input:
            self.running_mean_std_obs.load_state_dict(checkpoint['running_mean_std_obs'])
            self.running_mean_std_states.load_state_dict(checkpoint['running_mean_std_states'])
        if self.normalize_value:
            self.running_mean_std_value.load_state_dict(checkpoint['running_mean_std_value'])
        
        # Update num_seqs for inference if using RNN
        if self.is_rnn:
            self.model.actor_network.num_seqs = self.num_actors
            self.model.critic_network.num_seqs = self.num_actors
            # Reinitialize RNN states with correct batch size
            actor_states = self.model.actor_network.get_default_rnn_state()
            critic_states = self.model.critic_network.get_default_rnn_state()
            actor_states = [s.to(self.device) for s in actor_states]
            critic_states = [s.to(self.device) for s in critic_states]
            self.rnn_states = [actor_states, critic_states]

    def load_actor(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.actor_network.load_state_dict(checkpoint['actor_network'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if self.normalize_input:
            if self.model.actor_network.teacher:
                self.running_mean_std_states.load_state_dict(checkpoint['running_mean_std_states'])
            else:
                self.running_mean_std_obs.load_state_dict(checkpoint['running_mean_std_obs'])
        
        # Update num_seqs for inference if using RNN
        if self.is_rnn and self.model.actor_network.has_rnn:
            self.model.actor_network.num_seqs = self.num_actors
            # Reinitialize RNN states with correct batch size
            actor_states = self.model.actor_network.get_default_rnn_state()
            actor_states = [s.to(self.device) for s in actor_states]
            if self.rnn_states is not None:
                self.rnn_states[0] = actor_states

    def load_critic(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.critic_network.load_state_dict(checkpoint['critic_network'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic_lr = checkpoint.get('critic_lr', self.critic_lr)  # fallback to default if not saved
        if self.normalize_input:
            self.running_mean_std_states.load_state_dict(checkpoint['running_mean_std_states'])
        if self.normalize_value:
            self.running_mean_std_value.load_state_dict(checkpoint['running_mean_std_value'])
        
        # Update num_seqs for inference if using RNN
        if self.is_rnn and self.model.critic_network.has_rnn:
            self.model.critic_network.num_seqs = self.num_actors
            # Reinitialize RNN states with correct batch size
            critic_states = self.model.critic_network.get_default_rnn_state()
            critic_states = [s.to(self.device) for s in critic_states]
            if self.rnn_states is not None:
                self.rnn_states[1] = critic_states

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std_obs.train()
            self.running_mean_std_states.train()
        if self.normalize_value:
            self.running_mean_std_value.train()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std_obs.eval()
            self.running_mean_std_states.eval()
        if self.normalize_value:
            self.running_mean_std_value.eval()




