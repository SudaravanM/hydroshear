import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf

from rl.algo.ppo.ppo import PPO, _preproc_obs, policy_kl
from rl.models.actor_critic import ActorCritic
from rl.models.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl.utils.reformat import omegaconf_to_dict


class DistillPPO(PPO):
    """PPO with latent representation distillation from a teacher network"""
    
    def __init__(self, env, cfg, output_dir, teacher_ckpt_path=None):
        # Initialize base PPO
        super().__init__(env, cfg, output_dir)
        
        # Distillation configuration
        self.distill_config = cfg.train.ppo.get('distillation', {})
        self.use_latent_distillation = self.distill_config.get('use_latent_distillation', True)
        self.latent_loss_weight = self.distill_config.get('latent_loss_weight', 1.0)
        self.policy_distill_weight = self.distill_config.get('policy_distill_weight', 0.0)
        self.value_distill_weight = self.distill_config.get('value_distill_weight', 0.0)
        self.distill_loss_type = self.distill_config.get('loss_type', 'mse')  # mse, cosine, l1
        
        # Load teacher network if provided
        self.teacher_model = None
        self.teacher_running_mean_std_obs = None
        self.teacher_running_mean_std_states = None
        
        if teacher_ckpt_path is not None:
            self._load_teacher(teacher_ckpt_path, cfg)
            cprint(f"Loaded teacher model from: {teacher_ckpt_path}", "green")
        else:
            cprint("Warning: No teacher checkpoint provided for distillation", "yellow")
    
    def _load_teacher(self, teacher_ckpt_path, cfg):
        """Load teacher model from checkpoint"""
        # Create teacher model with same architecture
        teacher_actor_config = cfg.train.get('teacher_actor', cfg.train.actor)
        teacher_critic_config = cfg.train.get('teacher_critic', cfg.train.critic)
        
        # Set num_seqs for RNN
        if 'rnn' in teacher_actor_config:
            OmegaConf.set_struct(teacher_actor_config, False)
            teacher_actor_config['num_seqs'] = self.num_actors
            OmegaConf.set_struct(teacher_actor_config, True)
        if 'rnn' in teacher_critic_config:
            OmegaConf.set_struct(teacher_critic_config, False)
            teacher_critic_config['num_seqs'] = self.num_actors
            OmegaConf.set_struct(teacher_critic_config, True)
        
        self.teacher_model = ActorCritic(
            actor_obs_config=teacher_actor_config,
            critic_obs_config=teacher_critic_config
        )
        self.teacher_model.to(self.device)
        
        # Load teacher weights
        checkpoint = torch.load(teacher_ckpt_path, map_location=self.device)
        self.teacher_model.actor_network.load_state_dict(checkpoint['actor_network'])
        self.teacher_model.critic_network.load_state_dict(checkpoint['critic_network'])
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Load teacher's normalization statistics
        # Get per-channel observation lists from config
        per_channel_obs = self.ppo_config.get('per_channel_observations', None)
        per_channel_states = self.ppo_config.get('per_channel_states', None)
        
        self.teacher_running_mean_std_obs = RunningMeanStdObs(
            omegaconf_to_dict(teacher_actor_config.input_shape),
            per_channel_obs_list=per_channel_obs
        ).to(self.device)
        self.teacher_running_mean_std_states = RunningMeanStdObs(
            omegaconf_to_dict(teacher_critic_config.input_shape),
            per_channel_obs_list=per_channel_states
        ).to(self.device)
        
        if 'running_mean_std_obs' in checkpoint:
            self.teacher_running_mean_std_obs.load_state_dict(checkpoint['running_mean_std_obs'])
        if 'running_mean_std_states' in checkpoint:
            self.teacher_running_mean_std_states.load_state_dict(checkpoint['running_mean_std_states'])
        
        self.teacher_running_mean_std_obs.eval()
        self.teacher_running_mean_std_states.eval()
        
        cprint("Teacher model loaded and frozen successfully", "green")
    
    def _compute_representation_loss(self, student_features, teacher_features):
        """Compute loss for aligning student and teacher representations"""
        if self.distill_loss_type == 'mse':
            return F.mse_loss(student_features, teacher_features)
        elif self.distill_loss_type == 'l1':
            return F.l1_loss(student_features, teacher_features)
        elif self.distill_loss_type == 'cosine':
            # Cosine embedding loss (minimizes 1 - cosine_similarity)
            student_norm = F.normalize(student_features, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            cosine_sim = (student_norm * teacher_norm).sum(dim=-1)
            return (1.0 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown distillation loss type: {self.distill_loss_type}")
    
    def _extract_student_features(self, obs, states):
        """Extract intermediate features from student network"""
        # Filter observations and states
        filtered_obs = {k: v for k, v in obs.items() if k in self.actor_network_config.input_shape}
        filtered_states = {k: v for k, v in states.items() if k in self.critic_network_config.input_shape}
        
        # Normalize
        if self.normalize_input:
            filtered_obs = self.running_mean_std_obs(filtered_obs)
            filtered_states = self.running_mean_std_states(filtered_states)
        
        # Extract features from actor network
        if self.model.actor_network.teacher:
            actor_features = self.model.actor_network._process_teacher_observations(
                {'obs': filtered_obs, 'states': filtered_states}
            )
        else:
            actor_features = self.model.actor_network._process_student_observations(
                {'obs': filtered_obs, 'states': filtered_states}
            )
        
        # Apply main MLP if not using RNN (RNN case handled differently)
        if not self.model.actor_network.has_rnn:
            # Get intermediate representation before action heads
            actor_latent = self.model.actor_network.main_mlp(actor_features)
        else:
            actor_latent = actor_features
        
        return actor_latent
    
    def _extract_teacher_features(self, obs, states):
        """Extract intermediate features from teacher network"""
        if self.teacher_model is None:
            return None
        
        # Filter observations and states based on teacher's config
        teacher_actor_config = self.teacher_model.actor_obs_config
        filtered_obs = {k: v for k, v in obs.items() if k in teacher_actor_config.input_shape}
        filtered_states = {k: v for k, v in states.items() if k in teacher_actor_config.input_shape}
        
        # Normalize using teacher's statistics
        if self.normalize_input:
            filtered_obs = self.teacher_running_mean_std_obs(filtered_obs)
            filtered_states = self.teacher_running_mean_std_states(filtered_states)
        
        with torch.no_grad():
            # Extract features from teacher actor network
            if self.teacher_model.actor_network.teacher:
                teacher_features = self.teacher_model.actor_network._process_teacher_observations(
                    {'obs': filtered_obs, 'states': filtered_states}
                )
            else:
                teacher_features = self.teacher_model.actor_network._process_student_observations(
                    {'obs': filtered_obs, 'states': filtered_states}
                )
            
            # Apply main MLP if not using RNN
            if not self.teacher_model.actor_network.has_rnn:
                teacher_latent = self.teacher_model.actor_network.main_mlp(teacher_features)
            else:
                teacher_latent = teacher_features
        
        return teacher_latent
    
    def _get_teacher_policy_and_value(self, obs, states):
        """Get teacher's policy (mu, sigma) and value for distillation"""
        if self.teacher_model is None:
            return None, None, None, None
        
        # Filter and normalize observations
        teacher_actor_config = self.teacher_model.actor_obs_config
        teacher_critic_config = self.teacher_model.critic_obs_config
        
        filtered_obs = {k: v for k, v in obs.items() if k in teacher_actor_config.input_shape}
        filtered_states = {k: v for k, v in states.items() if k in teacher_critic_config.input_shape}
        
        if self.normalize_input:
            filtered_obs = self.teacher_running_mean_std_obs(filtered_obs)
            filtered_states = self.teacher_running_mean_std_states(filtered_states)
        
        obs_dict = {
            'obs': filtered_obs,
            'states': filtered_states,
            'rnn_states': None,
        }
        
        with torch.no_grad():
            teacher_mu, teacher_logstd, teacher_value, _ = self.teacher_model._actor_critic(obs_dict)
            
            # Apply same sigma transformation as in forward pass
            logstd_min, logstd_max = -2.0, 1.0
            teacher_logstd = logstd_min + 0.5 * (logstd_max - logstd_min) * (torch.tanh(teacher_logstd) + 1)
            teacher_sigma = torch.exp(teacher_logstd)
        
        return teacher_mu, teacher_sigma, teacher_value, None
    
    def train_epoch(self):
        """Override train_epoch to add distillation losses"""
        # Reset best_per_epoch_rewards
        self.best_per_epoch_rewards = -float('inf')
        
        # Rollout episodes
        start_t = time.time()
        self.set_eval()
        if not self.is_rnn:
            self.play_steps()
        else:
            self.play_steps_rnn()
        self.data_collect_time += time.time() - start_t
        
        # Train with distillation
        start_t = time.time()
        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls, grad_norms = [], [], [], [], [], []
        mus, sigmas = [], []
        latent_losses, policy_distill_losses, value_distill_losses = [], [], []
        
        max_mini_epochs = max(self.mini_epochs_actor, self.mini_epochs_critic)
        for epoch_idx in range(0, max_mini_epochs):
            ep_kls = []
            train_actor = epoch_idx < self.mini_epochs_actor
            train_critic = epoch_idx < self.mini_epochs_critic
            
            for i in range(len(self.experience_buffer)):
                input_dict = self.experience_buffer[i]
                value_preds = input_dict['values']
                old_action_log_probs = input_dict['neglogpacs']
                advantage = input_dict['advantages']
                old_mu = input_dict['mus']
                old_sigma = input_dict['sigmas']
                returns = input_dict['returns']
                actions = input_dict['actions']
                obs = input_dict['obses']
                states = input_dict['states']
                
                obs = _preproc_obs(obs)
                
                filtered_obs = {k: v for k, v in obs.items() if k in self.actor_network_config.input_shape}
                filtered_states = {k: v for k, v in states.items() if k in self.critic_network_config.input_shape}
                
                # For RNN training
                rnn_states_minibatch = None
                if self.is_rnn:
                    minibatch_size = actions.size(0)
                    num_seqs_in_batch = minibatch_size // self.seq_length
                    
                    if self.model.actor_network.rnn_name == 'lstm':
                        device = actions.device
                        actor_h = torch.zeros(self.model.actor_network.rnn_layers, num_seqs_in_batch, 
                                            self.model.actor_network.rnn_units, device=device)
                        actor_c = torch.zeros(self.model.actor_network.rnn_layers, num_seqs_in_batch,
                                            self.model.actor_network.rnn_units, device=device)
                        critic_h = torch.zeros(self.model.critic_network.rnn_layers, num_seqs_in_batch,
                                             self.model.critic_network.rnn_units, device=device)
                        critic_c = torch.zeros(self.model.critic_network.rnn_layers, num_seqs_in_batch,
                                             self.model.critic_network.rnn_units, device=device)
                        rnn_states_minibatch = [(actor_h, actor_c), (critic_h, critic_c)]
                
                # Standard PPO forward pass
                batch_dict = {
                    'prev_actions': actions,
                    'obs': self.running_mean_std_obs(filtered_obs) if self.normalize_input else filtered_obs,
                    'states': self.running_mean_std_states(filtered_states) if self.normalize_input else filtered_states,
                    'rnn_states': rnn_states_minibatch,
                    'seq_length': self.seq_length if self.is_rnn else 1,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                
                # Ensure consistent shapes
                old_action_log_probs = old_action_log_probs.squeeze(-1)
                value_preds = value_preds.squeeze(-1)
                advantage = advantage.squeeze(-1)
                returns = returns.squeeze(-1)
                values = values.squeeze(-1)
                
                # Standard PPO losses
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(mu)
                
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]
                
                # ===== DISTILLATION LOSSES =====
                latent_loss = torch.tensor(0.0, device=self.device)
                policy_distill_loss = torch.tensor(0.0, device=self.device)
                value_distill_loss = torch.tensor(0.0, device=self.device)
                
                if self.teacher_model is not None and train_actor:
                    # 1. Latent representation alignment
                    if self.use_latent_distillation and self.latent_loss_weight > 0:
                        student_latent = self._extract_student_features(obs, states)
                        teacher_latent = self._extract_teacher_features(obs, states)
                        
                        if teacher_latent is not None:
                            # If dimensions don't match, add a projection layer (simple linear projection)
                            if student_latent.shape[-1] != teacher_latent.shape[-1]:
                                # Create projection layer on the fly if needed
                                if not hasattr(self, 'latent_projection'):
                                    self.latent_projection = nn.Linear(
                                        student_latent.shape[-1], 
                                        teacher_latent.shape[-1]
                                    ).to(self.device)
                                student_latent = self.latent_projection(student_latent)
                            
                            latent_loss = self._compute_representation_loss(student_latent, teacher_latent)
                    
                    # 2. Policy distillation (KL between student and teacher policies)
                    if self.policy_distill_weight > 0:
                        teacher_mu, teacher_sigma, _, _ = self._get_teacher_policy_and_value(obs, states)
                        if teacher_mu is not None:
                            # KL divergence between Gaussian distributions
                            policy_distill_loss = policy_kl(teacher_mu, teacher_sigma, mu, sigma)
                    
                    # 3. Value distillation
                    if self.value_distill_weight > 0:
                        _, _, teacher_value, _ = self._get_teacher_policy_and_value(obs, states)
                        if teacher_value is not None:
                            value_distill_loss = F.mse_loss(values, teacher_value.squeeze(-1))
                
                # Combined losses
                actor_loss = (a_loss 
                            - entropy * self.entropy_coef 
                            + b_loss * self.bounds_loss_coef
                            + latent_loss * self.latent_loss_weight
                            + policy_distill_loss * self.policy_distill_weight)
                
                critic_loss = (0.5 * c_loss * self.critic_coef 
                             + value_distill_loss * self.value_distill_weight)
                
                # Optimize
                if train_actor:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=train_critic)
                
                if train_critic and not self.freeze_critic:
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                
                with torch.no_grad():
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
                
                if self.truncate_grads:
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
                
                # Track losses
                if train_actor:
                    a_losses.append(a_loss.detach())
                    ep_kls.append(kl.detach())
                    entropies.append(entropy.detach())
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss.detach())
                    mus.append(mu.detach())
                    sigmas.append(sigma.detach())
                    
                    # Track distillation losses
                    if self.teacher_model is not None:
                        latent_losses.append(latent_loss.detach())
                        if self.policy_distill_weight > 0:
                            policy_distill_losses.append(policy_distill_loss.detach())
                        if self.value_distill_weight > 0:
                            value_distill_losses.append(value_distill_loss.detach())
                
                if train_critic:
                    c_losses.append(c_loss.detach())
                
                self.experience_buffer.update_mu_sigma(mu.detach(), sigma.detach())
            
            # Update normalization statistics only once
            if self.normalize_input:
                self.running_mean_std_obs.eval()
                self.running_mean_std_states.eval()
            
            if len(ep_kls) > 0:
                mean_kls = torch.mean(torch.stack(ep_kls))
                kls.append(mean_kls.detach())
            else:
                mean_kls = torch.tensor(0.0, device=self.device)
                kls.append(mean_kls)
            
            # Update learning rates
            if train_actor and self.lr_schedule_actor == 'adaptive' and len(ep_kls) > 0:
                self.actor_lr = self.scheduler_actor.update(self.actor_lr, mean_kls.item())
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_lr
            if train_critic and not self.freeze_critic and self.lr_schedule_critic == 'adaptive' and len(ep_kls) > 0:
                self.critic_lr = self.scheduler_critic.update(self.critic_lr, mean_kls.item())
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_lr
        
        self.rl_train_time += time.time() - start_t
        
        # Store distillation losses for logging
        self.latent_losses = latent_losses
        self.policy_distill_losses = policy_distill_losses
        self.value_distill_losses = value_distill_losses
        
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas
    
    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas):
        """Override to add distillation loss logging"""
        # Call parent write_stats
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms, mus, sigmas)
        
        # Add distillation losses
        if hasattr(self, 'latent_losses') and len(self.latent_losses) > 0:
            self.writer.add_scalar('distillation/latent_loss', 
                                 torch.mean(torch.stack(self.latent_losses)).item(), 
                                 self.agent_steps)
        
        if hasattr(self, 'policy_distill_losses') and len(self.policy_distill_losses) > 0:
            self.writer.add_scalar('distillation/policy_distill_loss',
                                 torch.mean(torch.stack(self.policy_distill_losses)).item(),
                                 self.agent_steps)
        
        if hasattr(self, 'value_distill_losses') and len(self.value_distill_losses) > 0:
            self.writer.add_scalar('distillation/value_distill_loss',
                                 torch.mean(torch.stack(self.value_distill_losses)).item(),
                                 self.agent_steps)

