import hydra
import numpy as np
import omegaconf
import torch
import os
import copy
from isaacgym import gymapi, gymtorch, torch_utils, gymutil
import rl.tasks.factory.factory_control as fc
from rl.tasks.factory.factory_schema_class_task import FactoryABCTask
from rl.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from rl.tasks.amazon_drawer.drawer_env_pulling import DrawerEnvPulling
from rl.tasks.tacsl.tacsl_franka_control import TacSLFrankaControl
from rl.tasks.tacsl.tacsl_task_image_augmentation import TacSLTaskImageAugmentation
from rl.tacsl_sensors.shear_tactile_viz_utils import visualize_tactile_shear_image
import cv2
import csv

class DrawerTaskPulling(TacSLTaskImageAugmentation, TacSLFrankaControl, DrawerEnvPulling, FactoryABCTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self._get_task_yaml_params()
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        self._acquire_task_tensors()

        if self.cfg_task.env.use_isaac_gym_tactile:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            # Open finger to render nominal tactile sensor
            self.initialize_franka_robot_open_hand()
            # Initialize tactile sensors
            self.initialize_tactile_rgb_camera()

        if self.cfg_task.env.use_shear_force:
            assert self.cfg_task.env.use_gelsight, "shear force currently works only with gelsight fingers"
            num_divs = [self.cfg_task.env.num_shear_rows, self.cfg_task.env.num_shear_cols]
            self.initialize_penalty_based_tactile(num_divs=num_divs)
        
        if self.viewer is not None:
            self._set_viewer_params()
            
        if self.cfg_base.mode.export_scene:
            self.export_scene(label='drawer_task_pulling')
        
        self.set_friction_damping_params(joint_friction=self.cfg_task.env.joint_friction,
                                    joint_damping=self.cfg_task.env.joint_damping)

        self.image_obs_keys = [k for k, v in self.obs_dims.items() if len(v) > 2 and 'force_field' not in k and not k.endswith('_depth') and not k.endswith('_seg')]
        self.init_image_augmentation()

        self.reset_idx(torch.arange(self.num_envs))
    
    
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)

        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # keypoint tensors goal
        self.keypoints_goal = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), device=self.device)
        self.keypoints_drawer_handle_for_goal = torch.zeros_like(self.keypoints_goal, device=self.device)
        
        self.keypoints_drawer_handle_alignment = torch.zeros_like(self.keypoints_goal, device=self.device)
        self.keypoints_ee_alignment = torch.zeros_like(self.keypoints_goal, device=self.device)

        self.identity_quat = \
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)

        self._actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        local_keypoints_for_goal = torch.zeros_like(self.keypoints_goal, dtype=torch.float32, device=self.device)
        local_keypoints_for_goal[:, :, 0] = torch.linspace(0.0, 1.0, self.cfg_task.rl.num_keypoints, device=self.device) - 0.5
        
        drawer_handle_offset = -0.132957
        goal2drawer_pos = torch.tensor([self.cfg_task.rl.goal_drawer_dof + drawer_handle_offset, 0, 0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        goal2world_pos = torch_utils.tf_apply(
            self.drawer_quat,
            self.drawer_pos,
            goal2drawer_pos
        )
        goal2world_quat = self.drawer_quat
        
        self.keypoints_goal = torch_utils.tf_apply(
            goal2world_quat.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 4),
            goal2world_pos.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            local_keypoints_for_goal
        )
        
        self.keypoints_drawer_handle_for_goal = torch_utils.tf_apply(
            self.drawer_handle_quat.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 4),
            self.drawer_handle_pos.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            local_keypoints_for_goal
        )
        
        local_handle_keypoints_for_alignment = torch.zeros_like(self.keypoints_drawer_handle_alignment, dtype=torch.float32, device=self.device)
        local_handle_keypoints_for_alignment[:, :, 1] = torch.linspace(0.0, 1.0, self.cfg_task.rl.num_keypoints, device=self.device) - 0.5
        local_ee_keypoints_for_alignment = torch.zeros_like(self.keypoints_ee_alignment, dtype=torch.float32, device=self.device)
        local_ee_keypoints_for_alignment[:, :, 1] = torch.linspace(0.0, 1.0, self.cfg_task.rl.num_keypoints, device=self.device) - 0.5
        
        self.keypoints_drawer_handle_alignment = torch_utils.tf_apply(
            self.drawer_handle_quat.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 4),
            self.drawer_handle_pos.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            local_handle_keypoints_for_alignment
        )
        self.keypoints_ee_alignment = torch_utils.tf_apply(
            self.finger_centered_quat.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 4),
            self.finger_centered_pos.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            -local_ee_keypoints_for_alignment
        )
        
    def refresh_all_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        
    def pre_physics_step(self, actions):
        """Optionally reset environments at the end of episodes. Apply actions from policy as position/rotation targets."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        if self.cfg_task.randomize.get("use_force_perturb", False):
            apply_force_perturb_boolean = self.dof_pos[:,9] < self.dof_when_force_perturb
            
            drawer_handle = self.drawer_object.actor_handles['drawer']
            env_ptr = self.envs[0]
            rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, drawer_handle)
            body_start = self.gym.get_actor_rigid_body_index(env_ptr,drawer_handle,0,gymapi.DOMAIN_SIM)
            force_tensor = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
            
            # if no contact force between elastomer and drawer handle, set velocity of drawer to zero
            no_contact_envs = (self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_left']].norm(dim=-1) < 1e-3) | \
                                (self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_right']].norm(dim=-1) < 1e-3)
            self.dof_vel[apply_force_perturb_boolean & no_contact_envs, 9] = 0.0
            
            contact_envs = ~no_contact_envs
            
            if self.cfg_task.get("use_randomized_force_perturb", False):
                # force from noise range
                force_perturb_amount = torch.rand((self.num_envs,), device=self.device) * (self.cfg_task.randomize.force_perturb_range[1] - self.cfg_task.randomize.force_perturb_range[0]) + self.cfg_task.randomize.force_perturb_range[0]
                force_perturb_amount = force_perturb_amount[apply_force_perturb_boolean & contact_envs]
            else:
                force_perturb_amount = self.cfg_task.randomize.get("force_perturb_amount", -2.0)
            force_tensor[apply_force_perturb_boolean & contact_envs , body_start:body_start+len(rb_dict), 1] = force_perturb_amount
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_tensor),
                None,
                gymapi.ENV_SPACE
            )
            actor_ids_sim_int32 = torch.cat(
                (self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device),
                self.actor_ids_sim_tensors['drawer'].to(dtype=torch.int32, device=self.device)),
                dim=0
            )
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(actor_ids_sim_int32),
                                                len(actor_ids_sim_int32))
        
        # if self.dof_pos[0, 9] < -0.125:
        #     for env_id in range(self.num_envs):
        #         env_ptr = self.envs[env_id]
                
        #         rb_dict = self.gym.get_actor_rigid_body_dict(env_ptr, drawer_handle)
        #         body_start = self.gym.get_actor_rigid_body_index(env_ptr,drawer_handle,0,gymapi.DOMAIN_SIM)
        #         force_tensor = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
        #         # force_tensor[env_id, body_start:body_start+len(rb_dict), 2] = -(11.04974 - 5.52487) * 9.83  # increase weight
        #         force_tensor[env_id, body_start:body_start+len(rb_dict), 1] = -2
        #         # force_tensor[env_id, body_start:body_start+len(rb_dict), 1] = -1e100
        #         # force_tensor[env_id, :, :] = -1e10
        #         self.gym.apply_rigid_body_force_tensors(
        #             self.sim,
        #             gymtorch.unwrap_tensor(force_tensor),
        #             None,
        #             gymapi.ENV_SPACE
        #         )
            # drawer_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, drawer_handle)
            # drawer_rb_props[0].mass = 0.1  # make drawer static during reset
            # self.gym.set_actor_rigid_body_properties(env_ptr, drawer_handle, drawer_rb_props, recomputeInertia=True)
            
        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        pose_actions = self._actions[:, 0:6].clone()
        gripper_action = self._actions[:, 6].clone().unsqueeze(-1)
        # print(gripper_action[0,0])

        # # [x, y, z, rx, ry, rz]
        pose_actions[:, 0] = 0.0  # disable x
        pose_actions[:, 2] = 0.0  # disable z
        pose_actions[:, 3:6] = 0.0  # disable rot

        # gripper action is -1 to 1. map it to 0 to 0.08
        # gripper_action += 1.0
        # gripper_action /= 2.0
        # gripper_action *= self.cfg_task.env.franka_gripper_width_max

        # map -1 to 1 to 0.007 to 0.009
        gripper_action += 1.0 # now 0 to 2
        gripper_action /= 2.0 # now 0 to 1
        gripper_action = (gripper_action * 0.002) + 0.001 # now 0.001 to 0.003
        # gripper_action = 0.001
        # gripper_action = 0.005 + (0.005 * gripper_action)

        # if self.cfg_task.env.get('binary_grasp', False):
        #     gripper_action = 0.0 if gripper_action < 0.04 else self.cfg_task.env.franka_gripper_width_max

        if self.progress_buf[0] == 1:
            gripper_action[:] = 0.003
        # print(self.dof_pos[0,9], gripper_action[0,0], self._actions[0,1])

        # gripper_action = 0.0

        # clamp
        # gripper_action = torch.clamp(gripper_action, 0.0, self.cfg_task.env.franka_gripper_width_max)

        # print(pose_actions[0], gripper_action[0])

        self._apply_actions_as_ctrl_targets(actions=pose_actions,
                                            ctrl_target_gripper_dof_pos=gripper_action,
                                            do_scale=True)

        sim_dt_noise = self.cfg_task.env.get("sim_dt_noise", 0)
        if sim_dt_noise > 0.0:
            sim_params = self.gym.get_sim_params(self.sim)
            sim_params.dt = self.cfg_base.sim.dt * (1 + torch.rand(1) * sim_dt_noise)
            self.gym.set_sim_params(self.sim, sim_params)

        num_extra_control_steps = self.cfg_task.env.get("num_additional_control_steps", 0)  # for backward compatibility
        num_additional_control_steps_noise = self.cfg_task.env.get("num_additional_control_steps_noise", 0)
        if num_additional_control_steps_noise:
            num_extra_control_steps += torch.randint(0,
                                                     self.cfg_task.env.num_additional_control_steps_noise + 1,
                                                     (1,)
                                                     )[0].item()
        self.execute_control_loop(num_extra_control_steps)
            
    def reset(self):
        self.compute_observations()
        self.extras = {}

        obs_and_states_dict = dict()
        if self.use_dict_obs:
            obs_and_states_dict['obs'] = {
                k: torch.clamp(self.obs_dict[k], -self.clip_obs, self.clip_obs) for k in self.obs_dims
            }

            # asymmetric actor-critic
            if self.state_dims:
                obs_and_states_dict['states'] = {
                    k: torch.clamp(self.obs_dict[k], -self.clip_obs, self.clip_obs) for k in self.state_dims
                }
        else:
            obs_and_states_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                obs_and_states_dict['states'] = self.get_state()

        return obs_and_states_dict

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.refresh_all_tensors()
        self.compute_observations()
        self.compute_reward()

        self.visualize_keypoints(
            keypoints_vis=self.cfg_task.rl.visualize_keypoints,
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            # Check if plug is at the goal location within the socket
            task_success = self._check_success()
            # print('task success mean: ', torch.mean(task_success.float()).item())
            # print('task fail mean: ', torch.mean(self.fails.float()).item())
            # print("test mean: ", torch.mean( ((self.dof_pos[:, 9] <= self.cfg_task.rl.goal_drawer_dof) & (~self.fails)).float() ).item())
            # print()
            self.rew_buf[:] += task_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(task_success.float())


        self.prev_actions[:] = self._actions

        if self.cfg_base.mode.export_states:
            self.extract_poses()
    
    def compute_observations(self):
        """Compute observations."""

        if self.cfg_task.env.use_dict_obs:
            return self.compute_observations_dict_obs()

        return self.obs_buf  # shape = (num_envs, num_observations)
    

    def compute_observations_dict_obs(self):
        """
        Compute observations as a dictionary.

        Returns:
            obs_dict: Dictionary containing observations.
        """
        # print(self.finger_midpoint_pos[0])
        self.obs_dict['ee_pos'][:] = self.fingertip_midpoint_pos
        self.obs_dict['ee_quat'][:] = self.fingertip_midpoint_quat

        if 'dof_pos' in self.cfg_task.env.obsDims or 'dof_pos' in self.cfg_task.env.stateDims or 'dof_pos' in self.cfg_task.env.additional_obs:
            self.obs_dict['dof_pos'][:] = self.dof_pos

        if 'dof_vel' in self.cfg_task.env.obsDims or 'dof_vel' in self.cfg_task.env.stateDims or 'dof_vel' in self.cfg_task.env.additional_obs:
            self.obs_dict['dof_vel'][:] = self.dof_vel

        if 'ee_lin_vel' in self.cfg_task.env.obsDims or 'ee_lin_vel' in self.cfg_task.env.additional_obs or self.cfg_task.rl.asymmetric_observations:
            self.obs_dict['ee_lin_vel'][:] = self.fingertip_midpoint_linvel
            self.obs_dict['ee_ang_vel'][:] = self.fingertip_midpoint_angvel

        if 'drawer_handle_pos' in self.cfg_task.env.obsDims or 'drawer_handle_quat' in self.cfg_task.env.obsDims:
            self.obs_dict['drawer_handle_pos'][:] = self.drawer_handle_pos
            self.obs_dict['drawer_handle_quat'][:] = self.drawer_handle_quat

        
        if 'drawer_handle_left_elastomer_force' in self.cfg_task.env.obsDims or \
           'drawer_handle_right_elastomer_force' in self.cfg_task.env.obsDims:
            left_elastomer_and_drawer_handle_contact_force = self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_left']]
            right_elastomer_and_drawer_handle_contact_force = self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_right']]
            
            # left_fingerbox_and_drawer_handle_contact_force = self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['panda_fingerbox_left']]
            # right_fingerbox_and_drawer_handle_contact_force = self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['panda_fingerbox_right']]
            self.obs_dict['drawer_handle_left_elastomer_force'][:] = left_elastomer_and_drawer_handle_contact_force
            self.obs_dict['drawer_handle_right_elastomer_force'][:] = right_elastomer_and_drawer_handle_contact_force
        
        if 'eef_to_drawer_handle_pos' in self.cfg_task.env.obsDims or 'eef_to_drawer_handle_quat' in self.cfg_task.env.obsDims:
            # get drawer_handle2ee pose
            world2drawer_handle_quat, world2drawer_handle_pos = torch_utils.tf_inverse(
                self.drawer_handle_quat, self.drawer_handle_pos
            )
            ee2drawer_handle_quat, ee2drawer_handle_pos = torch_utils.tf_combine(
                world2drawer_handle_quat, world2drawer_handle_pos,
                self.fingertip_midpoint_quat, self.fingertip_midpoint_pos
            )
            drawer_handle2ee_quat, drawer_handle2ee_pos = torch_utils.tf_inverse(
                ee2drawer_handle_quat, ee2drawer_handle_pos
            )
            
            self.obs_dict['eef_to_drawer_handle_pos'][:] = drawer_handle2ee_pos
            self.obs_dict['eef_to_drawer_handle_quat'][:] = drawer_handle2ee_quat
            
        if 'drawer_box_left_fingerbox_contact_force' in self.cfg_task.env.obsDims or \
           'drawer_box_right_fingerbox_contact_force' in self.cfg_task.env.obsDims:
            left_fingerbox_and_drawer_box_contact_force = self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_fingerbox_left']]
            right_fingerbox_and_drawer_box_contact_force = self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_fingerbox_right']]
            
            self.obs_dict['drawer_box_left_fingerbox_contact_force'][:] = left_fingerbox_and_drawer_box_contact_force
            self.obs_dict['drawer_box_right_fingerbox_contact_force'][:] = right_fingerbox_and_drawer_box_contact_force

  
        if 'drawer_box_panda_leftfinger_contact_force' in self.cfg_task.env.obsDims or \
            'drawer_box_panda_rightfinger_contact_force' in self.cfg_task.env.obsDims:
            panda_leftfinger_and_drawer_box_contact_force = self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_leftfinger']]
            panda_rightfinger_and_drawer_box_contact_force = self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_rightfinger']]
            
            self.obs_dict['drawer_box_panda_leftfinger_contact_force'][:] = panda_leftfinger_and_drawer_box_contact_force
            self.obs_dict['drawer_box_panda_rightfinger_contact_force'][:] = panda_rightfinger_and_drawer_box_contact_force
        
        # tactile rgb
        if self.cfg_task.env.use_camera_obs:
        
            images = self.get_camera_image_tensors_dict()
            if self.cfg_task.env.use_isaac_gym_tactile:
                # Optionally subsample tactile image
                ssr = self.cfg_task.env.tactile_subsample_ratio
                for k in self.tactile_ig_keys:
                    images[k] = images[k][:, ::ssr, ::ssr]
            
            for cam in images:
                if cam in self.cfg_task.env.obsDims or cam in self.cfg_task.env.get('additional_obs', {}):
                    if cam.endswith('_depth'):
                        self.obs_dict[cam] = torch.clip(images[cam], 0, 2.0) # restrict depth to [0, 2] meters
                    elif cam.endswith('_seg'):
                        self.obs_dict[cam] = images[cam]
                        self.obs_dict[cam][self.obs_dict[cam] > 0] = 1  # NOTE: asssumes that we only have one segmentation id for binary mask
                    else:
                        if images[cam].dtype == torch.uint8:
                            self.obs_dict[cam][..., :3] = images[cam] / 255.
                        else:
                            self.obs_dict[cam][..., :3] = images[cam]
            if np.random.uniform(0,1) < 1.1:
                self.apply_image_augmentation_to_obs_dict()
            
            if 'left_tactile_camera_taxim_gray' in self.cfg_task.env.obsDims or 'left_tactile_camera_taxim_gray' in self.cfg_task.env.get('additional_obs', {}):
                # add left and right tactile camera gray
                left_gray = torch.mean(self.obs_dict['left_tactile_camera_taxim'][..., :3], dim=-1)
                right_gray = torch.mean(self.obs_dict['right_tactile_camera_taxim'][..., :3], dim=-1)
                if self.cfg_task.env.get('debug_vis', False) or self.record_frames:
                    vis_im_gray = torch.cat([left_gray, right_gray], dim=-1)[0] # (H, W*2)
                    vis_im_gray_np = vis_im_gray.cpu().numpy()
                    if self.cfg_task.env.get('debug_vis', False):
                        cv2.imshow('tactile_vis', vis_im_gray_np)
                        cv2.waitKey(1)
                    if self.record_frames:
                        if not os.path.isdir(self.record_frames_dir):
                            os.makedirs(self.record_frames_dir, exist_ok=True)
                        filename = f"tactile_vis_gray_{self.control_steps}.png"
                        cv2.imwrite(os.path.join(self.record_frames_dir, filename), (vis_im_gray_np * 255).astype(np.uint8))
                self.obs_dict['left_tactile_camera_taxim_gray'][:] = left_gray.unsqueeze(-1).expand(self.num_envs, left_gray.shape[1], left_gray.shape[2], 3)    # (num_envs, H, W, 3)
                self.obs_dict['right_tactile_camera_taxim_gray'][:] = right_gray.unsqueeze(-1).expand(self.num_envs, right_gray.shape[1], right_gray.shape[2], 3)  # (num_envs, H, W, 3)
        
            # left_vis_im = self.obs_dict['left_tactile_camera_taxim'][0]
            # right_vis_im = self.obs_dict['right_tactile_camera_taxim'][0]
            # vis_im = torch.cat([left_vis_im, right_vis_im], dim=1)  # (H, W*2, 3)
            # vis_im = (vis_im * 255.).to(torch.uint8).squeeze(0).cpu().numpy()
            # # resize to 80x120
            # vis_im = cv2.resize(vis_im, (120*4, 80*4), interpolation=cv2.INTER_LINEAR)
            # vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
            # vis_im = cv2.cvtColor(vis_im, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('nay', vis_im)
            # cv2.waitKey(1)
        
        if self.cfg_task.env.use_shear_force:
            if not self.cfg_task.env.use_shear_3d:

                if self.cfg_task.env.get('use_tacsl_shear', False):
                    tactile_force_field_dict = self.get_tactile_force_field_tensors_dict(debug=self.cfg_task.env.get('debug_vis', False), return_depth=False, normalize_shear=self.cfg_task.env.get("normalize_tacsl_shear_obs", False))

                    if self.cfg_task.env.use_tactile_field_obs:
                        for k in ['tactile_force_field_left', 'tactile_force_field_right']:
                            if k in self.cfg_task.env.obsDims:
                                self.obs_dict[k][:] = tactile_force_field_dict[k][..., 1:]
                                # if self.cfg_task.env.zero_out_normal_force_field_obs:
                                    # self.obs_dict[k][..., 0] *= 0.0
                    
                    # # visualize shear_images
                    if self.cfg_task.env.get('debug_vis', False) or self.record_frames:
                        shear_images = {k: torch.tensor(tactile_force_field_dict[k], device=self.device) for k in ['tactile_force_field_left_shear', 'tactile_force_field_right_shear'] if k in tactile_force_field_dict}

                        
                        shear_img_concat = np.concatenate([
                            shear_images['tactile_force_field_left_shear'].detach().cpu().numpy(),
                            shear_images['tactile_force_field_right_shear'].detach().cpu().numpy()
                        ], axis=1)
                        if self.cfg_task.env.get('debug_vis', False):
                            cv2.imshow('Shear Images', shear_img_concat)
                            cv2.waitKey(1)
                        if self.record_frames:
                            if not os.path.isdir(self.record_frames_dir):
                                os.makedirs(self.record_frames_dir, exist_ok=True)
                            filename = f"tacsl_shear_vis_{self.control_steps}.png"
                            cv2.imwrite(os.path.join(self.record_frames_dir, filename), (shear_img_concat * 255).astype(np.uint8))
            
                elif self.cfg_task.env.get("use_hydrosoft_model", False):
                    marker_displacement_dict = self.get_force_fields_dict(self.drawer_handle_quat, self.drawer_handle_pos) # hydrofots
                elif self.cfg_task.env.get("use_fots_model", False) or self.cfg_task.env.get("use_old_fots_model", False):
                    marker_displacement_dict = self.get_displacement_field_dict(self.drawer_handle_quat, self.drawer_handle_pos) # fots
                else:
                    raise ValueError(
                        "use_shear_force is on but no tactile backend is selected, so there is "
                        "no field to read. Set exactly one of use_tacsl_shear, use_hydrosoft_model, "
                        "use_fots_model or use_old_fots_model, or turn the tactile mode off "
                        "(task.env.student_tactile_mode=False). Without this the next line "
                        "dereferences an unbound dict."
                    )
                
                if self.cfg_task.env.use_tactile_field_obs and not self.cfg_task.env.get('use_tacsl_shear', False):
                    for k in ['elastomer_left', 'elastomer_right']:
                        shear = marker_displacement_dict[k][..., :2].flip(1, -1) # torch tensors need to flip this way
                        shear[..., 0] *= -1
                        if k == 'elastomer_left':
                            self.obs_dict[f'tactile_force_field_left'][:] = shear
                        else:
                            self.obs_dict[f'tactile_force_field_right'][:] = shear

                    # N0-zero, the primary no-touch control: keep the network, the observation
                    # dimensions and the tactile pipeline identical to H0 and zero only the VALUES
                    # the actor reads. N0 removes the tactile entries entirely, which also deletes
                    # two convolutional preprocessors, so it cannot separate "touch does not help"
                    # from "a different network does better". Defaults False, so no existing run
                    # changes behaviour.
                    if self.cfg_task.env.get('zero_out_tactile_field_obs', False):
                        self.obs_dict['tactile_force_field_left'][:] = 0.0
                        self.obs_dict['tactile_force_field_right'][:] = 0.0
                
                    if self.cfg_task.env.get('debug_vis', False) or self.record_frames:
                        left_marker_im = marker_displacement_dict['elastomer_left'][0].clone().cpu().numpy()
                        right_marker_im = marker_displacement_dict['elastomer_right'][0].clone().cpu().numpy()
                        leftboi = left_marker_im[..., :2][::-1, :, ::-1]
                        leftboi[..., 0] *= -1
                        
                        rightboi = right_marker_im[..., :2][::-1, :, ::-1]
                        rightboi[..., 0] *= -1

                        left_vis_image = visualize_tactile_shear_image(left_marker_im[..., 2], leftboi, shear_force_threshold=5.0, resolution=40)
                        right_vis_image = visualize_tactile_shear_image(right_marker_im[..., 2], rightboi, shear_force_threshold=5.0, resolution=40)
                        shear_vis = np.concatenate([left_vis_image, right_vis_image], axis=1)
                        if self.cfg_task.env.get('debug_vis', False):
                            cv2.imshow('marker_shear', shear_vis)
                            cv2.waitKey(1)
                        if self.record_frames:
                            if not os.path.isdir(self.record_frames_dir):
                                os.makedirs(self.record_frames_dir, exist_ok=True)
                            if self.cfg_task.env.get('use_hydrosoft_model', False):
                                filename = f"hydrosoft_marker_shear_{self.control_steps}.png"
                                
                                # this is for fig2
                                # also save left elastomer pose and right elastomer pose and indenter pose
                                # append / create a csv file
                                if not os.path.isdir(self.record_elastomer_left_pose_dir):
                                    os.makedirs(self.record_elastomer_left_pose_dir, exist_ok=True)
                                if not os.path.isdir(self.record_elastomer_right_pose_dir):
                                    os.makedirs(self.record_elastomer_right_pose_dir, exist_ok=True)
                                if not os.path.isdir(self.record_indenter_pose_dir):
                                    os.makedirs(self.record_indenter_pose_dir, exist_ok=True)
                                if not os.path.isdir(self.record_marker_shear_left_dir):
                                    os.makedirs(self.record_marker_shear_left_dir, exist_ok=True)
                                if not os.path.isdir(self.record_marker_shear_right_dir):
                                    os.makedirs(self.record_marker_shear_right_dir, exist_ok=True)
                                if not os.path.isdir(self.record_gripper_closed_dir):
                                    os.makedirs(self.record_gripper_closed_dir, exist_ok=True)
                                
                                with open(os.path.join(self.record_elastomer_left_pose_dir, 'poses.csv'), mode='a') as f:
                                    left_elastomer_link_id = self.get_link_handle(self.elastomer_actor_names[0], self.elastomer_link_names[0])
                                    left_elastomer_quat = self.body_quat[:, left_elastomer_link_id]
                                    left_elastomer_pos = self.body_pos[:, left_elastomer_link_id]
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(left_elastomer_quat[0].cpu().numpy().tolist() + left_elastomer_pos[0].cpu().numpy().tolist())
                                with open(os.path.join(self.record_elastomer_right_pose_dir, 'poses.csv'), mode='a') as f:
                                    right_elastomer_link_id = self.get_link_handle(self.elastomer_actor_names[1], self.elastomer_link_names[1])
                                    right_elastomer_quat = self.body_quat[:, right_elastomer_link_id]
                                    right_elastomer_pos = self.body_pos[:, right_elastomer_link_id]
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(right_elastomer_quat[0].cpu().numpy().tolist() + right_elastomer_pos[0].cpu().numpy().tolist())
                                with open(os.path.join(self.record_indenter_pose_dir, 'poses.csv'), mode='a') as f:
                                    indenter_quat = self.drawer_handle_quat
                                    indenter_pos = self.drawer_handle_pos
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(indenter_quat[0].cpu().numpy().tolist() + indenter_pos[0].cpu().numpy().tolist())
                                with open(os.path.join(self.record_gripper_closed_dir, 'gripper_closed.csv'), mode='a') as f:
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    # gripper_closed = 1 if self.dof_pos[0, 9] < 0.0027 else 0
                                    gripper_action = self._actions[:, 6].clone()
                                    gripper_action += 1.0 # now 0 to 2
                                    gripper_action /= 2.0 # now 0 to 1
                                    gripper_action = (gripper_action * 0.002) + 0.001 # now 0.001 to 0.003
                                    csv_writer.writerow([gripper_action[0].cpu().numpy().tolist()])
                                if hasattr(self, 'record_marker_shear_left_idx'):
                                    self.record_marker_shear_left_idx += 1
                                else:
                                    self.record_marker_shear_left_idx = 0
                                if hasattr(self, 'record_marker_shear_right_idx'):
                                    self.record_marker_shear_right_idx += 1
                                else:
                                    self.record_marker_shear_right_idx = 0
                                # save npy marker shear values
                                np.save(os.path.join(self.record_marker_shear_left_dir, f'shears_{self.record_marker_shear_left_idx}.npy'), marker_displacement_dict['elastomer_left'][0].cpu().numpy())
                                np.save(os.path.join(self.record_marker_shear_right_dir, f'shears_{self.record_marker_shear_right_idx}.npy'), marker_displacement_dict['elastomer_right'][0].cpu().numpy())
                                # with open(os.path.join(self.record_marker_shear_left_dir, 'shears.csv'), mode='a') as f:
                                #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                #     # flatten shear values
                                #     shear_values = marker_displacement_dict['elastomer_left'][0].cpu().numpy().flatten().tolist()
                                #     csv_writer.writerow(shear_values)
                                # with open(os.path.join(self.record_marker_shear_right_dir, 'shears.csv'), mode='a') as f:
                                #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                #     # flatten shear values
                                #     shear_values = marker_displacement_dict['elastomer_right'][0].cpu().numpy().flatten().tolist()
                                #     csv_writer.writerow(shear_values)
                                
                            elif self.cfg_task.env.get('use_fots_model', False):
                                filename = f"fots_marker_shear_{self.control_steps}.png"
                            else:
                                filename = f"marker_shear_{self.control_steps}.png"
                            cv2.imwrite(os.path.join(self.record_frames_dir, filename), (shear_vis * 255).astype(np.uint8))

        return self.obs_dict
    
    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if episode length expired."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at the current timestep."""
        
        # calc keypoint goal
        a, b = 50, 0.0001
        goal_keypoint_diff = self.keypoints_goal - self.keypoints_drawer_handle_for_goal
        goal_keypoint_dist = torch.norm(goal_keypoint_diff, p=2, dim=-1)  # shape = (num_envs, num_keypoints)
        goal_keypoint_dist_mean = torch.mean(goal_keypoint_dist, dim=-1)  # shape = (num_env
        
        goal_keypoint_reward = -goal_keypoint_dist_mean
        goal_keypoint_reward_exp = 1. / (torch.exp(a * goal_keypoint_reward) + b + torch.exp(-a * goal_keypoint_reward))
        
        
        # calc keypoint alignment
        alignment_keypoint_diff = self.keypoints_drawer_handle_alignment - self.keypoints_ee_alignment
        alignment_keypoint_dist = torch.norm(alignment_keypoint_diff, p=2, dim=-1)  # shape = (num_envs, num_keypoints)
        alignment_keypoint_dist_mean = torch.mean(alignment_keypoint_dist, dim=-1)  # shape = (num_envs)
        
        alignment_keypoint_reward = -alignment_keypoint_dist_mean
        alignment_keypoint_reward_exp = 1. / (torch.exp(a * alignment_keypoint_reward) + b + torch.exp(-a * alignment_keypoint_reward))
        
        # action penalize
        action_penalty = torch.norm(self._actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self._actions[:, :6] - self.prev_actions[:, :6], p=2, dim=-1) # action grad penalty but only for pose

        gripper_action = (self._actions[:, -1] + 1.0) / 2.0 * self.cfg_task.env.franka_gripper_width_max
        gripper_prev_action = (self.prev_actions[:, -1] + 1.0) / 2.0 * self.cfg_task.env.franka_gripper_width_max
        gripper_state_changed = (torch.abs(gripper_action - gripper_prev_action) > 0.001).float()
        gripper_action_grad_penalty = gripper_state_changed if self.cfg_task.env.get('binary_grasp', False) else torch.abs(gripper_action - gripper_prev_action)

        contact_elastomer_handle = torch.norm(self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_left']], p=2, dim=-1) \
            + torch.norm(self.contact_force_pairwise[:, self.drawer_handle_body_id_env, self.franka_body_ids_env['elastomer_right']], p=2, dim=-1)
        
        # elastomer_handle_no_contact_penalty = 0.0
        # if not (contact_elastomer_handle > 0.0):
        #     elastomer_handle_no_contact_penalty = 0.1

        contact_elastomer_handle = torch.clamp(contact_elastomer_handle, 0.0, 0.5)


        # drawer_box_left_fingerbox_contact_force, drawer_box_right_fingerbox_contact_force, drawer_box_panda_leftfinger_contact_force, drawer_box_panda_rightfinger_contact_force
        contact_penalty_drawer = torch.norm(self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_fingerbox_left']], p=2, dim=-1) \
            + torch.norm(self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_fingerbox_right']], p=2, dim=-1) \
            + torch.norm(self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_leftfinger']], p=2, dim=-1) \
            + torch.norm(self.contact_force_pairwise[:, self.drawer_box_body_id_env, self.franka_body_ids_env['panda_rightfinger']], p=2, dim=-1)
            

        # print(f"alignment_keypoint_dist_mean: {alignment_keypoint_dist_mean[0].item():.5f}")
        gripper_aligned_to_handle = (alignment_keypoint_dist_mean < 0.01).float()
        gripper_closed_reward = gripper_aligned_to_handle * torch.clamp(0.08 - gripper_action, min=0.0)
        gripper_closed_reward_exp = 1. / (torch.exp(a * gripper_closed_reward) + b + torch.exp(-a * gripper_closed_reward))

        gripper_misaligned_to_handle = 1.0 - gripper_aligned_to_handle
        gripper_misaligned_to_handle_penalty = gripper_misaligned_to_handle * torch.clamp(gripper_action, min=0.0)
        # print(f"contact_elastomer_handle: {contact_elastomer_handle[0].item():.5f}")
        
        # penalize gripper action (gripper_action < 0.0028) when it isn't yet force perturbed
        gripper_action = self._actions[:, 6].clone()
        gripper_action += 1.0 # now 0 to 2
        gripper_action /= 2.0 # now 0 to 1
        gripper_action = (gripper_action * 0.002) + 0.001 # now 0.001 to 0.003

        if self.progress_buf[0] == 1:
            gripper_action[:] = 0.003

        gripper_action_below_thresh = (gripper_action < 0.0029)
        dof_not_yet_perturbed = (self.dof_pos[:,9] > self.dof_when_force_perturb)
        dof_not_yet_perturbed_padded = (self.dof_pos[:,9] > self.dof_when_force_perturb + self.cfg_task.randomize.get("dof_padding_fail_preperturb", 0.0))
        gripper_action_perturb_penalty = gripper_action_below_thresh.float() * dof_not_yet_perturbed.float() * self.cfg_task.rl.get("gripper_force_perturb_penalty_scale", 0.01)
        
        gripper_action_postperturb_reward = (gripper_action <= 0.0015).float() * (~dof_not_yet_perturbed).float() * self.cfg_task.rl.get("gripper_post_perturb_reward_scale", 0.0)
        
        dense_gripper_action_postperturb_reward = torch.abs(self.contact_force_pairwise[:, self.franka_body_ids_env['elastomer_left'], self.drawer_handle_body_id_env, 2]) + \
                                                            torch.abs(self.contact_force_pairwise[:, self.franka_body_ids_env['elastomer_right'], self.drawer_handle_body_id_env, 2])
        dense_gripper_action_postperturb_reward = dense_gripper_action_postperturb_reward * (~dof_not_yet_perturbed).float() * self.cfg_task.rl.get("dense_gripper_post_perturb_reward_scale", 0.0)
        

        # if gripper_action_below_thresh and dof_not_yet_perturbed -> set to fail
        self.fails = self.fails | (gripper_action_below_thresh & dof_not_yet_perturbed_padded)
        # print("Progress Buff: ", self.progress_buf, "Gripper Action: ", gripper_action[0].item(), " DOF: ", self.dof_pos[0,9].item(), " FAIL: ", self.fails[0].item())
        
        self.rew_buf[:] = goal_keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          + goal_keypoint_reward_exp * self.cfg_task.rl.keypoint_reward_scale \
                          + alignment_keypoint_reward * self.cfg_task.rl.alignment_keypoint_reward_scale \
                          + alignment_keypoint_reward_exp * self.cfg_task.rl.alignment_keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          - action_grad_penalty * self.cfg_task.rl.action_gradient_penalty_scale \
                          - contact_penalty_drawer * self.cfg_task.rl.get("drawer_contact_penalty_scale", 0.01) \
                          - gripper_action_grad_penalty * self.cfg_task.rl.gripper_action_gradient_penalty_scale \
                          - (gripper_action_perturb_penalty if self.cfg_task.randomize.get("use_force_perturb", False) or self.cfg_task.rl.get("use_preperturb_penalty_anyways", False) else gripper_action_perturb_penalty * 0.0) \
                          + gripper_action_postperturb_reward \
                          + dense_gripper_action_postperturb_reward
                        #   - elastomer_handle_no_contact_penalty
                        #   + contact_elastomer_handle * self.cfg_task.rl.handle_grasp_reward_scale 
                        #   + gripper_closed_reward * self.cfg_task.rl.handle_grasp_reward_scale \
                        #   + gripper_closed_reward_exp * self.cfg_task.rl.gripper_closed_reward_scale \
                        #   - gripper_misaligned_to_handle_penalty * self.cfg_task.rl.gripper_closed_reward_scale \
                          
        
    def reset_idx(self, env_ids):
        """Reset specified environments."""
        # randomize when to apply force perturbation
        self.fails = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg_task.randomize.get("use_force_perturb", False):
            self.dof_when_force_perturb = torch.rand(self.num_envs, device=self.device) * -0.08 - 0.02 # between [-0.02, -0.10] m
        else:
            self.dof_when_force_perturb = torch.ones(self.num_envs, device=self.device) * -1.0  # -1.0 (impossible to reach)
        
        if self.cfg_task.randomize.randomize_compliance:
            if self.cfg_task.env.use_compliant_contact:
                # sample_compliance
                k_range = self.cfg_task.randomize.compliance_stiffness_range
                d_range = self.cfg_task.randomize.compliance_damping_range
                ks = k_range[0] + torch.rand(self.num_envs, device=self.device) * (k_range[1] - k_range[0])
                ds = d_range[0] + torch.rand(self.num_envs, device=self.device) * (d_range[1] - d_range[0])
                # set sampled compliance params for each env
                for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
                    self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                                      elastomer_link_name=elastomer_link_name,
                                                      compliance_stiffness=ks,
                                                      compliant_damping=ds,
                                                      use_acceleration_spring=False)

        # Randomize drawer joint friction
        if self.cfg_task.randomize.get('randomize_drawer_friction', True):
            friction_range = self.cfg_task.randomize.get('drawer_friction_range', [0.001, 0.01])
            for env_id in env_ids:
                env_ptr = self.env_ptrs[env_id]
                drawer_handle = self.actor_handles['drawer']
                drawer_dof_props = self.gym.get_actor_dof_properties(env_ptr, drawer_handle)
                # Randomize friction for the prismatic joint (index 0)
                sampled_friction = friction_range[0] + torch.rand(1).item() * (friction_range[1] - friction_range[0])
                drawer_dof_props['friction'][0] = sampled_friction
                self.gym.set_actor_dof_properties(env_ptr, drawer_handle, drawer_dof_props)

        # randomize controller parameters
        if self.cfg_task.randomize.randomize_ctrl_params:
            assert self.cfg_task.ctrl.ctrl_type == 'task_space_impedance', 'controller randomization currently works only for task_space_impedance'
            # use default controller params when randomizing initial state
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                                                            device=self.device).repeat((self.num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains, device=self.device).repeat(
                (self.num_envs, 1))

        if self.cfg_task.ige_dr.randomize:
            # use initial joint friction and damping during environment initialization
            for env_id in range(self.num_envs):
                env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']
                franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
                franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
                franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
                self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)

        self._reset_franka(env_ids)

        self.disable_gravity()  # to prevent plug from falling
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()
        
        if self.cfg_task.env.use_shear_force:
            if self.cfg_task.env.use_hydrosoft_model:
                self.reset_hydrofots()

        # self._move_gripper_to_dof_pos(gripper_dof_pos=self.cfg_task.env.get("franka_open_gripper_width", 0.0),
        #                               sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.003,
                                      sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        
        if self.cfg_task.env.use_shear_force:
            if self.cfg_task.env.use_fots_model or self.cfg_task.env.get("use_old_fots_model", False):
                self.reset_fots(self.drawer_handle_quat, self.drawer_handle_pos)
        
        self.enable_gravity(gravity_vec=self.cfg_base.sim.gravity)

        # finger_midpoint_to_drawer_handle_offset = self.drawer_handle_pos - self.finger_centered_pos
        # print(finger_midpoint_to_drawer_handle_offset[0])
        if self.cfg_task.env.use_shear_force:
            if self.cfg_task.env.use_hydrosoft_model:
                self.hydrofots_sensors[0].hydrosoft_forces[:, :, :2] *= 0.0 # set tangent forces to 0
                self.hydrofots_sensors[1].hydrosoft_forces[:, :, :2] *= 0.0 # set tangent forces to 0

        if self.cfg_task.randomize.randomize_ctrl_params:
            self.randomize_controller_params()

        if 'ige_dr' in self.cfg_task and self.cfg_task.ige_dr.randomize:
            # Must be executed before resetting self.reset_buf
            self.envs = self.env_ptrs  # DR code looks for self.envs
            self.apply_randomizations(self.cfg_task.ige_dr.randomization_params)
            self.set_gripper_friction_to_default()  # Don't randomize gripper friction/dynamic params, reset to default values
        
        self._reset_buffers(env_ids)

        self.reset_image_augmentation()
        
    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka to initial joint configuration
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:9] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)
        self.dof_pos[:, 9:] = 0.0  # drawer dof to reset

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids, 0:10] = 0.0

        actor_ids_sim_int32 = torch.cat(
            (self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[env_ids],
             self.actor_ids_sim_tensors['drawer'].to(dtype=torch.int32, device=self.device)[env_ids]),
            dim=0
        )
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(actor_ids_sim_int32),
                                              len(actor_ids_sim_int32))

        self._reset_franka_actuation(self.ctrl_target_dof_pos)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps,
                                     ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_open_gripper_width",
                                                                                       self.asset_info_franka_table.franka_gripper_width_max))
    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
    def _reset_object(self):
        """Reset root state of object."""
        pass    
    
    def _set_viewer_params(self):
        """Set viewer parameters."""

        # cam_pos = gymapi.Vec3(-0.432, -0.71,  0.7196)
        cam_pos = gymapi.Vec3(1.1, 0.0,  0.3196)
        cam_target = gymapi.Vec3(0., 0.0, 0.18)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _check_success(self):
        """Check for task success."""
        is_dof_past_thresh = self.dof_pos[:, 9] <= (self.cfg_task.rl.goal_drawer_dof + self.cfg_task.rl.get("goal_drawer_dof_success_padding", 0.0) + self.cfg_task.rl.get("goal_drawer_dof_success_padding", 0.0))
        task_success = (is_dof_past_thresh & (~self.fails)) if self.cfg_task.randomize.get("use_force_perturb", False) else is_dof_past_thresh

        # print(f"is_dof_past_thresh: {is_dof_past_thresh.float().mean().item():.3f}, fails: {self.fails.float().mean().item():.3f}")
        # task_success = (self.dof_pos[:, 9] <= self.cfg_task.rl.goal_drawer_dof)
        return task_success
    
    def visualize_keypoints(self, **kwargs):
        
        # Debug drawing only. Gated for two reasons:
        #  1. it calls gymutil.ArrowGeometry, which does NOT exist in the shipped
        #     IsaacGym_Preview_TacSL_Package (only Axes/Line/WireframeBBox/Box/Sphere),
        #     so stage 2 crashes the moment force perturbation is enabled;
        #  2. there is no viewer in headless training, so none of it is visible anyway.
        # The original guard, hasattr(self, 'dof_when_force_perturb'), becomes true
        # exactly when force perturbation is on, i.e. stage 2 onward.
        if self.headless or not kwargs.get('keypoints_vis', False):
            return

        # visualize plane based on self.dof_
        if hasattr(self, 'dof_when_force_perturb'):
            for env_idx in range(self.num_envs):
                plane_y = -self.dof_when_force_perturb[env_idx].item()
                plane_pose = gymapi.Transform()
                # plane_pose.p = gymapi.Vec3(0.5127, plane_y + self.drawer_pos[env_idx,1].item() + 0.29 / 2, 0.1578 + 0.01)
                plane_pose.p = gymapi.Vec3(0.5127, 0.29 / 2 - 0.15, 0.30 + 0.01)
                if self.dof_pos[env_idx,9] < self.dof_when_force_perturb[env_idx].item():
                    # plane_geom = gymutil.WireframeBoxGeometry(0.5, 0.0, 0.5, pose=plane_pose, color=color)
                    # arrow is pointing up, make it point in -y direction using quaternion
                    arrow_quat = torch_utils.quat_from_euler_xyz(torch.tensor([torch.pi/2]), torch.tensor([0]),torch.tensor([0])).flatten()
                    plane_pose.r = gymapi.Quat(arrow_quat[0].item(), arrow_quat[1].item(), arrow_quat[2].item(), arrow_quat[3].item())
                    plane_geom = gymutil.ArrowGeometry(scale=0.01, pose=plane_pose)
                    gymutil.draw_lines(plane_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)

            # for env_idx in range(self.num_envs):
            #     # add another plane for self.cfg_task.rl.goal_drawer_dof
            #     plane_y = -self.cfg_task.rl.goal_drawer_dof
            #     plane_pose = gymapi.Transform()
            #     plane_pose.p = gymapi.Vec3(0.5127, plane_y + self.drawer_pos[env_idx,1].item() + 0.29 / 2, 0.1578 + 0.01)
            #     plane_geom = gymutil.WireframeBoxGeometry(0.5, 0.0, 0.5, pose=plane_pose, color=(0.0, 0.0, 1.0))
            #     gymutil.draw_lines(plane_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)

        if kwargs['keypoints_vis']:
            for env_idx in range(self.num_envs):
                goal_keypoints = self.keypoints_goal[env_idx] 
                handle_keypoints = self.keypoints_drawer_handle_for_goal[env_idx]
                
                for i, point in enumerate(handle_keypoints):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(0.0, 0.0, 1.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
                
                for i, point in enumerate(goal_keypoints):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(1.0, 0.0, 0.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
                
                alignment_keypoints_ee = self.keypoints_ee_alignment[env_idx]
                alignment_keypoints_handle = self.keypoints_drawer_handle_alignment[env_idx]
                
                for i, point in enumerate(alignment_keypoints_handle):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(0.0, 1.0, 0.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
                    
                for i, point in enumerate(alignment_keypoints_ee):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(1.0, 1.0, 0.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
    