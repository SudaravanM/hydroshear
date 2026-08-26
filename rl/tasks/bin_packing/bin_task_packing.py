import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F

from isaacgym import gymutil
from isaacgym.gymutil import WireframeSphereGeometry

from isaacgym import gymapi, gymtorch, torch_utils
from rl.tasks.factory.factory_schema_class_task import FactoryABCTask
from rl.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from rl.tasks.bin_packing.bin_env_packing import BinEnvPacking
from rl.tasks.tacsl.tacsl_franka_control import TacSLFrankaControl
from rl.tasks.tacsl.tacsl_task_image_augmentation import TacSLTaskImageAugmentation
from rl.tacsl_sensors.shear_tactile_viz_utils import visualize_tactile_shear_image
import cv2
import csv
import os

class BinTaskPacking(TacSLTaskImageAugmentation, TacSLFrankaControl, BinEnvPacking, FactoryABCTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""
        self.cfg = cfg
        self._get_task_yaml_params()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # we can use cfg_task past this point since super().__init__() has been called
        
        # self.inhand_pos = torch.zeros_like(self.fingertip_midpoint_pos, device=self.device, dtype=self.fingertip_midpoint_pos.dtype)
        self.inhand_pos = torch.tensor(self.cfg_task.randomize.inhand_pos_initial, device=self.device, dtype=self.fingertip_midpoint_pos.dtype).unsqueeze(0).expand(self.num_envs, 3)  # (num_envs, 3)
        # self.inhand_pos = torch.tensor([0.0, 0.0, 0.025], device=self.device, dtype=self.fingertip_midpoint_pos.dtype).unsqueeze(0).expand(self.num_envs, 3)  # (num_envs, 3)
        
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

        if self.cfg_task.env.task_type == 'placement':
            # The placement task moves the peg to the tip of the placement pad, no insertion
            self.cfg_task.rl.insertion_frac = 0.0

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_base.mode.export_scene:
            self.export_scene(label='bin_task_insertion')

        self.set_friction_damping_params(joint_friction=self.cfg_task.env.joint_friction,
                                         joint_damping=self.cfg_task.env.joint_damping)

        self.image_obs_keys = [k for k, v in self.obs_dims.items() if len(v) > 2 and 'force_field' not in k and not k.endswith('_depth') and not k.endswith('_seg')]
        self.init_image_augmentation()

        self.reset_idx(torch.arange(self.num_envs))
    
    def refresh_all_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
    
    def _check_insertion_obj_upright(self):
        z_up = torch.zeros_like(self.insertion_obj_pos, device=self.device, dtype=torch.float32)
        z_up[:, 2] = 1.0  # (num_envs, 3)
        
        # cos(theta) = dot(v1,v2) / (||v1|| * ||v2||)
        # ||v1| = |v2| = 1.0
        obj_upright_vector = torch_utils.quat_apply(self.insertion_obj_quat, z_up)
        cos_angle_between_vectors = torch.sum(obj_upright_vector * z_up, dim=-1)
        
        angle_thresh = np.deg2rad(15)
        return cos_angle_between_vectors > np.cos(angle_thresh)  # (num_envs,)

    def _check_insertion_obj_centered(self, threshold_multiplier=0.5):
        # just check x,y between insertion object goal
        
        # get pose relative to goal
        world2goal_quat, world2goal_pos = torch_utils.tf_inverse(
            self.goals[:, 3:7],
            self.goals[:, :3]
        )
        insertion_obj2goal_quat, insertion_obj2goal_pos = torch_utils.tf_combine(
            world2goal_quat,
            world2goal_pos,
            self.insertion_obj_quat,
            self.insertion_obj_pos
        )
        xy_diff = insertion_obj2goal_pos
        # xy_diff = self.insertion_obj_pos[:, :2] - self.goals[:,:2].unsqueeze(0).to(self.device) # (num_envs, 2)
        xy_norm = torch.norm(xy_diff, p=2, dim=-1)  # (num_envs,)
        center_threshold = self.insertion_obj.size * threshold_multiplier
        return xy_norm < center_threshold  # (num_envs,)
    
    def _check_success(self):
        """Check for task success."""
        # return torch.logical_and(self._check_insertion_obj_upright(), self._check_insertion_obj_centered(threshold_multiplier=0.6))  # (num_envs,)
        # check if keypoint_dist is below threshold
        keypoint_dist = torch.norm(self.goal_keypoints - self.insertion_obj_keypoints, p=2, dim=-1)  # (num_envs, num_keypoints)
        goal_dist = torch.mean(keypoint_dist, dim=-1)  # (num_envs,)
        return goal_dist < self.cfg_task.rl.close_error_thresh
    
    def compute_observations_dict_obs(self):
        """
        Compute observations as a dictionary.

        Returns:
            obs_dict: Dictionary containing observations.
        """
        self.packed_objects_pos = self.root_pos[:, self.packed_objects.actor_id_env_list, 0:3]
        self.packed_objects_quat = self.root_quat[:, self.packed_objects.actor_id_env_list, 0:4]
        
        if 'ee_pos' in self.cfg_task.env.obsDims or 'ee_pos' in self.cfg_task.env.stateDims:
            self.obs_dict['ee_pos'][:] = self.fingertip_midpoint_pos
        if 'ee_quat' in self.cfg_task.env.obsDims or 'ee_quat' in self.cfg_task.env.stateDims:
            self.obs_dict['ee_quat'][:] = self.fingertip_midpoint_quat
        if 'ee_linvel' in self.cfg_task.env.obsDims or 'ee_linvel' in self.cfg_task.env.stateDims:
            self.obs_dict['ee_linvel'][:] = self.fingertip_midpoint_linvel
        if 'ee_angvel' in self.cfg_task.env.obsDims or 'ee_angvel' in self.cfg_task.env.stateDims:
            self.obs_dict['ee_angvel'][:] = self.fingertip_midpoint_angvel
        
        if 'insertion_object_pos' in self.cfg_task.env.obsDims or 'insertion_object_pos' in self.cfg_task.env.stateDims:
            self.obs_dict['insertion_object_pos'][:] = self.insertion_obj_pos
        if 'insertion_object_quat' in self.cfg_task.env.obsDims or 'insertion_object_quat' in self.cfg_task.env.stateDims:
            self.obs_dict['insertion_object_quat'][:] = self.insertion_obj_quat
        if 'insertion_object_linvel' in self.cfg_task.env.obsDims or 'insertion_object_linvel' in self.cfg_task.env.stateDims:
            self.obs_dict['insertion_object_linvel'][:] = self.insertion_obj_linvel
        if 'insertion_object_angvel' in self.cfg_task.env.obsDims or 'insertion_object_angvel' in self.cfg_task.env.stateDims:
            self.obs_dict['insertion_object_angvel'][:] = self.insertion_obj_angvel
        
        if 'packed_object_pose' in self.cfg_task.env.obsDims or 'packed_object_pose' in self.cfg_task.env.stateDims:
            packed_objects_vel  = torch.cat((self.packed_objects_linvel, self.packed_objects_angvel), dim=-1)
            packed_objects_pose = torch.cat((self.packed_objects_pos, self.packed_objects_quat), dim=-1)
            self.obs_dict['packed_object_pose'] =  packed_objects_pose.reshape(self.num_envs, -1)  # (num_envs, 7*num_cubes)
            self.obs_dict['packed_object_vel'] = packed_objects_vel.reshape(self.num_envs, -1)  # (num_envs, 6*num_cubes)
        
        if 'dof_pos' in self.cfg_task.env.obsDims or 'dof_pos' in self.cfg_task.env.stateDims:
            self.obs_dict['dof_pos'][:] = self.dof_pos
            # print(self.obs_dict['dof_pos'])

        if self.cfg_task.rl.add_contact_force_plug_decomposed or self.cfg_task.rl.add_contact_info_to_aac_states:
            # self.obs_dict['insertion_obj_socket_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.socket_body_id_env]
            # self.obs_dict['insertion_obj_socket_force'][:] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.socket_body_id_env]
            
            ### get forces between insertion object and bin + objects inside bin
            contact_forces_bin = torch.zeros((self.num_envs, len(self.packed_objects)+1, 3), device=self.device, dtype=torch.float32)
            for packed_obj_idx in range(len(self.packed_objects)):
                contact_forces_bin[:, packed_obj_idx, :] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.packed_objects.body_id_env_list[packed_obj_idx]]
            contact_forces_bin[:, -1, :] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.bin.body_id_env]
            self.obs_dict['insertion_object_bin_packed_force'] = contact_forces_bin.reshape(self.num_envs, -1)  # (num_envs, 3*(num_packed_objects+1))
            
            ### get force between insertion object and panda
            if self.cfg_task.env.use_compliant_contact:
                self.obs_dict['insertion_object_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['elastomer_left']] \
                    + self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['panda_fingerbox_left']] # (1,3)
                self.obs_dict['insertion_object_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['elastomer_right']] \
                    + self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['panda_fingerbox_right']]
                
            else:
                self.obs_dict['insertion_object_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['panda_leftfinger']]
                self.obs_dict['insertion_object_right_elastomer_force'][:] = self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.franka_body_ids_env['panda_rightfinger']]
        
        
        inverse_bin_quat, inverse_bin_translation = torch_utils.tf_inverse(
            self.bin_quat,
            self.bin_pos
        )
        if 'goal2bin_pos' in self.cfg_task.env.obsDims:
            _, goal2bin_translation = torch_utils.tf_combine(
                inverse_bin_quat,
                inverse_bin_translation,
                self.goals[:, 3:7],
                self.goals[:, :3]
            )
            self.obs_dict['goal2bin_pos'][:] = goal2bin_translation
        if 'ee2bin_pos' in self.cfg_task.env.obsDims or 'ee2bin_quat' in self.cfg_task.env.obsDims:
            ee2bin_quat, ee2bin_translation = torch_utils.tf_combine(
                inverse_bin_quat,
                inverse_bin_translation,
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos
            )
            if 'ee2bin_pos' in self.cfg_task.env.obsDims:
                self.obs_dict['ee2bin_pos'][:] = ee2bin_translation
            if 'ee2bin_quat' in self.cfg_task.env.obsDims:
                self.obs_dict['ee2bin_quat'][:] = ee2bin_quat
        
        
        if 'goal_quat' in self.cfg_task.env.obsDims or 'goal_quat' in self.cfg_task.env.stateDims:
            # set goals to insertion object orientation
            self.obs_dict['goal_quat'][:] = self.goals[:, 3:7]
        
        if 'goal_pos' in self.cfg_task.env.obsDims or 'goal_pos' in self.cfg_task.env.stateDims:
            # set goals to insertion object position
            self.obs_dict['goal_pos'][:] = self.goals[:, :3]
        
        if 'eef_to_goal_pos' in self.cfg_task.env.obsDims or 'eef_to_goal_pos' in self.cfg_task.env.stateDims:
            eef_to_goal_transform = torch_utils.tf_combine(
                *torch_utils.tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos),
                self.obs_dict['goal_quat'], self.obs_dict['goal_pos']
            )
            self.obs_dict['eef_to_goal_pos'][:] = eef_to_goal_transform[1]
            self.obs_dict['eef_to_goal_quat'][:] = eef_to_goal_transform[0]

            # print(eef_to_goal_transform)
            # print(self.obs_dict['eef_to_goal_pos'], self.obs_dict['eef_to_goal_quat'])
        
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
                self.obs_dict['left_tactile_camera_taxim_gray'][:] = left_gray.unsqueeze(-1)#.expand(self.num_envs, left_gray.shape[1], left_gray.shape[2], 3)    # (num_envs, H, W, 3)
                self.obs_dict['right_tactile_camera_taxim_gray'][:] = right_gray.unsqueeze(-1)#.expand(self.num_envs, right_gray.shape[1], right_gray.shape[2], 3)  # (num_envs, H, W, 3)
        
        if self.cfg_task.env.use_shear_force:
            if not self.cfg_task.env.use_shear_3d:

                if self.cfg_task.env.get('use_tacsl_shear', False):
                    tactile_force_field_dict = self.get_tactile_force_field_tensors_dict(debug=self.cfg_task.env.use_tactile_field_obs, return_depth=False, normalize_shear=self.cfg_task.env.get("normalize_tacsl_shear_obs", False))

                    if self.cfg_task.env.use_tactile_field_obs:
                        for k in ['tactile_force_field_left', 'tactile_force_field_right']:
                            self.obs_dict[k][:] = tactile_force_field_dict[k][..., 1:]
                            # if self.cfg_task.env.zero_out_normal_force_field_obs:
                            #     self.obs_dict[k][..., 0] *= 0.0

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
                    aug_vector = torch.zeros((self.num_envs, 3), device=self.device)
                    
                    # if bookshelving do in world coord
                    if self.cfg_task.name == "BookShelvingTask":
                        aug_vector[:, 2] = -self.hydrofots_sensors[0].hydroshear_gravity_noise.squeeze(-1)  # deform by 1 cm downwards
                    else:
                        aug_vector[:, 2] = self.hydrofots_sensors[0].hydroshear_gravity_noise.squeeze(-1)  # deform by 1 cm downwards
                        aug_vector = torch_utils.quat_apply(self.fingertip_midpoint_quat, aug_vector)
                    
                    # aug pos and quat runs extra hydrofots step model forces that are not captured from IsaacGym simulator such as gravity effects
                    # can also be used to jitter the shear forces and create some randomization
                    
                    # aug_plug_quat = self.insertion_obj_quat
                    # aug_plug_pos = self.insertion_obj_pos + aug_vector
                    
                    '''
                        bXa = transform of a in b frame
                        eeXplug = eeXworld * worldXplug
                        eeXplughat = eeXplug * plugXplughat
                        worldXplughat = worldXee * eeXplughat
                    '''
                    world2ee_quat, world2ee_pos = torch_utils.tf_inverse(self.fingergrasp_quat, self.fingergrasp_pos)
                    plug2ee_quat, plug2ee_pos = torch_utils.tf_combine(
                        world2ee_quat,
                        world2ee_pos,
                        self.insertion_obj_quat,
                        self.insertion_obj_pos
                    )
                    #rotate in y direction
                    # how much to rotate
                    inhand_roty = torch_utils.quat_from_euler_xyz(
                        torch.zeros((self.num_envs,), device=self.device),
                        torch.zeros((self.num_envs,), device=self.device) + self.hydrofots_sensors[0].hydroshear_gravity_rot_noise.squeeze(-1),
                        torch.zeros((self.num_envs,), device=self.device)
                    )
                    rotated_plug2ee_quat, rotated_plug2ee_pos = torch_utils.tf_combine(
                        inhand_roty,
                        torch.zeros_like(plug2ee_pos),
                        plug2ee_quat,
                        plug2ee_pos
                    )
                    
                    rotated_plug2world_quat, rotated_plug2world_pos = torch_utils.tf_combine(
                        self.fingergrasp_quat,
                        self.fingergrasp_pos,
                        rotated_plug2ee_quat,
                        rotated_plug2ee_pos
                    )
                    
                    aug_plug_quat = rotated_plug2world_quat
                    aug_plug_pos = rotated_plug2world_pos + aug_vector
                    
                    marker_displacement_dict = self.get_force_fields_dict(self.insertion_obj_quat, self.insertion_obj_pos, aug_indenter_quat=aug_plug_quat, aug_indenter_pos=aug_plug_pos) # hydrofots
                elif self.cfg_task.env.get("use_fots_model", False) or self.cfg_task.env.get("use_old_fots_model", False):
                    marker_displacement_dict = self.get_displacement_field_dict(self.insertion_obj_quat, self.insertion_obj_pos) # fots
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
                                    indenter_quat = self.insertion_obj_quat
                                    indenter_pos = self.insertion_obj_pos
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(indenter_quat[0].cpu().numpy().tolist() + indenter_pos[0].cpu().numpy().tolist())
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
    
    def execute_terminal_primitive(self):
        """Execute terminal primitive actions."""
        if self.cfg_task.env.task_type == 'screwing':
            # do screw primitive
            self.apply_screw_primitive()

        if self.cfg_task.env.task_type in ['placement', 'screwing']:
            # open-gripper and lift
            self._open_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)
            self._lift_gripper(gripper_dof_pos=0.1, sim_steps=self.cfg_task.env.num_gripper_close_sim_steps//2)
    
    ## FactoryABCTask methods
    #########################################
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask
        
    def _acquire_task_tensors(self): 
        """Acquire tensors."""
        
        self.goal_keypoints = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device)
        self.insertion_obj_keypoints = torch.zeros_like(self.goal_keypoints, dtype=torch.float32, device=self.device)
        self.insertion_obj_keypoints_vis = torch.zeros((self.num_envs, len(self.packed_objects), 8, 3), dtype=torch.float32, device=self.device)  # (num_envs, num_packed_objects, num_keypoints, 3)
        
        # keypoints will be arranged in a bounding-box fashion around the object all determined by object.sizes (3,)
        self.packed_objects_keypoints = torch.zeros((self.num_envs, len(self.packed_objects), 8, 3), dtype=torch.float32, device=self.device)  # (num_envs, num_packed_objects, num_keypoints, 3)
        self.packed_goal_keypoints = torch.zeros((self.num_envs, len(self.packed_objects), self.cfg_task.rl.num_keypoints, 3), dtype=torch.float32, device=self.device)  # (num_envs, num_packed_objects, num_keypoints, 3)
        
        # NOTE: these tensors will be related to the task / calculating the reward
        self._actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        
    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # NOTE: these tensors will be related to calculating the reward
        local_keypoints = torch.zeros_like(self.goal_keypoints, dtype=torch.float32, device=self.device)
        
        if not hasattr(self, 'goals'):
            self.goals = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device) # NOTE: this is just a placeholder, will be set in _reset_object()
            self.goals[:, -1] = 1.0 # set w component of quaternion to 1.0
        if not hasattr(self, 'packed_goals'):
            self.packed_goals = torch.zeros((self.num_envs, len(self.packed_objects), 7), dtype=torch.float32, device=self.device)
            self.packed_goals[:, :, -1] = 1.0
        if not hasattr(self, 'keypoint_dirs'):
            self.keypoint_dirs = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        
        local_keypoints[torch.arange(self.num_envs, device=self.device), :, self.keypoint_dirs] = torch.linspace(0.0, 1.0, self.cfg_task.rl.num_keypoints, device=self.device) - 0.5

        goals = self.goals.unsqueeze(1).expand(self.num_envs, self.cfg_task.rl.num_keypoints, 7)  # (num_envs, num_keypoints, 7)
        self.goal_keypoints[:, :, :3] = torch_utils.tf_apply(goals[:, :, 3:7], goals[:, :, :3], local_keypoints)
        # self.goal_keypoints[:, :, :3] = local_keypoints + self.goals.unsqueeze(1)[:, :, :3]
        for kpt_idx in range(self.cfg_task.rl.num_keypoints):
            self.insertion_obj_keypoints[:, kpt_idx, :] = torch_utils.tf_combine(
                self.insertion_obj_quat,
                self.insertion_obj_pos,
                self.identity_quat,
                -local_keypoints[:, kpt_idx, :]
            )[1]
        
        # packed object keypoints
        packed_obj_sizes = torch.tensor(self.packed_objects.sizes)
        lower = (-packed_obj_sizes / 2.0)
        upper = (packed_obj_sizes / 2.0)
        local_keypoints = torch.stack([
            torch.stack([lower[:,0], lower[:,1], lower[:,2]]),
            torch.stack([upper[:,0], lower[:,1], lower[:,2]]),
            torch.stack([lower[:,0], upper[:,1], lower[:,2]]),
            torch.stack([upper[:,0], upper[:,1], lower[:,2]]),
            torch.stack([lower[:,0], lower[:,1], upper[:,2]]),
            torch.stack([upper[:,0], lower[:,1], upper[:,2]]),
            torch.stack([lower[:,0], upper[:,1], upper[:,2]]),
            torch.stack([upper[:,0], upper[:,1], upper[:,2]])
        ]).permute(2,0,1).to(self.device).to(torch.float32)  # (num_packed_objects, 8, 3)
        local_geom_centers = self.packed_objects.geometric_centers.unsqueeze(1).expand(len(self.packed_objects), 8, 3).to(self.device).to(torch.float32)  # (num_envs, num_packed_objects, 3)
        local_keypoints = local_keypoints + local_geom_centers
        local_keypoints = local_keypoints.unsqueeze(0).expand(self.num_envs, len(self.packed_objects), 8, 3)  # (num_envs, num_packed_objects, num_keypoints, 3)
        packed_obj_goal_pos = self.packed_goals[:, :, :3].unsqueeze(2).expand(self.num_envs, len(self.packed_objects), 8, 3)  # (num_envs, num_packed_objects, num_keypoints, 3)
        packed_obj_goal_quat = self.packed_goals[:, :, 3:7].unsqueeze(2).expand(self.num_envs, len(self.packed_objects), 8, 4)  # (num_envs, num_packed_objects, num_keypoints, 4)
        self.packed_goal_keypoints = torch_utils.tf_apply(
            packed_obj_goal_quat,
            packed_obj_goal_pos,
            local_keypoints
        ) # (num_envs, num_packed_objects, num_keypoints, 3)
        
        self.packed_objects_pos = self.root_pos[:, self.packed_objects.actor_id_env_list, 0:3]
        self.packed_objects_quat = self.root_quat[:, self.packed_objects.actor_id_env_list, 0:4]
        self.packed_objects_keypoints = torch_utils.tf_apply(
            self.packed_objects_quat.unsqueeze(2).expand(self.num_envs, len(self.packed_objects), 8, 4),
            self.packed_objects_pos.unsqueeze(2).expand(self.num_envs, len(self.packed_objects), 8, 3),
            local_keypoints
        )
    
    def pre_physics_step(self, actions):
        """Optionally reset environments at the end of episodes. Apply actions from policy as position/rotation targets."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self._actions,
                                            ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                            do_scale=True)

        sim_dt_noise = self.cfg_task.env.get("sim_dt_noise", 0)
        if sim_dt_noise > 0.0:
            sim_params = self.gym.get_sim_params(self.sim)
            sim_params.dt = self.cfg_base.sim.dt * (1 + torch.rand(1) * sim_dt_noise)
            self.gym.set_sim_params(self.sim, sim_params)

        num_extra_control_steps = self.cfg_task.env.get("num_additional_control_steps", 0)  # for backward compatibility
        num_additional_control_steps_noise = self.cfg_task.env.get("num_additional_control_steps_noise", 0)
        if num_additional_control_steps_noise:
            num_extra_control_steps += torch.randint(
                0,
                self.cfg_task.env.num_additional_control_steps_noise + 1,
                (1,)
            )[0].item()
        
        self.execute_control_loop(num_extra_control_steps)
    
    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            self.execute_terminal_primitive()

        self.refresh_all_tensors()
        self.compute_observations()
        self.compute_reward()
        
        gamma = self.cfg_task.rl.get('discount_factor', 1.0)
        self.rew_buf[:] *= torch.pow(torch.tensor(gamma), torch.tensor(self.progress_buf[0]))
        self.visualize_keypoints(
            keypoints_vis=self.cfg_task.rl.visualize_keypoints,
            goal_vis=self.cfg_task.rl.visualize_goal_pose,
            packed_objects_vis=self.cfg_task.rl.get('visualize_packed_keypoints', False),
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            task_success = self._check_success()
            # print(f"task_success: {task_success}")
            self.rew_buf[:] += task_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(task_success.float())

        # task_success = self._check_success()
        # # print(f"task_success: {task_success}")
        # self.rew_buf[:] += task_success * self.cfg_task.rl.success_bonus
        # self.extras['successes'] = torch.mean(task_success.float())

        self.prev_actions[:] = self._actions

        if self.cfg_base.mode.export_states:
            self.extract_poses()

    def compute_observations(self):
        """Compute observations."""

        if self.cfg_task.env.use_dict_obs:
            return self.compute_observations_dict_obs()

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""
        self._update_reset_buf()
        self._update_rew_buf()
    
    def _update_rew_buf(self):
        """Compute reward at the current timestep."""
        
        keypoint_diff = self.goal_keypoints - self.insertion_obj_keypoints
        keypoint_dist = torch.mean(torch.norm(keypoint_diff, p=2, dim=-1),dim=-1)  # shape = (num_envs, num_keypoints)
        keypoint_reward = -keypoint_dist
        
        a, b = 300, 0.0001
        a, b = 50, 0.0001
        keypoint_reward_exp = 1. / (torch.exp(a * keypoint_reward) + b + torch.exp(-a * keypoint_reward))
        
        if self.cfg_task.rl.get('use_packed_obj_reward', False):
            # (num_envs, num_packed_objects, num_keypoints, 3)
            packed_obj_dist = torch.sum(torch.mean(torch.norm(self.packed_goal_keypoints - self.packed_objects_keypoints,dim=-1), dim=-1),dim=-1)
            packed_obj_reward = -packed_obj_dist

            # a_packed, b_packed = 50, 0.0001
            a_packed = self.cfg_task.rl.get('a_packed', 50)
            b_packed = self.cfg_task.rl.get('b_packed', 0.0001)
            packed_obj_reward_exp = 1. / (torch.exp(a_packed * packed_obj_reward) + b_packed + torch.exp(-a_packed * packed_obj_reward))
        else:
            packed_obj_reward = 0.0
            packed_obj_reward_exp = 0.0
        
        is_obj_centered = self._check_insertion_obj_centered(threshold_multiplier=0.5)
        if self.cfg_task.rl.use_shaped_keypoint_reward:
            '''
                If object if not precisely at hole, don't try to insert.
                if it does insert (insert near gap but not actually in gap),
                -> we need to show that this is a failure case
            '''
            
            world2goal_quat, world2goal_pos = torch_utils.tf_inverse(
                self.goals[:, 3:7],
                self.goals[:, :3]
            )
            insertion_obj2goal_quat, insertion_obj2goal_pos = torch_utils.tf_combine(
                world2goal_quat,
                world2goal_pos,
                self.insertion_obj_quat,
                self.insertion_obj_pos
            )
            z_dist = insertion_obj2goal_pos[:, 2]
            
            z_dist[z_dist > 0] = 0.0 # only consider z_dist below goal (inserted)
            z_dist = -z_dist # make (cube z pos below goal) positive
            
            z_dist = (1.0 - is_obj_centered.float()) * z_dist # only consider z_dist if object is not centered
            z_dist = z_dist.squeeze()
                        
            keypoint_reward[z_dist > 0] *= 10.0 # penalize for edge case
        
        # action penalty
        action_penalty = torch.norm(self._actions, p=2, dim=-1)
        action_grad_penalty = torch.norm(self._actions - self.prev_actions, p=2, dim=-1)
        
        contact_penalty = torch.norm(self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.bin.body_id_env], p=2, dim=-1)
        contact_force_table = torch.norm(self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.table_body_id], p=2, dim=-1)
        
        contact_cube = torch.norm(self.contact_force_pairwise[:, self.insertion_obj.body_id_env, self.packed_objects.body_id_env_list], p=2, dim=-1).sum(dim=-1)
        
        # if insertion_obj is centered, reduce the contact penalty by a scale factor
        if self.cfg_task.rl.use_shaped_contact_pen:
            contact_pen_reduction_scalar = self.cfg_task.rl.contact_pen_reduction_scalar  # = 0.001
            contact_penalty = (contact_penalty * (1 - is_obj_centered.float()) +
                                contact_penalty * (is_obj_centered.float() * contact_pen_reduction_scalar))
            contact_force_table = (contact_force_table * (1 - is_obj_centered.float()) +
                                contact_force_table * (is_obj_centered.float() * contact_pen_reduction_scalar))
            contact_cube = (contact_cube * (1 - is_obj_centered.float()) +
                                contact_cube * (is_obj_centered.float() * contact_pen_reduction_scalar))
        
        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          + keypoint_reward_exp * self.cfg_task.rl.keypoint_reward_scale \
                          + packed_obj_reward * self.cfg_task.rl.get('packed_obj_keypoint_reward_scale', 1.0) \
                          + packed_obj_reward_exp * self.cfg_task.rl.get('packed_obj_keypoint_reward_scale', 1.0) \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          - action_grad_penalty * self.cfg_task.rl.action_gradient_penalty_scale \
                          - contact_force_table * self.cfg_task.rl.contact_penalty_scale \
                          - contact_penalty * self.cfg_task.rl.contact_penalty_scale \
                          - contact_cube * self.cfg_task.rl.contact_penalty_scale

    def _update_reset_buf(self):
        """Assign environments for reset if episode length expired."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        
        # randomize compliance during reset
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
        self._reset_object(env_ids)

        self.disable_gravity()  
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()
        
        if self.cfg_task.env.use_shear_force:
            if self.cfg_task.env.use_hydrosoft_model:
                self.reset_hydrofots()

        self._move_gripper_to_dof_pos(gripper_dof_pos=self.cfg_task.env.get("franka_close_gripper_width", 0.0),
                                      sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        
        if self.cfg_task.env.use_shear_force:
            if self.cfg_task.env.get("use_fots_model", False) or self.cfg_task.env.get("use_old_fots_model", False):
                self.reset_fots(self.plug_quat, self.plug_pos)
        
        self.enable_gravity(gravity_vec=self.cfg_base.sim.gravity)
        
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
    
    def visualize_keypoints(self, **kwargs):
        
        if kwargs['keypoints_vis']:
            for env_idx in range(self.num_envs):
                obj_keypoints = self.insertion_obj_keypoints[env_idx] # tensor of shape (4, 3)
                # visualize insertion object keypoints
                for i, point in enumerate(obj_keypoints):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(0.0, 1.0, 0.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)

                goal_keypoints = self.goal_keypoints[env_idx] # tensor of shape (4, 3)
                for i, point in enumerate(goal_keypoints):
                    point_pose = gymapi.Transform()
                    point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    ball_geom = gymutil.WireframeSphereGeometry(radius=0.01, pose=point_pose, color=(1.0, 0.0, 0.0))
                    gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)

        if kwargs['packed_objects_vis']:
            for env_idx in range(self.num_envs):
                for obj_idx in range(len(self.packed_objects)):
                    for i, point in enumerate(self.packed_objects_keypoints[env_idx, obj_idx]):
                        point_pose = gymapi.Transform()
                        point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                        ball_geom = gymutil.WireframeSphereGeometry(radius=0.005, pose=point_pose, color=(0.0, 0.0, 1.0))
                        gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
                    
                    for i, point in enumerate(self.packed_goal_keypoints[env_idx, obj_idx]):
                        point_pose = gymapi.Transform()
                        point_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                        ball_geom = gymutil.WireframeSphereGeometry(radius=0.005, pose=point_pose, color=(0.0, 1.0, 0.0))
                        gymutil.draw_lines(ball_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)

        if kwargs['goal_vis']:
            # Get insertion object dimensions (handles both Cube and Book)
            if hasattr(self.insertion_obj, 'params'):
                # Book: params is [x, y, z] dimensions
                obj_dims = torch.tensor(self.insertion_obj.params)
            else:
                # Cube: size is a scalar
                obj_dims = torch.tensor([self.insertion_obj.size] * 3)
            lower = -obj_dims / 2.0
            upper = obj_dims / 2.0
            bbox_template = torch.stack([lower, upper], dim=0)
            for env_idx in range(self.num_envs):
                goal_pos = self.goals[env_idx, :3]
                goal_pose = gymapi.Transform()
                goal_pose.p = gymapi.Vec3(goal_pos[0],goal_pos[1], goal_pos[2])
                goal_pose.r = gymapi.Quat(self.goals[env_idx, 3],
                                          self.goals[env_idx, 4],
                                          self.goals[env_idx, 5],
                                          self.goals[env_idx, 6])
                bbox = gymutil.WireframeBBoxGeometry(bbox_template, goal_pose, (1.0, 1.0, 0.0))
                coordinate_frame_geom = gymutil.AxesGeometry(scale=1.0, pose=goal_pose)
                # gymutil.draw_lines(coordinate_frame_geom, self.gym, self.viewer, self.env_ptrs[env_idx], None)
                gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_idx], None)
        return

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka to initial joint configuration
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))

        self._reset_franka_actuation(self.ctrl_target_dof_pos)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)  # probably not needed
        self.refresh_all_tensors()

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps,
                                     ctrl_target_gripper_dof_pos=self.cfg_task.env.get("franka_open_gripper_width",
                                                                                       self.asset_info_franka_table.franka_gripper_width_max))

    def _reset_object(self, env_ids):
        """Reset root states of objects."""
        self.gym.simulate(self.sim)
        self.refresh_base_tensors()

        reset_ids = []

        
        bin_pose = self.bin_pose
        bin_quat = torch.tensor([bin_pose.r.x, bin_pose.r.y, bin_pose.r.z, bin_pose.r.w], dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        bin_pos = torch.tensor([bin_pose.p.x, bin_pose.p.y, bin_pose.p.z], dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
        
        upper_pos_noise = torch.tensor(self.cfg_task.randomize.preset_object_pos_noise, dtype=torch.float32)
        lower_pos_noise = torch.tensor(self.cfg_task.randomize.preset_object_pos_noise, dtype=torch.float32) * -1.0

        bin_quat_noise = torch_utils.quat_from_euler_xyz(torch.zeros(self.num_envs), torch.zeros(self.num_envs), torch.rand(self.num_envs) * self.cfg_task.randomize.preset_object_yaw_noise * torch.pi / 180.0)
        bin_pos_noise = (lower_pos_noise - upper_pos_noise) * torch.rand((self.num_envs, 3)) + upper_pos_noise

        bin_quat, bin_pos = torch_utils.tf_combine(
            bin_quat, 
            bin_pos,
            bin_quat_noise,
            bin_pos_noise
        )
        

        # reset bin pose
        self.root_pos[env_ids, self.bin.actor_id_env, :3] = bin_pos.to(self.device)
        self.root_quat[env_ids, self.bin.actor_id_env, :4] = bin_quat.to(self.device)

        # Add z-direction noise to bin position (only translation, no x/y or rotation)
        bin_z_noise = 2 * (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        bin_z_noise = bin_z_noise * self.cfg_task.randomize.get("bin_pos_z_noise", 0.0)
        self.root_pos[env_ids, self.bin.actor_id_env, 2] += bin_z_noise[env_ids]

        self.root_linvel[env_ids, self.bin.actor_id_env, :] = 0.0
        self.root_angvel[env_ids, self.bin.actor_id_env, :] = 0.0
        reset_ids.append(self.actor_ids_sim_tensors['bin'][env_ids])
        
        
        # reset packed objects
        object_poses = []
        self.goals = []
        self.packed_goals = []
        self.keypoint_dirs = []
        for env_idx in range(self.num_envs):
            self.packed_objects.reset_colors(self.env_ptrs[env_idx]) 
            cube_poses, goal, keypoint_dir, packed_goals = self.packed_objects.initialize_poses(
                squish_goal=self.cfg_task.randomize.preset_object_squish_goal,
                randomize=self.cfg_task.randomize.preset_object_randomization,
                bin_pos=bin_pos[env_idx],
                bin_quat=bin_quat[env_idx],
                return_desired_packed_poses=True
            )
            cube_poses_torch = torch.zeros((len(cube_poses), 7), dtype=torch.float32, device=self.device)
            for i in range(len(cube_poses)):
                cube_poses_torch[i, 0] = cube_poses[i].p.x
                cube_poses_torch[i, 1] = cube_poses[i].p.y
                cube_poses_torch[i, 2] = cube_poses[i].p.z
                cube_poses_torch[i, 3] = cube_poses[i].r.x
                cube_poses_torch[i, 4] = cube_poses[i].r.y
                cube_poses_torch[i, 5] = cube_poses[i].r.z
                cube_poses_torch[i, 6] = cube_poses[i].r.w
            object_poses.append(cube_poses_torch)
            self.goals.append(goal)
            self.packed_goals.append(packed_goals)
            self.keypoint_dirs.append(keypoint_dir)
        self.goals = torch.stack(self.goals, dim=0).to(self.device)  # (num_envs, 3)
        object_poses = torch.stack(object_poses, dim=0)  # (num_envs, num_objects, 7)
        self.packed_goals = torch.stack(self.packed_goals).to(self.device)  # (num_envs, num_objects, 7)
        self.keypoint_dirs = torch.tensor(self.keypoint_dirs, dtype=torch.int64, device=self.device) # (num_envs,)

        # Adjust goals for bin z-noise (since goals are defined relative to bin)
        self.goals[:, 2] += bin_z_noise  # Add z-noise to goal z-position
        self.packed_goals[:, :, 2] += bin_z_noise.unsqueeze(-1)  # Add z-noise to packed goals z-positions
        
        
        for obj_idx in range(len(self.packed_objects)):
            self.root_pos[env_ids, self.packed_objects.actor_id_env_list[obj_idx], :3] = object_poses[env_ids, obj_idx, :3]
            self.root_quat[env_ids, self.packed_objects.actor_id_env_list[obj_idx], :4] = object_poses[env_ids, obj_idx, 3:7]
            
            self.root_linvel[env_ids, self.packed_objects.actor_id_env_list[obj_idx], :] = 0.0
            self.root_angvel[env_ids, self.packed_objects.actor_id_env_list[obj_idx], :] = 0.0

            reset_ids.append(self.actor_ids_sim_tensors[f'obj_{obj_idx}'][env_ids])
        
        noise_insertobj2ee_translation = (2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)) # [-1, 1] (N, 3)
        noise_insertobj2ee_translation = noise_insertobj2ee_translation * torch.tensor(self.cfg_task.randomize.get("insertion_obj_noise_pos_in_gripper", [0.0, 0.0, 0.0]), device=self.device).unsqueeze(0)
        noise_insertobj2ee_translation[:, 1] = 0.0 # set noise y to zero
        
        noise_insertobj2ee_euler = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        noise_insertobj2ee_euler *= torch.tensor(self.cfg_task.randomize.insertion_obj_noise_rot_in_gripper,
                                                  device=self.device).expand(self.num_envs, 3)
        noise_insertobj2ee_quat = torch_utils.quat_from_euler_xyz(noise_insertobj2ee_euler[:, 0],
                                                                  noise_insertobj2ee_euler[:, 1],
                                                                 noise_insertobj2ee_euler[:, 2])
        noise_insertobj2ee_translation = noise_insertobj2ee_translation + self.inhand_pos.expand(self.num_envs, 3) # (N, 3)
        
        noise_insertobj2world_quat, noise_insertobj2world_translation = torch_utils.tf_combine(
            self.fingertip_midpoint_quat, self.fingertip_midpoint_pos,
            noise_insertobj2ee_quat, noise_insertobj2ee_translation
        )
        
        world2ee_quat, world2ee_pos = torch_utils.tf_inverse(self.fingergrasp_quat, self.fingergrasp_pos)
        plug2ee_quat, plug2ee_pos = torch_utils.tf_combine(
            world2ee_quat,
            world2ee_pos,
            noise_insertobj2world_quat,
            noise_insertobj2world_translation
        )
        
        inhand_roty_noise_lower, inhand_roty_noise_upper = self.cfg_task.randomize.get("insertion_obj_inhand_rotation_noise_range", [0.0, 0.0])
        inhand_roty_noise = (inhand_roty_noise_upper - inhand_roty_noise_lower) * torch.rand((self.num_envs,), dtype=torch.float32, device=self.device) + inhand_roty_noise_lower
        inhand_roty = self.cfg_task.randomize.get("insertion_obj_inhand_rotation", 0.0) + inhand_roty_noise
        inhand_roty = torch_utils.quat_from_euler_xyz(torch.zeros_like(inhand_roty), inhand_roty, torch.zeros_like(inhand_roty))
        rotated_plug2ee_quat, rotated_plug2ee_pos = torch_utils.tf_combine(
            inhand_roty,
            torch.zeros_like(plug2ee_pos),
            plug2ee_quat,
            plug2ee_pos
        )
        
        rotated_plug2world_quat, rotated_plug2world_pos = torch_utils.tf_combine(
            self.fingergrasp_quat,
            self.fingergrasp_pos,
            rotated_plug2ee_quat,
            rotated_plug2ee_pos
        )
        
        noise_insertobj2world_translation = rotated_plug2world_pos
        noise_insertobj2world_quat = rotated_plug2world_quat
        
        # z-axis of end-effector is facing away from robot. we want to flip such that z-axis of box is facing towards robot end-effector
        flip_z_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        noise_insertobj2world_quat = torch_utils.quat_mul(noise_insertobj2world_quat, flip_z_quat)
        
        self.root_pos[env_ids, self.insertion_obj.actor_id_env, 0:3] = noise_insertobj2world_translation
        self.root_quat[env_ids, self.insertion_obj.actor_id_env, 0:4] = noise_insertobj2world_quat
        self.root_linvel[env_ids, self.insertion_obj.actor_id_env, :] = 0.0
        self.root_angvel[env_ids, self.insertion_obj.actor_id_env, :] = 0.0
        
        reset_ids.append(self.actor_ids_sim_tensors['plug'][env_ids])
        reset_ids = torch.cat(reset_ids, dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(reset_ids),
            len(reset_ids)
        )
        
    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def _set_viewer_params(self):
        """Set viewer parameters."""

        # cam_pos = gymapi.Vec3(1.1, 0.0,  0.7196)
        # cam_target = gymapi.Vec3(0., 0.0, -0.48)

        cam_pos = gymapi.Vec3(0.1, -0.1, 0.5)
        cam_target = gymapi.Vec3(0.8, 0.0, -0.1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    #########################################