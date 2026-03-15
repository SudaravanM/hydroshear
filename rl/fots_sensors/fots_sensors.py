from isaacgym.torch_utils import tf_combine, tf_inverse, tf_apply
from rl.tacsl_sensors.tacsl_sensors import quaternion_to_axis_angle
import os
import numpy as np
import torch


import torch.nn.functional as F
from rl.hydroshear_sensors.hydroshear_sensors import MeshSDFSensor



class FotsSensor:
    def __init__(self,
                 fots_cfg,
                 elastomer_urdf_root,
                 elastomer_urdf,
                 elastomer_link_name,
                 indenter_urdf_root,
                 indenter_urdf,
                 indenter_link_name,
                 max_steps=1000,
                 device='cpu', indenter_sdf_sensor = None,
                 use_original_implementation=False):
        self.elastomer_sdf_sensor = MeshSDFSensor(elastomer_urdf_root,elastomer_urdf, elastomer_link_name, device)
        self.indenter_sdf_sensor  = MeshSDFSensor(indenter_urdf_root,indenter_urdf, indenter_link_name, device) if indenter_sdf_sensor is None else indenter_sdf_sensor
        self.fots_cfg = fots_cfg
        self.max_steps = max_steps
        
        if self.fots_cfg is not None:
            self.lambda_d = torch.tensor(self.fots_cfg.get("lambda_d", 1.0), device=device)
            self.lambda_s = torch.tensor(self.fots_cfg.get("lambda_s", 1.0), device=device)
            self.lambda_t = torch.tensor(self.fots_cfg.get("lambda_t", 1.0), device=device)
            
            self.dilate_scale = torch.tensor(self.fots_cfg.get("dilate_scale", 1.0), device=device)
            self.shear_scale = torch.tensor(self.fots_cfg.get("shear_scale", 1.0), device=device)
            self.twist_scale = torch.tensor(self.fots_cfg.get("twist_scale", 1.0), device=device)
        
        self.use_original_implementation = use_original_implementation
        self.device = device
        
    def initialize(self, num_envs, cop2indenter_quat = None, cop2indenter_pos = None):
        '''
        NOTE: 
        - one weakness of FOTS implementation is that it uses object frame coordinate to compute gaussian kernel for shear/twist
        - instead, we have to use the center of contact patch from the initial grasp as the reference pose to track instead of object frame
        '''
        self.cop2indenter_quat = torch.tensor([1,0,0,0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(num_envs,4) if (cop2indenter_quat is None) or self.use_original_implementation else cop2indenter_quat
        self.cop2indenter_pos = torch.tensor([0,0,0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(num_envs,3) if (cop2indenter_pos is None) or self.use_original_implementation else cop2indenter_pos
        self.indenter2elastomer_tf_history_se2 = torch.zeros((num_envs, self.max_steps, 3), device=self.device)
        self.index = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)
        self.save_indenter2elastomer_quat = torch.zeros((num_envs, self.max_steps, 4), device=self.device)
        
    def get_old_marker_dilation(self, tactile_pts_in_elastomer, tactile_pts_in_indenter):
        # tactile points queried on indenter sdf
        tactile_pts_sdf,_ = self.indenter_sdf_sensor.get_sdf(tactile_pts_in_indenter) # (num_envs, num_tactile_pts)
        tactile_pts_height = F.relu(-tactile_pts_sdf) # (num_envs, num_tactile_pts)
        
        # handle no contact case (reset)
        tactile_contact_sum = (tactile_pts_height > 0).sum(dim=1) # (num_envs,)
        self.index[tactile_contact_sum == 0] = torch.zeros((tactile_contact_sum == 0).sum(), device=self.device, dtype=torch.int32)
        self.indenter2elastomer_tf_history_se2[tactile_contact_sum == 0, :, :] = torch.zeros((tactile_contact_sum == 0).sum(), self.max_steps, 3, device=self.device)


        num_envs, num_tactile_pts = tactile_pts_height.shape
        dvec = self.dilate_scale * (tactile_pts_in_elastomer.unsqueeze(2) - tactile_pts_in_elastomer.unsqueeze(1)) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1) # (num_envs, num_tactile_pts, num_tactile_pts)
        
        gaussian_exp = torch.exp(-self.lambda_d * norm_dvec).unsqueeze(-1).expand(num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3)
        Mdilate = (h * dvec * gaussian_exp).sum(dim=2) # (num_envs, num_tactile_pts, 3)
        
        return Mdilate
    
    def get_old_marker_shear(self, tactile_pts_in_elastomer, indenter2elastomer_quat, indenter2elastomer_pos):
        num_envs = tactile_pts_in_elastomer.shape[0]
        self.save_indenter2elastomer_quat[torch.arange(num_envs), self.index, :] = indenter2elastomer_quat

        # check if double-cover occurs
        prev_save_indenter2elastomer_quat = self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0] - 1, :]
        curr_save_indenter2elastomer_quat = self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0], :]
        dot_products = torch.sum(prev_save_indenter2elastomer_quat * curr_save_indenter2elastomer_quat, dim=-1)
        sign_corrections = torch.where(dot_products < 0, -1.0, 1.0)
        curr_save_indenter2elastomer_quat = curr_save_indenter2elastomer_quat * sign_corrections.unsqueeze(-1)
        self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0], :] = curr_save_indenter2elastomer_quat

        self.indenter2elastomer_tf_history_se2[torch.arange(num_envs), self.index, :] = self.pose2se2(self.save_indenter2elastomer_quat[torch.arange(num_envs), self.index, :], indenter2elastomer_pos)


        # se2
        indenter2elastomer_se2 = self.indenter2elastomer_tf_history_se2[torch.arange(num_envs), self.index, :] # (num_envs, 3)
        indenter2elastomer_se2_0 = self.indenter2elastomer_tf_history_se2[:, 0, :]
        delta_se2_xy = self.shear_scale * (indenter2elastomer_se2[:, :2] - indenter2elastomer_se2_0[:, :2]) # (num_envs, 2)
        delta_se2_theta = indenter2elastomer_se2[:, 2] - indenter2elastomer_se2_0[:, 2] # (num_envs,)

        delta_se2_theta = torch.clamp(delta_se2_theta, -180 * torch.pi / 180.0, 180 * torch.pi / 180.0)
        
        # (M - G) but object center (G) is zero so it becomes M
        gaussian_exp_s = torch.exp(-self.lambda_s * torch.norm((tactile_pts_in_elastomer[:, :, :2] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, :2]), dim=-1)) # (num_envs, num_tactile_pts)
        gaussian_exp_t = torch.exp(-self.lambda_t * torch.norm((tactile_pts_in_elastomer[:, :, :2] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, :2]), dim=-1)) # (num_envs, num_tactile_pts)
        
        Mshear = delta_se2_xy.unsqueeze(1) * gaussian_exp_s.unsqueeze(-1) # (num_envs, num_tactile_pts, 2)
        
        dx = self.twist_scale * (tactile_pts_in_elastomer[:, :, 0] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, 0])
        dy = self.twist_scale * (tactile_pts_in_elastomer[:, :, 1] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, 1])
        
        rotx_dx = (dx * torch.cos(delta_se2_theta).unsqueeze(-1) - dy * torch.sin(delta_se2_theta).unsqueeze(-1)) - dx
        roty_dy = (dx * torch.sin(delta_se2_theta).unsqueeze(-1) + dy * torch.cos(delta_se2_theta).unsqueeze(-1)) - dy
        
        Mtwist = torch.stack([rotx_dx, roty_dy], dim=-1) * gaussian_exp_t.unsqueeze(-1) # (num_envs, num_tactile_pts, 2)
        
        self.index += 1

        return Mshear + Mtwist
        
    def get_marker_dilation(self, tactile_pts_in_elastomer, tactile_pts_in_indenter):
        # tactile points queried on indenter sdf
        tactile_pts_sdf,_ = self.indenter_sdf_sensor.get_sdf(tactile_pts_in_indenter) # (num_envs, num_tactile_pts)
        tactile_pts_height = F.relu(-tactile_pts_sdf) # (num_envs, num_tactile_pts)
        
        # handle no contact case (reset)
        tactile_contact_sum = (tactile_pts_height > 0).sum(dim=1) # (num_envs,)
        self.index[tactile_contact_sum == 0] = torch.zeros((tactile_contact_sum == 0).sum(), device=self.device, dtype=torch.int32)
        self.indenter2elastomer_tf_history_se2[tactile_contact_sum == 0, :, :] = torch.zeros((tactile_contact_sum == 0).sum(), self.max_steps, 3, device=self.device)


        num_envs, num_tactile_pts = tactile_pts_height.shape
        dvec = (tactile_pts_in_elastomer.unsqueeze(2) - tactile_pts_in_elastomer.unsqueeze(1)) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1) # (num_envs, num_tactile_pts, num_tactile_pts)
        
        gaussian_exp = torch.exp(-self.lambda_d * norm_dvec).unsqueeze(-1).expand(num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3)
        Mdilate = (self.dilate_scale * h * dvec * gaussian_exp).sum(dim=2) # (num_envs, num_tactile_pts, 3)
        
        return Mdilate
    
    def pose2se2(self, quat, pos):
        '''
        NOTE: this is under the assumption tangent is xy in elastomer frame and z is normal direction
        '''
        object2elastomer_xy = pos[:, :2]  # (num_envs, 2)
        object2elastomer_theta = quaternion_to_axis_angle(quat)[:, 2]
        
        object2elastomer_se2 = torch.zeros((pos.shape[0], 3), device=self.device)
        object2elastomer_se2[:, :2] = object2elastomer_xy
        object2elastomer_se2[:, 2] = object2elastomer_theta
        return object2elastomer_se2
    
    def get_marker_shear(self, tactile_pts_in_elastomer, indenter2elastomer_quat, indenter2elastomer_pos):
        num_envs = tactile_pts_in_elastomer.shape[0]
        self.save_indenter2elastomer_quat[torch.arange(num_envs), self.index, :] = indenter2elastomer_quat

        # check if double-cover occurs
        prev_save_indenter2elastomer_quat = self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0] - 1, :]
        curr_save_indenter2elastomer_quat = self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0], :]
        dot_products = torch.sum(prev_save_indenter2elastomer_quat * curr_save_indenter2elastomer_quat, dim=-1)
        sign_corrections = torch.where(dot_products < 0, -1.0, 1.0)
        curr_save_indenter2elastomer_quat = curr_save_indenter2elastomer_quat * sign_corrections.unsqueeze(-1)
        self.save_indenter2elastomer_quat[torch.arange(num_envs, device=self.device)[self.index > 0], self.index[self.index > 0], :] = curr_save_indenter2elastomer_quat

        self.indenter2elastomer_tf_history_se2[torch.arange(num_envs), self.index, :] = self.pose2se2(self.save_indenter2elastomer_quat[torch.arange(num_envs), self.index, :], indenter2elastomer_pos)


        # se2
        indenter2elastomer_se2 = self.indenter2elastomer_tf_history_se2[torch.arange(num_envs), self.index, :] # (num_envs, 3)
        indenter2elastomer_se2_0 = self.indenter2elastomer_tf_history_se2[:, 0, :]
        delta_se2_xy = indenter2elastomer_se2[:, :2] - indenter2elastomer_se2_0[:, :2] # (num_envs, 2)
        delta_se2_theta = indenter2elastomer_se2[:, 2] - indenter2elastomer_se2_0[:, 2] # (num_envs,)

        # delta_se2_xy = indenter2elastomer_se2[:, :2] - indenter2elastomer_se2_0[:, :2] # (num_envs, 2)
        # indenter2elastomer__to__indenter2elastomer0_quat = quat_mul(
        #     quat_conjugate(self.save_indenter2elastomer_quat[:, 0, :]),
        #     self.save_indenter2elastomer_quat[torch.arange(num_envs), self.index, :]
        # )
        # indenter2elastomer__to__indenter2elastomer0_angleaxis = quaternion_to_axis_angle(indenter2elastomer__to__indenter2elastomer0_quat)
        # # rotate angleaxis to be in elastomer frame
        # indenter2elastomer__to__indenter2elastomer0_in_elastomer_angleaxis = tf_apply(
        #     self.save_indenter2elastomer_quat[:, 0, :],
        #     torch.zeros((num_envs, 3), device=self.device),
        #     indenter2elastomer__to__indenter2elastomer0_angleaxis
        # )
        # delta_se2_theta = indenter2elastomer__to__indenter2elastomer0_in_elastomer_angleaxis[:, 2]

        # wrap angle to [-pi, pi]
        # delta_se2_theta = torch.atan2(torch.sin(delta_se2_theta), torch.cos(delta_se2_theta))

        delta_se2_theta = torch.clamp(delta_se2_theta, -180 * torch.pi / 180.0, 180 * torch.pi / 180.0)
        # print(indenter2elastomer_se2[:, 2] * 180 / torch.pi, indenter2elastomer_se2_0[:, 2] * 180 / torch.pi)
        # if delta_se2_theta.abs().max() > torch.pi/2:
        # (M - G) but object center (G) is zero so it becomes M
        gaussian_exp_s = torch.exp(-self.lambda_s * torch.norm((tactile_pts_in_elastomer[:, :, :2] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, :2]), dim=-1)) # (num_envs, num_tactile_pts)
        gaussian_exp_t = torch.exp(-self.lambda_t * torch.norm((tactile_pts_in_elastomer[:, :, :2] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, :2]), dim=-1)) # (num_envs, num_tactile_pts)
        
        Mshear = self.shear_scale * delta_se2_xy.unsqueeze(1) * gaussian_exp_s.unsqueeze(-1) # (num_envs, num_tactile_pts, 2)
        
        dx = tactile_pts_in_elastomer[:, :, 0] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, 0]
        dy = tactile_pts_in_elastomer[:, :, 1] - indenter2elastomer_se2_0.unsqueeze(1)[:, :, 1]
        
        rotx_dx = (dx * torch.cos(delta_se2_theta).unsqueeze(-1) - dy * torch.sin(delta_se2_theta).unsqueeze(-1)) - dx
        roty_dy = (dx * torch.sin(delta_se2_theta).unsqueeze(-1) + dy * torch.cos(delta_se2_theta).unsqueeze(-1)) - dy
        
        Mtwist = self.twist_scale * torch.stack([rotx_dx, roty_dy], dim=-1) * gaussian_exp_t.unsqueeze(-1) # (num_envs, num_tactile_pts, 2)
        
        self.index += 1

        return Mshear + Mtwist
    
    def calculate_marker_motion(self, tactile_pts_in_elastomer, indenter2elastomer_quat, indenter2elastomer_pos):
        elastomer2indenter_quat, elastomer2indenter_pos = tf_inverse(indenter2elastomer_quat, indenter2elastomer_pos)
        
        num_envs, num_tactile_pts, _ = tactile_pts_in_elastomer.shape
        elastomer2indenter_quat = elastomer2indenter_quat.unsqueeze(1).expand(num_envs, num_tactile_pts, 4)
        elastomer2indenter_pos = elastomer2indenter_pos.unsqueeze(1).expand(num_envs, num_tactile_pts, 3)
        
        tactile_pts_in_indenter = tf_apply(elastomer2indenter_quat, elastomer2indenter_pos, tactile_pts_in_elastomer)
        cop2elastomer_quat, cop2elastomer_pos = tf_combine(indenter2elastomer_quat, indenter2elastomer_pos, self.cop2indenter_quat, self.cop2indenter_pos)
        if self.use_original_implementation:
            Mdilate = self.get_old_marker_dilation(tactile_pts_in_elastomer, tactile_pts_in_indenter)
            # use cop2elastomer instead of indenter2elastomer
            Mshear_twist = self.get_old_marker_shear(tactile_pts_in_elastomer, cop2elastomer_quat, cop2elastomer_pos)
        else:
            
            Mdilate = self.get_marker_dilation(tactile_pts_in_elastomer, tactile_pts_in_indenter)
            # use cop2elastomer instead of indenter2elastomer
            Mshear_twist = self.get_marker_shear(tactile_pts_in_elastomer, cop2elastomer_quat, cop2elastomer_pos)
        Mshear_twist = torch.cat([Mshear_twist, torch.zeros((num_envs, num_tactile_pts, 1), device=self.device)], dim=-1)
        
        return Mdilate + Mshear_twist
    
from typing import List
class FotsFieldSensor:
    def __init__(self, fots_sensors: List[FotsSensor], elastomer_link_names: List[str], elastomer_actor_names: List[str]):
        assert len(fots_sensors) == len(elastomer_link_names) == len(elastomer_actor_names)
        
        self.fots_sensors = fots_sensors
        self.elastomer_link_names = elastomer_link_names
        self.elastomer_actor_names = elastomer_actor_names
        
        self.num_divs = [7, 9]
        self.num_tactile_pts = self.num_divs[0] * self.num_divs[1]

        self.local_elastomer_pts = [
            torch.tensor(sensor.elastomer_sdf_sensor.generate_tactile_points(margin=0.003, local_z_dir=-1, num_divs=self.num_divs)).to(self.device).to(torch.float32)
            for sensor in self.fots_sensors
        ]
    
    def reset_fots(self, indenter_quat, indenter_pos):
        indenter2elastomer_tf_list, elastomer2indenter_tf_list = self.get_indenter2elastomer_tf_fots(indenter_quat, indenter_pos)
        for sensor_idx in range(len(self.fots_sensors)):
            elastomerpts_in_elastomer = torch.tensor(self.local_elastomer_pts[sensor_idx]).unsqueeze(0).expand(self.num_envs, self.num_tactile_pts, 3)
            # get elastomerpts in indenter
            # use plug_quat and plug_pos for indenter
            elastomer2indenter_quat, elastomer2indenter_pos = elastomer2indenter_tf_list[sensor_idx]
            elastomerpts_in_indenter = tf_apply(
                elastomer2indenter_quat.unsqueeze(1).expand(self.num_envs, self.num_tactile_pts, 4),
                elastomer2indenter_pos.unsqueeze(1).expand(self.num_envs, self.num_tactile_pts, 3),
                elastomerpts_in_elastomer
            )
            # get sdf values
            sdf_elastomerpts_in_indenter, _ = self.fots_sensors[sensor_idx].indenter_sdf_sensor.get_sdf(elastomerpts_in_indenter)
            # find out which elastomerpts are in contact with indenter and avg... if there are no pts in contact then set to zero rot zero pos
            in_contact_mask = sdf_elastomerpts_in_indenter < 0.0  # (num_envs, num_tactile_pts)
            num_contact_pts_per_env = torch.sum(in_contact_mask.float(), dim=-1)  # (num_envs,)
            masked_elastomerpts_in_indenter = elastomerpts_in_indenter * in_contact_mask.unsqueeze(-1).float()
            cop2indenter_pos = torch.zeros((self.num_envs, 3), device=self.device)
            cop2indenter_pos[num_contact_pts_per_env > 0] = torch.sum(masked_elastomerpts_in_indenter[num_contact_pts_per_env > 0], dim=1) / num_contact_pts_per_env[num_contact_pts_per_env > 0].unsqueeze(-1).float()
            
            # set to identity quaternion first
            cop2indenter_quat = torch.zeros((self.num_envs, 4), device=self.device)
            cop2indenter_quat[:, 3] = 1.0
            
            self.fots_sensors[sensor_idx].initialize(self.num_envs, cop2indenter_quat=cop2indenter_quat, cop2indenter_pos=cop2indenter_pos)
            
    
    def get_link_handle(self, actor_name, link_name):
        link_handle = self.gym.find_actor_rigid_body_handle(
            self.env_ptrs[0], self.actor_handles[actor_name], link_name)
        return link_handle
    
    def get_indenter2elastomer_tf_fots(self, indenter_quat, indenter_pos):
        indenter2elastomer_tf_list = []
        elastomer2indenter_tf_list = []
        
        for i in range(len(self.fots_sensors)):
            elastomer_link_id = self.get_link_handle(self.elastomer_actor_names[i], self.elastomer_link_names[i])
            elastomer_body_quat = self.body_quat[:, elastomer_link_id].expand(self.num_envs, 4)
            elastomer_body_pos = self.body_pos[:, elastomer_link_id].expand(self.num_envs, 3)
            
            inv_elastomer_quat, inv_elastomer_pos = tf_inverse(elastomer_body_quat, elastomer_body_pos)
            # world2elastomer @ indenter2world -> indenter2elastomer
            indenter2elastomer_quat, indenter2elastomer_pos = tf_combine(inv_elastomer_quat, inv_elastomer_pos, indenter_quat, indenter_pos)
            elastomer2indenter_quat, elastomer2indenter_pos = tf_inverse(indenter2elastomer_quat, indenter2elastomer_pos)
            indenter2elastomer_tf_list.append((indenter2elastomer_quat, indenter2elastomer_pos))
            elastomer2indenter_tf_list.append((elastomer2indenter_quat, elastomer2indenter_pos))
        return indenter2elastomer_tf_list, elastomer2indenter_tf_list
    
    def get_displacement_field_dict(self, indenter_quat, indenter_pos):
        indenter2elastomer_tf_list, elastomer2indenter_tf_list = self.get_indenter2elastomer_tf_fots(indenter_quat, indenter_pos)
        
        marker_displacements_dict = {}
        for i, sensor in enumerate(self.fots_sensors):
            elastomer_pts_in_elastomer = torch.tensor(self.local_elastomer_pts[i], device=self.device).unsqueeze(0).expand(self.num_envs, self.num_tactile_pts, 3) # (num_envs, num_elastomer_pts, 3)
            indenter2elastomer_quat, indenter2elastomer_pos = indenter2elastomer_tf_list[i]
            marker_displacement = sensor.calculate_marker_motion(elastomer_pts_in_elastomer, indenter2elastomer_quat, indenter2elastomer_pos)
            marker_displacement = marker_displacement.reshape(self.num_envs, self.num_divs[1], self.num_divs[0], 3)
            marker_displacements_dict[self.elastomer_link_names[i]] = marker_displacement
        
        return marker_displacements_dict