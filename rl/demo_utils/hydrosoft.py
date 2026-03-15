
import torch

import pytorch_volumetric as pv

# from hydroshear.utils.torch_utils import tf_combine, tf_inverse, tf_apply
# from hydroshear.utils.torch_utils import quat_apply, quat_conjugate, quat_mul
import torch.nn.functional as F
import numpy as np


class HydroSoftSensor:
    '''
        TactileFieldSensor gets sdf of indenter and queries elastomer points.
        HydroSoftSensor is the reverse where it gets sdf of elastomer and queries indenter points.
    '''
    def __init__(self, mu = 2.0, lambda_d = 5e4, lambda_s = 5e4, shear_scale = 50, dilate_scale = 10, normal_axis=2):
        super().__init__()
        if not hasattr(self, 'device'):
            self.device = 'cpu'
        
        # initialize coefficients
        self.I = 1
        self.K = 1
        self.E = 1
        self.A = 1
        self.mu = mu
        
        self.lambda_s = lambda_s
        self.lambda_d = lambda_d
        
        self.shear_scale = shear_scale
        self.dilate_scale = dilate_scale
        
        self.hydrosoft_forces = None
        
        
        self.normal_axis = normal_axis  # default z axis
        self.tangent_axes = [i for i in range(3) if i != normal_axis]
    
    
    def initialize(self, num_envs, num_obj_pts):
        self.hydrosoft_forces = torch.zeros((num_envs, num_obj_pts, 3), device=self.device)
        self.prev_sdf = None
        self.prev_grasped_obj_pts_elastomer = None
    
    def step_hydrosoft_forces(self, grasped_obj_pts_in_elastomer, grasped_obj_sdf, reverse_z=False):
        sdf = grasped_obj_sdf
        
        #displacement
        if self.prev_grasped_obj_pts_elastomer is None:
            self.prev_grasped_obj_pts_elastomer = grasped_obj_pts_in_elastomer
        if self.prev_sdf is None:
            self.prev_sdf = sdf
        
        # alpha_d = (F.relu(-sdf) - F.relu(-self.prev_sdf)) / (self.prev_sdf - sdf + 1e-8)
        # displacement = alpha_d.unsqueeze(-1) * (self.prev_grasped_obj_pts_elastomer - grasped_obj_pts_in_elastomer) # (num_envs, num_pts, 3)
        alpha_d = -torch.divide(F.relu(-self.prev_sdf) - F.relu(-sdf), self.prev_sdf - sdf)
        alpha_isnan = torch.isnan(alpha_d)
        alpha_d[alpha_isnan] = -0.5 * (torch.sign(sdf[alpha_isnan]) - 1)
        displacement = alpha_d.unsqueeze(-1) * (self.prev_grasped_obj_pts_elastomer - grasped_obj_pts_in_elastomer)
        
        # NOTE: assume +z is out of elastomer
        # self.hydrosoft_forces[:,:,2] += self.E * self.A * displacement[:, :, 2]
        # self.hydrosoft_forces[:,:,:2] += self.K * self.A * displacement[:, :, :2]
        # fn = self.hydrosoft_forces[:,:,2]
        # ft = self.hydrosoft_forces[:,:,:2]
        fn = self.hydrosoft_forces[:,:,self.normal_axis] + self.E * self.A * displacement[:, :, self.normal_axis]
        ft = self.hydrosoft_forces[:,:,self.tangent_axes] + self.K * self.A * displacement[:, :, self.tangent_axes]
        
        # fn_bar = F.relu(fn)
        fn_bar = F.relu(fn * (-1 if reverse_z else 1))
        
        norm_ft = torch.norm(ft, dim=-1) # (num_envs, num_pts)
        # norm_ft[norm_ft < 1e-8] = 1
        # ft_bar = torch.zeros_like(ft)
        # ft_bar[norm_ft > 1e-8] = torch.clamp(self.mu * fn_bar[norm_ft > 1e-8] / norm_ft[norm_ft > 1e-8], 1).unsqueeze(-1) * ft[norm_ft > 1e-8] 
        # ft_bar = torch.minimum(self.mu * fn_bar / (norm_ft + 1e-8), torch.ones_like(norm_ft)).unsqueeze(-1) * ft
        ft_bar = torch.minimum(self.mu * fn_bar, norm_ft).unsqueeze(-1) * ft / (norm_ft.unsqueeze(-1) + 1e-8)
        
        fbar = torch.cat([ft_bar, fn_bar.unsqueeze(-1) * (-1 if reverse_z else 1)], dim=-1)
        fbar = torch.heaviside(-sdf, values=torch.tensor([0.0], device=self.device)).unsqueeze(-1) * fbar # zero out forces if not in contact

        self.hydrosoft_forces = fbar
        
        self.prev_grasped_obj_pts_elastomer = grasped_obj_pts_in_elastomer
        self.prev_sdf = sdf

        return fbar
    
    def get_dilation_displacement(self, tactile_pts_elastomer, tactile_pts_height):
        '''
        tactile_pts_elastomer: (num_envs, num_tactile_pts, 3)
        tactile_pts_height: (num_envs, num_tactile_pts)
        '''
        # sum_i delta_h_i (M - C_i) * exp(-lambda_d * \|M - C_i\|^2)
        num_envs, num_tactile_pts, _ = tactile_pts_elastomer.shape
        dvec = (tactile_pts_elastomer.unsqueeze(2) - tactile_pts_elastomer.unsqueeze(1)) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1) # (num_envs, num_tactile_pts, num_tactile_pts)

        gaussian_exp = torch.exp(-self.lambda_d * norm_dvec**2).unsqueeze(-1).expand(num_envs, num_tactile_pts, num_tactile_pts, 3) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        Mdilate = (self.dilate_scale * h * dvec * gaussian_exp).sum(dim=2)
        
        return Mdilate
    
    def get_hydrosoft_displacement(self, tactile_pts_elastomer, obj_pts_elastomer, obj_pts_sdf, reverse_z=False):
        grasped_obj_pts_in_elastomer = obj_pts_elastomer # (num_envs, num_pts, 3)
        
        
        fbar = self.step_hydrosoft_forces(grasped_obj_pts_in_elastomer, obj_pts_sdf, reverse_z)
        
        # assume fbar is just marker displacement from elastomer
        affected_marker_positions_ = grasped_obj_pts_in_elastomer + fbar # get markers affected during motion (num_envs, num_pts, 3)
        
        # obj heights also acts as a mask because when not in contact, fbar is 0
        obj_pts_height = fbar[:, :, self.normal_axis] # (num_envs, num_pts)
        
        #tactile_pts_elastomer: (num_envs, num_tactile_pts, 3)
        num_envs, num_pts, _ = obj_pts_elastomer.shape
        num_tactile_pts = tactile_pts_elastomer.shape[1]
        
        tactile_pts_elastomer = tactile_pts_elastomer.unsqueeze(1).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        affected_marker_positions = affected_marker_positions_.unsqueeze(2).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        fbar = fbar.unsqueeze(2).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        norm_elastomerpts_hydropts = torch.norm(tactile_pts_elastomer - affected_marker_positions, dim=-1) # (num_envs, num_pts, num_tactile_pts)
        gaussian_exp = torch.exp(-self.lambda_s * norm_elastomerpts_hydropts**2).unsqueeze(-1).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts)
        
        h = obj_pts_height.unsqueeze(-1).unsqueeze(-1).expand(num_envs, num_pts, num_tactile_pts, 3)
        Mshear = torch.sum(self.shear_scale * h * -1 * fbar * gaussian_exp * (-1 if reverse_z else 1), dim=1) # (num_envs, num_tactile_pts, 3)
        
        return Mshear, affected_marker_positions_
    
    def get_marker_displacement(self, tactile_pts_elastomer, tactile_pts_height, obj_pts_elastomer, obj_pts_sdf):
        
        Mdilate = self.get_dilation_displacement(tactile_pts_elastomer, tactile_pts_height)
        Mshear = self.get_hydrosoft_displacement(tactile_pts_elastomer, obj_pts_elastomer, obj_pts_sdf)
        
        return Mdilate + Mshear