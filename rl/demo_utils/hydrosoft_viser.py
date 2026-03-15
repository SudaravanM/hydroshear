# https://github.com/mujocolab/mjlab/blob/dcfce29254d02f2a847897bf1854d8a0ea25d5e9/src/mjlab/viewer/viser.py#L856-L897
# Refer to above to add force arrows to viser


import torch

import pytorch_volumetric as pv

import torch.nn.functional as F
import numpy as np
import open3d as o3d

class HydroSoftSensor:
    '''
        TactileFieldSensor gets sdf of indenter and queries elastomer points.
        HydroSoftSensor is the reverse where it gets sdf of elastomer and queries indenter points.
    '''
    def __init__(self, server):
        super().__init__()
        if not hasattr(self, 'device'):
            self.device = 'cpu'
        
        # initialize coefficients
        self.I = 1
        self.K = 1
        self.E = 1
        self.A = 1
        self.mu = 1e20
        
        # these are calibrated irl params
        # self.lambda_d = 1500
        # self.lambda_s = 2e5
        # self.dilate_scale = 30769.2308
        # self.shear_scale  = 1230769.23
        
        # params for visual
        self.lambda_d = 100_000
        self.lambda_s = 2e4
        self.dilate_scale = 30769.2308 * 20
        self.shear_scale  = 1230769.23 / 30
        
        # self.lambda_d = 2000
        # self.lambda_s = 40000
        # self.dilate_scale = 38461.5385
        # self.shear_scale  = 790000
        
        self.hydrosoft_forces = None
        
        self.server = server
        self.force_arrows = []
    
    def heaviside(self, x):
        return 0.5 * (torch.sign(x) + 1)
    
    def initialize(self, num_envs, num_obj_pts):
        self.hydrosoft_forces = torch.zeros((num_envs, num_obj_pts, 3), device=self.device)
        self.prev_sdf = None
        self.prev_grasped_obj_pts_elastomer = None
    
    def step_hydrosoft_forces(self, grasped_obj_pts_in_elastomer, grasped_obj_sdf):
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
        fn = self.hydrosoft_forces[:,:,2] + self.E * self.A * displacement[:, :, 2]
        ft = self.hydrosoft_forces[:,:,:2] + self.K * self.A * displacement[:, :, :2]
        
        fn_bar = F.relu(fn)
        norm_ft = torch.norm(ft, dim=-1) # (num_envs, num_pts)
        # ft_bar = torch.minimum((self.mu * fn_bar) / (norm_ft + 1e-8), torch.ones_like(norm_ft)).unsqueeze(-1) * ft
        normalize_ft = torch.divide(ft, norm_ft.unsqueeze(-1))
        normalize_ft[torch.isnan(normalize_ft)] = 0.0
        ft_bar = torch.minimum(self.mu * fn_bar, norm_ft).unsqueeze(-1) * normalize_ft
        # print("Sum of projected points:", ((self.mu * fn_bar / (norm_ft + 1e-8)) < 1.0).sum())
        
        
        fbar = torch.cat([ft_bar, fn_bar.unsqueeze(-1)], dim=-1) # (num_envs, num_pts, 3)
        fbar = self.heaviside(-sdf).unsqueeze(-1) * fbar

        self.hydrosoft_forces = fbar
        
        # Clear previous force arrows
        for arrow_handle in self.force_arrows:
            try:
                arrow_handle.remove()
            except (KeyError, RuntimeError):
                # Handle already removed or doesn't exist
                pass
        self.force_arrows = []
        
        # Add new force arrows
        np_fbar = fbar.squeeze(0).cpu().numpy()
        np_pts = grasped_obj_pts_in_elastomer.squeeze(0).cpu().numpy()
        for i in range(np_fbar.shape[0]):
            force = np_fbar[i]
            if np.linalg.norm(force) <= 1e-3:
                continue
            
            arrow_name = f"force_arrow_{i}"
            force_magnitude = np.linalg.norm(force)
            
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=1e-5 * 10,
                cone_radius=1e-3 * 1e-2 * 30,
                cylinder_height=(np.linalg.norm(force) - 1e-3),  # scale height by magnitude
                cone_height=(1e-3),  # scale height by magnitude
            )
            arrow.paint_uniform_color([1, 0, 0])
            arrow.translate(np_pts[i])
            
            arrow_origin = np_pts[i]
            shear_direction = force / np.linalg.norm(force)  # normalize shear direction
            # use https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            # makes use of rodrigues formula to solve the rotation matrix
            v = np.cross(np.array([0, 0, 1]), shear_direction)
            c = np.array([0, 0, 1]) @ shear_direction # cos(theta) where theta is angle between vectors since both are unit vectors
            def skew(v):
                """Create skew-symmetric matrix from vector v."""
                return np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
            skew_v = skew(v)
            skew_v_squared = skew_v @ skew_v
            rotm = np.eye(3) + skew_v + skew_v_squared * (1 / (1 + c))
            arrow.rotate(rotm, center=arrow_origin)
            
            handle = self.server.scene.add_mesh_simple(
                name=arrow_name,
                vertices=np.asarray(arrow.vertices),
                faces=np.asarray(arrow.triangles),
                color=(255, 0, 0),
                # material="standard",
                # flat_shading=True,
                # cast_shadow=False,
                # receive_shadow=False,
                # wireframe=False,
                # side="double",
                opacity=1.0,
            )
            
            # Create arrow pointing in force direction
            # handle = self.server.scene.add_spline_catmull_rom(
            #     arrow_name,
            #     positions=np.array([
            #         np_pts[i],
            #         np_pts[i] + force_direction * force_magnitude
            #     ]),
            #     color=(255, 0, 0),
            #     line_width=2.0,
            # )
            
            
            self.force_arrows.append(handle)
        
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
        dvec = tactile_pts_elastomer.unsqueeze(2) - tactile_pts_elastomer.unsqueeze(1) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1)**2 # (num_envs, num_tactile_pts, num_tactile_pts)

        gaussian_exp = torch.exp(-self.lambda_d * norm_dvec).unsqueeze(-1).expand(num_envs, num_tactile_pts, num_tactile_pts, 3) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        Mdilate = (self.dilate_scale * h * dvec * gaussian_exp).sum(dim=2)
        
        return Mdilate
    
    def get_hydrosoft_displacement(self, tactile_pts_elastomer, obj_pts_elastomer, obj_pts_sdf):
        grasped_obj_pts_in_elastomer = obj_pts_elastomer # (num_envs, num_pts, 3)
        
        
        fbar = self.step_hydrosoft_forces(grasped_obj_pts_in_elastomer, obj_pts_sdf)
        
        # assume fbar is just marker displacement from elastomer
        affected_marker_positions = grasped_obj_pts_in_elastomer + fbar # get markers affected during motion (num_envs, num_pts, 3)
        
        # obj heights also acts as a mask because when not in contact, fbar is 0
        obj_pts_height = fbar[:, :, -1] # (num_envs, num_pts)
        
        #tactile_pts_elastomer: (num_envs, num_tactile_pts, 3)
        num_envs, num_pts, _ = obj_pts_elastomer.shape
        num_tactile_pts = tactile_pts_elastomer.shape[1]
        
        tactile_pts_elastomer = tactile_pts_elastomer.unsqueeze(1).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        affected_marker_positions = affected_marker_positions.unsqueeze(2).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        fbar = fbar.unsqueeze(2).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts, 3)
        norm_elastomerpts_hydropts = torch.norm(tactile_pts_elastomer - affected_marker_positions, dim=-1) # (num_envs, num_pts, num_tactile_pts)
        gaussian_exp = torch.exp(-self.lambda_s * norm_elastomerpts_hydropts**2).unsqueeze(-1).expand(num_envs, num_pts, num_tactile_pts, 3) # (num_envs, num_pts, num_tactile_pts)
        
        h = obj_pts_height.unsqueeze(-1).unsqueeze(-1).expand(num_envs, num_pts, num_tactile_pts, 3)
        Mshear = torch.sum(self.shear_scale * h * -1 * fbar * gaussian_exp, dim=1) # (num_envs, num_tactile_pts, 3)
        
        return Mshear
    
    def get_marker_displacement(self, tactile_pts_elastomer, tactile_pts_height, obj_pts_elastomer, obj_pts_sdf):
        
        Mdilate = self.get_dilation_displacement(tactile_pts_elastomer, tactile_pts_height)
        Mshear = self.get_hydrosoft_displacement(tactile_pts_elastomer, obj_pts_elastomer, obj_pts_sdf)
        self.Mshear = Mshear
        
        return Mdilate + Mshear

