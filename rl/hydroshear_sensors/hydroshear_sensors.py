from isaacgym.torch_utils import tf_combine, tf_inverse, tf_apply

import os
import numpy as np
from urdfpy import URDF

import torch
import torch.nn.functional as F

import trimesh
import open3d as o3d
import pytorch_volumetric as pv

import point_cloud_utils as pcu
# import time
# import cProfile
# import pstats

from termcolor import cprint

class MeshSDFSensor:
    '''
    Class for calculating the signed distance field (SDF) of a mesh and querying points on the mesh.
    Uses PyTorch Volumetric for SDF calculation.
    '''
    def __init__(self, urdf_root, urdf_path, link_name, device='cpu', resolution=0.001):
        urdf = URDF.load(os.path.join(urdf_root, urdf_path))
        
        self.device = device
        self.mesh_path = urdf.link_map[link_name].collisions[0].geometry.mesh.filename
        self.mesh_pv = pv.MeshObjectFactory(os.path.join(urdf_root, self.mesh_path))

        # trimesh_mesh = urdf.link_map[link_name].collisions[0].geometry.mesh.meshes[0]
        # self.mesh_pv._mesh = o3d.geometry.TriangleMesh(
        #     vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
        #     triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces)
        # )
        # self.mesh_pv._mesht = None
        self.mesh_pv._mesh.compute_vertex_normals()
        
        self.mesh = urdf.link_map[link_name].collisions[0].geometry.mesh.meshes[0]
        
        self.mesh_pv.precompute_sdf()
        # self.mesh_sdf = pv.MeshSDF(self.mesh_pv)
        self.mesh_sdf_gt = pv.MeshSDF(self.mesh_pv)
        self.mesh_sdf = pv.CachedSDF(
            object_name=link_name, 
            resolution=resolution, 
            range_per_dim=self.mesh_pv.bounding_box(padding=0.1), 
            gt_sdf=self.mesh_sdf_gt,
            device=self.device,
            cache_path=f'{link_name}_cached_sdf.pkl',
            clean_cache=False
        )

        
        del urdf # save memory
    
    def get_sdf(self, points_in_mesh):
        '''
        Takes in points in mesh frame and return sdf values / normals at those points.
        '''
        sdf, normal = self.mesh_sdf(points_in_mesh)
        return sdf, normal

    def sample_points(self, num_points):
        points, normals, _ = pv.sample_mesh_points(self.mesh_pv, num_points=num_points, dbpath='model_points_cache.pkl', device=self.device, dtype=torch.float32)
        os.remove('model_points_cache.pkl')
        return points, normals

    def poisson_sample_points(self, radius, initial_num_points=int(1e8)):
        points, normals, _ = pv.sample_mesh_points(self.mesh_pv, num_points=initial_num_points, dbpath='model_points_cache.pkl', device=self.device, dtype=torch.float32)
        os.remove('model_points_cache.pkl')
        idx = torch.tensor(pcu.downsample_point_cloud_poisson_disk(points.cpu().numpy(), radius=radius, target_num_samples=-1))
        return points[idx], normals[idx]

    def generate_tactile_points_trimesh(self, num_divs=[20, 30], margin=0.003, local_z_dir = 1):
        
        mesh_dims = np.diff(self.mesh.bounds, axis=0).squeeze()
        slim_axis = np.argmin(mesh_dims)   # determine flat axis of elastomer
        self.slim_axis = slim_axis
        
        axis_idxs = list(range(3))
        axis_idxs.remove(slim_axis)     # remove slim idx
        div_sz = (mesh_dims[axis_idxs] - margin * 2.) / (np.array(num_divs) + 1)
        tactile_points_dx = min(div_sz)
        axis_idxs.append(slim_axis)
        
        center = (self.mesh.bounds[0] + self.mesh.bounds[1]) / 2.
        
        local_x_pts = np.linspace(center[axis_idxs[0]] - tactile_points_dx * (num_divs[0] + 1.) / 2., center[axis_idxs[0]] + tactile_points_dx * (num_divs[0] + 1.) / 2., num_divs[0] + 2)[1:-1]
        local_y_pts = np.linspace(center[axis_idxs[1]] - tactile_points_dx * (num_divs[1] + 1.) / 2., center[axis_idxs[1]] + tactile_points_dx * (num_divs[1] + 1.) / 2., num_divs[1] + 2)[1:-1]
        local_xv, local_yv = np.meshgrid(local_x_pts, local_y_pts)
        local_zv = np.zeros_like(local_xv) + center[slim_axis]
        
        local_pts_list = [None, None, None]
        local_pts_list[axis_idxs[0]] = local_xv
        local_pts_list[axis_idxs[1]] = local_yv
        local_pts_list[slim_axis] = local_zv
        
        pts3d = np.stack(local_pts_list, axis=-1).reshape(-1, 3)
        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        ray_dir = np.array([0, 0, 0])
        ray_dir[slim_axis] = local_z_dir
        
        index_tri, index_ray, locations = mesh_data.intersects_id(pts3d,
                                                                  np.tile([ray_dir], (pts3d.shape[0], 1)),
                                                                  return_locations=True, multiple_hits=False)
        tactile_points = locations[index_ray.argsort()]
        return tactile_points

    def generate_tactile_points(self, num_divs=[20, 30], margin=0.003, local_z_dir = 1):

        # generate grid on elastomer
        bbox = self.mesh_pv._mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        dims = bbox.get_extent()
        slim_axis = np.argmin(dims) # this is axis penetration will happen along
        self.slim_axis = slim_axis
        
        # determine gap between adjacent tactile points
        axis_idxs = list(range(3))
        axis_idxs.remove(slim_axis)
        div_sz = (dims[axis_idxs] - margin * 2.) / (np.array(num_divs) + 1)
        tactile_points_dx = min(div_sz)
        axis_idxs.append(slim_axis)
        # sample points on the flat plane
        center = bbox.get_center()
        
        local_x_pts = np.linspace(center[axis_idxs[0]] - tactile_points_dx * (num_divs[0] + 1.) / 2., center[axis_idxs[0]] + tactile_points_dx * (num_divs[0] + 1.) / 2., num_divs[0] + 2)[1:-1]
        local_y_pts = np.linspace(center[axis_idxs[1]] - tactile_points_dx * (num_divs[1] + 1.) / 2., center[axis_idxs[1]] + tactile_points_dx * (num_divs[1] + 1.) / 2., num_divs[1] + 2)[1:-1]
        local_xv, local_yv = np.meshgrid(local_x_pts, local_y_pts)
        local_zv = np.zeros_like(local_xv) + center[slim_axis]
        
        local_pts_list = [None, None, None]
        local_pts_list[axis_idxs[0]] = local_xv
        local_pts_list[axis_idxs[1]] = local_yv
        local_pts_list[slim_axis] = local_zv
        
        # create rays
        pts3d = np.stack(local_pts_list, axis=-1).reshape(-1, 3)
        slim_ray = np.array([0., 0., 0.], dtype=np.float32)
        slim_ray[slim_axis] = local_z_dir
        ray3d = slim_ray.reshape(1, 3).repeat(pts3d.shape[0], axis=0)
        rays  = np.concatenate([pts3d, ray3d], axis=-1) # (num_rays, 6)
        rays  = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        ans = self.mesh_pv._raycasting_scene.cast_rays(rays)
        local_zv = ans['t_hit'].numpy() * local_z_dir + center[slim_axis].flatten()
        
        local_pts_list[slim_axis] = local_zv.reshape(local_xv.shape)
        tactile_points = np.stack(local_pts_list, axis=-1).reshape(-1, 3)
        
        return tactile_points
    def get_normal_axis(self):
        assert hasattr(self, 'slim_axis'), "Must run generate_tactile_points() before get_slim_axis()"
        return self.slim_axis
    def get_tangent_axes(self):
        assert hasattr(self, 'slim_axis'), "Must run generate_tactile_points() before get_tangent_axes()"
        tangent_axes = list(range(3))
        tangent_axes.remove(self.slim_axis)
        return tangent_axes

class HydroFotsSensor:
    '''
        This class implements the HydroFOTS sensor and integrates it into IsaacGym Framework.
        Vocabulary:
        - elastomer: deformable material on a tactile sensor whose deformation is measured to measure force interaction between robot and object.
        - indenter: object used to indent (apply force to) the elastomer.
    '''
    
    def __init__(
        self, 
        hydrofots_cfg,
        elastomer_urdf_root,
        elastomer_urdf,
        elastomer_link_name,
        indenter_urdf_root,
        indenter_urdf,
        indenter_link_name,
        device='cpu', indenter_sdf_sensor = None,
        elastomer_resolution=0.001,
        indenter_resolution=0.001,
    ):
        self.elastomer_sdf_sensor = MeshSDFSensor(elastomer_urdf_root,elastomer_urdf, elastomer_link_name, device, elastomer_resolution)
        self.indenter_sdf_sensor  = MeshSDFSensor(indenter_urdf_root,indenter_urdf, indenter_link_name, device, indenter_resolution) if indenter_sdf_sensor is None else indenter_sdf_sensor
        
        self.device = device
        self.hydrofots_cfg = hydrofots_cfg

        # initialize coefficients from config
        if hydrofots_cfg is not None:
            self.I = torch.tensor(hydrofots_cfg.get('I', 1), device=device)
            self.K = torch.tensor(hydrofots_cfg.get('K', 1), device=device)
            self.E = torch.tensor(hydrofots_cfg.get('E', 1), device=device)
            self.A = torch.tensor(hydrofots_cfg.get('A', 1), device=device)
            self.mu = torch.tensor(hydrofots_cfg.get('mu', 10000.0), device=device)
            self.lambda_s = torch.tensor(hydrofots_cfg.get('lambda_s', 300), device=device)
            self.lambda_d = torch.tensor(hydrofots_cfg.get('lambda_d', 700), device=device)
            self.dilate_scale = torch.tensor(hydrofots_cfg.get('dilate_scale', 1000 / 0.065), device=device)
            self.shear_scale = torch.tensor(hydrofots_cfg.get('shear_scale', 1000 / 0.065), device=device)
            self.hydroshear_gravity_noise = torch.tensor(hydrofots_cfg.get('hydroshear_gravity_noise', 0.00025), device=device)
            self.hydroshear_gravity_rot_noise = torch.tensor(hydrofots_cfg.get('hydroshear_gravity_rot_noise', 0.0), device=device)
        
        self.randomize_coefficients = hydrofots_cfg.get('randomize_coefficients', False)
        self.randomize_every_step = hydrofots_cfg.get('randomize_every_step', False)
        self.randomize_every_episode = hydrofots_cfg.get('randomize_every_episode', False)
        
        self.hydrosoft_forces = None
        self.prev_sdf = None
        self.prev_indenter_pts_in_elastomer = None

        # FFT-accelerated dilation (opt-in; configured via enable_fft_dilation()).
        # For a regular planar taxel grid the O(N^2) pairwise dilation sum is an
        # exact 2D convolution, evaluated with an FFT in O(N log N). See
        # fft_dilation.py and the paper's Appendix A.7. Disabled by default; the
        # FFT only pays off past a few hundred taxels (below that the dense path
        # is faster), so we gate on num_tactile_pts >= fft_dilation_min_pts.
        self._fft_dilation = None
        self.use_fft_dilation = hydrofots_cfg.get('use_fft_dilation', False) if hydrofots_cfg is not None else False
        self.fft_dilation_min_pts = hydrofots_cfg.get('fft_dilation_min_pts', 256) if hydrofots_cfg is not None else 256

        # GEMM-accelerated shear (opt-in). Shear sums over irregular indenter
        # points so it is NOT an FFT/convolution, but it factors into a batched
        # matmul that is exact for any layout, faster, and avoids the dense
        # (E, P, N, 3) tensor that OOMs at scale. See fft_shear.py / FFT.md.
        self.use_matmul_shear = hydrofots_cfg.get('use_matmul_shear', False) if hydrofots_cfg is not None else False

        cprint(f"HydroFOTS coefficients: {self.mu=}, {self.lambda_s=}, {self.lambda_d=}, {self.dilate_scale=}, {self.shear_scale=}", "green")
        
    def initialize(self, num_envs, num_indenter_pts, randomization_dict=None):
        self.hydrosoft_forces = torch.zeros((num_envs, num_indenter_pts, 3), device=self.device)
        self.aug_hydrosoft_forces = torch.zeros((num_envs, num_indenter_pts, 3), device=self.device)
        self.prev_sdf = None
        self.prev_indenter_pts_in_elastomer = None

        if self.randomize_coefficients:
            if randomization_dict is None:
                mu_range = self.hydrofots_cfg.get('mu_range', [0.5, 0.75])
                lambda_d_range = self.hydrofots_cfg.get('lambda_d_range', [11000, 22000])
                lambda_s_range = self.hydrofots_cfg.get('lambda_s_range', [7000, 13000])
                dilate_scale_range = self.hydrofots_cfg.get('dilate_scale_range', [1000 / 0.065 * 0.8, 1000 / 0.065 * 1.2])
                shear_scale_range = self.hydrofots_cfg.get('shear_scale_range', [1000 / 0.065 * 0.9, 1000 / 0.065 * 1.1])
                hydroshear_gravity_range = self.hydrofots_cfg.get('hydroshear_gravity_range', [0.0001, 0.00025])
                hydroshear_gravity_rot_range = self.hydrofots_cfg.get('hydroshear_gravity_rot_range', [0.0, 0.001])

                self.mu = torch.tensor(mu_range[0] + torch.rand(num_envs, device=self.device) * (mu_range[1] - mu_range[0]), device=self.device).unsqueeze(-1)
                self.lambda_d = torch.tensor(lambda_d_range[0] + torch.rand(num_envs, device=self.device) * (lambda_d_range[1] - lambda_d_range[0]), device=self.device).unsqueeze(-1)
                self.lambda_s = torch.tensor(lambda_s_range[0] + torch.rand(num_envs, device=self.device) * (lambda_s_range[1] - lambda_s_range[0]), device=self.device).unsqueeze(-1)
                self.dilate_scale = torch.tensor(dilate_scale_range[0] + torch.rand(num_envs, device=self.device) * (dilate_scale_range[1] - dilate_scale_range[0]), device=self.device).unsqueeze(-1)
                self.shear_scale = torch.tensor(shear_scale_range[0] + torch.rand(num_envs, device=self.device) * (shear_scale_range[1] - shear_scale_range[0]), device=self.device).unsqueeze(-1)
                self.hydroshear_gravity_noise = torch.tensor(hydroshear_gravity_range[0] + torch.rand(num_envs, device=self.device) * (hydroshear_gravity_range[1] - hydroshear_gravity_range[0]), device=self.device).unsqueeze(-1)
                self.hydroshear_gravity_rot_noise = torch.tensor(hydroshear_gravity_rot_range[0] + torch.rand(num_envs, device=self.device) * (hydroshear_gravity_rot_range[1] - hydroshear_gravity_rot_range[0]), device=self.device).unsqueeze(-1)
            else:
                self.mu = randomization_dict.get('mu', self.mu)
                self.lambda_d = randomization_dict.get('lambda_d', self.lambda_d)
                self.lambda_s = randomization_dict.get('lambda_s', self.lambda_s)
                self.dilate_scale = randomization_dict.get('dilate_scale', self.dilate_scale)
                self.shear_scale = randomization_dict.get('shear_scale', self.shear_scale)
                self.hydroshear_gravity_noise = randomization_dict.get('hydroshear_gravity_noise', self.hydroshear_gravity_noise)
                self.hydroshear_gravity_rot_noise = randomization_dict.get('hydroshear_gravity_rot_noise', self.hydroshear_gravity_rot_noise)

        left_randomization_dict = {
            'mu': self.mu,
            'lambda_d': self.lambda_d,
            'lambda_s': self.lambda_s,
            'dilate_scale': self.dilate_scale,
            'shear_scale': self.shear_scale,
            'hydroshear_gravity_noise': self.hydroshear_gravity_noise,
            'hydroshear_gravity_rot_noise': self.hydroshear_gravity_rot_noise,
        }

        # cprint(f"HydroShear coefficients:", "green")
        # cprint(f"mu {self.mu}", "green")
        # cprint(f"lambda_s {self.lambda_s}", "green")
        # cprint(f"lambda_d {self.lambda_d}", "green")
        # cprint(f"dilate_scale {self.dilate_scale / 1000 * 0.065} * 1000 / 0.065", "green")
        # cprint(f"shear_scale {self.shear_scale / 1000 * 0.065} * 1000 / 0.065", "green")
        # cprint(f"hydroshear gravity effect {self.hydroshear_gravity_noise}", "green")

        return left_randomization_dict


    def step_hydrosoft_forces(self, sdf, indenter_pts_in_elastomer, reverse_z=True):
        '''
        Calculate hydrosoft forces based on SDF.
        '''
        if self.prev_sdf is None:
            self.prev_sdf = sdf
            self.prev_indenter_pts_in_elastomer = indenter_pts_in_elastomer
        
        # calculate displacement coefficient (0 < alpha_d < 1)
        # alpha_d = (F.relu(-sdf) - F.relu(-self.prev_sdf)) / (self.prev_sdf - sdf + 1e-8)
        # displacement = alpha_d.unsqueeze(-1) * (self.prev_indenter_pts_in_elastomer - indenter_pts_in_elastomer) # (num_envs, num_pts, 3)
        alpha_d = -torch.divide(F.relu(-self.prev_sdf) - F.relu(-sdf), self.prev_sdf - sdf)
        alpha_isnan = torch.isnan(alpha_d)
        alpha_d[alpha_isnan] = -0.5 * (torch.sign(sdf[alpha_isnan]) - 1)
        displacement = alpha_d.unsqueeze(-1) * (self.prev_indenter_pts_in_elastomer - indenter_pts_in_elastomer)


        normal_axis = self.elastomer_sdf_sensor.get_normal_axis()
        tangent_axes = self.elastomer_sdf_sensor.get_tangent_axes()
        
        fn = self.hydrosoft_forces[:, :, normal_axis] + self.E * self.A * displacement[:, :, normal_axis]
        ft = self.hydrosoft_forces[:, :, tangent_axes] + (self.K * self.A * displacement[:, :, tangent_axes])
        
        # print("SDF:", sdf.min(), sdf.max())
        # print("fn:", fn.min(), fn.max())
        
        fn_bar = F.relu(fn * (-1 if reverse_z else 1))
        norm_ft = torch.norm(ft, dim=-1)
        # ft_bar = torch.minimum(self.mu * fn_bar / (norm_ft + 1e-8), torch.ones_like(norm_ft)).unsqueeze(-1) * ft
        ft_bar = torch.minimum(self.mu * fn_bar, norm_ft).unsqueeze(-1) * ft / (norm_ft.unsqueeze(-1) + 1e-8)
        
        # print("fn_bar:", fn_bar.min(), fn_bar.max())
        
        # og: (ftx, fn, fty)
        # (ftx, fty, fn)
        # (0,2,1)
        
        # og: (fn, ftx, fty)
        # (ftx, fty, fn)
        # (1,2,0)
        
        # fn = 1
        # ft = 0,2
        fbar = torch.cat([ft_bar, fn_bar.unsqueeze(-1) * (-1 if reverse_z else 1)], dim=-1)

        # index_axes = [tangent_axes[0], tangent_axes[1], normal_axis]
        # original_fbar = fbar.clone() 
        # # fbar = fbar[:, :, index_axes] = fbar[:, :, [0,1,2]]
        # fbar[:, :, index_axes[0]] = original_fbar[:, :, 0]
        # fbar[:, :, index_axes[1]] = original_fbar[:, :, 1]
        # fbar[:, :, index_axes[2]] = original_fbar[:, :, 2]
        
        fbar = torch.heaviside(-sdf, values=torch.tensor([0.0], device=self.device)).unsqueeze(-1) * fbar # zero out forces if not in contact
        
        return fbar
    
    def get_marker_shear(self, tactile_pts_in_elastomer, indenter_pts_in_elastomer, aug_indenter_pts_in_elastomer=None, reverse_z=True):
        sdf, _ = self.elastomer_sdf_sensor.get_sdf(indenter_pts_in_elastomer) # (num_envs, num_pts)
        fbar = self.step_hydrosoft_forces(sdf, indenter_pts_in_elastomer) # (num_envs, num_indenter_pts, 3)
        self.hydrosoft_forces = fbar
        self.prev_indenter_pts_in_elastomer = indenter_pts_in_elastomer
        self.prev_sdf = sdf
        
        if aug_indenter_pts_in_elastomer is not None:
            sdf, _ = self.elastomer_sdf_sensor.get_sdf(aug_indenter_pts_in_elastomer) # (num_envs, num_pts)
            fbar = self.step_hydrosoft_forces(sdf, aug_indenter_pts_in_elastomer)
            self.aug_hydrosoft_forces = fbar
        
        normal_axis = self.elastomer_sdf_sensor.get_normal_axis()

        # Fast path: factor the shear sum into a batched matmul. Mathematically
        # identical to the dense sum below (see fft_shear.py) for any point
        # layout, but never materializes the (E, P, N, 3) tensor -> faster and
        # far lighter on memory (P >> N).
        if self.use_matmul_shear:
            from .fft_shear import shear_matmul
            return shear_matmul(
                tactile_pts_in_elastomer, indenter_pts_in_elastomer, fbar,
                self.lambda_s, self.shear_scale,
                normal_axis=normal_axis, reverse_z=reverse_z,
            )

        affected_marker_positions = indenter_pts_in_elastomer + fbar # get markers affected during motion (num_envs, num_pts, 3)

        indenter_pts_height = fbar[:, :, normal_axis]

        num_envs, num_indenter_pts, _ = indenter_pts_in_elastomer.shape
        num_tactile_pts = tactile_pts_in_elastomer.shape[1]

        tactile_pts_in_elastomer = tactile_pts_in_elastomer.unsqueeze(1).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        affected_marker_positions = affected_marker_positions.unsqueeze(2).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        fbar = fbar.unsqueeze(2).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        norm_elastomer2hydrosoft_pts = torch.norm(tactile_pts_in_elastomer - affected_marker_positions, dim=-1) # (num_envs, num_indenter_pts, num_tactile_pts)
        h = indenter_pts_height.unsqueeze(-1).unsqueeze(-1).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
    
        gaussian_exp = torch.exp(-self.lambda_s.unsqueeze(-1) * norm_elastomer2hydrosoft_pts**2).unsqueeze(-1).expand(num_envs, num_indenter_pts, num_tactile_pts, 3)
        Mshear = torch.sum(self.shear_scale.unsqueeze(-1).unsqueeze(-1) * h * -1 * fbar * gaussian_exp * (-1 if reverse_z else 1), dim=1) # (num_envs, num_tactile_pts, 3)
        return Mshear
        
    def enable_fft_dilation(self, tactile_pts_in_elastomer, grid_shape, atol_planar=1e-4):
        '''
        Precompute the FFT dilation operator for a regular planar taxel grid.

        Args:
            tactile_pts_in_elastomer: (num_tactile_pts, 3) or (num_envs, num_tactile_pts, 3)
                tactile points in the elastomer frame, laid out row-major as
                (r * grid_shape[1] + c), matching generate_tactile_points().
            grid_shape: (H, W) = (num_divs[1], num_divs[0]) = (rows, cols).
            atol_planar: max normal-axis deviation (m) tolerated before warning
                that the grid is non-planar (FFT becomes an approximation).

        The taxel spacing and tangent/normal axes are inferred from the points.
        '''
        from .fft_dilation import DilationFFT, infer_grid_geometry

        pts = tactile_pts_in_elastomer
        if pts.dim() == 3:
            pts = pts[0]
        pts = pts.detach()
        info = infer_grid_geometry(pts, grid_shape, atol_planar=atol_planar)
        if not info['is_planar']:
            cprint(f"[HydroFOTS] WARNING: taxel grid not planar (normal-axis spread "
                   f"{info['planar_residual']:.2e} m > {atol_planar:.1e} m); FFT dilation "
                   f"is an approximation of the dense sum.", "yellow")
        self._fft_dilation = DilationFFT(
            grid_shape, info['spacing'], tangent_axes=info['tangent_axes'],
            device=self.device, dtype=pts.dtype,
        )
        self._fft_grid_shape = tuple(grid_shape)
        cprint(f"[HydroFOTS] FFT dilation enabled: grid={tuple(grid_shape)}, "
               f"spacing={info['spacing']}, tangent_axes={info['tangent_axes']}, "
               f"planar_residual={info['planar_residual']:.2e} m", "green")
        return info

    def get_marker_dilation(self, tactile_pts_in_elastomer, tactile_pts_in_indenter):
        # tactile points queried on indenter sdf
        tactile_pts_sdf,_ = self.indenter_sdf_sensor.get_sdf(tactile_pts_in_indenter) # (num_envs, num_tactile_pts)
        tactile_pts_height = F.relu(-tactile_pts_sdf) # (num_envs, num_tactile_pts)

        num_envs, num_tactile_pts = tactile_pts_height.shape

        # Fast path: 2D-FFT convolution on a regular planar grid. Mathematically
        # identical to the dense sum below (see fft_dilation.py). Only used when
        # explicitly enabled and the grid is large enough for FFT to pay off.
        if (self.use_fft_dilation and self._fft_dilation is not None
                and num_tactile_pts >= self.fft_dilation_min_pts):
            return self._fft_dilation(tactile_pts_height, self.lambda_d, self.dilate_scale)

        dvec = tactile_pts_in_elastomer.unsqueeze(2) - tactile_pts_in_elastomer.unsqueeze(1) # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1)**2 # (num_envs, num_tactile_pts, num_tactile_pts)

        gaussian_exp = torch.exp(-self.lambda_d.unsqueeze(-1) * norm_dvec).unsqueeze(-1).expand(num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3)
        Mdilate = (self.dilate_scale.unsqueeze(-1).unsqueeze(-1) * h * dvec * gaussian_exp).sum(dim=2) # (num_envs, num_tactile_pts, 3)

        return Mdilate
    
    def get_marker_displacement(self, tactile_pts_in_indenter, tactile_pts_in_elastomer, indenter_pts_in_elastomer, aug_indenter_pts_in_elastomer=None):
        # with cProfile.Profile() as profile:
        Mdilate = self.get_marker_dilation(tactile_pts_in_elastomer, tactile_pts_in_indenter)
        # stats = pstats.Stats(profile)
        # stats.sort_stats(pstats.SortKey.CUMULATIVE)
        # stats.print_stats()
        
        # with cProfile.Profile() as profile:
        Mshear  = self.get_marker_shear(tactile_pts_in_elastomer, indenter_pts_in_elastomer, aug_indenter_pts_in_elastomer=aug_indenter_pts_in_elastomer)
        # stats = pstats.Stats(profile)
        # stats.sort_stats(pstats.SortKey.CUMULATIVE)
        # stats.print_stats()
        # Mdilate = torch.zeros_like(Mshear)
        
        return Mdilate + Mshear
    
from typing import List
class HydroFotsFieldSensor:
    def __init__(self, hydrofots_sensors: List[HydroFotsSensor], elastomer_link_names: List[str], elastomer_actor_names: List[str], visualize=False):
        assert len(hydrofots_sensors) == len(elastomer_link_names) == len(elastomer_actor_names), "Length of hydrofots_sensors, elastomer_link_names, and elastomer_actor_names must be the same"
        
        self.hydrofots_sensors = hydrofots_sensors
        self.elastomer_link_names = elastomer_link_names
        self.elastomer_actor_names = elastomer_actor_names
        
        self.num_divs = [7, 9]
        self.num_tactile_pts = self.num_divs[0] * self.num_divs[1]
        # self.local_indenter_pts, _ = self.hydrofots_sensors[0].indenter_sdf_sensor.sample_points(self.num_indenter_pts) # sample indenter points once and reuse for all sensors
        self.local_indenter_pts = self.hydrofots_sensors[0].indenter_sdf_sensor.poisson_sample_points(radius=0.00075, initial_num_points=int(1e6))[0].cpu().numpy()
        self.num_indenter_pts = self.local_indenter_pts.shape[0]
        self.local_elastomer_pts = [
            torch.tensor(sensor.elastomer_sdf_sensor.generate_tactile_points(margin=0.003, local_z_dir=-1, num_divs=self.num_divs)).to(self.device).to(torch.float32)
            for sensor in self.hydrofots_sensors
        ]

        # Precompute FFT dilation operators for sensors that opted in. grid_shape
        # is (rows, cols) = (num_divs[1], num_divs[0]), matching the row-major
        # (r*W + c) layout produced by generate_tactile_points(). Gated at call
        # time by fft_dilation_min_pts, so this is a no-op below that threshold.
        grid_shape = (self.num_divs[1], self.num_divs[0])
        for sensor, pts in zip(self.hydrofots_sensors, self.local_elastomer_pts):
            if getattr(sensor, 'use_fft_dilation', False):
                sensor.enable_fft_dilation(pts, grid_shape)

        self.visualize = visualize
        if visualize:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            
            self.vis = o3d.visualization.Visualizer()
            
            self.left_elastomer_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.asarray(hydrofots_sensors[0].elastomer_sdf_sensor.mesh_pv._mesh.vertices)),
                triangles=o3d.utility.Vector3iVector(np.asarray(hydrofots_sensors[0].elastomer_sdf_sensor.mesh_pv._mesh.triangles))
            )
            
            self.right_elastomer_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.asarray(hydrofots_sensors[1].elastomer_sdf_sensor.mesh_pv._mesh.vertices)),
                triangles=o3d.utility.Vector3iVector(np.asarray(hydrofots_sensors[1].elastomer_sdf_sensor.mesh_pv._mesh.triangles))
            )
                    
            self.obj_pcd = o3d.geometry.PointCloud()
            self.obj_pcd.points = o3d.utility.Vector3dVector(self.local_indenter_pts)
            self.obj_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.create_window(window_name='both', width=1600, height=800)
            self.arrows = []
            
            self.left_elastomer2indenter = None
            self.right_elastomer2indenter = None
            
            # add elastomer tactile points
            self.left_elastomer_pts = o3d.geometry.PointCloud()
            self.left_elastomer_pts.points = o3d.utility.Vector3dVector(self.local_elastomer_pts[0].cpu().numpy())
            self.left_elastomer_pts.paint_uniform_color([0, 1, 0])
            self.vis.add_geometry(self.left_elastomer_pts)
            
            self.right_elastomer_pts = o3d.geometry.PointCloud()
            self.right_elastomer_pts.points = o3d.utility.Vector3dVector(self.local_elastomer_pts[1].cpu().numpy())
            self.right_elastomer_pts.paint_uniform_color([0, 1, 0])
            self.vis.add_geometry(self.right_elastomer_pts)
            
            
            # self.vis.add_geometry(origin)
            self.vis.add_geometry(self.obj_pcd)
            # self.vis.add_geometry(self.left_elastomer_mesh)
            # self.vis.add_geometry(self.right_elastomer_mesh)
            
        
    def reset_hydrofots(self):
        randomization_dict = None
        for sensor in self.hydrofots_sensors:
            randomization_dict = sensor.initialize(self.num_envs, self.num_indenter_pts, randomization_dict)
        if self.visualize:
            self.left_elastomer2indenter = None
            self.right_elastomer2indenter = None
        
    def get_link_handle(self, actor_name, link_name):
        link_handle = self.gym.find_actor_rigid_body_handle(
            self.env_ptrs[0], self.actor_handles[actor_name], link_name)
        return link_handle
    
    def get_indenter2elastomer_tf(self, indenter_quat, indenter_pos):
        indenter2elastomer_tf_list = []
        elastomer2indenter_tf_list = []
        
        for i in range(len(self.hydrofots_sensors)):
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
    
    def get_force_fields_dict(self, indenter_quat, indenter_pos, aug_indenter_quat = None, aug_indenter_pos = None):
        indenter2elastomer_tf_list, elastomer2indenter_tf_list = self.get_indenter2elastomer_tf(indenter_quat, indenter_pos)
        
        aug_indenter_pts_in_elastomer = None
        if aug_indenter_quat is not None and aug_indenter_pos is not None:
            aug_indenter2elastomer_tf_list, _ = self.get_indenter2elastomer_tf(aug_indenter_quat, aug_indenter_pos)
        
        marker_displacements_dict = {}
        for i, sensor in enumerate(self.hydrofots_sensors):
            elastomer_pts_in_elastomer = torch.tensor(self.local_elastomer_pts[i], device=self.device).unsqueeze(0).expand(self.num_envs, self.num_tactile_pts, 3) # (num_envs, num_elastomer_pts, 3)
            
            elastomer2indenter_quat, elastomer2indenter_pos = elastomer2indenter_tf_list[i]
            elastomer2indenter_quat = elastomer2indenter_quat.unsqueeze(1).expand(self.num_envs, self.num_tactile_pts, 4)
            elastomer2indenter_pos = elastomer2indenter_pos.unsqueeze(1).expand(self.num_envs, self.num_tactile_pts, 3)
            elastomer_pts_in_indenter = tf_apply(elastomer2indenter_quat, elastomer2indenter_pos, elastomer_pts_in_elastomer) # (num_envs, num_elastomer_pts, 3)
            
            indenter_pts_in_indenter = torch.tensor(self.local_indenter_pts, device=self.device).unsqueeze(0).expand(self.num_envs, self.num_indenter_pts, 3) # (num_envs, num_indenter_pts, 3)
            indenter2elastomer_quat, indenter2elastomer_pos = indenter2elastomer_tf_list[i]
            indenter2elastomer_quat = indenter2elastomer_quat.unsqueeze(1).expand(self.num_envs, self.num_indenter_pts, 4)
            indenter2elastomer_pos = indenter2elastomer_pos.unsqueeze(1).expand(self.num_envs, self.num_indenter_pts, 3)
            indenter_pts_in_elastomer = tf_apply(indenter2elastomer_quat, indenter2elastomer_pos, indenter_pts_in_indenter) # (num_envs, num_indenter_pts, 3)
            
            if aug_indenter_quat is not None and aug_indenter_pos is not None:
                aug_indenter_pts_in_indenter = torch.tensor(self.local_indenter_pts, device=self.device).unsqueeze(0).expand(self.num_envs, self.num_indenter_pts, 3) # (num_envs, num_indenter_pts, 3)
                aug_indenter2elastomer_quat, aug_indenter2elastomer_pos = aug_indenter2elastomer_tf_list[i]
                aug_indenter2elastomer_quat = aug_indenter2elastomer_quat.unsqueeze(1).expand(self.num_envs, self.num_indenter_pts, 4)
                aug_indenter2elastomer_pos = aug_indenter2elastomer_pos.unsqueeze(1).expand(self.num_envs, self.num_indenter_pts, 3)
                aug_indenter_pts_in_elastomer = tf_apply(aug_indenter2elastomer_quat, aug_indenter2elastomer_pos, aug_indenter_pts_in_indenter) # (num_envs, num_indenter_pts, 3)
            
                # before getting displacement check out indenter points and elastomer mesh
                # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                # indenter_pcd = o3d.geometry.PointCloud()
                # indenter_pcd.points = o3d.utility.Vector3dVector(indenter_pts_in_elastomer[0].cpu().numpy())
                # indenter_pcd.paint_uniform_color([1, 0, 0])
                # elastomer_pcd = o3d.geometry.PointCloud()
                # elastomer_pcd.points = o3d.utility.Vector3dVector(elastomer_pts_in_elastomer[0].cpu().numpy())
                # elastomer_pcd.paint_uniform_color([0, 1, 0])
                # aug_indenter_pcd = o3d.geometry.PointCloud()
                # aug_indenter_pcd.points = o3d.utility.Vector3dVector(aug_indenter_pts_in_elastomer[0].cpu().numpy()) 
                # aug_indenter_pcd.paint_uniform_color([0, 0, 1])
                # o3d.visualization.draw_geometries([origin, indenter_pcd, aug_indenter_pcd, elastomer_pcd, sensor.elastomer_sdf_sensor.mesh_pv._mesh])
            
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            # elastomer_pcd = o3d.geometry.PointCloud()
            # elastomer_pcd.points = o3d.utility.Vector3dVector(elastomer_pts_in_indenter[0].cpu().numpy())
            # o3d.visualization.draw_geometries([origin, elastomer_pcd, sensor.indenter_sdf_sensor.mesh_pv._mesh])
            
            marker_displacement = sensor.get_marker_displacement(elastomer_pts_in_indenter, elastomer_pts_in_elastomer, indenter_pts_in_elastomer, aug_indenter_pts_in_elastomer=aug_indenter_pts_in_elastomer)
            
            marker_displacement = marker_displacement.reshape(self.num_envs, self.num_divs[1], self.num_divs[0], 3)
            
            marker_displacements_dict[self.elastomer_link_names[i]] = marker_displacement
        
        if self.visualize:
            local_indenter_pts = self.local_indenter_pts
            self.obj_pcd.points = o3d.utility.Vector3dVector(local_indenter_pts)
            
            left_elastomer2indenter_quat, left_elastomer2indenter_pos = elastomer2indenter_tf_list[0]
            left_elastomer2indenter_quat = left_elastomer2indenter_quat[0]
            left_elastomer2indenter_pos = left_elastomer2indenter_pos[0]
            
            right_elastomer2indenter_quat, right_elastomer2indenter_pos = elastomer2indenter_tf_list[1]
            right_elastomer2indenter_quat = right_elastomer2indenter_quat[0]
            right_elastomer2indenter_pos = right_elastomer2indenter_pos[0]
            
            # transform meshes
            left_elastomer_mesh_vertices = np.asarray(self.hydrofots_sensors[0].elastomer_sdf_sensor.mesh_pv._mesh.vertices).copy()
            transformed_left_vertices = tf_apply(
                left_elastomer2indenter_quat.unsqueeze(0).expand(left_elastomer_mesh_vertices.shape[0],4),
                left_elastomer2indenter_pos.unsqueeze(0).expand(left_elastomer_mesh_vertices.shape[0],3),
                torch.tensor(left_elastomer_mesh_vertices, device=self.device, dtype=torch.float32)
            )
            self.left_elastomer_mesh.vertices = o3d.utility.Vector3dVector(transformed_left_vertices.cpu().numpy())
            
            right_elastomer_mesh_vertices = np.asarray(self.hydrofots_sensors[1].elastomer_sdf_sensor.mesh_pv._mesh.vertices).copy()
            transformed_right_vertices = tf_apply(
                right_elastomer2indenter_quat.unsqueeze(0).expand(right_elastomer_mesh_vertices.shape[0],4),
                right_elastomer2indenter_pos.unsqueeze(0).expand(right_elastomer_mesh_vertices.shape[0],3),
                torch.tensor(right_elastomer_mesh_vertices, device=self.device, dtype=torch.float32)
            )
            self.right_elastomer_mesh.vertices = o3d.utility.Vector3dVector(transformed_right_vertices.cpu().numpy())
            
            # self.vis.update_geometry(self.left_elastomer_mesh)
            # self.vis.update_geometry(self.right_elastomer_mesh)
                
            
            
            # visualize elastomer pts
            left_elastomer_pts_in_indenter = tf_apply(
                left_elastomer2indenter_quat.unsqueeze(0).expand(self.local_elastomer_pts[0].shape[0],4),
                left_elastomer2indenter_pos.unsqueeze(0).expand(self.local_elastomer_pts[0].shape[0],3),
                self.local_elastomer_pts[0]
            ).cpu().numpy()
            self.left_elastomer_pts.points = o3d.utility.Vector3dVector(left_elastomer_pts_in_indenter)
            right_elastomer_pts_in_indenter = tf_apply(
                right_elastomer2indenter_quat.unsqueeze(0).expand(self.local_elastomer_pts[1].shape[0],4),
                right_elastomer2indenter_pos.unsqueeze(0).expand(self.local_elastomer_pts[1].shape[0],3),
                self.local_elastomer_pts[1]
            ).cpu().numpy()
            self.right_elastomer_pts.points = o3d.utility.Vector3dVector(right_elastomer_pts_in_indenter)
            self.vis.update_geometry(self.left_elastomer_pts)
            self.vis.update_geometry(self.right_elastomer_pts)
            
            
            if len(self.arrows) > 0:
                for arrow in self.arrows:
                    self.vis.remove_geometry(arrow, reset_bounding_box=False)
                self.arrows = []
            
            np_fbar_left = self.hydrofots_sensors[0].aug_hydrosoft_forces[0].cpu().numpy().reshape(-1, 3)
            np_fbar_right = self.hydrofots_sensors[1].aug_hydrosoft_forces[0].cpu().numpy().reshape(-1, 3)
            np_fbar = np.concatenate([np_fbar_left, np_fbar_right], axis=0)
            
            left_matrix = tf_apply(
                left_elastomer2indenter_quat.unsqueeze(0).expand(3,4),
                torch.zeros((3,3), device=self.device),
                torch.eye(3, device=self.device)
            ).cpu().numpy()
            right_matrix = tf_apply(
                right_elastomer2indenter_quat.unsqueeze(0).expand(3,4),
                torch.zeros((3,3), device=self.device),
                torch.eye(3, device=self.device)
            ).cpu().numpy()
            
            # print("Fnormal min max")
            # print(np_fbar_left[:,2].min(), np_fbar_left[:,2].max())
            # print(np_fbar_right[:,2].min(), np_fbar_right[:,2].max())
            # print()
            
            for i in range(np_fbar.shape[0]):
                if np.linalg.norm(np_fbar[i]) < 1e-7:
                    continue
                rot_matrix = left_matrix[:3,:3].T if i < np_fbar_left.shape[0] else right_matrix[:3,:3].T
                force = rot_matrix @ np_fbar[i]
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=1e-5 * 10,
                    cone_radius=1.5 * 1e-5 * 20,
                    cylinder_height=(np.linalg.norm(force) + 1e-8),  # scale height by magnitude
                    cone_height=(2e-4),  # scale height by magnitude
                )
                arrow.paint_uniform_color([1, 0, 0] if i < np_fbar_left.shape[0] else [0, 0, 1])
                arrow.translate(local_indenter_pts[i % np_fbar_left.shape[0]])
                
                arrow_origin = local_indenter_pts[i % np_fbar_left.shape[0]]
                shear_direction = force / np.linalg.norm(force)
                
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
                
                self.vis.add_geometry(arrow, reset_bounding_box=False)
                self.arrows.append(arrow)         
            self.vis.update_geometry(self.obj_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
                        
            
        return marker_displacements_dict