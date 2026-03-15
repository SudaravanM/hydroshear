import os
import numpy as np
import torch

import open3d as o3d
import pytorch_volumetric as pv

# from hydroshear.utils.torch_utils import tf_combine, tf_inverse, tf_apply
# import torch.nn.functional as F
import trimesh
import point_cloud_utils as pcu

class MeshSDFSensor:
    '''
    Class for calculating the signed distance field (SDF) of a mesh and querying points on the mesh.
    Uses PyTorch Volumetric for SDF calculation.
    '''
    def __init__(self, mesh_path, device='cpu'):
        self.device = device
        self.mesh_path = mesh_path
        self.mesh_pv = pv.MeshObjectFactory(self.mesh_path)

        self.mesh_pv._mesh.compute_vertex_normals()
        # self.mesh = urdf.link_map[link_name].visuals[0].geometry.mesh.meshes[0]
        
        self.mesh_pv.precompute_sdf()
        self.mesh_sdf = pv.MeshSDF(self.mesh_pv)
        
    
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