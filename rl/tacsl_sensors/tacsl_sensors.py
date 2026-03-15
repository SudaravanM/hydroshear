'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Tactile Sensing Modules
----------------
Implementation of visuo-tactile sensing module to generate tactile RGB image or tactile force field.
'''

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import tf_combine, tf_inverse
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul
from isaacgym import torch_utils as tu

from rl.tacsl_sensors.tactile_utils.gelsight_render import gelsightRender

import os
import itertools
import numpy as np
import torch
import trimesh
from urdfpy import URDF
import yaml

from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from collections.abc import Iterable
# from torch_cluster.fps import fps
import skimage
import cv2
import open3d as o3d

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :-1], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., -1:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :-1] / sin_half_angles_over_angles

def deproject_rgbd_image(depth, projection_matrix, camera_view_matrix, tool_segmentation, device='cpu'):
    projection_matrix = projection_matrix.to(device)  # (4, 4)
    camera_view_matrix = camera_view_matrix.to(device)  # (4, 4)
    depth = depth.to(device)  # (height, width)
    tool_segmentation = tool_segmentation.to(device)  # (height, width)

    vinv = camera_view_matrix.T.to(device)
    vinv[:3, 3] = -vinv[:3, :3].T @ vinv[:3, 3]  # invert translation part
    vinv[:3,:3] = vinv[:3,:3].T
    
    fu = 2.0 / projection_matrix[0, 0]
    fv = 2.0 / projection_matrix[1, 1]

    width = depth.shape[1]
    centerU = width / 2
    height = depth.shape[0]
    centerV = height / 2

    xx, yy = torch.meshgrid(torch.arange(width, device=depth.device), torch.arange(height, device=device), indexing='xy')    
    xx = xx[tool_segmentation] # (height, width) -> (num_points,)
    yy = yy[tool_segmentation] # (height, width) -> (num_points,)
    ones = torch.ones_like(xx, device=device, dtype=torch.float32)
    zz = depth[tool_segmentation] # (height, width) -> (num_points,)
    pts3d = torch.stack((xx.float(), yy.float(), ones), dim=-1) # (num_points, 3)
    pts3d[:,0] = -(pts3d[:,0] - centerU) / width * fu * zz
    pts3d[:,1] = (pts3d[:,1] - centerV) / height * fv * zz
    pts3d[:,2] = zz
    pts3d = pts3d @ vinv[:3, :3].T  + vinv[:3, 3].unsqueeze(0)  # (num_points, 3)
    
    return pts3d[:, :3]


def get_camera_config(sensor_type, tactile_camera_name, actor_name, attach_link_name):
    """
    Get the camera configuration based on the sensor type.

    Args:
        sensor_type (str): Type of the sensor (e.g., 'gelsight_r15', 'gs_mini').
        tactile_camera_name (str): Name of the tactile camera.
        actor_name (str): Name of the actor on which the tactile sensor is attached.
        attach_link_name (str): Name of the link to which the camera is attached.

    Returns:
        dict: Tactile camera configuration.
    """
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if sensor_type == 'gelsight_r15':
        config_filepath = os.path.join(parent_dir, 'configs', 'gelsight_r15.yaml')
    elif sensor_type == 'gs_mini':
        config_filepath = os.path.join(parent_dir, 'configs', 'gs_mini.yaml')
    else:
        raise NotImplementedError(f'Sensor type {sensor_type} not recognized')

    with open(config_filepath) as stream:
        tactile_camera_config = yaml.safe_load(stream)
    tip_to_cam_pos = [0.0, 0.0, -tactile_camera_config['camera_dist']]
    tip_to_cam_quat = R.from_euler('xyz', tactile_camera_config['tip_to_cam_euler']).as_quat()
    tactile_camera_config['camera_pose'] = [tip_to_cam_pos, tip_to_cam_quat.tolist()]
    tactile_camera_config['image_size'] = [tactile_camera_config['height'], tactile_camera_config['width']]

    fx, fy = tactile_camera_config['camera_dist'], tactile_camera_config['camera_dist']
    fov_x = 2 * np.arctan(tactile_camera_config['cx'] / fx) * 180 / np.pi
    fov_y = 2 * np.arctan(tactile_camera_config['cy'] / fy) * 180 / np.pi
    tactile_camera_config['horizontal_fov'] = float(fov_x)

    tactile_camera_config['name'] = tactile_camera_name
    tactile_camera_config['actor_name'] = actor_name
    tactile_camera_config['attach_link_name'] = attach_link_name
    return tactile_camera_config


class TactileBase:
    """
    Base class for tactile sensors.
    Provides methods for configuring compliant dynamics.
    """
    def configure_compliant_dynamics(self, actor_handle, elastomer_link_name,
                                     compliance_stiffness, compliant_damping, use_acceleration_spring=False):
        """
        Configure the compliant dynamics for a given actor and link.

        Args:
            actor_handle: Handle for the actor.
            elastomer_link_name (str): Name of the elastomer link.
            compliance_stiffness (float or list): Compliance stiffness value(s).
            compliant_damping (float or list): Compliant damping value(s).
            use_acceleration_spring (bool): Whether to use acceleration spring.
        """
        if not isinstance(compliance_stiffness, Iterable):
            compliance_stiffness = [compliance_stiffness] * len(self.env_ptrs)
        if not isinstance(compliant_damping, Iterable):
            compliant_damping = [compliant_damping] * len(self.env_ptrs)

        for env_id, env_ptr in enumerate(self.env_ptrs):
            body_name_idx_map = self.gym.get_actor_rigid_body_dict(env_ptr, actor_handle)
            body_names = self.gym.get_actor_rigid_body_names(env_ptr, actor_handle)
            elastomer_body_id = body_name_idx_map[elastomer_link_name]
            assert body_names[elastomer_body_id] == elastomer_link_name, 'order of rigid body does not agree'
            rs_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
            rb_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, actor_handle)
            elastomer_shape_id = rb_shape_indices[elastomer_body_id].start

            if "friction_elastomer" in self.cfg_task.env:
                rs_props[elastomer_shape_id].friction = self.cfg_task.env.friction_elastomer

            if self.cfg_task.randomize.get('randomize_friction', False):
                friction_range = self.cfg_task.randomize.get('elastomer_friction_factor_range', [0.5, 1.0])
                friction = friction_range[0] + torch.rand(self.num_envs, device=self.device) * (friction_range[1] - friction_range[0])
                rs_props[elastomer_shape_id].friction = friction[env_id]

            rs_props[elastomer_shape_id].compliance = compliance_stiffness[env_id]
            rs_props[elastomer_shape_id].compliant_damping = compliant_damping[env_id]
            rs_props[elastomer_shape_id].use_acceleration_spring = use_acceleration_spring
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, rs_props)

    def set_compliant_dynamics_for_tactile_sensors(self, tactile_sensor_configs):
        """
        Set the compliant dynamics for multiple tactile sensors.

        Args:
            tactile_sensor_configs (list): List of tactile sensor configurations.
        """
        for tactile_sensor_config in tactile_sensor_configs:
            self.configure_compliant_dynamics(actor_handle=tactile_sensor_config['actor_handle'],
                                              elastomer_link_name=tactile_sensor_config['elastomer_link_name'],
                                              compliance_stiffness=tactile_sensor_config['compliance_stiffness'],
                                              compliant_damping=tactile_sensor_config['compliant_damping'],
                                              use_acceleration_spring=tactile_sensor_config['use_acceleration_spring'])


class CameraSensor:
    """
    Class for managing camera sensors.
    Provides methods for creating and managing camera actors and tensors.
    """
    def create_camera_actors(self, camera_spec_dict):
        """
        Create camera actors based on the camera specification dictionary.
        # Note: This should be called once, as IsaacGym's global camera indexing expects all cameras of env 0 be created before env 1, and so on.

        Args:
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            tuple: List of camera handles and list of camera tensors.
        """
        camera_handles_list = []
        camera_tensors_list = []

        for i in range(self.num_envs):
            env_ptr = self.env_ptrs[i]
            env_camera_handles = self.setup_env_cameras(env_ptr, camera_spec_dict)
            camera_handles_list.append(env_camera_handles)

            env_camera_tensors = self.create_tensors_for_env_cameras(env_ptr, env_camera_handles, camera_spec_dict)
            camera_tensors_list.append(env_camera_tensors)
        return camera_handles_list, camera_tensors_list
    def get_camera_parameters(self, env_camera_handles, device='cpu'):
        '''
        Args:
            env_camera_handles (list of dict): list of dictionaries of camera handles for the environment.
        '''
        camera_projection_dict = {}
        camera_view_dict = {}
        
        cam_names = list(env_camera_handles[0].keys())
        for cam_name in cam_names:
            camera_projection_dict[cam_name] = np.zeros((self.num_envs, 4, 4), dtype=np.float32)
            camera_view_dict[cam_name] = np.zeros((self.num_envs, 4, 4), dtype=np.float32)
            
        for i in range(self.num_envs):
            env_ptr = self.env_ptrs[i]
            env_camera_handle = env_camera_handles[i]
                
            for cam_name, camera_handle in env_camera_handle.items():
                projection_matrix = np.matrix(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle))
                view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle))
                
                camera_projection_dict[cam_name][i] = projection_matrix
                camera_view_dict[cam_name][i] = view_matrix
        for cam_name in cam_names:
            camera_projection_dict[cam_name] = torch.tensor(camera_projection_dict[cam_name], device=device)
            camera_view_dict[cam_name] = torch.tensor(camera_view_dict[cam_name], device=device)
        return camera_projection_dict, camera_view_dict

    def get_visual_pcd(self, images = None, subsample_num=500):
        if images is None:
            images = self.get_camera_image_tensors_dict()
        # get segmentations done first
        seg_dict = {}
        for cam in images:
            if cam.endswith('_seg'):
                cam_name = cam.replace('_seg', '')
                seg_dict[cam_name] = images[cam]
                seg_dict[cam_name][seg_dict[cam_name] > 0] = 1  # NOTE: asssumes that we only have one segmentation id for binary mask
        # get camera parameters loaded
        camera_projection_dict, camera_view_dict = self.get_camera_parameters(self.camera_handles_list, device=self.device)
        
        # get env point clouds
        obj_pcd = []
        hack_rot = torch.tensor([[-1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]], device=self.device).float()
        for idx_env in range(self.num_envs):
            pcd = []
            for cam in images:
                if cam.endswith('_depth'):
                    cam_name = cam.replace('_depth', '')
                    depth = images[cam]
                    seg = seg_dict[cam_name] > 0 # mask
                    
                    # get params
                    proj_matrix = camera_projection_dict[cam_name] # (env, 4, 4)
                    view_matrix  = camera_view_dict[cam_name] # (env, 4, 4)
                    points = deproject_rgbd_image(depth[idx_env], proj_matrix[idx_env], view_matrix[idx_env] @ hack_rot, seg[idx_env], device=self.device)
                    
                    pcd.append(points)
            pcd = torch.concatenate(pcd, dim=0) # (num_points, 3)
            
            
            pcd = torch.cat([pcd, torch.zeros((subsample_num - pcd.shape[0], 3), device=self.device)], dim=0) if pcd.shape[0] < subsample_num else pcd
            subsample_idx = torch.multinomial(torch.ones(pcd.shape[0], device=device, dtype=pcd.dtype) / pcd.shape[0], num_samples=subsample_num, replacement=False)
            # subsample_idx = fps(pcd, ratio= (subsample_num+10) / pcd.shape[0], random_start=True)[:subsample_num]
            assert subsample_idx.shape[0] == subsample_num, f"Subsampled point cloud has {subsample_idx.shape[0]} points, expected {subsample_num}."
            pcd = pcd[subsample_idx, :]  # (num_points, 3)
            

            
            pcd = pcd.to(self.device)
            obj_pcd.append(pcd)
        
        obj_pcd = torch.stack(obj_pcd, dim=0)  # (num_envs, num_points, 3)
        return obj_pcd
        
    def create_tensors_for_env_cameras(self, env_ptr, env_camera_handles, camera_spec_dict):
        """
        Create tensors for environment cameras.

        Args:
            env_ptr: Pointer to the environment.
            env_camera_handles (dict): Dictionary of camera handles.
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            dict: Dictionary of environment camera tensors.
        """
        env_camera_tensors = {}
        for name in env_camera_handles:
            camera_handle = env_camera_handles[name]
            if camera_spec_dict[name].image_type == 'rgb':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_COLOR)
            elif camera_spec_dict[name].image_type == 'depth':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_DEPTH)
            elif camera_spec_dict[name].image_type == 'all':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_COLOR)
                camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_DEPTH)
                camera_tensor_seg = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_SEGMENTATION)
                torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
                torch_camera_tensor_depth = gymtorch.wrap_tensor(camera_tensor_depth)
                torch_camera_tensor_seg = gymtorch.wrap_tensor(camera_tensor_seg)
                env_camera_tensors[name] = torch_camera_tensor
                env_camera_tensors[name + '_depth'] = torch_camera_tensor_depth
                env_camera_tensors[name + '_seg'] = torch_camera_tensor_seg
                continue
            else:
                raise NotImplementedError(f'Camera type {camera_spec_dict[name].image_type} not supported')

            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)

            # store references to the tensor that gets updated when render_all_camera_sensors
            env_camera_tensors[name] = torch_camera_tensor
        return env_camera_tensors

    def setup_env_cameras(self, env_ptr, camera_spec_dict):
        """
        Set up environment cameras.

        Args:
            env_ptr: Pointer to the environment.
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            dict: Dictionary of camera handles.
        """
        camera_handles = {}
        for name, camera_spec in camera_spec_dict.items():
            camera_properties = gymapi.CameraProperties()
            camera_properties.height = camera_spec.image_size[0]
            camera_properties.width = camera_spec.image_size[1]
            camera_properties.enable_tensors = True
            camera_properties.horizontal_fov = camera_spec.horizontal_fov
            if 'near_plane' in camera_spec:
                camera_properties.near_plane = camera_spec.near_plane

            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
            camera_handles[name] = camera_handle

            if camera_spec.is_body_camera:
                actor_handle = self.gym.find_actor_handle(env_ptr, camera_spec.actor_name)
                robot_body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle,
                                                                          camera_spec.attach_link_name)

                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_ptr,
                    robot_body_handle,
                    gymapi.Transform(gymapi.Vec3(*camera_spec.camera_pose[0]),
                                     gymapi.Quat(*camera_spec.camera_pose[1])),
                    gymapi.FOLLOW_TRANSFORM,
                )
            else:
                transform = gymapi.Transform(gymapi.Vec3(*camera_spec.camera_pose[0]),
                                             gymapi.Quat(*camera_spec.camera_pose[1]))
                self.gym.set_camera_transform(camera_handle, env_ptr, transform)
        return camera_handles

    def get_camera_image_tensors_dict(self):
        """
        Get the dictionary of camera image tensors.

        Returns:
            dict: Dictionary of camera image tensors.
        """
        # transforms and information must be communicated from the physics simulation into the graphics system
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        camera_image_tensors_dict = dict()

        for name in self.camera_spec_dict:
            camera_spec = self.camera_spec_dict[name]
            if camera_spec['image_type'] == 'rgb':
                num_channels = 3
                camera_images = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1], num_channels),
                    device=self.device, dtype=torch.uint8)
                for id in np.arange(self.num_envs):
                    camera_images[id] = self.camera_tensors_list[id][name][:, :, :num_channels].clone()
            elif camera_spec['image_type'] == 'depth':
                num_channels = 1
                camera_images = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                    device=self.device, dtype=torch.float)
                for id in np.arange(self.num_envs):
                    # Note that isaac gym returns negative depth
                    # See: https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/graphics.html?highlight=image_depth#camera-image-types
                    camera_images[id] = self.camera_tensors_list[id][name][:, :].clone() * -1.
                    camera_images[id][camera_images[id] == np.inf] = 0.0
            elif camera_spec['image_type'] == 'all':
                
                ## RGB, Depth, Seg save
                num_channels_rgb = 3
                camera_images = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1], num_channels_rgb),
                    device=self.device, dtype=torch.uint8)
                camera_depth  = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                    device=self.device, dtype=torch.float)
                camera_seg    = torch.zeros(
                    (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                    device=self.device, dtype=torch.int32)
                for id in np.arange(self.num_envs):
                    camera_images[id] = self.camera_tensors_list[id][name][:, :, :num_channels_rgb].clone()
                    
                    # Note that isaac gym returns negative depth
                    # See: https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/graphics.html?highlight=image_depth#camera-image-types
                    camera_depth[id] = self.camera_tensors_list[id][name + '_depth'][:, :].clone() * -1.
                    camera_depth[id][camera_depth[id] == np.inf] = 0.0
                    
                    camera_seg[id] = self.camera_tensors_list[id][name + '_seg'].clone()
                camera_image_tensors_dict[name] = camera_images
                camera_image_tensors_dict[name + '_depth'] = camera_depth
                camera_image_tensors_dict[name + '_seg'] = camera_seg
                continue
            else:
                raise NotImplementedError(f'Image type {camera_spec["image_type"]} not supported!')
            camera_image_tensors_dict[name] = camera_images

        return camera_image_tensors_dict


class TactileRGBSensor(TactileBase, CameraSensor):
    """
    Class for simulating tactile RGB sensors.
    Inherits from TactileBase and CameraSensor.
    """
    def __init__(self):
        super().__init__()
        self.taxim_gelsight = None
        self.has_tactile_rgb = False
        self.nominal_tactile = None

    def get_tactile_rgb_camera_configs(self, tactile_sensor_configs):
        """
        Get the tactile RGB camera configurations.

        Args:
            tactile_sensor_configs (list): List of tactile sensor configurations.

        Returns:
            dict: Dictionary of tactile camera specifications.
        """
        camera_spec_dict = dict()

        for tactile_sensor_config in tactile_sensor_configs:
            camera_config = get_camera_config(tactile_sensor_config['sensor_type'],
                                              tactile_sensor_config['tactile_camera_name'],
                                              tactile_sensor_config['actor_name'],
                                              tactile_sensor_config['attach_link_name'])
            camera_config = OmegaConf.create(camera_config)
            tactile_camera_spec_dict = {camera_config['name']: camera_config}
            camera_spec_dict.update(tactile_camera_spec_dict)

        self.taxim_gelsight = gelsightRender(tactile_sensor_configs[0]['sensor_type'], device=self.device)
        self.has_tactile_rgb = True

        return camera_spec_dict

    def initialize_tactile_rgb_camera(self):
        """
        Initialize the tactile RGB camera by capturing a nominal tactile image with no indentation on the elastomer.
        """
        image_dict = self.get_camera_image_tensors_dict()
        self.nominal_tactile = {k: image_dict[k][:1] for k in image_dict.keys() if 'tactile' in k}

    def visualize_tactile_camera_frame(self, tip_to_cam_pos, tip_to_cam_quat):
        gelsight_handle = self.actor_handles['gelsight_finger']
        elastomer_tip_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], gelsight_handle,
                                                                          'elastomer_tip')
        tip_to_cam_pos_tensor = torch.tensor(tip_to_cam_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        tip_to_cam_quat_tensor = torch.tensor(tip_to_cam_quat, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        world_to_camera_trans_new = tf_combine(self.body_quat[:, elastomer_tip_link_handle],
                                               self.body_pos[:, elastomer_tip_link_handle],
                                               tip_to_cam_quat_tensor,
                                               tip_to_cam_pos_tensor)
        for env_id in range(len(self.env_ptrs)):
            self.draw_frame_in_env(env_id, (world_to_camera_trans_new[1][env_id],
                                            world_to_camera_trans_new[0][env_id]))

    def get_camera_image_tensors_dict(self):
        """
        Get the dictionary of camera image tensors, including tactile RGB images.

        Returns:
            dict: Dictionary of camera image tensors.
        """
        camera_image_tensors_dict = super().get_camera_image_tensors_dict()

        # Compute tactile RGB from tactile depth
        if hasattr(self, 'has_tactile_rgb') and self.nominal_tactile:
            for k in self.nominal_tactile:
                depth_image = self.nominal_tactile[k] - camera_image_tensors_dict[k]    # depth_image_delta
                taxim_render_all = self.taxim_gelsight.render_tensorized(depth_image)
                camera_image_tensors_dict[f'{k}_taxim'] = taxim_render_all
        return camera_image_tensors_dict


class TactileFieldSensor(TactileBase):
    """
    Class for simulating tactile field sensors.
    Inherits from TactileBase.
    """
    def __init__(self):
        super().__init__()
        self.tactile_pos_local, self.tactile_quat_local = None, None
        self.sdf_tool = None
        self.tactile_shear_field_configs_dict = None
        self.sdf, self.indenter_mesh, self.indenter_mesh_local_tf = None, None, None

        # initialize coefficients
        self.tactile_kn = 1.
        self.tactile_damping = 0.003
        self.tactile_mu = 2.
        self.tactile_kt = 0.1

    def setup_tactile_force_field(self, sdf_tool, num_tactile_rows, num_tactile_cols, tactile_shear_field_configs):
        """
        Set up the tactile force field sensing.

        Args:
            sdf_tool (str): Tool for signed distance field (SDF) calculation.
            num_tactile_rows (int): Number of rows of tactile points.
            num_tactile_cols (int): Number of columns of tactile points.
            tactile_shear_field_configs (dict): Configuration for the tactile shear field.

        Returns:
            tensor: SDF tensor.
        """
        self.sdf_tool = sdf_tool
        self.tactile_shear_field_configs_dict = self.post_process_shear_field_configs(tactile_shear_field_configs)
        # Set up tactile sdf tensors
        sdf_tensor = None
        if sdf_tool == 'physx':
            # This should be called before calling self.gym.prepare_sim(self.sim)
            num_queries_per_env = num_tactile_rows * num_tactile_cols
            _sdf_tensor = self.gym.acquire_sdf_view_tensor(self.sim, 1, num_queries_per_env)
            sdf_tensor = gymtorch.wrap_tensor(_sdf_tensor)
        return sdf_tensor

    def get_elastomer_to_tip_transform(self, actor_name, elastomer_link_name, elastomer_tip_link_name):
        """
        Get the transformation from the elastomer to the tip.
        Note: This is the same for a given sensor_type so should probably go into a config file

        Args:
            actor_name (str): Name of the actor.
            elastomer_link_name (str): Name of the elastomer link.
            elastomer_tip_link_name (str): Name of the elastomer tip link.

        Returns:
            tuple: Quaternion and position of the transformation.
        """
        elastomer_parent_actor_handle = self.actor_handles[actor_name]
        elastomer_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], elastomer_parent_actor_handle, elastomer_link_name)
        elastomer_tip_link_handle = self.gym.find_actor_rigid_body_handle(self.env_ptrs[0], elastomer_parent_actor_handle,
                                                                          elastomer_tip_link_name)
        assert elastomer_link_handle > -1, 'elastomer_link_handle does not exist'
        assert elastomer_tip_link_handle > -1, 'elastomer_tip_link_handle does not exist'
        elastomer_to_tip_link_trans = tf_combine(*tf_inverse(self.body_quat[:, elastomer_link_handle],
                                                             self.body_pos[:, elastomer_link_handle]),
                                                 self.body_quat[:, elastomer_tip_link_handle],
                                                 self.body_pos[:, elastomer_tip_link_handle])
        elastomer_to_tip_link_quat, elastomer_to_tip_link_pos = elastomer_to_tip_link_trans
        return elastomer_to_tip_link_quat, elastomer_to_tip_link_pos

    def generate_tactile_points(self, elastomer_parent_urdf_path, elastomer_link_name, elastomer_tip_link_name,
                                elastomer_actor_name, num_divs=[20, 25], margin=0.003, visualize=False):
        """
        Generate tactile points on the elastomer.

        Args:
            elastomer_parent_urdf_path (str): Path to the elastomer parent URDF.
            elastomer_link_name (str): Name of the elastomer link.
            elastomer_tip_link_name (str): Name of the elastomer tip link.
            elastomer_actor_name (str): Name of the elastomer actor.
            num_divs (list): Number of divisions for the tactile points.
            margin (float): Margin for the tactile points.
            visualize (bool): Whether to visualize the points.

        Returns:
            tuple: Tactile points positions and quaternions.
        """
        robot = URDF.load(elastomer_parent_urdf_path)
        mesh = robot.link_map[elastomer_link_name].visuals[0].geometry.mesh.meshes[0]

        # generate grid on elastomer
        elastomer_dims = np.diff(mesh.bounds, axis=0).squeeze()
        slim_axis = np.argmin(elastomer_dims)   # determine flat axis of elastomer
        _, elastomer_to_tip_link_pos = self.get_elastomer_to_tip_transform(
            actor_name=elastomer_actor_name,
            elastomer_link_name=elastomer_link_name,
            elastomer_tip_link_name=elastomer_tip_link_name)

        # determine gap between adjacent tactile points
        axis_idxs = list(range(3))
        axis_idxs.remove(slim_axis)     # remove slim idx
        div_sz = (elastomer_dims[axis_idxs] - margin * 2.) / (np.array(num_divs) + 1)
        tactile_points_dx = min(div_sz)

        # sample points on the flat plane
        planar_grid_points = []
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.
        idx = 0
        for axis_i in range(3):
            if axis_i == slim_axis:
                # On the slim axis, place a point far away so ray is pointing at the elastomer tip
                planar_grid_points.append([np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())])
            else:
                axis_grid_points = np.linspace(center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.) / 2., center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.) / 2., num_divs[idx] + 2)
                planar_grid_points.append(axis_grid_points[1:-1])  # leave out the extreme corners
                idx += 1

        grid_corners = itertools.product(planar_grid_points[0], planar_grid_points[1], planar_grid_points[2])
        grid_corners = np.array(list(grid_corners))

        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        ray_dir = np.array([0, 0, 0])
        ray_dir[slim_axis] = -np.sign(elastomer_to_tip_link_pos[0][slim_axis].item())   # ray point towards elastomer
        index_tri, index_ray, locations = mesh_data.intersects_id(grid_corners,
                                                                  np.tile([ray_dir], (grid_corners.shape[0], 1)),
                                                                  return_locations=True, multiple_hits=False)

        if visualize:
            query_pointcloud = trimesh.PointCloud(locations, colors=(0., 0., 1.))

            trimesh.Scene([mesh, query_pointcloud]).show()

        if len(index_ray) != len(grid_corners):
            raise ValueError("Fewer number of tactile points")

        tactile_points = locations[index_ray.argsort()]
        tactile_points_pos_tensor = torch.tensor(tactile_points, dtype=torch.float32, device=self.device)
        rotation = (0, 0, -np.pi)
        rotation = R.from_euler('xyz', rotation).as_quat()
        tactile_points_quat_tensor = torch.tensor(rotation, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(tactile_points), 1)
        return tactile_points_pos_tensor, tactile_points_quat_tensor

    def get_batched_points_in_world(self, tactile_points_pos_local, tactile_points_quat_local, elastomer_link_id):
        num_tactile_points = tactile_points_pos_local.shape[1]
        body_pos = self.body_pos[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 3)
        body_quat = self.body_quat[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 4)
        tactile_points_pos_tmp = tactile_points_pos_local
        tactile_points_quat_tmp = tactile_points_quat_local
        quat_list_per_env, pos_list_per_env = tf_combine(body_quat, body_pos,
                                                         tactile_points_quat_tmp, tactile_points_pos_tmp)

        return pos_list_per_env, quat_list_per_env
    def get_tactile_points_in_world(self, tactile_points_pos_local, tactile_points_quat_local, elastomer_link_id):
        """
        Get the tactile points in the world frame.

        Args:
            tactile_points_pos_local (tensor): Local positions of the tactile points.
            tactile_points_quat_local (tensor): Local quaternions of the tactile points.
            elastomer_link_id (int): ID of the elastomer link.

        Returns:
            tuple: Positions and quaternions of the tactile points in the world frame.
        """
        num_tactile_points = tactile_points_pos_local.shape[0]
        body_pos = self.body_pos[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 3)
        body_quat = self.body_quat[:, elastomer_link_id].unsqueeze(1).expand(self.num_envs, num_tactile_points, 4)
        tactile_points_pos_tmp = tactile_points_pos_local.unsqueeze(0).expand(self.num_envs, num_tactile_points, 3)
        tactile_points_quat_tmp = tactile_points_quat_local.unsqueeze(0).expand(self.num_envs, num_tactile_points, 4)
        quat_list_per_env, pos_list_per_env = tf_combine(body_quat, body_pos,
                                                         tactile_points_quat_tmp, tactile_points_pos_tmp)

        return pos_list_per_env, quat_list_per_env

    ### Trimesh
    def construct_sdf_with_trimesh(self, mesh):
        """
        Construct the signed distance field (SDF) using trimesh.

        Args:
            mesh (trimesh.Trimesh): Trimesh object.

        Returns:
            trimesh.proximity.ProximityQuery: Proximity query object.
        """
        proximity_query = trimesh.proximity.ProximityQuery(mesh)
        return proximity_query

    def query_distance_and_normal_with_trimesh(self, sdf, query_points, visualize=False):
        """
        Query distance and normal using trimesh.

        Args:
            sdf (trimesh.proximity.ProximityQuery): Proximity query object.
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()

        distance_np = sdf.signed_distance(query_points_np)
        mask = (distance_np > 0.) # positive for penetration

        distance = torch.tensor(distance_np, dtype = dtype, device = self.device)
        normal = torch.zeros(query_points.shape, dtype = dtype, device = self.device)
        if mask.sum() > 0:
            closest_np, _, _ = sdf.on_surface(query_points_np[mask])

            closest = torch.tensor(closest_np, dtype = dtype, device = self.device)

            normal[mask] = tu.normalize(closest - query_points[mask])

            if visualize:
                pointcloud_other_query = trimesh.PointCloud(query_points_np[~mask], colors=(0., 0., 1.))
                pointcloud_query = trimesh.PointCloud(query_points_np[mask], colors=(0., 1., 0.))
                pointcloud_closest = trimesh.PointCloud(closest_np, colors=(1., 0., 0.))

                trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()

        return mask, distance, normal

    ###  pysdf
    def construct_sdf_with_pysdf(self, mesh):
        """
        Construct the signed distance field (SDF) using pysdf.

        Args:
            mesh (trimesh.Trimesh): Trimesh object.

        Returns:
            SDF: SDF object.
        """
        from pysdf import SDF
        sdf = SDF(mesh.vertices, mesh.faces)
        return sdf

    def query_distance_and_normal_with_pysdf(self, sdf, query_points, visualize=False):
        """
        Query distance and normal using pysdf.

        Args:
            sdf (SDF): SDF object.
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        dtype = query_points.dtype
        query_points_np = query_points.cpu().numpy()

        distance_np = sdf(query_points_np)
        mask = (distance_np > 0.) # positive for penetration

        distance = torch.tensor(distance_np, dtype = dtype, device = self.device)
        normal = torch.zeros(query_points.shape, dtype = dtype, device = self.device)

        if mask.sum() > 0:
            eps = 1e-5
            grad_x = sdf(query_points_np[mask] + np.array([eps, 0., 0.])) - sdf(query_points_np[mask] - np.array([eps, 0., 0.]))
            grad_y = sdf(query_points_np[mask] + np.array([0., eps, 0.])) - sdf(query_points_np[mask] - np.array([0., eps, 0.]))
            grad_z = sdf(query_points_np[mask] + np.array([0., 0., eps])) - sdf(query_points_np[mask] - np.array([0., 0., eps]))
            grad = torch.tensor(np.stack((grad_x, grad_y, grad_z), axis = 1), dtype = dtype, device = self.device) # check dimension
            normal[mask] = -tu.normalize(grad)

            if visualize:
                closest_np = query_points_np[mask] + np.expand_dims(distance_np[mask], axis = -1) * normal[mask].cpu().numpy()
                pointcloud_other_query = trimesh.PointCloud(query_points_np[~mask], colors=(0., 0., 1.))
                pointcloud_query = trimesh.PointCloud(query_points_np[mask], colors=(0., 1., 0.))
                pointcloud_closest = trimesh.PointCloud(closest_np, colors=(1., 0., 0.))

                trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query, pointcloud_closest]).show()

        return mask, distance, normal

    ### PhysX SDF
    def query_distance_and_normal_with_physx(self, query_points, visualize=False):
        """
        Query distance and normal using PhysX.

        Args:
            query_points (tensor): Points to query.
            visualize (bool): Whether to visualize the results.

        Returns:
            tuple: Mask, distance, and normal vectors.
        """
        self.gym.refresh_sdf_view_tensor(self.sim,
                                         gymtorch.unwrap_tensor(self.sdf_shape_global_ids_per_env),
                                         gymtorch.unwrap_tensor(query_points))
        distance = -self.sdf_tensor[:, ..., 3]
        normal = self.sdf_tensor[:, ..., :3]
        mask = (distance > 0.) # positive for penetration

        if mask.sum() > 0 and visualize:
            viz_env_id = 0
            pointcloud_other_query = trimesh.PointCloud(query_points[viz_env_id][~mask[viz_env_id]].cpu().numpy(),
                                                        colors=(0., 0., 1.))
            pointcloud_query = trimesh.PointCloud(query_points[viz_env_id][mask[viz_env_id]].cpu().numpy(),
                                                  colors=(0., 1., 0.))

            trimesh.Scene([self.indenter_mesh, pointcloud_other_query, pointcloud_query]).show()
        return mask, distance, normal

    def load_sdf_oracle_of_indenter(self, indenter_urdf_path, indenter_actor_name, indenter_rb_name):
        """
        Load the SDF oracle of the indenter.

        Args:
            indenter_urdf_path (str): Path to the indenter URDF.
            indenter_actor_name (str): Name of the indenter actor.
            indenter_rb_name (str): Name of the indenter rigid body.

        Returns:
            tuple: SDF object, indenter mesh, and indenter mesh local transformation.
        """
        robot = URDF.load(indenter_urdf_path)
        indenter_mesh = robot.links[-1].visuals[0].geometry.geometry.meshes[0]
        origin = robot.links[-1].visuals[0].origin
        tf_mat = origin
        tf_pos = tf_mat[0:3, 3]
        tf_quat = R.from_matrix(tf_mat[0:3, 0:3]).as_quat()
        indenter_mesh_local_tf = (torch.tensor([tf_quat], dtype=torch.float, device=self.device), torch.tensor([tf_pos], dtype=torch.float, device=self.device))

        if self.sdf_tool == 'trimesh':
            sdf = self.construct_sdf_with_trimesh(indenter_mesh)
        elif self.sdf_tool == 'pysdf':
            sdf = self.construct_sdf_with_pysdf(indenter_mesh)
        elif self.sdf_tool == 'physx':
            # intialize the sdf object indices
            self.sdf_shape_global_ids_per_env = torch.zeros((self.num_envs, 1), dtype=torch.int32, device=self.device)
            indenter_actor_handle = self.actor_handles[indenter_actor_name]
            for env_id, env_ptr in enumerate(self.env_ptrs):
                indenter_rb_id_actor = self.gym.find_actor_rigid_body_index(env_ptr, indenter_actor_handle, indenter_rb_name, gymapi.DOMAIN_ACTOR)
                indenter_rb_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, indenter_actor_handle)
                indenter_rb_shape_id_actor = indenter_rb_shape_indices[indenter_rb_id_actor].start
                indenter_rb_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, indenter_actor_handle)
                self.sdf_shape_global_ids_per_env[env_id, 0] = indenter_rb_shape_props[indenter_rb_shape_id_actor].global_index
            sdf = None
        else:
            raise NotImplementedError

        return sdf, indenter_mesh, indenter_mesh_local_tf

    def initialize_penalty_based_tactile(self, num_divs):
        # Use one of the sensor configs to set-up the force-field computation
        # Assumption here is that there is a single type of sensor, though there can be multiple instances e.g. left, right sensor
        sensor_config = list(self.tactile_shear_field_configs_dict.values())[0]
        self.tactile_pos_local, self.tactile_quat_local = self.generate_tactile_points(
            elastomer_parent_urdf_path=sensor_config['elastomer_parent_urdf_path'],
            elastomer_link_name=sensor_config['elastomer_link_name'],
            elastomer_tip_link_name=sensor_config['elastomer_tip_link_name'],
            elastomer_actor_name=sensor_config['elastomer_actor_name'],
            num_divs=num_divs, visualize=False)

        # initialize sdf
        self.sdf, self.indenter_mesh, self.indenter_mesh_local_tf = self.load_sdf_oracle_of_indenter(
            indenter_urdf_path=sensor_config['indenter_urdf_path'],
            indenter_actor_name=sensor_config['indenter_actor_name'],
            indenter_rb_name=sensor_config['indenter_link_name'])

        # initialize coefficients
        self.tactile_kn = 1.
        self.tactile_damping = 0.003
        self.tactile_mu = 2.
        self.tactile_kt = 0.1

    def query_collision(self, sdf, tf_sdf, sdf_linvel_world, sdf_angvel_world, points_world, velocity_world):
        """
        Query collisions in the SDF.

        Args:
            sdf: Signed-distance field of the object.
            tf_sdf: (pos, quat) of the object/SDF frame.
            sdf_linvel_world: Linear velocity of the SDF object in the world frame.
            sdf_angvel_world: Angular velocity of the SDF object in the world frame.
            points_world: Points in the world frame.
            velocity_world: Velocities of the points in the world frame.

        Returns:
            tuple: Depth, depth_dot, normal, and vt (all in the world frame).
        """
        num_points_per_env = points_world.shape[1]

        tf_sdf = (tf_sdf[0].unsqueeze(1).expand([self.num_envs, num_points_per_env, 4]),
                  tf_sdf[1].unsqueeze(1).expand([self.num_envs, num_points_per_env, 3]))
        sdf_linvel_world = sdf_linvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])
        sdf_angvel_world = sdf_angvel_world.unsqueeze(1).expand([self.num_envs, num_points_per_env, 3])

        tf_sdf_inv = tu.tf_inverse(tf_sdf[0], tf_sdf[1])
        
        # compute points in the object frame
        points_sdf = tu.tf_apply(tf_sdf_inv[0], tf_sdf_inv[1], points_world)

        # visualize self.grasped_object_mesh_pts and points_world        
        # import open3d as o3d
        # object_mesh_pcd = o3d.geometry.PointCloud()
        # object_mesh_pcd.points = o3d.utility.Vector3dVector(self.grasped_object_mesh_pts.cpu().numpy().reshape(-1, 3))
        # elastomer_pcd = o3d.geometry.PointCloud()
        # elastomer_pcd.points = o3d.utility.Vector3dVector(points_sdf[0].cpu().numpy().reshape(-1, 3))
        # o3d.visualization.draw_geometries([object_mesh_pcd, elastomer_pcd])

        # compute depth
        if self.sdf_tool == 'trimesh':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_trimesh(sdf, points_sdf.view(-1, 3))
        elif self.sdf_tool == 'pysdf':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_pysdf(sdf, points_sdf.view(-1, 3))
        elif self.sdf_tool == 'physx':
            collision_mask_flatten, depth_flatten, normal_flatten_sdf = \
                self.query_distance_and_normal_with_physx(points_sdf.view(self.num_envs, 1, -1, 3))
        else:
            raise NotImplementedError

        depth = depth_flatten.reshape(points_world.shape[:-1])
        depth = depth.clamp(min=0., max=None)

        # compute other returned values
        normal_world = torch.zeros(points_world.shape, device=self.device)
        depth_dot = torch.zeros(points_world.shape[:-1], device=self.device)
        vt_world = torch.zeros(points_world.shape, device = self.device)

        if collision_mask_flatten.sum() > 0:
            normal_sdf = normal_flatten_sdf.reshape(normal_world.shape)
            normal_world = tu.quat_apply(tf_sdf[0], normal_sdf)

            '''
            x = R.T (xw - p)
            xdot = Rdot.T (xw - p) + R.T (xwdot - pdot)
                 = R.T [w].T (xw - p) + R.T (xwdot - pdot)
                 = R.T (-[w] (xw - p) + xwdot - pdot)
                 = R.T ((xw - p) x [w] + xwdot - pdot)
            '''
            velocity_sdf = tu.quat_apply(tf_sdf_inv[0],
                                         torch.cross(points_world - tf_sdf[1], sdf_angvel_world, dim = -1) +
                                         velocity_world - sdf_linvel_world)

            '''
            ddot = dd/dx * dx/dt = n.T * xdot
            '''
            depth_dot = torch.sum(normal_sdf * velocity_sdf, dim = -1)

            '''
            xc_world = R * xc + p
            xcdot_world = [w] R xc + R xcdot + pdot
                        = [w] R xc + pdot
            '''
            closest_points_sdf = points_sdf + depth.unsqueeze(-1) * normal_sdf
            closest_points_velocity_world = torch.cross(sdf_angvel_world, tu.quat_apply(tf_sdf[0], closest_points_sdf)) + sdf_linvel_world
            relative_velocity_world = velocity_world - closest_points_velocity_world

            vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim = -1, keepdim = True)

        return depth, depth_dot, normal_world, vt_world

    def post_process_shear_field_configs(self, tactile_shear_field_configs):
        def get_link_handle(actor_name, link_name):
            link_handle = self.gym.find_actor_rigid_body_handle(
                self.env_ptrs[0], self.actor_handles[actor_name], link_name)
            return link_handle

        for tactile_shear_field_config in tactile_shear_field_configs:
            tactile_shear_field_config['elastomer_link_rb_id'] = get_link_handle(
                tactile_shear_field_config['elastomer_actor_name'], tactile_shear_field_config['elastomer_link_name'])
            tactile_shear_field_config['indenter_link_rb_id'] = get_link_handle(
                tactile_shear_field_config['indenter_actor_name'], tactile_shear_field_config['indenter_link_name'])

        tactile_shear_field_configs_dict = dict()
        for config in tactile_shear_field_configs:
            tactile_shear_field_configs_dict[config['name']] =  config
        return tactile_shear_field_configs_dict

    def get_tactile_shear_force_fields(self):
        tactile_force_field = dict()
        for key, config in self.tactile_shear_field_configs_dict.items():
            indenter_link_id = config['indenter_link_rb_id']
            elastomer_link_id = config['elastomer_link_rb_id']
            result = self.get_penalty_based_tactile_forces(indenter_link_id, elastomer_link_id)
            tactile_force_field[key] = result
        return tactile_force_field

    def get_tactile_points_velocities(self, elastomer_link_id):
        elastomer_angvel_world = self.body_angvel[:, elastomer_link_id]
        elastomer_linvel_world = self.body_linvel[:, elastomer_link_id]
        elastomer_quat_world = self.body_quat[:, elastomer_link_id]
        num_tactile_points = self.tactile_pos_local.shape[0]
        tactile_velocity_world = torch.cross(
            elastomer_angvel_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 3)),
            tu.quat_apply(elastomer_quat_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 4)),
                       self.tactile_pos_local.expand((self.num_envs, num_tactile_points, 3)))) \
                                 + elastomer_linvel_world.unsqueeze(1).expand((self.num_envs, num_tactile_points, 3))
        return tactile_velocity_world

    def get_penalty_based_tactile_forces(self, indenter_link_id, elastomer_link_id):
        """
        Get the penalty-based tactile forces.

        Ref: https://openreview.net/forum?id=6BIffCl6gsM

        Args:
            indenter_link_id (int): ID of the indenter link.
            elastomer_link_id (int): ID of the elastomer link.

        Returns:
            tuple: Interpenetration depth, tactile normal force, and tactile shear force.
        """

        # acquire sdf related variables
        sdf_tf = tf_combine(self.body_quat[:, indenter_link_id],
                            self.body_pos[:, indenter_link_id],
                            self.indenter_mesh_local_tf[0].expand(self.num_envs, 4),
                            self.indenter_mesh_local_tf[1].expand(self.num_envs, 3))

        sdf_angvel_world, sdf_linvel_world = self.body_angvel[:, indenter_link_id], self.body_linvel[:, indenter_link_id]

        self.tactile_pos_world, self.tactile_quat_world = self.get_tactile_points_in_world(
            self.tactile_pos_local, self.tactile_quat_local, elastomer_link_id
        )
        tactile_velocity_world = self.get_tactile_points_velocities(elastomer_link_id)
        # print(tactile_velocity_world.abs().sum())

        depth, depth_dot, normal_world, vt_world = self.query_collision(self.sdf, sdf_tf, sdf_linvel_world, sdf_angvel_world,
                                                   self.tactile_pos_world, tactile_velocity_world)

        # compute tactile forces in world frame
        '''compute contact force'''
        fc_norm = self.tactile_kn * depth #- self.tactile_damping * depth_dot * depth
        fc_world = fc_norm.unsqueeze(-1) * normal_world

        '''compute frictional force'''
        vt_norm = vt_world.norm(dim=-1)
        ft_static_norm = self.tactile_kt * vt_norm
        ft_dynamic_norm = self.tactile_mu * fc_norm
        ft_world = -torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)

        '''net tactile force'''
        tactile_force_world = fc_world + ft_world

        '''tactile force in tactile frame'''
        quat_tactile_inv = tu.quat_conjugate(self.tactile_quat_world)
        tactile_force_tactile = tu.quat_apply(quat_tactile_inv, tactile_force_world)

        # tactile_normal_force = -tactile_force_tactile[..., 2]
        # tactile_shear_force = tactile_force_tactile[..., 0:2]
        tactile_normal_axis = torch.tensor([0., 1., 0.], device=self.device)
        tactile_shear_x_axis = torch.tensor([-1., 0., 0.], device=self.device)
        tactile_shear_y_axis = torch.tensor([0., 0., 1.], device=self.device)
        # tactile_normal_force = -tactile_force_tactile[..., 1] # NOTE: the tactile frame has y as normal direction, to be changed
        # tactile_shear_force = tactile_force_tactile[..., 0:3:2]
        tactile_normal_force = -(tactile_normal_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force_x = (tactile_shear_x_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force_y = (tactile_shear_y_axis.view(1, 1, -1) * tactile_force_tactile).sum(-1)
        tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)

        return depth, tactile_normal_force, tactile_shear_force

    def get_tactile_pcd(self, tactile_force_field_dict, tactile_depth_dict=None, zero_normal=False):
        tactile_pcd_local_left = self.tactile_pos_local.clone() # (N, 3)
        tactile_pcd_local_left = tactile_pcd_local_left.unsqueeze(0).repeat(self.num_envs, 1, 1)  # (num_envs, num_tactile_points, 3)
        tactile_pcd_local_right = tactile_pcd_local_left.clone()  # (num_envs, num_tactile_points, 3)
        
        if not (tactile_depth_dict is None):
            left_depth_flat = tactile_depth_dict['tactile_force_field_left']    # (num_envs, N, 1)
            right_depth_flat = tactile_depth_dict['tactile_force_field_right']  # (num_envs, N)
            tactile_pcd_local_left[..., 1]  += left_depth_flat  # deform the tactile point cloud to show penetration
            tactile_pcd_local_right[..., 1] += right_depth_flat
        
        num_tactile_points = tactile_pcd_local_left.shape[1]
        tactile_quat_local = self.tactile_quat_local.unsqueeze(0).expand(self.num_envs, num_tactile_points, 4)
        
        # get tactile points in world frame
        tactile_pcd_world_left, tactile_quat_world_left = self.get_batched_points_in_world(
            tactile_pcd_local_left, tactile_quat_local, self.tactile_shear_field_configs_dict['tactile_force_field_left']['elastomer_link_rb_id']
        )
        tactile_pcd_world_right, tactile_quat_world_right = self.get_batched_points_in_world(
            tactile_pcd_local_right, tactile_quat_local, self.tactile_shear_field_configs_dict['tactile_force_field_right']['elastomer_link_rb_id']
        )
        
        # get rotation from tactile to world
        tactile_quat_local_inv = quat_conjugate(tactile_quat_local)  # (num_envs, num_tactile_points, 4)
        tactile_quat_tactile2world_left = quat_mul(tactile_quat_world_left, tactile_quat_local_inv)
        tactile_quat_tactile2world_right = quat_mul(tactile_quat_world_right, tactile_quat_local_inv)
        
        shear_left_tactile_frame = tactile_force_field_dict['tactile_force_field_left'].reshape(self.num_envs, -1, 3) # (num_envs, num_tactile_points, 3)
        shear_right_tactile_frame = tactile_force_field_dict['tactile_force_field_right'].reshape(self.num_envs, -1, 3) # (num_envs, num_tactile_points, 3)
        
        if zero_normal:
            shear_left_tactile_frame[..., 0] *= 0.0
            shear_right_tactile_frame[..., 0] *= 0.0
        
        # align shear force to tactile frame
        shear_left_tactile_frame = shear_left_tactile_frame[..., [1,0,2]]
        shear_right_tactile_frame = shear_right_tactile_frame[..., [1,0,2]]  # (num_envs, num_tactile_points, 3)
        
        # rotate shear force to world frame
        tactile_shear_left_world = quat_apply(tactile_quat_tactile2world_left, shear_left_tactile_frame)  # (num_envs, num_tactile_points, 3)
        tactile_shear_right_world = quat_apply(tactile_quat_tactile2world_right, shear_right_tactile_frame)  # (num_envs, num_tactile_points, 3)
        
        tactile_pcd_world = torch.concatenate([tactile_pcd_world_left, tactile_pcd_world_right], dim=1)  # (num_envs, num_tactile_points * 2, 3)
        tactile_shear_world = torch.concatenate([tactile_shear_left_world, tactile_shear_right_world], dim=1)  # (num_envs, num_tactile_points * 2, 3)
        return tactile_pcd_world, tactile_shear_world
    
    def get_o3d_pcd(self, nrows, ncols, tactile_depth_dict, object_pcd, tactile_pcd, tactile_shear, env_vis=0):
        tactile_pcd_color = []
        # get colors
        for key in ['tactile_force_field_left', 'tactile_force_field_right']:
            tactile_depth = tactile_depth_dict[key][env_vis]  # (num_envs, N)
            tactile_depth_im = tactile_depth.reshape((nrows, ncols)).cpu().numpy()
            stretch = skimage.exposure.rescale_intensity(tactile_depth_im, in_range='image', out_range=(0,255)).astype(np.uint8)
            stretch = cv2.merge([stretch,stretch,stretch])
            
            color1 = (0, 0, 255)     #red
            color2 = (0, 165, 255)   #orange
            color3 = (0, 255, 255)   #yellow
            color4 = (255, 255, 0)   #cyan
            color5 = (255, 0, 0)     #blue
            color6 = (128, 64, 64)   #violet
            colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
            
            lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
            result_colored = cv2.LUT(stretch, lut)
            tactile_pcd_color.append(result_colored.reshape(-1, 3))
        tactile_pcd_color = np.concatenate(tactile_pcd_color, axis=0)  # (num_tactile_points * 2, 3)
        tactile_pcd_color = tactile_pcd_color / 255.0  # normalize to [0, 1]
        
        shear_o3d = []
        for shear_idx in range(nrows * ncols * 2):
            shear = tactile_shear[shear_idx]
            if np.linalg.norm(shear) < 1e-4:
                continue
            arrow_origin = tactile_pcd[shear_idx]  # (3,)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=1e-5 * 10,
                cone_radius=1.5 * 1e-5 * 20,
                cylinder_height=(np.linalg.norm(shear)) * 0.1 * 2,  # scale height by magnitude
                cone_height=(np.linalg.norm(shear)) * 0.05 * 2,  # scale height by magnitude
            )
            # translate arrow
            # color arrow
            arrow.translate(arrow_origin)
            arrow.paint_uniform_color(tactile_pcd_color[shear_idx, :].tolist())  # use the color of the tactile point
            
            # rotate arrow to point in the direction of shear
            shear_direction = shear / np.linalg.norm(shear)  # normalize shear direction
            # copilot implementation
            # rotm = R.from_rotvec(np.cross([0, 0, 1], shear_direction)).as_matrix()  # rotation from z-axis to shear direction
            
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
            
            shear_o3d.append(arrow)
        # start visualizing
        obj_pcd_o3d = o3d.geometry.PointCloud()
        obj_pcd_o3d.points = o3d.utility.Vector3dVector(object_pcd)
        obj_pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])  # gray color
        
        tactile_pcd_o3d = o3d.geometry.PointCloud()
        tactile_pcd_o3d.points = o3d.utility.Vector3dVector(tactile_pcd)
        tactile_pcd_o3d.colors = o3d.utility.Vector3dVector(tactile_pcd_color)  # set colors
        
        return obj_pcd_o3d, tactile_pcd_o3d, shear_o3d