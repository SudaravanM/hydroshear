from collections import defaultdict
import hydra
import numpy as np
import os
import torch
import cv2

from omegaconf import OmegaConf

from isaacgym import gymapi
from rl.tasks.factory.factory_schema_class_env import FactoryABCEnv
from rl.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from rl.tasks.tacsl.tacsl_base import TacSLBase
from rl.tasks.tacsl.tacsl_env_insertion import TacSLSensors
from rl.utils.urdf_object import ObjectABC
from rl.tasks.tacsl.tacsl_env_insertion import TacSLSensors
from rl.hydroshear_sensors.hydroshear_sensors import HydroFotsSensor, HydroFotsFieldSensor
from rl.fots_sensors.fots_sensors import FotsSensor, FotsFieldSensor

class Drawer(ObjectABC):
    def __init__(self, gym):
        super().__init__(gym)
        self.texture_filename = './assets/furniture/mesh/textures/wood1.jpeg'
        
    def initialize_pose(self):
        raise NotImplementedError("drawer does not need to initialize poses, it is static")
    
    def setup_asset(self, sim):
        drawer_options = gymapi.AssetOptions()
        # density please
        drawer_options.density = 1000  # default = 1000.0
        drawer_options.flip_visual_attachments = False
        drawer_options.fix_base_link = True
        drawer_options.thickness = 0.0  # default = 0.02
        drawer_options.armature = 0.0  # default = 0.0
        drawer_options.use_physx_armature = True
        drawer_options.linear_damping = 0.0  # default = 0.0
        drawer_options.max_linear_velocity = 1000.0  # default = 1000.0
        drawer_options.angular_damping = 0.0  # default = 0.5
        drawer_options.max_angular_velocity = 64.0  # default = 64.0
        drawer_options.disable_gravity = False
        drawer_options.enable_gyroscopic_forces = True
        drawer_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        drawer_options.use_mesh_materials = False
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        drawer_asset  = self.gym.load_asset(sim, urdf_root, 'drawer.urdf', drawer_options)
        self.urdf_path = os.path.join(urdf_root, 'drawer.urdf')
        self.assets['drawer'] = drawer_asset
        
        texture_img = cv2.imread(self.texture_filename)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        texture_img = np.dstack((texture_img, np.ones((texture_img.shape[0], texture_img.shape[1]), dtype=np.uint8) * 255))  # add alpha channel
        H,W, _ = texture_img.shape
        texture_img = texture_img.reshape((H, W*4))
        self.texture_handle = self.gym.create_texture_from_buffer(sim, W, H, texture_img)
        # self.texture_handle = self.gym.create_texture_from_file(sim, self.texture_filename)
        
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id, pose=gymapi.Transform()):
        ## create drawer 
        drawer_handle = self.gym.create_actor(env_ptr, self.assets['drawer'], pose, 'drawer', collision_group_id, 0, 0)
        self.actor_handles['drawer'] = drawer_handle
        self.actor_ids_sim['drawer'].append(actor_count)
        
        drawer_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, drawer_handle)
        drawer_shape_props[0].friction = 0.01 # higher
        drawer_shape_props[0].rolling_friction = 0.0  # default = 0.0
        drawer_shape_props[0].torsion_friction = 0.0  # default = 0.0
        drawer_shape_props[0].restitution = 0.0  # default = 0.0
        drawer_shape_props[0].compliance = 0.0  # default = 0.0
        drawer_shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, drawer_handle, drawer_shape_props)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, 'drawer', gymapi.DOMAIN_ENV)
            self.drawer_rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        self.drawer_rb_names[0], gymapi.DOMAIN_ENV)
        
        actor_rigid_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, self.actor_id_env, self.drawer_rb_names[0], gymapi.DOMAIN_ACTOR)
        self.gym.set_rigid_body_texture(env_ptr, self.actor_handles['drawer'], actor_rigid_body_id_env, gymapi.MESH_VISUAL_AND_COLLISION, self.texture_handle)
        
        actor_count += 1
        return actor_count


class DrawerEnvPulling(TacSLBase, TacSLSensors, HydroFotsFieldSensor, FotsFieldSensor, FactoryABCEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self._get_env_yaml_params()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        self.nominal_tactile = None

        elastomer_urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'tacsl', 'urdf')
        indenter_urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        
        hydrofots_cfg = self.cfg_task.sensor.hydroshear
        old_fots_cfg = self.cfg_task.sensor.get("oldfots", None)
        fots_cfg = self.cfg_task.sensor.fots

        if self.cfg_task.env.get('use_shear_force', False):
            if self.cfg_task.env.get('use_hydrosoft_model', False):
                hydrofots_left = HydroFotsSensor(
                    hydrofots_cfg=hydrofots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_left',
                    indenter_urdf=self.plug_file,
                    indenter_link_name='drawer_handle',
                    device=self.device,
                    elastomer_resolution=self.cfg_task.env.get('elastomer_resolution', 0.001),
                    indenter_resolution=self.cfg_task.env.get('indenter_resolution', 0.001)
                )
                hydrofots_right = HydroFotsSensor(
                    hydrofots_cfg=hydrofots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_right',
                    indenter_urdf=self.plug_file,
                    indenter_link_name='drawer_handle',
                    device=self.device,
                    indenter_sdf_sensor=hydrofots_left.indenter_sdf_sensor, # share sdf sensor between left and right
                    elastomer_resolution=self.cfg_task.env.get('elastomer_resolution', 0.001),
                    indenter_resolution=self.cfg_task.env.get('indenter_resolution', 0.001)
                )
                HydroFotsFieldSensor.__init__(self, 
                    hydrofots_sensors=[hydrofots_left, hydrofots_right],
                    elastomer_link_names=['elastomer_left', 'elastomer_right'],
                    elastomer_actor_names=['franka', 'franka'],
                    # visualize=True
                )
            elif self.cfg_task.env.get('use_fots_model', False) or self.cfg_task.env.get('use_old_fots_model', False):
                sensor_fots_cfg = fots_cfg if self.cfg_task.env.get('use_fots_model', False) else old_fots_cfg
                fots_left = FotsSensor(
                    fots_cfg=sensor_fots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_left',
                    indenter_urdf=self.plug_file,
                    indenter_link_name='drawer_handle',
                    device=self.device,
                    use_original_implementation=self.cfg_task.env.get('use_old_fots_model', False)
                )
                
                fots_right = FotsSensor(
                    fots_cfg=fots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_right',
                    indenter_urdf=self.plug_file,
                    indenter_link_name='drawer_handle',
                    device=self.device,
                    indenter_sdf_sensor=fots_left.indenter_sdf_sensor, # share sdf sensor between left and right
                    use_original_implementation=self.cfg_task.env.get('use_old_fots_model', False)
                )
                FotsFieldSensor.__init__(self,
                    fots_sensors=[fots_left, fots_right],
                    elastomer_link_names=['elastomer_left', 'elastomer_right'],
                    elastomer_actor_names=['franka', 'franka'],
                )

    # override TacSLSensors method
    def _compose_tactile_force_field_configs(self):
        tactile_shear_field_config = dict([
            ('name', 'tactile_force_field_left'),
            ('elastomer_actor_name', 'franka'), ('elastomer_link_name', 'elastomer_left'),
            ('elastomer_tip_link_name', 'elastomer_tip_left'),
            ('elastomer_parent_urdf_path', self.asset_file_paths["franka"]),
            ('indenter_urdf_path', self.asset_file_paths["drawer"]),
            ('indenter_actor_name', 'drawer'), ('indenter_link_name', 'drawer_handle'),
            ('actor_handle', self.actor_handles['franka']),
            ('compliance_stiffness', self.cfg_task.env.compliance_stiffness),
            ('compliant_damping', self.cfg_task.env.compliant_damping),
            ('use_acceleration_spring', False)
        ])
        tactile_shear_field_config_left = tactile_shear_field_config.copy()
        tactile_shear_field_config_right = tactile_shear_field_config.copy()
        tactile_shear_field_config_right['name'] = 'tactile_force_field_right'
        tactile_shear_field_config_right['elastomer_link_name'] = 'elastomer_right'
        tactile_shear_field_config_right['elastomer_tip_link_name'] = 'elastomer_tip_right'
        tactile_shear_field_configs = [tactile_shear_field_config_left, tactile_shear_field_config_right]
        return tactile_shear_field_configs

    def set_friction_damping_params(self, joint_friction=None, joint_damping=None):
        """
        Set friction and damping parameters for the robot joints.

        Args:
            joint_friction: Friction values for the joints.
            joint_damping: Damping values for the joints.
        """
        if joint_friction is None and joint_damping is None:
            return

        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)

            if joint_friction is not None:
                franka_dof_props['friction'][:9] = joint_friction[:9]
            if joint_damping is not None:
                franka_dof_props['damping'][:9] = joint_damping[:9]

            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)
        
    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/DrawerEnvPulling.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting
    
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.asset_info_franka_table.table_depth * 0.6,
                            -self.asset_info_franka_table.table_width * 0.6,
                            0.0)
        upper = gymapi.Vec3(self.asset_info_franka_table.table_depth * 0.6,
                            self.asset_info_franka_table.table_width * 0.6,
                            self.asset_info_franka_table.table_height)
        num_per_row = int(np.sqrt(self.num_envs))

        self.drawer_object = Drawer(self.gym)
        


        self.print_sdf_warning()
        self.assets = dict()
        self.asset_file_paths = dict()
        self.assets['franka'], self.assets['table'] = self.import_franka_assets()
        self._import_env_assets()
        self._create_actors(lower, upper, num_per_row)
        self.parse_controller_spec()
        
        self.plug_actor_id_env = self.drawer_object.actor_id_env
        self.asset_file_paths['drawer'] = self.drawer_object.urdf_path
        self.plug_file = 'drawer.urdf'
        self._create_sensors()
        
    def _import_env_assets(self):
        """Set plug and socket asset options. Import assets."""
        self.drawer_object.setup_asset(self.sim)
        self.assets.update(self.drawer_object.assets)
    
    def _create_actors(self, lower, upper, num_per_row):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = 0.0
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = self.asset_info_franka_table.robot_base_to_table_offset_x
        table_pose.p.y = 0.0
        table_pose.p.z = -self.asset_info_franka_table.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.table_handles = []
        self.shape_ids = []
        self.table_actor_ids_sim = []  # within-sim indices
        self.env_subassembly_id = []
        self.actor_handles = {}
        self.actor_ids_sim = defaultdict(list)
        self.rbs_com = defaultdict(list)
        actor_count = 0

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, self.assets['franka'], franka_pose, 'franka',
                                                      i + self.num_envs, 0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, self.assets['franka'], franka_pose, 'franka', i, 0, 0)
            self.actor_handles['franka'] = franka_handle
            self.actor_ids_sim['franka'].append(actor_count)
            actor_count += 1
            

            table_handle = self.gym.create_actor(env_ptr, self.assets['table'], table_pose, 'table', i, 0, 0)
            self.actor_handles['table'] = table_handle
            self.actor_ids_sim['table'].append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'wsg50_base_link', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            rb_ids = [link7_id, hand_id, left_finger_id, right_finger_id]
            rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.assets['franka'])
            self.shape_ids = [rb_shape_indices[rb_id].start for rb_id in rb_ids]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            
            
            pose = gymapi.Transform()
            # pose.p.x = 0.5127
            # pose.p.y = -0.1773 - (0.30/2) + 0.013
            # pose.p.z = 0.1578 + 0.01

            # drawer = fingertip_midpoint + box_dimy/2 + box_urdfoffsety + gap (due to fingertip misalignment and fingerbox mesh inaccuracies)
            pose.p.x = 0.50889933
            pose.p.y = -0.20576764 - (0.225/2) - 0.007 - 0.006
            pose.p.z = 0.1469082 + 0.01 + 0.00564
            # rotate z by +90 deg
            pose.r = gymapi.Quat.from_euler_zyx(0, 0.0, -np.pi/2)
            # pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            actor_count = self.drawer_object.create_actor(self.env_ptrs[0], env_ptr, actor_count, i, pose)
        
        self.actor_handles.update(self.drawer_object.actor_handles)
        self.actor_ids_sim.update(self.drawer_object.actor_ids_sim)

        if self.cfg_task.env.use_compliant_contact:
            # Set compliance params
            self.set_elastomer_compliance(self.cfg_task.env.compliance_stiffness, self.cfg_task.env.compliant_damping)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.actor_ids_sim_tensors = {key: torch.tensor(self.actor_ids_sim[key], dtype=torch.int32, device=self.device)
                                      for key in self.actor_ids_sim.keys()}

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.rbs_com_tensors = {key: torch.tensor(self.rbs_com[key], dtype=torch.float, device=self.device) for key in
                                self.rbs_com.keys()}

        # For extracting body pos/quat, force, and Jacobian
        drawer_handle = self.drawer_object.actor_handles['drawer']
        self.drawer_handle_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, drawer_handle, 'drawer_handle', gymapi.DOMAIN_ENV)
        
        self.drawer_box_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, drawer_handle, 'drawer_box', gymapi.DOMAIN_ENV)
        
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'wsg50_base_link',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_fingerbox_left',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_fingerbox_right', gymapi.DOMAIN_ENV)
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                               'panda_leftfingertip',
                                                                               gymapi.DOMAIN_ENV)
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_rightfingertip',
                                                                                gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        self.finger_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_finger_centered',
                                                                                gymapi.DOMAIN_ENV)
        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'wsg50_base_link',
                                                                           gymapi.DOMAIN_ACTOR)
        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                  'panda_leftfinger',
                                                                                  gymapi.DOMAIN_ACTOR)
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_rightfinger',
                                                                                   gymapi.DOMAIN_ACTOR)
        self.left_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                     'panda_leftfingertip',
                                                                                     gymapi.DOMAIN_ACTOR)
        self.right_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                      'panda_rightfingertip',
                                                                                      gymapi.DOMAIN_ACTOR)
        self.fingertip_centered_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                         'panda_fingertip_centered',
                                                                                         gymapi.DOMAIN_ACTOR)
        self.table_body_id = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles['table'],
                                                                  'box', gymapi.DOMAIN_ENV)
        self.franka_body_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
        self.franka_body_ids_env = dict()
        for b_name in self.franka_body_names:
            self.franka_body_ids_env[b_name] = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                                    self.actor_handles['franka'],
                                                                                    b_name, gymapi.DOMAIN_ENV)
    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        """Set elastomer compliance parameters."""
        for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
            self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                              elastomer_link_name=elastomer_link_name,
                                              compliance_stiffness=compliance_stiffness,
                                              compliant_damping=compliant_damping,
                                              use_acceleration_spring=False)
        pass
    
    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.franka_base_pos = self.root_pos[:, self.franka_actor_id_env, 0:3]
        self.franka_base_quat = self.root_quat[:, self.franka_actor_id_env, 0:4]
        
        self.drawer_pos = self.root_pos[:, self.drawer_object.actor_id_env, 0:3]
        self.drawer_quat = self.root_quat[:, self.drawer_object.actor_id_env, 0:4]
        
        self.drawer_handle_pos = self.body_pos[:, self.drawer_handle_body_id_env, 0:3]
        self.drawer_handle_quat = self.body_quat[:, self.drawer_handle_body_id_env, 0:4]
        
        self.plug_quat = self.drawer_handle_quat
        self.plug_pos = self.drawer_handle_pos

        self.drawer_handle_linvel = self.body_linvel[:, self.drawer_handle_body_id_env, 0:3]
        self.drawer_handle_angvel = self.body_angvel[:, self.drawer_handle_body_id_env, 0:3]

        self.finger_centered_pos = self.body_pos[:, self.finger_centered_body_id_env, 0:3]
        self.finger_centered_quat = self.body_quat[:, self.finger_centered_body_id_env, 0:4]
        
    def refresh_env_tensors(self):
        """Refresh tensors."""
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)