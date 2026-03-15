from rl.tasks.bin_packing.bin_task_packing import BinTaskPacking
from rl.tasks.book_shelving.book_shelving_env import BookShelvingEnv
import torch
from isaacgym import gymapi

class BookShelvingTask(BookShelvingEnv, BinTaskPacking):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.cfg = cfg
        self._get_task_yaml_params()
        
        # only call __init__ from BookShelvingEnv side, instead of using super()'s MRO, we enforce which __init__ is called
        BookShelvingEnv.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # we can use cfg_task past this point since super().__init__() has been called
        
        self.grasped_object_mesh_pts = self.insertion_obj.get_surface_points(self.cfg_task.env.num_visual_pcd).to(self.device)
        self.packed_single_object_mesh_pts = self.packed_objects.get_surface_points(self.cfg_task.env.num_packed_obj_pcd).to(self.device)
        
        # self.inhand_pos = torch.tensor([0.0, 0.0, 0.02], device=self.device, dtype=torch.float32)
        self.inhand_pos = torch.tensor(self.cfg_task.randomize.inhand_pos_initial, device=self.device, dtype=torch.float32)
        
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
            self.export_scene(label='book_shelving_task')

        self.set_friction_damping_params(joint_friction=self.cfg_task.env.joint_friction,
                                         joint_damping=self.cfg_task.env.joint_damping)

        self.image_obs_keys = [k for k, v in self.obs_dims.items() if len(v) > 2 and 'force_field' not in k and not k.endswith('_depth') and not k.endswith('_seg')]
        self.init_image_augmentation()

        self.reset_idx(torch.arange(self.num_envs))

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(0.0, -0.4,  1.1)
        cam_target = gymapi.Vec3(0.5, 0.0, 0.8)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)