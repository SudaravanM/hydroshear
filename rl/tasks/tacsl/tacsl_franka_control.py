import torch
from isaacgym import torch_utils, gymtorch
import rl.tasks.factory.factory_control as fc

class TacSLFrankaControl:
    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """
        Apply actions from policy as position/rotation targets.

        Args:
            actions: Actions to be applied.
            ctrl_target_gripper_dof_pos: Target gripper DOF position.
            do_scale: Boolean indicating if action scaling is applied.
        """
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def apply_screw_primitive(self):
        """Apply screw primitive."""
        self.refresh_all_tensors()
        target_dof = self.dof_pos.clone()
        target_dof[:, 6] += 1.

        DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
        DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
        joint_prop_gains = torch.tensor(DEFAULT_K_GAINS, device=self.device) / 100.
        joint_deriv_gains = torch.tensor(DEFAULT_D_GAINS, device=self.device) / 100.

        # Step sim
        sim_steps = self.cfg_task.env.num_gripper_close_sim_steps * 4
        for _ in range(sim_steps):
            self.refresh_all_tensors()
            self.dof_torque[:, 0:7] = joint_prop_gains * (target_dof - self.dof_pos)[:, 0:7] + \
                                      joint_deriv_gains * (0.0 - self.dof_vel[:, 0:7])
            # keep prev torque applied by gripper
            self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.dof_torque),
                                                            gymtorch.unwrap_tensor(self.actor_ids_sim_tensors['franka']),
                                                            len(self.actor_ids_sim_tensors['franka']))
            self.render()
            self.gym.simulate(self.sim)
    
    def _move_to_target_pose_and_gripper_width(self, target_fingertip_midpoint_pos, target_fingertip_midpoint_quat,
                                               gripper_dof_pos, sim_steps=20, gentle_gripper_close=False):
        """Move arm to target end-effector pose, and gripper to target width using task-space controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = target_fingertip_midpoint_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = target_fingertip_midpoint_quat

        # Step sim
        for step_i in range(sim_steps):
            self.refresh_all_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            if gentle_gripper_close:
                # break gripper motion into steps. Try to move to target in half the num of steps left.
                num_steps_left = sim_steps - step_i
                target_gripper_i = (gripper_dof_pos - self.gripper_dof_pos) / (0.5 * num_steps_left)
                target_gripper_pos = self.gripper_dof_pos + target_gripper_i
            else:
                target_gripper_pos = gripper_dof_pos
            self._apply_actions_as_ctrl_targets(delta_hand_pose, target_gripper_pos, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)
            if self.cfg_task.env.get('use_shear_force', False):
                if self.cfg_task.env.get('use_hydrosoft_model', False):
                    self.get_force_fields_dict(self.plug_quat, self.plug_pos)
                    
    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        # Keep current end-effector pose as target end-effector pose, when moving the gripper joint
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()

        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)
        
    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_midpoint_pos.detach().clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_midpoint_quat.detach().clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += lift_distance
        self._move_to_target_pose_and_gripper_width(self.ctrl_target_fingertip_midpoint_pos,
                                                    self.ctrl_target_fingertip_midpoint_quat,
                                                    gripper_dof_pos, sim_steps=sim_steps)
    
    def _reset_franka_actuation(self, ctrl_target_dof_pos):
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'].flatten()
        zeros = torch.zeros_like(self.dof_torque)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(zeros),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(ctrl_target_dof_pos),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        
    def _randomize_gripper_pose(self, env_ids, sim_steps, ctrl_target_gripper_dof_pos=0.0):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )

        # NOTE: sometimes this gives -quat which is the same as quat
        # but this will give crazy axis-angle-error
        # we have to check if the negative version is close to the other quat and flip

        # roll, pitch, yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        # print(roll[0], pitch[0], yaw[0])
        # quat = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)
        # self.ctrl_target_fingertip_midpoint_quat = quat
        # self.ctrl_target_fingertip_midpoint_quat = self.fingertip_midpoint_quat.clone()


        # Step sim and render
        for _ in range(sim_steps):
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')
            _, axis_angle_error2 = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=-self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')
            flip_idx = torch.norm(axis_angle_error,p=2,dim=-1) > torch.norm(axis_angle_error2,p=2,dim=-1)
            axis_angle_error[flip_idx] = axis_angle_error2[flip_idx]

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions, ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos, do_scale=False)

            self.gym.simulate(self.sim)
            self.refresh_all_tensors()
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.actor_ids_sim_tensors['franka'][env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._reset_franka_actuation(self.dof_pos.clone())

        self.refresh_all_tensors()
        
    def randomize_controller_params(self):

        k_gains_min = torch.tensor(self.cfg_task.randomize.task_prop_gains_min, dtype=torch.float32, device=self.device)
        k_gains_max = torch.tensor(self.cfg_task.randomize.task_prop_gains_max, dtype=torch.float32, device=self.device)
        ctrl_param_noise = torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self.cfg_ctrl['task_prop_gains'] = k_gains_min + ctrl_param_noise * (k_gains_max - k_gains_min)
        self.cfg_ctrl['task_deriv_gains'] = 2 * torch.sqrt(self.cfg_ctrl['task_prop_gains'])

        # Scale down the rotation gains of the controller. Stabilizes motion for the default franka urdf.
        self.cfg_ctrl['task_deriv_gains'][:, 3:] /= 10.     # reduce the damping of the rotation action params
        
    def execute_control_loop(self, num_control_steps):
        """
        Execute the control loop for a specified number of steps.

        Args:
            num_control_steps: Number of control steps to execute.
        """
        for _ in range(num_control_steps):
            # execute previous control signal
            self.gym.simulate(self.sim)

            # refresh tensors
            self.refresh_all_tensors()

            # generate new control signal
            self.generate_ctrl_signals()
            
    def initialize_franka_robot_open_hand(self):
        """
        Initialize Franka robot to default dof position.
        """
        self.dof_pos[:, 0:7] = torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device)
        self.dof_pos[:, 7:] = self.cfg_task.env.get("franka_open_gripper_width",
                                                    self.asset_info_franka_table.franka_gripper_width_max)

        self.ctrl_target_dof_pos[:] = self.dof_pos[:]
        self.dof_vel[:, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.actor_ids_sim_tensors['franka'].to(dtype=torch.int32, device=self.device)[:]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh_all_tensors()
        
    def set_gripper_friction_to_default(self):
        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)
            franka_dof_props['friction'][7:9] = self.cfg_task.env.default_gripper_joint_friction
            franka_dof_props['damping'][7:9] = self.cfg_task.env.default_gripper_joint_damping
            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)