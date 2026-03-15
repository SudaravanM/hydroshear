import isaacgym

import os
import sys
from dataclasses import dataclass
from omegaconf import OmegaConf
from hydra import initialize_config_dir
import tyro
from rl.algo.ppo.ppo import PPO
from rl.tasks import isaacgym_task_map
from rl.utils.reformat import omegaconf_to_dict
from rl.utils.misc import load_config_from_checkpoint, set_seed


@dataclass
class Config:
    ckpt_path: str
    visualize_keypoints: bool = False
    visualize_goal_pose: bool = False
    visualize_packed_keypoints: bool = False
    debug_vis: bool = False
    save_video: bool = False
    headless: bool = False

    
    # Randomization flags
    randomize_peg_rotation: bool = False
    randomize_peg_translation: bool = False
    randomize_socket_translation: bool = False
    randomize_compliance: bool = False
    randomize_friction: bool = False
    randomize_control: bool = False
    randomize_shear_coefficients: bool = False

    # Override flags
    fix_compliance: bool = False
    fix_friction: bool = False
    fix_hydroshear: bool = False
    fix_fots: bool = False
    fix_peg_translation: bool = False
    fix_gripper_width: bool = False
    use_hydrosoft_model: bool = False
    use_tacsl_model: bool = False
    use_fots_model: bool = False
    add_touch_obs: bool = False
    add_tactile_rgb_obs: bool = False

    # Noisy observation flags
    noisy_socket_pos_obs: bool = False


def main():
    args = tyro.cli(Config)

    print(f"Checkpoint path: {args.ckpt_path}")
    print(f"Checkpoint exists: {os.path.exists(args.ckpt_path)}")

    cfg = load_config_from_checkpoint(args.ckpt_path)
    print(f"Using config from checkpoint: {cfg is not None}")
    
    cfg.task.env.numEnvs = 1
    cfg.headless = args.headless
    cfg.force_render = not cfg.headless

    set_seed(cfg.seed)

    cfg.task.rl.visualize_keypoints = args.visualize_keypoints
    cfg.task.rl.visualize_goal_pose = args.visualize_goal_pose
    cfg.task.rl.visualize_packed_keypoints = args.visualize_packed_keypoints

    cfg.task.env.stateDims = {
        **cfg.task.env.stateDims,
        'socket_force': [1],
        'socket_force_smooth': [1],
    }

    cfg.task.env.debug_vis = args.debug_vis

    if args.add_touch_obs:
        cfg.task.env.use_camera = False
        cfg.task.env.use_camera_obs = args.add_tactile_rgb_obs
        cfg.task.env.use_isaac_gym_tactile = args.add_tactile_rgb_obs
        cfg.task.env.use_gelsight = True

        cfg.task.env.use_shear_force = True
        cfg.task.env.use_tactile_field_obs = True
        cfg.task.env.zero_out_normal_force_field_obs = True

        cfg.task.env.obsDims = {
            **cfg.task.env.obsDims,
            'tactile_force_field_left': [9, 7, 2],
            'tactile_force_field_right': [9, 7, 2],
        }
        cfg.task.env.use_hydrosoft_model = args.use_hydrosoft_model
        cfg.task.env.use_tacsl_shear = args.use_tacsl_model
        cfg.task.env.use_fots_model = args.use_fots_model

        if args.add_tactile_rgb_obs:
            cfg.task.env.obsDims.update({
                'left_tactile_camera_taxim': [80, 60, 3],
                'right_tactile_camera_taxim': [80, 60, 3],
                'left_tactile_camera_taxim_gray': [80, 60, 1],
                'right_tactile_camera_taxim_gray': [80, 60, 1],
            })

    # Apply randomization flags
    if args.randomize_peg_rotation:
        print("Rotation randomization enabled")
        cfg.task.randomize.plug_noise_rot_in_gripper = [0.0, 0.628318, 0.0]
    
    if args.randomize_peg_translation:
        print("Translation randomization enabled")
        cfg.task.randomize.plug_pos_in_gripper_noise_xy = [0.005, 0.01]

    if args.randomize_socket_translation:
        cfg.task.randomize.socket_pos_xyz_noise = [0.1, 0.1, 0.02]

    if args.noisy_socket_pos_obs:
        cfg.task.randomize.socket_pos_xyz_obs_noise = [0.005, 0.005, 0.005]
    
    if args.randomize_compliance:
        print("Compliance randomization enabled")
        cfg.task.randomize.randomize_compliance = True
        cfg.task.randomize.compliance_stiffness_range = [150, 350]
        cfg.task.randomize.compliance_damping_range = [0.0, 1.0]
    
    if args.randomize_friction:
        print("Friction randomization enabled")
        cfg.task.randomize.randomize_friction = True
        cfg.task.randomize.elastomer_friction_factor_range = [0.5, 1.0]

    if args.randomize_control:
        print("Control randomization enabled")
        cfg.task.randomize.randomize_ctrl_params = True

    if args.randomize_shear_coefficients:
        print("Shear coefficients randomization enabled")
        cfg.sensor.randomize_coefficients = True
        cfg.sensor.randomize_every_episode = True

    if args.fix_compliance:
        print("Fix compliance enabled")
        cfg.task.randomize.randomize_compliance = False
        cfg.task.env.compliance_stiffness = 300.0
        cfg.task.env.compliant_damping = 0.0

    if args.fix_friction:
        print("Fix friction enabled")
        cfg.task.randomize.randomize_friction = False
        cfg.task.env.elastomer_friction_factor = 1.0

    if args.fix_gripper_width:
        print("Fix gripper width enabled")
        cfg.task.env.franka_close_gripper_width = 0.03

    if args.fix_hydroshear:
        cfg.task.sensor.hydroshear.randomize_coefficients = False
        cfg.task.sensor.hydroshear.randomize_every_episode = False
        cfg.task.sensor.hydroshear.mu = 0.5
        cfg.task.sensor.hydroshear.lambda_d = 20000
        cfg.task.sensor.hydroshear.lambda_s = 10800
        cfg.task.sensor.hydroshear.dilate_scale = 1000 / 0.065 * 1
        cfg.task.sensor.hydroshear.shear_scale = 1000 / 0.065 * 1
        cfg.task.sensor.hydroshear.hydroshear_gravity_effect = 0.00025

    if args.fix_fots:
        cfg.task.sensor.fots.randomize_coefficients = False
        cfg.task.sensor.fots.randomize_every_episode = False
        cfg.task.sensor.fots.dilate_scale = 1000 / 0.065 * 1
        cfg.task.sensor.fots.shear_scale = 1000 / 0.065 * 1
        cfg.task.sensor.fots.twist_scale = 1000 / 0.065 * 1

    if args.fix_peg_translation:
        print("Fix peg translation enabled")
        cfg.task.randomize.plug_pos_in_gripper_noise_xy = [0.0, 0.0]

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../configs"))
    print(f"Config directory: {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task),
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=False,
            force_render=cfg.force_render,
        )
    env.reset()

    rl_output_dir = os.environ["RL_OUTPUT_PATH"] if "RL_OUTPUT_PATH" in os.environ else "./outputs"
    output_dir = os.path.join(rl_output_dir, "0_hydroshear", "play", cfg.wandb_name)
    agent = eval(cfg.train.algo)(env=env, cfg=cfg, output_dir=output_dir)

    # Testing fully trained policies
    agent.load_model(args.ckpt_path)

    agent.test()
if __name__ == "__main__":
    main()