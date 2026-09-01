import isaacgym

import os
import hydra
import wandb
import datetime
from dataclasses import dataclass
from pathlib import Path
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose

import torch

from rl.tasks import isaacgym_task_map
from rl.algo.ppo.ppo import PPO
from rl.utils.reformat import omegaconf_to_dict, print_dict
from rl.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

# cfg.task.env.use_tacsl_shear = False
# cfg.task.env.use_hydrosoft_model = False

@hydra.main(version_base="1.1",config_name='config', config_path='../../../configs')
def main(cfg: DictConfig):

    aacd: bool = True
    if hasattr(cfg, "ckpt_path") and len(cfg.ckpt_path) > 0:
        teacher_ckpt_path = to_absolute_path(cfg.ckpt_path)
        cont_ckpt_path = to_absolute_path(cfg.ckpt_path)
    else:
        teacher_ckpt_path = None
        cont_ckpt_path = None

    continue_training: bool = False

    # These two select the sensor MODE. Each fans out to several coupled cfg fields below, so
    # they are deliberately not set independently: use_tactile_shear drives both use_shear_force
    # and use_tactile_field_obs, and use_tactile_rgb also drives numEnvs. Keeping that coupling
    # is what makes an invalid combination unreachable.
    # They are read from the task config (falling back to the upstream defaults) so that an
    # ablation such as N0 is recorded in the run's own config.yaml, instead of living in an
    # unversioned edit to this file that no artifact captures.
    use_tactile_shear: bool = bool(cfg.task.env.get("student_tactile_mode", True))
    use_tactile_rgb: bool = bool(cfg.task.env.get("student_rgb_mode", False))
    headless: bool = True

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed)
    cfg.headless = headless
    cfg.force_render = not cfg.headless
    cfg.rollout = False
    cfg.multi_gpu = False

    cfg.task.env.use_camera = False
    cfg.task.env.use_camera_obs = use_tactile_rgb
    cfg.task.env.use_isaac_gym_tactile = use_tactile_rgb
    cfg.task.env.use_gelsight = True

    cfg.task.env.use_shear_force = use_tactile_shear
    cfg.task.env.use_tactile_field_obs = use_tactile_shear
    cfg.task.env.zero_out_normal_force_field_obs = True
    # cfg.task.env.franka_urdf_file = "tacsl_franka_gelsight_mini.urdf"
    cfg.task.env.tactile_subsample_ratio = 4

    cfg.task.randomize.randomize_color_channel = False
    cfg.task.randomize.use_ep_image_aug = True
    cfg.task.randomize.use_t_color_aug = False
    cfg.task.randomize.use_diff_tactile_img = False
    cfg.task.randomize.concat_tactile_plain = False

    cfg.task.rl.add_contact_info_to_aac_states = True
    cfg.task.rl.add_contact_force_plug_decomposed = True
    cfg.task.env.use_compliant_contact = True
    cfg.task.rl.asymmetric_observations = True

    # cfg.task.env.visualize_tactile_point_clouds = False

    cfg.task.env.numEnvs = 128 if use_tactile_rgb else 256

    if cfg.wandb_activate:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = wandb.init(
            project=cfg.wandb_project,
            config=omegaconf_to_dict(cfg),
            sync_tensorboard=True,
            name=f"{cfg.wandb_name}_{datetime_str}",
            resume="allow",
            monitor_gym=True,
        )
    obs_config = {
        'ee_pos': [3],
        'ee_quat': [4],
        'eef_to_socket_pos': [3],
        'eef_to_socket_quat': [4],
        'tactile_force_field_left': [9, 7, 2],
        'tactile_force_field_right': [9, 7, 2],
    }

    if use_tactile_rgb:
        obs_config.update(
            left_tactile_camera_taxim=[80, 60, 3],
            right_tactile_camera_taxim=[80, 60, 3],
            left_tactile_camera_taxim_gray=[80, 60, 1],
            right_tactile_camera_taxim_gray=[80, 60, 1]
        )

    # cfg.task.env.obsDims = obs_config
    # cfg.task.env.stateDims = states_config

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

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

    # RS_C0_PRODUCER: matched C(0) TRAINING calibration (GT-PREF-007's open limitation).
    # Replaces ONLY the tactile field producer with a zero-cost stand-in of identical shape,
    # dtype and device. Everything else is untouched and still runs: tactile CNNs, actor,
    # critic, AACD teacher, stochastic action sampling, experience buffer, PPO updates, resets.
    # T_train(B) - T_train(C0) then gives the producer cost measured IN TRAINING CONTEXT,
    # rather than borrowed from a differently-scoped rollout run.
    # Absent unless the env var is set, so existing runs are bit-identical.
    if os.environ.get("RS_C0_PRODUCER"):
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tactile_backend_seam import LatencyDummyBackend
        _c0 = LatencyDummyBackend(env.num_envs, env.device, latency_ms=0.0)
        env.get_force_fields_dict = _c0.get_force_fields_dict
        print("[RS_C0_PRODUCER] tactile producer -> zero-cost stand-in. "
              "THROUGHPUT CALIBRATION ONLY; the learning outcome of this run is meaningless.",
              flush=True)

    rl_output_dir = os.environ["RL_OUTPUT_PATH"] if "RL_OUTPUT_PATH" in os.environ else "./outputs"
    task_name = cfg.task_name
    output_dir = os.path.join(rl_output_dir, "1_hydroshear", task_name, cfg.wandb_name)
    agent: PPO = eval(cfg.train.algo)(env=env, cfg=cfg, output_dir=output_dir)

    if continue_training and cont_ckpt_path is not None:
        agent.load_model(cont_ckpt_path)
    else:
        if aacd and teacher_ckpt_path is not None:
            agent.load_critic(teacher_ckpt_path)
    
    agent.train()


if __name__ == "__main__":
    main()