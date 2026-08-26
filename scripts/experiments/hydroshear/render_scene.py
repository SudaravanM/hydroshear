"""Render a VIDEO of the drawer scene from a checkpoint: what the robot actually does.

The recorded-trajectory GIF shows what the policy FEELS. This shows what it DOES, by turning on
one of the environment's own scene cameras and grabbing an RGB frame every control step.

No environment changes are needed. DrawerEnvPulling already inherits TacSLSensors, whose
_create_sensors() builds camera actors whenever cfg.task.env.use_camera is true; the drawer
configs simply never switch it on. Cameras also work headless: vec_task only disables graphics
when enableCameraSensors is false AND headless is true.

  python scripts/experiments/hydroshear/render_scene.py --ckpt_path <ckpt.pth> \
      --episodes 1 --num-envs 2 --out outputs/scene

Writes <out>/scene_<ckpt>.gif plus the raw frames, and prints where the drawer got to.
"""
import isaacgym  # noqa: F401  MUST precede torch

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from hydra import initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rl.algo.ppo.ppo import PPO                                     # noqa
from rl.tasks import isaacgym_task_map                              # noqa
from rl.utils.reformat import omegaconf_to_dict                     # noqa
from rl.utils.misc import load_config_from_checkpoint, set_seed     # noqa


@dataclass
class Config:
    ckpt_path: str
    episodes: int = 1
    num_envs: int = 2
    out: str = "outputs/scene"
    view: str = "front"          # front | side | wrist
    height: int = 480
    width: int = 640
    stride: int = 2              # keep every Nth frame in the gif
    fps: int = 15


# world cameras copied from configs/task/HydroShearTrainBase.yaml, at a sane resolution
VIEWS = {
    "front": dict(is_body_camera=False, actor_name=None, attach_link_name=None,
                  camera_pose=[[0.8, 0.0, 0.5],
                               [-2.58819045e-01, 1.58480958e-17, 9.65925826e-01, 5.91458986e-17]]),
    "side":  dict(is_body_camera=False, actor_name=None, attach_link_name=None,
                  camera_pose=[[0.3, -0.6797, 0.7099],
                               [-0.1830127, 0.1830127, 0.6830127, 0.6830127]]),
    "wrist": dict(is_body_camera=True, actor_name="franka", attach_link_name="panda_hand",
                  camera_pose=[[0.045, 0, 0.04], [0.5, 0, 0.866, 0]]),
}


def main():
    args = tyro.cli(Config)
    assert args.view in VIEWS, f"view must be one of {list(VIEWS)}"
    cfg = load_config_from_checkpoint(args.ckpt_path)
    assert cfg is not None, f"no config.yaml beside {args.ckpt_path}"

    cfg.task.env.numEnvs = args.num_envs
    cfg.headless = True
    cfg.force_render = False
    with open_dict(cfg):
        # switching this on is what makes _create_sensors() build the camera actors, and what
        # keeps graphics enabled in headless mode (vec_task.py: enableCameraSensors)
        cfg.task.env.use_camera = True
        cfg.task.env.use_camera_obs = False          # we grab frames ourselves; not policy input
        cfg.task.env.camera_configs = OmegaConf.create([dict(
            name=args.view, image_size=[args.height, args.width], image_type="rgb",
            horizontal_fov=75.0, **VIEWS[args.view])])
    set_seed(cfg.seed)

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=True, virtual_screen_capture=False, force_render=False)
    obs_dict = env.reset()

    agent = eval(cfg.train.algo)(env=env, cfg=cfg, output_dir=os.path.join(args.out, "_agent"))
    agent.load_model(args.ckpt_path)
    agent.set_eval()
    print(f"  camera '{args.view}' {args.height}x{args.width}, {args.num_envs} envs", flush=True)

    frames, dof, episodes, step = [], [], 0, 0
    while episodes < args.episodes:
        fo = {k: v for k, v in obs_dict["obs"].items() if k in agent.actor_network_config.input_shape}
        fs = {k: v for k, v in obs_dict["states"].items() if k in agent.critic_network_config.input_shape}
        mu, _, _, rnn_states = agent.model.act_inference({
            "obs": agent.running_mean_std_obs(fo) if agent.normalize_input else fo,
            "states": agent.running_mean_std_states(fs) if agent.normalize_input else fs,
            "rnn_states": agent.rnn_states if agent.is_rnn else None})
        agent.rnn_states = rnn_states

        if step % args.stride == 0:
            img = env.get_camera_image_tensors_dict()[args.view]      # (E, H, W, 3) uint8
            frames.append(img.detach().cpu().numpy().copy())
            dof.append(float(env.dof_pos[0, 9]) * 1e3 if env.dof_pos.shape[1] > 9 else 0.0)

        obs_dict, rewards, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        step += 1
        # match training: clear recurrent state at the episode boundary
        if agent.is_rnn and bool(torch.as_tensor(done).any()):
            di = torch.as_tensor(done).nonzero(as_tuple=False)
            for st in agent.rnn_states:
                for s in st:
                    s[:, di, :] = 0.0 * s[:, di, :]
        if bool(torch.as_tensor(done).any()) and "successes" in info:
            episodes += 1
            print(f"  episode {episodes}  success {float(info['successes']):.4f}  (step {step})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.basename(args.ckpt_path).replace(".pth", "")
    arr = np.stack(frames)                                            # (T, E, H, W, 3)
    np.save(os.path.join(args.out, f"scene_{stem}_frames.npy"), arr[:, 0])

    # tile the environments side by side so several attempts are visible at once
    tiled = [np.concatenate([f[e] for e in range(min(arr.shape[1], 3))], axis=1) for f in arr]
    try:
        from PIL import Image
        ims = [Image.fromarray(f) for f in tiled]
        gif = os.path.join(args.out, f"scene_{stem}.gif")
        ims[0].save(gif, save_all=True, append_images=ims[1:],
                    duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"\n  saved {gif}  ({len(ims)} frames, {os.path.getsize(gif)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  gif failed ({e}); frames are in the .npy")
    print(f"  drawer travelled {dof[0]:.1f} -> {dof[-1]:.1f} mm over {step} steps")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
