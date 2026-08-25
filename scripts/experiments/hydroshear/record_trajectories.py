"""Record BACKEND-AGNOSTIC trajectories from a HydroShear drawer policy.

Why this exists
---------------
The tactile field is not primitive data. In HydroShear it is computed, every step, as a pure
function of one transform per finger:

    indenter2elastomer = inv(elastomer_world_pose) o handle_world_pose        (get_indenter2elastomer_tf)

Everything after that (SDF, kernel, marker displacement) is one backend's opinion about contact
physics. So if a dataset stores `tactile_force_field_*`, it is welded to HydroFots forever. If it
stores the TRANSFORM, the tactile channel can be re-rendered later with HydroFots, FOTS, TacSL,
RealSim, or a learned surrogate, and the same trajectories can train an LSTM today or a
transformer tomorrow.

That is the whole point: the trajectories are the asset, not the policy.

Recording the transform also makes the backend comparison paired: the SAME motion rendered two
ways, rather than two separately-generated datasets that differ for uncontrolled reasons.

What is stored, per step, per env
---------------------------------
    rel_pos_left/right   [3]   indenter (handle) position in that elastomer's frame
    rel_quat_left/right  [4]   and its orientation
    handle_pos/quat      [3,4] world, kept so the transform can be re-derived or audited
    elast_l/r_pos/quat   [3,4] world, same reason
    gripper_width        [2]   the franka finger DOFs; the policy's action target
    drawer_dof           [1]   task progress
    actions              [7]
    rewards, dones, successes

Usage
-----
    python record_trajectories.py --ckpt_path <teacher.pth> --episodes 4 --num-envs 128 \
                                  --out ~/hydroshear/outputs/trajectories
"""
import isaacgym  # noqa: F401  MUST precede torch

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from hydra import initialize_config_dir
from isaacgym.torch_utils import tf_combine, tf_inverse

# import paths taken verbatim from scripts/experiments/hydroshear/play_hydroshear.py
from rl.algo.ppo.ppo import PPO                                     # noqa
from rl.tasks import isaacgym_task_map                              # noqa
from rl.utils.reformat import omegaconf_to_dict                     # noqa
from rl.utils.misc import load_config_from_checkpoint, set_seed     # noqa


@dataclass
class Config:
    ckpt_path: str
    episodes: int = 4
    num_envs: int = 128
    out: str = "trajectories"
    headless: bool = True
    overrides: str = ""      # comma-separated key=value applied to the checkpoint config, e.g.
                             # "task.env.use_shear_force=True,task.env.use_hydrosoft_model=True,task.env.use_shear_3d=False"


def main():
    args = tyro.cli(Config)
    cfg = load_config_from_checkpoint(args.ckpt_path)
    cfg.task.env.numEnvs = args.num_envs
    cfg.headless = args.headless
    cfg.force_render = not cfg.headless
    if args.overrides:
        from omegaconf import OmegaConf, open_dict
        with open_dict(cfg):
            for kv in args.overrides.split(","):
                k, v = kv.split("=", 1)
                OmegaConf.update(cfg, k.strip(), OmegaConf.create({"v": v.strip()})["v"], force_add=True)
        print("  overrides:", args.overrides, flush=True)
    set_seed(cfg.seed)

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless, virtual_screen_capture=False, force_render=cfg.force_render)
    obs_dict = env.reset()

    agent = eval(cfg.train.algo)(env=env, cfg=cfg,
                                 output_dir=os.path.join(args.out, "_agent"))
    agent.load_model(args.ckpt_path)
    agent.set_eval()

    # Resolve the two elastomer links once. Same lookup HydroFots uses.
    link_ids = {}
    for name in ("elastomer_left", "elastomer_right"):
        link_ids[name] = env.gym.find_actor_rigid_body_handle(
            env.env_ptrs[0], env.actor_handles["franka"], name)
    print(f"  elastomer link ids: {link_ids}", flush=True)

    # If the env computes a HydroFots field each step, capture the RAW (num_envs, 9, 7, 3) output
    # by wrapping the seam. Never call get_force_fields_dict a second time: get_marker_shear
    # stores prev_sdf / prev_indenter_pts each call, so a re-call corrupts the hysteresis term.
    has_hydro = hasattr(env, "hydrofots_sensors") and hasattr(env, "get_force_fields_dict")
    if has_hydro:
        _orig = env.get_force_fields_dict
        def _wrapped(*a, **k):
            out = _orig(*a, **k); env._last_fields = out; return out
        env.get_force_fields_dict = _wrapped
        print(f"  HydroFots present: {len(env.hydrofots_sensors)} sensors; raw fields will be recorded", flush=True)

    def contact_geometry(i):
        """Depth and tangential centroid of the handle inside elastomer i, from THEIR SDF.

        After each step the sensor holds this step's sdf over the sampled indenter points
        (prev_sdf) and those points in the elastomer frame (prev_indenter_pts_in_elastomer).
        depth = max penetration = relu(-min sdf); centroid = mean of penetrating points' xy.
        """
        sen = env.hydrofots_sensors[i]
        sdf = sen.prev_sdf; pts = sen.prev_indenter_pts_in_elastomer
        if sdf is None or pts is None:
            z = np.zeros((env.num_envs, 1), np.float32)
            return z, np.zeros((env.num_envs, 2), np.float32), z.astype(np.int32)
        pen = (-sdf).clamp(min=0.0)                                   # (E, P)
        depth = pen.max(dim=1).values.unsqueeze(1)                    # (E, 1)
        inside = (sdf < 0).float()                                    # (E, P)
        n = inside.sum(dim=1, keepdim=True)
        cen = (pts[..., :2] * inside.unsqueeze(-1)).sum(dim=1) / n.clamp(min=1)   # (E, 2)
        return n2(depth), n2(cen), n.squeeze(1).detach().cpu().numpy().astype(np.int32)

    def relative_tf():
        """inv(elastomer_pose) o handle_pose, per finger. Mirrors get_indenter2elastomer_tf."""
        h_q, h_p = env.drawer_handle_quat, env.drawer_handle_pos
        out = {}
        for name, lid in link_ids.items():
            e_q = env.body_quat[:, lid].expand(env.num_envs, 4)
            e_p = env.body_pos[:, lid].expand(env.num_envs, 3)
            inv_q, inv_p = tf_inverse(e_q, e_p)
            r_q, r_p = tf_combine(inv_q, inv_p, h_q, h_p)
            out[name] = (r_q.clone(), r_p.clone(), e_q.clone(), e_p.clone())
        return out

    rec = {k: [] for k in (
        "rel_pos_left", "rel_quat_left", "rel_pos_right", "rel_quat_right",
        "elast_left_pos", "elast_left_quat", "elast_right_pos", "elast_right_quat",
        "handle_pos", "handle_quat", "gripper_width", "drawer_dof",
        "actions", "rewards", "dones", "tactile_left", "tactile_right",
        "hydro_raw_left", "hydro_raw_right", "depth_left", "depth_right",
        "contact_xy_left", "contact_xy_right", "n_pen_left", "n_pen_right")}
    successes, episodes, step = [], 0, 0
    n2 = lambda t: t.detach().cpu().numpy().astype(np.float32)

    while episodes < args.episodes:
        fo = {k: v for k, v in obs_dict["obs"].items() if k in agent.actor_network_config.input_shape}
        fs = {k: v for k, v in obs_dict["states"].items() if k in agent.critic_network_config.input_shape}
        input_dict = {
            "obs": agent.running_mean_std_obs(fo) if agent.normalize_input else fo,
            "states": agent.running_mean_std_states(fs) if agent.normalize_input else fs,
            "rnn_states": agent.rnn_states if agent.is_rnn else None,
        }
        mu, _, _, rnn_states = agent.model.act_inference(input_dict)
        agent.rnn_states = rnn_states
        actions = torch.clamp(mu, -1.0, 1.0)

        tf = relative_tf()
        rec["rel_quat_left"].append(n2(tf["elastomer_left"][0]))
        rec["rel_pos_left"].append(n2(tf["elastomer_left"][1]))
        rec["elast_left_quat"].append(n2(tf["elastomer_left"][2]))
        rec["elast_left_pos"].append(n2(tf["elastomer_left"][3]))
        rec["rel_quat_right"].append(n2(tf["elastomer_right"][0]))
        rec["rel_pos_right"].append(n2(tf["elastomer_right"][1]))
        rec["elast_right_quat"].append(n2(tf["elastomer_right"][2]))
        rec["elast_right_pos"].append(n2(tf["elastomer_right"][3]))
        rec["handle_pos"].append(n2(env.drawer_handle_pos))
        rec["handle_quat"].append(n2(env.drawer_handle_quat))
        rec["gripper_width"].append(n2(env.dof_pos[:, 7:9]))
        rec["drawer_dof"].append(n2(env.dof_pos[:, 9:10]))
        rec["actions"].append(n2(actions))

        # The backend's own field for THIS step, if the env computes one (student checkpoints do,
        # teachers do not). Stored in addition to the geometry, never instead of it: it is what
        # makes a backend comparison paired, the same trajectory rendered two ways.
        for k, key in (("tactile_left", "tactile_force_field_left"), ("tactile_right", "tactile_force_field_right")):
            rec[k].append(n2(obs_dict["obs"][key]) if key in obs_dict["obs"] else np.zeros((env.num_envs, 0), np.float32))
        obs_dict, rewards, done, info = env.step(actions)
        # the backend field and contact geometry that belong to the NEW state (computed in
        # compute_observations during this step)
        if has_hydro and getattr(env, "_last_fields", None) is not None:
            rec["hydro_raw_left"].append(n2(env._last_fields["elastomer_left"]))
            rec["hydro_raw_right"].append(n2(env._last_fields["elastomer_right"]))
            for i, f in ((0, "left"), (1, "right")):
                d, c, nn_ = contact_geometry(i)
                rec[f"depth_{f}"].append(d); rec[f"contact_xy_{f}"].append(c); rec[f"n_pen_{f}"].append(nn_)
        else:
            for k in ("hydro_raw_left", "hydro_raw_right", "depth_left", "depth_right",
                      "contact_xy_left", "contact_xy_right", "n_pen_left", "n_pen_right"):
                rec[k].append(np.zeros((env.num_envs, 0), np.float32))
        rec["rewards"].append(n2(rewards))
        rec["dones"].append(done.detach().cpu().numpy().astype(np.uint8))
        step += 1
        if bool(torch.as_tensor(done).any()) and "successes" in info:
            successes.append(float(info["successes"]))
            episodes += 1
            print(f"  episode {episodes}/{args.episodes}  success {successes[-1]:.4f}  "
                  f"(step {step})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stacked = {k: np.stack(v) for k, v in rec.items()}   # (T, num_envs, ...)
    meta = dict(
        ckpt=args.ckpt_path, task=cfg.task_name, num_envs=args.num_envs,
        episodes=args.episodes, steps=step, successes=successes,
        tactile_stored=bool(stacked["tactile_left"].shape[-1] > 0),
        hydro_raw_stored=bool(stacked["hydro_raw_left"].shape[-1] > 0),
        overrides=args.overrides,
        note=("geometry is the primary record. rel_{pos,quat}_{left,right} is "
              "inv(elastomer_world) o handle_world, the exact input HydroShear's "
              "get_indenter2elastomer_tf produces, so any backend can render from it. "
              "tactile_{left,right} additionally holds the env's own post-processed field "
              "(the student's observation, [9,7,2]) when the checkpoint's env computes one. "
              "hydro_raw_{left,right} is the RAW HydroFots (9,7,3) field in the elastomer frame, "
              "depth_* the max SDF penetration of the handle into the gel (m), contact_xy_* the "
              "centroid of the penetrating indenter points in the elastomer frame (m): the "
              "renderer's inputs, from the reference backend's own SDF."),
    )
    out = os.path.join(args.out, f"traj_{os.path.basename(args.ckpt_path).replace('.pth','')}.npz")
    np.savez_compressed(out, meta=meta, **stacked)
    mb = os.path.getsize(out) / 1e6
    print(f"\n  saved {out}  ({mb:.1f} MB)")
    print(f"  shapes: T={step} envs={args.num_envs}  "
          f"rel_pos_left {stacked['rel_pos_left'].shape}")
    print(f"  mean success over {len(successes)} episodes: {np.mean(successes):.4f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
