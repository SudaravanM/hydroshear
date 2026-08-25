"""Record BACKEND-AGNOSTIC trajectories from a HydroShear policy.

Why this exists
---------------
The tactile field is not primitive data. In HydroShear it is computed, every step, as a pure
function of one transform per finger:

    indenter2elastomer = inv(elastomer_world_pose) o indenter_world_pose     (get_indenter2elastomer_tf)

Everything after that (SDF, kernel, marker displacement) is one backend's opinion about contact
physics. So if a dataset stores `tactile_force_field_*`, it is welded to HydroFots forever. If it
stores the TRANSFORM, the tactile channel can be re-rendered later with HydroFots, FOTS, TacSL,
RealSim, or a learned surrogate, and the same trajectories can train an LSTM today or a
transformer tomorrow.

That is the whole point: the trajectories are the asset, not the policy.

Recording the transform also makes the backend comparison paired: the SAME motion rendered two
ways, rather than two separately-generated datasets that differ for uncontrolled reasons.

Alignment contract
------------------
Row i of EVERY per-state array describes the same simulator state t_i: the geometry, the world
poses, the joint state, the field the policy observed, the raw backend field, and the SDF
contact geometry. `actions[i]` is the action chosen at t_i; `rewards[i]` and `dones[i]` belong to
the transition t_i -> t_{i+1}. `check_trajectory.py` verifies this on a saved file.

What is stored, per step, per env
---------------------------------
    rel_pos_left/right     [3]     indenter position in that elastomer's frame
    rel_quat_left/right    [4]     and its orientation (xyzw)
    indenter_pos/quat      [3,4]   world; drawer handle, plug, or bin object depending on the task
    elast_l/r_pos/quat     [3,4]   world, kept so the transform can be re-derived or audited
    dof_pos                [D]     all joint positions
    gripper_width          [2]     the franka finger DOFs (dof 7, 8); the policy's action target
    drawer_dof             [1]     task progress (drawer task only; empty otherwise)
    actions                [A]
    rewards, dones
    tactile_left/right     [9,7,2] the field the policy observed, when the env computes one
    hydro_raw_left/right   [9,7,3] the RAW HydroFots field (tangent, tangent, normal), same step
    depth_left/right       [1]     max SDF penetration of the indenter into the gel (m)
    contact_xy_left/right  [2]     centroid of the penetrating indenter points, gel tangent plane (m)
    n_pen_left/right       [1]     number of penetrating indenter points

Usage
-----
    python scripts/experiments/hydroshear/record_trajectories.py --ckpt_path <ckpt.pth> \
        --episodes 4 --num-envs 128 --out ~/hydroshear/outputs/trajectories

    # replay on a machine without a free GPU (Isaac Gym CPU pipeline, model on CPU):
    ... --overrides "sim_device=cpu,rl_device=cpu,pipeline=cpu,task.sim.use_gpu_pipeline=False,task.sim.physx.use_gpu=False"
"""
import isaacgym  # noqa: F401  MUST precede torch

import functools
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
                             # "task.env.use_shear_force=True,task.env.use_hydrosoft_model=True"
                             # (leave use_tactile_field_obs off for a teacher: its obs_dict has no
                             # tactile keys, and the seam is still called without that flag)


def main():
    args = tyro.cli(Config)
    cfg = load_config_from_checkpoint(args.ckpt_path)
    assert cfg is not None, f"no config.yaml beside {os.path.dirname(os.path.dirname(args.ckpt_path))}/nn"
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
    if str(cfg.rl_device).startswith("cpu"):
        # checkpoints were saved from cuda tensors; PPO.load_model calls torch.load without map_location
        torch.load = functools.partial(torch.load, map_location="cpu")
    set_seed(cfg.seed)

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless, virtual_screen_capture=False, force_render=cfg.force_render)

    # If the env computes a HydroFots field, capture the RAW (num_envs, 9, 7, 3) output by wrapping
    # the seam. Installed BEFORE env.reset() so the very first state's field is captured too.
    # Never call get_force_fields_dict a second time: get_marker_shear stores prev_sdf /
    # prev_indenter_pts each call, so a re-call makes prev == current and corrupts the hysteresis
    # term (0/0, silently replaced by a step function).
    has_hydro = hasattr(env, "hydrofots_sensors") and hasattr(env, "get_force_fields_dict")
    if has_hydro:
        _orig = env.get_force_fields_dict
        def _wrapped(*a, **k):
            out = _orig(*a, **k); env._last_fields = out; return out
        env.get_force_fields_dict = _wrapped
        print(f"  HydroFots present: {len(env.hydrofots_sensors)} sensors; raw fields will be recorded", flush=True)

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

    # The indenter is whatever each task passes to the seam: every task exposes it as plug_*
    # (the drawer env aliases its handle to plug_*), bin packing as insertion_obj_*.
    if hasattr(env, "plug_quat"):
        indenter_pose = lambda: (env.plug_quat, env.plug_pos)
    elif hasattr(env, "insertion_obj_quat"):
        indenter_pose = lambda: (env.insertion_obj_quat, env.insertion_obj_pos)
    else:
        raise RuntimeError(f"{cfg.task_name}: no plug_* or insertion_obj_* pose on the env")

    # The gel's tangent plane, from the sensor itself (it derives the normal as the thin axis of
    # the elastomer mesh); do not assume it is xy.
    tangent_axes = []
    if has_hydro:
        for sen in env.hydrofots_sensors:
            tangent_axes.append(list(sen.elastomer_sdf_sensor.get_tangent_axes()))
        print(f"  gel tangent axes per sensor: {tangent_axes}", flush=True)

    n2 = lambda t: t.detach().cpu().numpy().astype(np.float32)

    def contact_geometry(i):
        """Depth and tangential centroid of the indenter inside elastomer i, from THEIR SDF.

        After each seam call the sensor holds that state's sdf over the sampled indenter points
        (prev_sdf) and those points in the elastomer frame (prev_indenter_pts_in_elastomer).
        depth = max penetration = relu(-sdf); centroid = mean of penetrating points' tangent coords.
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
        cen = (pts[..., tangent_axes[i]] * inside.unsqueeze(-1)).sum(dim=1) / n.clamp(min=1)   # (E, 2)
        return n2(depth), n2(cen), n.squeeze(1).detach().cpu().numpy().astype(np.int32)

    def relative_tf():
        """inv(elastomer_pose) o indenter_pose, per finger. Mirrors get_indenter2elastomer_tf."""
        h_q, h_p = indenter_pose()
        out = {}
        for name, lid in link_ids.items():
            e_q = env.body_quat[:, lid].expand(env.num_envs, 4)
            e_p = env.body_pos[:, lid].expand(env.num_envs, 3)
            inv_q, inv_p = tf_inverse(e_q, e_p)
            r_q, r_p = tf_combine(inv_q, inv_p, h_q, h_p)
            out[name] = (r_q.clone(), r_p.clone(), e_q.clone(), e_p.clone())
        return out

    def record_state(rec):
        """Everything that describes the CURRENT simulator state, all from the same step."""
        tf = relative_tf()
        rec["rel_quat_left"].append(n2(tf["elastomer_left"][0]))
        rec["rel_pos_left"].append(n2(tf["elastomer_left"][1]))
        rec["elast_left_quat"].append(n2(tf["elastomer_left"][2]))
        rec["elast_left_pos"].append(n2(tf["elastomer_left"][3]))
        rec["rel_quat_right"].append(n2(tf["elastomer_right"][0]))
        rec["rel_pos_right"].append(n2(tf["elastomer_right"][1]))
        rec["elast_right_quat"].append(n2(tf["elastomer_right"][2]))
        rec["elast_right_pos"].append(n2(tf["elastomer_right"][3]))
        h_q, h_p = indenter_pose()
        rec["indenter_pos"].append(n2(h_p))
        rec["indenter_quat"].append(n2(h_q))
        rec["dof_pos"].append(n2(env.dof_pos))
        rec["gripper_width"].append(n2(env.dof_pos[:, 7:9]))
        rec["drawer_dof"].append(n2(env.dof_pos[:, 9:10]) if env.dof_pos.shape[1] > 9
                                 else np.zeros((env.num_envs, 0), np.float32))
        # the field the policy observed for this state (student checkpoints; teachers have none)
        for k, key in (("tactile_left", "tactile_force_field_left"), ("tactile_right", "tactile_force_field_right")):
            rec[k].append(n2(obs_dict["obs"][key]) if key in obs_dict["obs"] else np.zeros((env.num_envs, 0), np.float32))
        # the backend's raw field and contact geometry for this state: produced by the seam call
        # inside the compute_observations that yielded obs_dict, i.e. the same state as above
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

    rec = {k: [] for k in (
        "rel_pos_left", "rel_quat_left", "rel_pos_right", "rel_quat_right",
        "elast_left_pos", "elast_left_quat", "elast_right_pos", "elast_right_quat",
        "indenter_pos", "indenter_quat", "dof_pos", "gripper_width", "drawer_dof",
        "actions", "rewards", "dones", "tactile_left", "tactile_right",
        "hydro_raw_left", "hydro_raw_right", "depth_left", "depth_right",
        "contact_xy_left", "contact_xy_right", "n_pen_left", "n_pen_right")}
    successes, episodes, step = [], 0, 0

    while episodes < args.episodes:
        # mirrors ppo.test()
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

        record_state(rec)                       # state t_i, before it is advanced
        rec["actions"].append(n2(actions))      # chosen at t_i

        obs_dict, rewards, done, info = env.step(actions)   # transition t_i -> t_{i+1}
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
        tangent_axes=tangent_axes, overrides=args.overrides,
        note=("geometry is the primary record. rel_{pos,quat}_{left,right} is "
              "inv(elastomer_world) o indenter_world, the exact input HydroShear's "
              "get_indenter2elastomer_tf produces, so any backend can render from it. "
              "Row i of every per-state array is the same simulator state; actions[i] was "
              "chosen there; rewards[i]/dones[i] belong to the transition to row i+1. "
              "tactile_{left,right} is the env's own post-processed field (the student's "
              "observation, [9,7,2]) when the checkpoint's env computes one; "
              "hydro_raw_{left,right} the RAW HydroFots (9,7,3) field in the elastomer frame, "
              "channels (tangent, tangent, normal); depth_* the max SDF penetration of the "
              "indenter into the gel (m); contact_xy_* the centroid of the penetrating indenter "
              "points in the gel tangent plane (m): the renderer's inputs, from the reference "
              "backend's own SDF."),
    )
    out = os.path.join(args.out, f"traj_{os.path.basename(args.ckpt_path).replace('.pth','')}.npz")
    np.savez_compressed(out, meta=meta, **stacked)
    mb = os.path.getsize(out) / 1e6
    print(f"\n  saved {out}  ({mb:.1f} MB)")
    print(f"  shapes: T={step} envs={args.num_envs}  "
          f"rel_pos_left {stacked['rel_pos_left'].shape}  hydro_raw_left {stacked['hydro_raw_left'].shape}")
    print(f"  mean success over {len(successes)} episodes: {np.mean(successes):.4f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
