"""Extract the HydroShear two-gel geometry CONTRACT from a live Isaac environment.

Why live rather than URDF arithmetic: reconstructing the final gel poses from the URDF chain plus
a separately-read gripper DOF re-derives what Isaac already computed, and any mismatch (asset
scaling, an extra transform, the actual runtime DOF) would be silent. Dumping the world poses
Isaac actually instantiated, then composing them relative to the handle, folds in the joint chain,
the current DOF, asset transforms, rotations and scaling automatically.

Reference frame definition: the state after reset and after the task's own gripper-closing phase
(num_gripper_move_sim_steps), BEFORE randomized perturbation -- i.e. the state drawer pulling
begins from. If contact is not established there, we advance to the first frame where BOTH gels
report contact and RECORD THAT CHOICE explicitly in the output.

Writes handle-relative transforms T_H^L and T_H^R, which are what the FLASH two-gel scene should
be built from.
"""
import isaacgym  # noqa: F401  MUST precede torch

import argparse, json, os, sys
import numpy as np
import torch
from hydra import initialize_config_dir
from isaacgym import torch_utils

from rl.tasks import isaacgym_task_map                              # noqa: E402
from rl.utils.reformat import omegaconf_to_dict                     # noqa: E402
from rl.utils.misc import load_config_from_checkpoint, set_seed     # noqa: E402


def q2R(q):
    """xyzw -> 3x3, matching isaacgym's convention."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def rel(hq, hp, bq, bp):
    """T_H^B = (T_W^H)^-1 T_W^B, using the codebase's own tf helpers."""
    iq, ip = torch_utils.tf_inverse(hq, hp)
    return torch_utils.tf_combine(iq, ip, bq, bp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="H0 checkpoint; its config defines the env")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--max-extra-steps", type=int, default=60)
    ap.add_argument("--out", default="/home/sudhirmts/two_gel_geometry.json")
    a = ap.parse_args()

    cfg = load_config_from_checkpoint(a.ckpt)
    cfg.task.env.numEnvs = a.envs
    cfg.headless = True; cfg.force_render = False
    cfg.seed = set_seed(cfg.seed)
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device, sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id, headless=True,
            virtual_screen_capture=False, force_render=False)
    env.reset()

    hid = env.drawer_handle_body_id_env
    lid = env.franka_body_ids_env["elastomer_left"]
    rid = env.franka_body_ids_env["elastomer_right"]

    def contact_norms():
        cf = env.contact_force_pairwise
        return (cf[:, hid, lid].norm(dim=-1), cf[:, hid, rid].norm(dim=-1))

    zero = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    ln, rn = contact_norms()
    chosen, extra = "post_reset_pre_perturbation", 0
    while extra < a.max_extra_steps and not bool(((ln > 1e-3) & (rn > 1e-3)).all()):
        env.step(zero); extra += 1
        ln, rn = contact_norms()
        chosen = "first_frame_with_bilateral_contact"
    print("  reference frame: %s  (extra steps advanced: %d)" % (chosen, extra))
    print("  contact |F| left  %s" % ln.tolist())
    print("  contact |F| right %s" % rn.tolist())

    hp, hq = env.body_pos[:, hid, 0:3], env.body_quat[:, hid, 0:4]
    lp, lq = env.body_pos[:, lid, 0:3], env.body_quat[:, lid, 0:4]
    rp, rq = env.body_pos[:, rid, 0:3], env.body_quat[:, rid, 0:4]
    Lq, Lp = rel(hq, hp, lq, lp)
    Rq, Rp = rel(hq, hp, rq, rp)

    e = 0
    print("\n  === HANDLE-RELATIVE TRANSFORMS, env %d (THE CONTRACT) ===" % e)
    print("  T_H^L  pos %s" % np.round(Lp[e].cpu().numpy(), 6).tolist())
    print("         quat(xyzw) %s" % np.round(Lq[e].cpu().numpy(), 6).tolist())
    print("  T_H^R  pos %s" % np.round(Rp[e].cpu().numpy(), 6).tolist())
    print("         quat(xyzw) %s" % np.round(Rq[e].cpu().numpy(), 6).tolist())
    sep = float(torch.norm(Lp[e] - Rp[e]))
    print("  gel separation |L-R| = %.6f m" % sep)

    RL, RR = q2R(Lq[e].cpu().numpy()), q2R(Rq[e].cpu().numpy())
    Rrel = RR.T @ RL
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    print("\n  === LEFT->RIGHT ROTATION (URDF predicts pi about Y) ===")
    print("  relative rotation angle: %.3f deg  (expect ~180)" % ang)
    print("  L local axes (cols x,y,z):\n%s" % np.round(RL, 4))
    print("  R local axes (cols x,y,z):\n%s" % np.round(RR, 4))

    dofs = env.dof_pos[e].cpu().numpy().tolist() if hasattr(env, "dof_pos") else None
    out = {
        "reference_frame": chosen, "extra_steps": extra, "num_envs": env.num_envs,
        "body_ids": {"handle": int(hid), "elastomer_left": int(lid), "elastomer_right": int(rid)},
        "world": {k: {"pos": v[0][e].cpu().numpy().tolist(), "quat_xyzw": v[1][e].cpu().numpy().tolist()}
                  for k, v in {"handle": (hp, hq), "left": (lp, lq), "right": (rp, rq)}.items()},
        "handle_relative": {
            "T_H_left":  {"pos": Lp[e].cpu().numpy().tolist(), "quat_xyzw": Lq[e].cpu().numpy().tolist()},
            "T_H_right": {"pos": Rp[e].cpu().numpy().tolist(), "quat_xyzw": Rq[e].cpu().numpy().tolist()}},
        "gel_separation_m": sep,
        "left_to_right_rotation_deg": ang,
        "contact_force_norm": {"left": ln.tolist(), "right": rn.tolist()},
        "dof_pos_env0": dofs,
        "all_envs_agree": {
            "T_H_left_pos_std": Lp.std(dim=0).cpu().numpy().tolist(),
            "T_H_right_pos_std": Rp.std(dim=0).cpu().numpy().tolist()},
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print("\n  written %s" % a.out)
    print("  NOTE: this is the geometry CONTRACT. The FLASH scene must be built from")
    print("  handle_relative, not re-derived from the URDF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
