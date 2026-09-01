"""Thread C: the R0 behavioural evaluator. Does a policy actually do adaptive grip modulation?

Success rate cannot answer that. GT-TASK-001: grip is HALF the drawer action space, so a policy
can score well on a width that happens to work. This logs what success hides.

Premises it consumes (see ground_truths/):
  GT-CDESIGN-003  the six quantities and the event definitions, frozen 2026-08-27
  GT-CDESIGN-005  the deterministic scenario suite
  GT-TASK-011     the perturbation is DRAWER-POSITION triggered, threshold ~ U[-0.10,-0.02] per episode
  GT-TASK-012     self.fails is anti-pre-clamp, NOT handle loss
  GT-TASK-013     the task's OWN grip constants: 0.0029 tightened, 0.0015 bonus, range 0.001-0.003
  GT-TASK-015     clamping is unconditionally legal past -0.095 m (the late-clamp window)

Determinism: the per-episode perturbation threshold is normally redrawn at random. This patches
reset_idx to impose a FIXED per-env threshold, which is what makes "early" and "late" scenarios
reproducible. Threshold, not timestamp: GT-TASK-011.

  python eval_behavior.py --ckpt_path <path> --episodes 5 --num-envs 128 --force-amount -2.0
"""
import isaacgym  # noqa: F401  MUST precede torch

import os
import sys
import json
from dataclasses import dataclass
from hydra import initialize_config_dir
import tyro
import numpy as np
import torch

from rl.algo.ppo.ppo import PPO                                        # noqa
from rl.tasks import isaacgym_task_map                                 # noqa
from rl.utils.reformat import omegaconf_to_dict                        # noqa
from rl.utils.misc import load_config_from_checkpoint, set_seed        # noqa
try:
    from rl.algo.ppo.distillation_ppo import DistillationPPO           # noqa
except Exception:
    DistillationPPO = None
try:
    from rl.algo.ppo.distill_ppo import DistillPPO                     # noqa
except Exception:
    DistillPPO = None

# ---------------------------------------------------------------------------------------------
# FROZEN 2026-08-27, BEFORE ANY TRACE WAS VIEWED (GT-CDESIGN-003).
# The first three are the TASK'S OWN constants, lifted from its reward function rather than
# chosen by us, which is a stronger form of "frozen" than a number we picked.
GRIP_TIGHT_THRESH = 0.0029   # drawer_task_pulling.py:671, the pre-perturb fail test
GRIP_BONUS_THRESH = 0.0015   # :675, earns the post-perturb bonus
GRIP_MIN, GRIP_MAX = 0.001, 0.003    # :211, tightest .. loosest
LATE_CLAMP_DOF = -0.095      # GT-TASK-015: past here, clamping can never trip the fail rule
# our own, and stated numerically before any data existed:
TIGHTEN_SUSTAIN_K = 3        # steps the tightening must persist to count as t_tighten
CUE_SIGMA_K = 4.0            # tactile cue = shear rises this many sd above its pre-perturb baseline
CUE_MIN_BASELINE_STEPS = 5   # need this many pre-perturb steps before a cue can be declared
# ---------------------------------------------------------------------------------------------


@dataclass
class Config:
    ckpt_path: str
    episodes: int = 5
    num_envs: int = 128
    headless: bool = True
    force_amount: float = -2.0
    """Perturbation magnitude, N along env-frame y. The suite's light/medium/strong axis."""
    thresholds: str = "sweep"
    """'sweep' spreads the per-env perturbation threshold evenly over [-0.10,-0.02]; or a float."""
    tag: str = "H0"
    out_dir: str = ""
    log_fields: bool = True
    """Log the full policy-consumed 9x7x2 fields per finger per step. Required by GT-CDESIGN-008:
    v1 collapsed them to a scalar at capture time, so the cue family could not be tried offline
    (GT-BEHAV-006 correction)."""


def grip_from_action(a7):
    """The task's own mapping, drawer_task_pulling.py:209-211. a in [-1,1] -> width in [0.001,0.003]."""
    return ((a7 + 1.0) / 2.0) * 0.002 + 0.001


def main():
    args = tyro.cli(Config)
    cfg = load_config_from_checkpoint(args.ckpt_path)
    cfg.task.env.numEnvs = args.num_envs
    cfg.headless = args.headless
    cfg.force_render = not cfg.headless
    cfg.task.randomize.force_perturb_amount = args.force_amount
    cfg.task.randomize.use_randomized_force_perturb = False   # determinism
    set_seed(cfg.seed)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless, virtual_screen_capture=False, force_render=cfg.force_render)

    dev = env.device
    N = args.num_envs
    if args.thresholds == "sweep":
        fixed_thresh = torch.linspace(-0.02, -0.10, N, device=dev)
    else:
        fixed_thresh = torch.full((N,), float(args.thresholds), device=dev)

    # Impose the scenario threshold at every reset. Without this the env redraws it at random
    # (GT-TASK-011) and no two episodes are comparable.
    _orig_reset_idx = env.reset_idx
    def _patched_reset_idx(env_ids):
        _orig_reset_idx(env_ids)
        if hasattr(env, "dof_when_force_perturb"):
            env.dof_when_force_perturb[env_ids] = fixed_thresh[env_ids]
    env.reset_idx = _patched_reset_idx

    obs_dict = env.reset()
    if hasattr(env, "dof_when_force_perturb"):
        env.dof_when_force_perturb[:] = fixed_thresh

    out_dir = args.out_dir or os.path.join(
        os.environ.get("RL_OUTPUT_PATH", "./outputs"), "0_hydroshear", "threadC")
    os.makedirs(out_dir, exist_ok=True)
    agent = eval(cfg.train.algo)(env=env, cfg=cfg, output_dir=out_dir)
    agent.load_model(args.ckpt_path)
    agent.set_eval()

    eL = env.franka_body_ids_env['elastomer_left']
    eR = env.franka_body_ids_env['elastomer_right']
    hB = env.drawer_handle_body_id_env

    print(f"\n  THREAD C behavioural eval   tag={args.tag}  algo={cfg.train.algo}  envs={N}  "
          f"episodes={args.episodes}  force={args.force_amount} N", flush=True)
    print(f"  thresholds: {fixed_thresh.min():.4f} .. {fixed_thresh.max():.4f} m "
          f"({'swept' if args.thresholds=='sweep' else 'fixed'})", flush=True)

    LOG = {k: [] for k in ("dof", "grip_cmd", "fnL", "fnR", "contact", "shear", "fails", "prog")}
    FLD = {"fieldL": [], "fieldR": []}   # full 9x7x2, the input the cue family operates on
    rates, ep_bounds, step = [], [], 0

    while len(rates) < args.episodes:
        fo = {k: v for k, v in obs_dict['obs'].items() if k in agent.actor_network_config.input_shape}
        fs = {k: v for k, v in obs_dict['states'].items() if k in agent.critic_network_config.input_shape}
        input_dict = {
            'obs': agent.running_mean_std_obs(fo) if agent.normalize_input else fo,
            'states': agent.running_mean_std_states(fs) if agent.normalize_input else fs,
            'rnn_states': agent.rnn_states if agent.is_rnn else None,
        }
        mu, _, _, rnn_states = agent.model.act_inference(input_dict)
        agent.rnn_states = rnn_states
        act = torch.clamp(mu, -1.0, 1.0)

        # --- log BEFORE stepping: state the action was chosen from, plus the commanded grip ---
        cf = env.contact_force_pairwise
        fL = cf[:, eL, hB].norm(dim=-1)
        fR = cf[:, eR, hB].norm(dim=-1)
        # handle contact, using the env's OWN test (drawer_task_pulling.py:143-144)
        in_contact = (cf[:, hB, eL].norm(dim=-1) >= 1e-3) & (cf[:, hB, eR].norm(dim=-1) >= 1e-3)
        # C0, the KNOWN-FAILING control cue (GT-BEHAV-006): mean per-taxel magnitude.
        sh = torch.zeros(N, device=dev)
        for k in ('tactile_force_field_left', 'tactile_force_field_right'):
            if k in obs_dict['obs']:
                t = obs_dict['obs'][k]
                sh = sh + t.reshape(N, -1, t.shape[-1]).norm(dim=-1).mean(dim=-1)
        if args.log_fields:
            for k, dst in (('tactile_force_field_left', 'fieldL'),
                           ('tactile_force_field_right', 'fieldR')):
                if k in obs_dict['obs']:
                    FLD[dst].append(obs_dict['obs'][k].detach().to(torch.float32).cpu())
        LOG["dof"].append(env.dof_pos[:, 9].clone())
        LOG["grip_cmd"].append(grip_from_action(act[:, 6]).clone())
        LOG["fnL"].append(fL.clone()); LOG["fnR"].append(fR.clone())
        LOG["contact"].append(in_contact.clone())
        LOG["shear"].append(sh.clone())
        LOG["fails"].append(env.fails.clone() if hasattr(env, "fails") else torch.zeros(N, dtype=torch.bool, device=dev))
        LOG["prog"].append(env.progress_buf.clone())

        obs_dict, rewards, done, info = env.step(act)
        step += 1
        if agent.is_rnn and bool(torch.as_tensor(done).any()):
            di = torch.as_tensor(done).nonzero(as_tuple=False)
            for st in agent.rnn_states:
                for x in st:
                    x[:, di, :] = 0.0 * x[:, di, :]
        if bool(torch.as_tensor(done).any()) and 'successes' in info:
            rates.append(float(info['successes']))
            ep_bounds.append(step)
            print(f"  episode {len(rates):3d}  success {rates[-1]:.4f}  (step {step})", flush=True)

    D = {k: torch.stack(v).cpu().numpy() for k, v in LOG.items()}   # [T, N]
    for k, v in FLD.items():
        if v:
            D[k] = torch.stack(v).numpy()          # [T, N, 9, 7, 2]
            print(f"  {k}: {D[k].shape}  {D[k].nbytes/1e6:.0f} MB uncompressed")
    np.savez_compressed(
        os.path.join(out_dir, f"trace_{args.tag}_f{abs(args.force_amount):g}.npz"),
        thresh=fixed_thresh.cpu().numpy(), ep_bounds=np.array(ep_bounds),
        rates=np.array(rates), force_amount=args.force_amount, tag=args.tag,
        constants=json.dumps(dict(
            GRIP_TIGHT_THRESH=GRIP_TIGHT_THRESH, GRIP_BONUS_THRESH=GRIP_BONUS_THRESH,
            GRIP_MIN=GRIP_MIN, GRIP_MAX=GRIP_MAX, LATE_CLAMP_DOF=LATE_CLAMP_DOF,
            TIGHTEN_SUSTAIN_K=TIGHTEN_SUSTAIN_K, CUE_SIGMA_K=CUE_SIGMA_K,
            CUE_MIN_BASELINE_STEPS=CUE_MIN_BASELINE_STEPS)),
        **D)
    print(f"\n  wrote {out_dir}/trace_{args.tag}_f{abs(args.force_amount):g}.npz   "
          f"shape T={D['dof'].shape[0]} N={D['dof'].shape[1]}"
          f"{'  WITH FIELDS' if args.log_fields else ''}")
    print(f"  mean success {np.mean(rates):.4f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
