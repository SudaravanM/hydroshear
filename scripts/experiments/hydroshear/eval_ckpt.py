"""Bounded, SCORED evaluation of a HydroShear checkpoint.

Their play_hydroshear.py runs `while True` and never scores: the success-rate print in
ppo.test() is commented out, and the 0.94 in the filename is a TRAINING-time number, not
something the play script reproduces. This runs a fixed number of episodes and reports the
success rate using THEIR criterion, extras['successes'], which the task publishes on the
final step of each episode as the mean over envs.

It also raises numEnvs: play forces numEnvs=1, so one episode is a single 0/1 sample. With
128 envs each episode is a 128-sample estimate.

  python eval_ckpt.py --ckpt_path <path> --episodes 5 --num-envs 128 --headless
"""
import isaacgym  # noqa: F401  MUST precede torch; the bindings patch tensor interop

import os
import sys
from dataclasses import dataclass
from hydra import initialize_config_dir
import tyro
import numpy as np
import torch

# import paths taken verbatim from scripts/experiments/hydroshear/play_hydroshear.py
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


@dataclass
class Config:
    ckpt_path: str
    episodes: int = 5
    num_envs: int = 128
    headless: bool = True


def main():
    args = tyro.cli(Config)
    cfg = load_config_from_checkpoint(args.ckpt_path)
    cfg.task.env.numEnvs = args.num_envs
    cfg.headless = args.headless
    cfg.force_render = not cfg.headless
    set_seed(cfg.seed)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless, virtual_screen_capture=False, force_render=cfg.force_render)
    obs_dict = env.reset()

    out_dir = os.path.join(os.environ.get("RL_OUTPUT_PATH", "./outputs"), "0_hydroshear", "eval")
    agent = eval(cfg.train.algo)(env=env, cfg=cfg, output_dir=out_dir)
    agent.load_model(args.ckpt_path)
    agent.set_eval()

    print(f"\n  algo={cfg.train.algo}  task={cfg.task_name}  envs={args.num_envs}  "
          f"episodes={args.episodes}", flush=True)

    rates, step = [], 0
    while len(rates) < args.episodes:
        # mirrors ppo.test() exactly
        fo = {k: v for k, v in obs_dict['obs'].items() if k in agent.actor_network_config.input_shape}
        fs = {k: v for k, v in obs_dict['states'].items() if k in agent.critic_network_config.input_shape}
        input_dict = {
            'obs': agent.running_mean_std_obs(fo) if agent.normalize_input else fo,
            'states': agent.running_mean_std_states(fs) if agent.normalize_input else fs,
            'rnn_states': agent.rnn_states if agent.is_rnn else None,
        }
        mu, _, _, rnn_states = agent.model.act_inference(input_dict)
        agent.rnn_states = rnn_states
        obs_dict, rewards, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        step += 1
        # Zero the recurrent state of every env that just finished, exactly as the training
        # rollout does (ppo.py play_steps, `zero_rnn_on_done`, always true for an RNN policy).
        # ppo.test() omits this, so hidden state leaks across the episode boundary and every
        # episode after the first is evaluated from a corrupted state: on the drawer student
        # that read 0.945 for episode 1 and 0.383 to 0.492 for episodes 2 to 8. The checkpoint's
        # own best_sr was measured WITH this reset, so without it the two are not comparable.
        if agent.is_rnn and bool(torch.as_tensor(done).any()):
            done_idx = torch.as_tensor(done).nonzero(as_tuple=False)
            for states_tuple in agent.rnn_states:      # actor_states, critic_states
                for s in states_tuple:                 # h, c for LSTM
                    s[:, done_idx, :] = 0.0 * s[:, done_idx, :]
        # extras is a PERSISTENT dict: once 'successes' is written at the episode's final
        # step it stays there on every later step. Keying on its presence counts one episode
        # many times. Episode length is constant across envs, so `done` marks the real boundary.
        if bool(torch.as_tensor(done).any()) and 'successes' in info:
            r = float(info['successes'])
            rates.append(r)
            print(f"  episode {len(rates):3d}  success rate {r:.4f}   (step {step})", flush=True)

    a = np.array(rates)
    print(f"\n  === RESULT over {len(a)} episodes x {args.num_envs} envs ===")
    print(f"  mean success rate : {a.mean():.4f}")
    print(f"  per-episode       : {np.array2string(a, precision=3)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
