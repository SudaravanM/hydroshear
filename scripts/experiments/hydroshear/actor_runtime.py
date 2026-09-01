"""The real H0 actor, extracted verbatim from eval_ckpt.py so the F4 harness measures the
policy that actually trained -- tactile CNNs, LSTM and all.

Why a shared module rather than a second implementation: the recurrent-reset bug (eval_ckpt.py
lines 84-95) cost us a whole set of evaluation numbers once, because ppo.test() omits the
`zero_rnn_on_done` that the training rollout performs. A hand-written copy of the loop inside
the harness would be a second chance to make exactly that mistake, and a throughput number
carrying it would look perfectly healthy. So the loop lives once, here, and validate_actor.py
proves this module agrees with the evaluator it was extracted from.

NOT a benchmark. Timing produced with this module is only trustworthy once validate_actor.py
reports PASS on both tests.
"""
import os

import torch


class ActorRuntime:
    """obs -> tactile CNN -> recurrent policy -> action, with per-env recurrent reset."""

    def __init__(self, agent):
        self.agent = agent
        self.n_resets = 0
        self.n_acts = 0

    @classmethod
    def from_checkpoint(cls, ckpt_path, env, cfg, out_dir=None):
        # import paths verbatim from eval_ckpt.py / play_hydroshear.py; resolved by NAME rather
        # than eval() so a config naming an algo we did not import fails loudly here instead of
        # raising NameError somewhere further down.
        from rl.algo.ppo.ppo import PPO                               # noqa: F401
        algos = {"PPO": PPO}
        try:
            from rl.algo.ppo.distillation_ppo import DistillationPPO  # noqa: F401
            algos["DistillationPPO"] = DistillationPPO
        except Exception:
            pass
        try:
            from rl.algo.ppo.distill_ppo import DistillPPO            # noqa: F401
            algos["DistillPPO"] = DistillPPO
        except Exception:
            pass
        name = cfg.train.algo
        if name not in algos:
            raise RuntimeError(f"cfg.train.algo={name!r} is not one of {sorted(algos)}; "
                               f"add its import here rather than widening the lookup")
        out_dir = out_dir or os.path.join(
            os.environ.get("RL_OUTPUT_PATH", "./outputs"), "0_hydroshear", "f4")
        agent = algos[name](env=env, cfg=cfg, output_dir=out_dir)
        agent.load_model(ckpt_path)
        agent.set_eval()
        return cls(agent)

    def act(self, obs_dict):
        """Mirrors ppo.test() / eval_ckpt.py exactly. Deterministic: returns mu, not a sample."""
        a = self.agent
        fo = {k: v for k, v in obs_dict['obs'].items()
              if k in a.actor_network_config.input_shape}
        fs = {k: v for k, v in obs_dict['states'].items()
              if k in a.critic_network_config.input_shape}
        input_dict = {
            'obs': a.running_mean_std_obs(fo) if a.normalize_input else fo,
            'states': a.running_mean_std_states(fs) if a.normalize_input else fs,
            'rnn_states': a.rnn_states if a.is_rnn else None,
        }
        mu, _, _, rnn_states = a.model.act_inference(input_dict)
        a.rnn_states = rnn_states
        self.n_acts += 1
        return torch.clamp(mu, -1.0, 1.0)

    def reset_done(self, done):
        """Zero the recurrent state of every env that just finished.

        The training rollout (ppo.py play_steps, zero_rnn_on_done) always does this for an RNN
        policy; ppo.test() does not. Omitting it leaks hidden state across the episode boundary
        and every episode after the first is evaluated from a corrupted state -- 0.945 then
        0.383-0.492 on the drawer student. The checkpoint's own best_sr was measured WITH the
        reset, so without it the two are not comparable.
        """
        a = self.agent
        if not a.is_rnn:
            return 0
        d = torch.as_tensor(done)
        if not bool(d.any()):
            return 0
        done_idx = d.nonzero(as_tuple=False)
        for states_tuple in a.rnn_states:          # actor_states, critic_states
            for s in states_tuple:                 # h, c for LSTM
                s[:, done_idx, :] = 0.0 * s[:, done_idx, :]
        self.n_resets += 1
        return int(d.sum())

    def rnn_snapshot(self):
        """Flat clone of the recurrent state, for equivalence testing."""
        a = self.agent
        if not a.is_rnn or a.rnn_states is None:
            return None
        return [s.detach().clone() for tup in a.rnn_states for s in tup]

    def stats(self):
        return {"acts": self.n_acts, "rnn_resets": self.n_resets,
                "is_rnn": bool(self.agent.is_rnn),
                "normalize_input": bool(self.agent.normalize_input)}
