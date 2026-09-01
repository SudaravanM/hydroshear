"""Gate for actor_runtime.py. No throughput number from the actor-enabled harness is
trustworthy until BOTH tests below report PASS.

  Test 1  ACTOR EQUIVALENCE   an identical frozen observation sequence, replayed through the
                              normal evaluator path and through ActorRuntime, must produce the
                              same actions. Catches wrong observation filtering, missing input
                              normalisation, a mis-threaded recurrent state, and a missing or
                              misplaced zero_rnn_on_done.
  Test 2  BEHAVIOURAL SANITY  the checkpoint run through ActorRuntime must reproduce its own
                              recorded success rate within sampling variation.

On failure this script prints diagnostics and exits non-zero. It does NOT adjust tolerances or
retry: a policy-loading or LSTM-reset bug that gets tuned until the numbers look right produces
a fake latency budget, which is worse than no budget.

  python scripts/experiments/hydroshear/validate_actor.py --ckpt <path> --envs 64 --steps 40
"""
import isaacgym  # noqa: F401  MUST precede torch

import argparse
import json
import os
import sys

import numpy as np
import torch
from hydra import initialize_config_dir

from rl.tasks import isaacgym_task_map                              # noqa: E402
from rl.utils.reformat import omegaconf_to_dict                     # noqa: E402
from rl.utils.misc import load_config_from_checkpoint, set_seed     # noqa: E402

from actor_runtime import ActorRuntime                              # noqa: E402

ATOL = 1e-5      # actions are clamped to [-1,1]; identical maths must land far inside this
RTOL = 1e-4


def build(ckpt, num_envs, seed=None):
    cfg = load_config_from_checkpoint(ckpt)
    cfg.task.env.numEnvs = num_envs
    cfg.headless = True
    cfg.force_render = False
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=True, virtual_screen_capture=False, force_render=False)
    return env, cfg


def clone_obs(d):
    return {kk: {k: v.detach().clone() for k, v in d[kk].items()} for kk in ('obs', 'states')}


def test1_equivalence(ckpt, envs, steps):
    """Record a real observation sequence with the INLINE evaluator loop, then replay it
    through ActorRuntime and compare actions step by step."""
    print(f"\n  TEST 1  actor equivalence   envs={envs} steps={steps}", flush=True)
    env, cfg = build(ckpt, envs, seed=1234)
    obs_dict = env.reset()

    rt_ref = ActorRuntime.from_checkpoint(ckpt, env, cfg)
    a = rt_ref.agent

    frozen, ref_actions, dones = [], [], []
    for _ in range(steps):
        frozen.append(clone_obs(obs_dict))
        # --- inline, exactly as eval_ckpt.py / ppo.test() writes it ---
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
        act = torch.clamp(mu, -1.0, 1.0)
        ref_actions.append(act.detach().clone())
        obs_dict, _, done, _ = env.step(act)
        d = torch.as_tensor(done)
        dones.append(d.detach().clone())
        if a.is_rnn and bool(d.any()):
            di = d.nonzero(as_tuple=False)
            for st in a.rnn_states:
                for s in st:
                    s[:, di, :] = 0.0 * s[:, di, :]

    # --- replay the SAME sequence through ActorRuntime, fresh recurrent state ---
    rt = ActorRuntime.from_checkpoint(ckpt, env, cfg)
    worst, worst_i = 0.0, -1
    for i, (od, ref) in enumerate(zip(frozen, ref_actions)):
        got = rt.act(od)
        e = float((got - ref).abs().max())
        if e > worst:
            worst, worst_i = e, i
        rt.reset_done(dones[i])

    n_boundaries = int(sum(bool(torch.as_tensor(x).any()) for x in dones))
    exercised = rt.stats()["rnn_resets"] > 0
    ok = (worst <= ATOL) and exercised
    print(f"    max |a_harness - a_evaluator| = {worst:.3e}   (tol {ATOL:.1e})")
    print(f"    worst step {worst_i}   rnn_resets {rt.stats()['rnn_resets']} "
          f"(reference applied {n_boundaries})")
    if not exercised:
        print(f"    INCONCLUSIVE: no episode boundary in {steps} steps, so zero_rnn_on_done was")
        print(f"    never executed. max_episode_length is 256; run at least that many steps or")
        print(f"    this test says nothing about the reset path -- the exact path that produced")
        print(f"    the 0.497-vs-0.9365 error. Treated as FAIL.")
    print(f"    is_rnn={rt.stats()['is_rnn']}  normalize_input={rt.stats()['normalize_input']}")
    if not ok:
        print("    FAIL  the harness actor does not reproduce the evaluator.")
        print("    Do NOT benchmark. Likely causes, in order: observation filtering against")
        print("    input_shape, running_mean_std applied to the wrong dict, recurrent state not")
        print("    threaded back, or zero_rnn_on_done applied at the wrong point in the loop.")
    return ok, {"max_abs_action_diff": worst, "worst_step": worst_i,
                "episode_boundaries": n_boundaries, "reset_path_exercised": exercised,
                **rt.stats()}


def test2_behaviour(ckpt, envs, episodes, expected, band):
    """Run the checkpoint through ActorRuntime and reproduce its own success rate."""
    print(f"\n  TEST 2  behavioural sanity   envs={envs} episodes={episodes}", flush=True)
    env, cfg = build(ckpt, envs, seed=4321)
    obs_dict = env.reset()
    rt = ActorRuntime.from_checkpoint(ckpt, env, cfg)

    rates, step = [], 0
    while len(rates) < episodes and step < episodes * 2000:
        act = rt.act(obs_dict)
        obs_dict, _, done, info = env.step(act)
        step += 1
        rt.reset_done(done)
        if bool(torch.as_tensor(done).any()) and 'successes' in info:
            r = float(info['successes'])
            rates.append(r)
            print(f"    episode {len(rates):3d}   success {r:.4f}   (step {step})", flush=True)

    arr = np.array(rates)
    mean = float(arr.mean()) if len(arr) else float('nan')
    print(f"    mean success {mean:.4f} +/- {float(arr.std()):.4f} over {len(arr)} episodes")
    if expected is None:
        print("    no --expected given; recording only, NOT a pass")
        return None, {"episodes": rates, "mean": mean, "expected": None}
    ok = abs(mean - expected) <= band
    print(f"    expected {expected:.4f}  band +/-{band:.4f}   -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        first = rates[0] if rates else float('nan')
        print(f"    first episode {first:.4f}, later mean "
              f"{float(np.mean(rates[1:])) if len(rates) > 1 else float('nan'):.4f}")
        print("    If episode 1 is healthy and later episodes are ~half, the recurrent reset is")
        print("    not taking effect -- that is the exact signature of the known bug.")
    return ok, {"episodes": rates, "mean": mean, "expected": expected, "band": band}


def child(a, which):
    """One test per process. Isaac Gym does not tolerate building a second environment after
    the first is torn down (f4_harness.py relies on the same rule): doing so exits the process
    mid-test and the run silently reports partial results."""
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "--one", "--test", str(which),
           "--ckpt", a.ckpt, "--envs", str(a.envs), "--steps", str(a.steps),
           "--episodes", str(a.episodes), "--band", str(a.band)]
    if a.expected is not None:
        cmd += ["--expected", str(a.expected)]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=a.timeout, cwd=os.getcwd())
    for line in p.stdout.splitlines():
        if line.startswith("VAL_JSON "):
            return json.loads(line[len("VAL_JSON "):]), p.stdout
    tail = "; ".join((p.stderr or "").strip().splitlines()[-4:])
    return None, f"rc={p.returncode} {tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--test", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--expected", type=float, default=None,
                    help="known success rate for this checkpoint")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--out", default="./validate_actor.json")
    ap.add_argument("--only", default="1,2")
    a = ap.parse_args()

    if a.one:
        if a.test == 1:
            ok, d = test1_equivalence(a.ckpt, a.envs, a.steps)
        else:
            ok, d = test2_behaviour(a.ckpt, a.envs, a.episodes, a.expected, a.band)
        print("VAL_JSON " + json.dumps({"pass": (None if ok is None else bool(ok)), **d}),
              flush=True)
        sys.stdout.flush()
        os._exit(0)

    print("=" * 74)
    print("  ACTOR VALIDATION GATE -- no throughput number is valid until both PASS")
    print("=" * 74)
    res, allok = {}, True
    which = [x.strip() for x in a.only.split(",")]

    if "1" in which:
        d, out = child(a, 1)
        if d is None:
            print(f"  TEST 1 could not run: {out}")
            json.dump({"test1_equivalence": {"pass": False, "error": out}},
                      open(a.out, "w"), indent=2)
            sys.exit(2)
        print(out.rstrip())
        res["test1_equivalence"] = d
        allok = allok and bool(d.get("pass"))
        if not d.get("pass"):
            print("\n  STOPPING after Test 1 failure. Test 2 would not be interpretable.")
            json.dump(res, open(a.out, "w"), indent=2)
            sys.exit(2)

    if "2" in which:
        d, out = child(a, 2)
        if d is None:
            print(f"  TEST 2 could not run: {out}")
            res["test2_behaviour"] = {"pass": False, "error": out}
            allok = False
        else:
            print(out.rstrip())
            res["test2_behaviour"] = d
            if d.get("pass") is not None:
                allok = allok and bool(d["pass"])

    res["gate"] = "PASS" if allok else "FAIL"
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n  GATE: {res['gate']}      written {a.out}")
    print("  " + ("Actor mode may now be benchmarked." if allok else
                  "Do NOT benchmark. Leave diagnostics for review; do not tune to pass."))
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
