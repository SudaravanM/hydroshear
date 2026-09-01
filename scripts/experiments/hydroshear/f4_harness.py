"""F4: end-to-end agent-steps/s, and the LATENCY BUDGET a tactile backend must meet.

The only measurement that chooses the architecture (GT-DESIGN-003): not solver throughput, but
agent transitions per second summed over environments, through the real Isaac seam.

Three arms:
    A  Isaac, tactile disabled          the floor: what the world costs without touch at all
    B  Isaac + HydroFots                the reference the direct backend must approach
    C  Isaac + a latency-parameterised stand-in

Arm C is why this is worth building before RealSim is stable. Sweeping the stand-in's latency and
measuring end-to-end throughput yields the BUDGET -- how many ms per control step a tactile backend
may spend for a given agent-steps/s target. F3's per-substep numbers are then judged against a
measured budget instead of an assumed one, and the budget is useful whichever way the
direct-vs-surrogate decision goes.

  python scripts/experiments/hydroshear/f4_harness.py --arms A,B --steps 200
  python scripts/experiments/hydroshear/f4_harness.py --arms C --latency 0,5,20,50,100 --steps 200
"""
import isaacgym  # noqa: F401  MUST precede torch

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from rl.tasks import isaacgym_task_map                 # noqa: E402
from rl.utils.reformat import omegaconf_to_dict        # noqa: E402
from rl.utils.misc import set_seed                     # noqa: E402

TOUCH = "DrawerTaskPullingStudent"
NOTOUCH = "DrawerTaskPullingStudentNoTouch"


def build_env(task_name, num_envs, hydrofots=True):
    """Mirror train_student_aacd.py's flag coupling. Setting the coupled fields by hand would
    measure a configuration that never trains."""
    d = os.path.abspath("configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=d, version_base="1.1"):
        cfg = compose(config_name="config",
                      overrides=[f"task={task_name}",
                                 "train=hydroshear/drawer_pulling/student_lstm"])
    m = bool(cfg.task.env.get("student_tactile_mode", True))
    cfg.seed = set_seed(cfg.seed)
    cfg.headless = True
    cfg.force_render = False
    cfg.task.env.use_camera = False
    cfg.task.env.use_camera_obs = False
    cfg.task.env.use_isaac_gym_tactile = False
    cfg.task.env.use_gelsight = True
    cfg.task.env.use_shear_force = m
    cfg.task.env.use_tactile_field_obs = m
    cfg.task.env.zero_out_normal_force_field_obs = True
    cfg.task.env.tactile_subsample_ratio = 4
    cfg.task.rl.add_contact_info_to_aac_states = True
    cfg.task.rl.add_contact_force_plug_decomposed = True
    cfg.task.env.use_compliant_contact = True
    cfg.task.rl.asymmetric_observations = True
    if m and hydrofots:
        cfg.task.env.use_hydrosoft_model = True
        cfg.task.sensor.hydroshear.randomize_coefficients = False
    cfg.task.env.numEnvs = num_envs
    with initialize_config_dir(config_dir=d, version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=True, virtual_screen_capture=False, force_render=False)
    return env, cfg


def build_env_actor(ckpt, num_envs, hydrofots=True):
    """Actor mode MUST build the environment from the checkpoint's own config. The hand-composed
    config above mirrors train_student_aacd.py, but only the checkpoint's config is guaranteed to
    produce the observation shapes the saved actor expects."""
    from rl.utils.misc import load_config_from_checkpoint
    cfg = load_config_from_checkpoint(ckpt)
    cfg.task.env.numEnvs = num_envs
    cfg.headless = True
    cfg.force_render = False
    if not hydrofots:
        cfg.task.env.use_hydrosoft_model = False
    cfg.seed = set_seed(cfg.seed)
    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.1"):
        env = isaacgym_task_map[cfg.task_name](
            cfg=omegaconf_to_dict(cfg.task), rl_device=cfg.rl_device,
            sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id,
            headless=True, virtual_screen_capture=False, force_render=False)
    return env, cfg


def time_arm(env, steps, warmup, backend=None, actor=None):
    """actor=None measures the environment alone: PREFLIGHT, optimistic, no policy cost.
    actor set measures obs -> tactile CNN -> recurrent policy -> action -> step -> rnn reset,
    which is the rollout the trainer actually performs."""
    if actor is None:
        act = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        for _ in range(warmup):
            env.step(act)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    else:
        obs = env.reset()
        for _ in range(warmup):
            a_ = actor.act(obs)
            obs, _, done, _ = env.step(a_)
            actor.reset_done(done)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            a_ = actor.act(obs)
            obs, _, done, _ = env.step(a_)
            actor.reset_done(done)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    ms = 1000.0 * wall / steps
    return {"ms_per_control_step": round(ms, 3),
            "control_steps_per_s": round(1000.0 / ms, 3),
            "agent_steps_per_s": round(env.num_envs * 1000.0 / ms, 1),
            "throughput_kind": ("rollout_with_policy" if actor else "env_only_preflight"),
            "actor": (actor.stats() if actor else None),
            "backend": (backend.stats() if backend else None)}


def run_one(a):
    ckpt = getattr(a, "actor", None) or None
    if ckpt and a.arm == "A":
        print("F4_JSON " + json.dumps({"arm": "A", "error":
              "actor mode does not apply to arm A: NoTouch has a different observation space "
              "than the checkpoint's actor. Use C(0) as the matched no-backend baseline."}),
              flush=True)
        os._exit(0)
    rt = None
    if a.arm == "C":
        from tactile_backend_seam import LatencyDummyBackend
        if ckpt:
            env, cfg = build_env_actor(ckpt, a.envs, hydrofots=True)
        else:
            env, cfg = build_env(TOUCH, a.envs, hydrofots=True)
        be = LatencyDummyBackend(env.num_envs, env.device, latency_ms=a.latency)
        # substitute the stand-in for the real seam method
        env.get_force_fields_dict = be.get_force_fields_dict
        env.reset()
        if ckpt:
            from actor_runtime import ActorRuntime
            rt = ActorRuntime.from_checkpoint(ckpt, env, cfg)
        r = time_arm(env, a.steps, a.warmup, backend=be, actor=rt)
    else:
        if ckpt:
            env, cfg = build_env_actor(ckpt, a.envs, hydrofots=(a.arm == "B"))
        else:
            env, cfg = build_env(NOTOUCH if a.arm == "A" else TOUCH, a.envs,
                                 hydrofots=(a.arm == "B"))
        env.reset()
        if ckpt:
            from actor_runtime import ActorRuntime
            rt = ActorRuntime.from_checkpoint(ckpt, env, cfg)
        r = time_arm(env, a.steps, a.warmup, actor=rt)
    r.update(arm=a.arm, envs=a.envs, latency_ms=(a.latency if a.arm == "C" else None))
    print("F4_JSON " + json.dumps(r), flush=True)
    sys.stdout.flush()
    os._exit(0)


def child(a, arm, latency=0.0):
    cmd = [sys.executable, os.path.abspath(__file__), "--one", "--arm", arm,
           "--envs", str(a.envs), "--steps", str(a.steps), "--warmup", str(a.warmup),
           "--latency", str(latency)]
    if getattr(a, "actor", None):
        cmd += ["--actor", a.actor]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout,
                       cwd=os.getcwd())
    for l in p.stdout.splitlines():
        if l.startswith("F4_JSON "):
            return json.loads(l[len("F4_JSON "):]), None
    return None, f"rc={p.returncode} " + "; ".join((p.stderr or "").strip().splitlines()[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--latency", default="0")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="./f4_results.json")
    ap.add_argument("--actor", default=None,
                    help="checkpoint path; enables the real H0 actor. Requires validate_actor.py "
                         "to have reported GATE: PASS on this checkpoint.")
    a = ap.parse_args()
    if a.one:
        a.latency = float(a.latency)
        return run_one(a)

    rows = []
    print(f"\n  F4 harness, {a.envs} envs, {a.steps} timed steps. One process per arm: Isaac does")
    print(f"  not tolerate in-process teardown.\n")
    kind = "ROLLOUT (real actor)" if a.actor else "F4-PREFLIGHT - NO POLICY FORWARD"
    print(f"  {kind}   -- {'includes' if a.actor else 'EXCLUDES'} policy forward; "
          f"EXCLUDES the PPO update either way.")
    print(f"  {'arm':<34}{'ms/step':>10}{'agent-steps/s':>16}{'vs H0 train':>12}")
    print("  " + "-" * 74)
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        lat = [float(x) for x in a.latency.split(",")] if arm == "C" else [0.0]
        for L in lat:
            r, err = child(a, arm, L)
            if err:
                print(f"  {('arm ' + arm):<34}{'FAILED':>10}  {err[:34]}")
                rows.append({"arm": arm, "latency_ms": L, "error": err})
                continue
            name = {"A": "A  Isaac, no tactile",
                    "B": "B  Isaac + HydroFots",
                    "C": f"C  Isaac + stand-in @{L:g} ms"}[arm]
            rows.append(r)
            print(f"  {name:<34}{r['ms_per_control_step']:>10.2f}{r['agent_steps_per_s']:>16.1f}"
                  f"{r['agent_steps_per_s']/800.0:>11.2f}x", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2)
    ok = [r for r in rows if "error" not in r]
    A = next((r for r in ok if r["arm"] == "A"), None)
    B = next((r for r in ok if r["arm"] == "B"), None)
    C0 = next((r for r in ok if r["arm"] == "C" and r.get("latency_ms") == 0.0), None)
    base, blabel = (C0, "C(0)") if C0 else (A, "A")
    if base and B:
        share = B["ms_per_control_step"] - base["ms_per_control_step"]
        print(f"\n  HydroFots field generation = B - {blabel} = {share:.2f} ms of a "
              f"{B['ms_per_control_step']:.2f} ms control step "
              f"({100*share/B['ms_per_control_step']:.1f}%).")
        if C0 and A:
            print(f"  C(0) - A = {C0['ms_per_control_step']-A['ms_per_control_step']:.2f} ms is "
                  f"the tactile OBSERVATION path, which B - A wrongly charges to the producer.")
        if not C0:
            print("  NOTE: no C(0) in this run, so this is B - A and still confounds the "
                  "producer with the tactile observation path. Prefer B - C(0).")
        print("  This is what HydroFots costs. It is NOT automatically the budget a direct")
        print("  backend gets: the budget follows from the TARGET training throughput.")
    C = [r for r in ok if r["arm"] == "C"]
    if len(C) > 1:
        print(f"\n  LATENCY BUDGET (stand-in): agent-steps/s against backend cost")
        for r in sorted(C, key=lambda x: x["latency_ms"]):
            band = ("DIRECT-CANDIDATE" if r["agent_steps_per_s"] >= 400
                    else "GRAY" if r["agent_steps_per_s"] >= 200 else "SURROGATE-LIKELY")
            print(f"     {r['latency_ms']:>6.0f} ms -> {r['agent_steps_per_s']:>8.1f} "
                  f"agent-steps/s   {band}")
        print("  Read off the largest latency still inside the target band: that is what the")
        print("  tactile backend has to beat, measured rather than assumed.")
    print(f"\n  written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
