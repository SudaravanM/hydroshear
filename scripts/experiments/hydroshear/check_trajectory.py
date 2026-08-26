"""Verify a trajectory file written by record_trajectories.py. numpy only; no simulator.

Checks, per side where the data exists:
  ALIGN    the field the policy observed at row i equals the env's post-processing of the RAW
           backend field at row i (drawer_task_pulling.py:476-477: take channels :2, flip rows
           and channels, negate channel 0). Rows i-1 and i+1 are reported for contrast, so a
           one-step offset between geometry and field is visible, not just a mismatch.
  TF       rel_pos/rel_quat re-derive from the stored world poses as inv(elastomer) o indenter.
  CONTACT  depth > 0 and n_pen > 0 exactly when the raw field is nonzero (the sensor zeroes
           forces outside contact), so the SDF geometry describes the same contact as the field.
  QUAT     unit quaternions; DONES episode boundaries are at fixed-length intervals.

    python scripts/experiments/hydroshear/check_trajectory.py <traj.npz>
"""
import sys
import numpy as np


def q_conj(q):
    return q * np.array([-1, -1, -1, 1], np.float32)


def q_mul(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4)); bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz], -1)


def q_rot(q, v):
    vq = np.concatenate([v, np.zeros_like(v[..., :1])], -1)
    return q_mul(q_mul(q, vq), q_conj(q))[..., :3]


def env_postprocess(raw):
    """drawer_task_pulling.py:476-477 / tacsl equivalent: raw (T,E,9,7,3) -> obs (T,E,9,7,2)."""
    f = raw[..., :2][:, :, ::-1, :, ::-1].copy()
    f[..., 0] *= -1
    return f


def main(path):
    d = np.load(path, allow_pickle=True); m = d["meta"].item()
    print({k: m[k] for k in m if k != "note"})
    T = d["rel_pos_left"].shape[0]
    results = []

    def report(name, ok, detail):
        results.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name:<8} {detail}")

    for side in ("left", "right"):
        # ---- TF
        e_q, e_p = d[f"elast_{side}_quat"], d[f"elast_{side}_pos"]
        h_q, h_p = d["indenter_quat"], d["indenter_pos"]
        rel_p = q_rot(q_conj(e_q), h_p - e_p)
        rel_q = q_mul(q_conj(e_q), h_q)
        dp = np.abs(rel_p - d[f"rel_pos_{side}"]).max()
        dq = np.minimum(np.abs(rel_q - d[f"rel_quat_{side}"]), np.abs(rel_q + d[f"rel_quat_{side}"])).max()
        report(f"TF {side}", dp < 1e-5 and dq < 1e-4, f"max |dpos| {dp:.2e} m, max |dquat| {dq:.2e}")
        qn = np.linalg.norm(d[f"rel_quat_{side}"], axis=-1)
        report(f"QUAT {side}", np.abs(qn - 1).max() < 1e-4, f"norm in [{qn.min():.6f}, {qn.max():.6f}]")

        # ---- ALIGN
        raw, obs = d[f"hydro_raw_{side}"], d[f"tactile_{side}"]
        if raw.shape[-1] == 0 or obs.shape[-1] == 0:
            print(f"  [SKIP] ALIGN {side}: raw stored={raw.shape[-1] > 0}, obs stored={obs.shape[-1] > 0}")
        else:
            f = env_postprocess(raw)
            e0 = np.abs(f - obs).max()
            em = np.abs(f[1:] - obs[:-1]).max()     # zero if raw lags obs by one step
            ep = np.abs(f[:-1] - obs[1:]).max()     # zero if raw is one step AHEAD of obs (the old bug)
            scale = np.abs(obs).max()
            ok = e0 < 1e-5 * max(scale, 1.0) and scale > 0
            report(f"ALIGN {side}", ok,
                   f"max|obs[i]-f(raw[i])| {e0:.2e}; contrast: raw behind {em:.2e}, raw ahead {ep:.2e}; |obs| max {scale:.3f}")

        # ---- CONTACT
        dep, npen = d[f"depth_{side}"], d[f"n_pen_{side}"]
        if raw.shape[-1] == 0 or dep.shape[-1] == 0:
            print(f"  [SKIP] CONTACT {side}")
        else:
            field_on = np.abs(raw).sum(axis=(2, 3, 4)) > 0          # (T, E)
            pen_on = npen > 0
            depth_on = dep[..., 0] > 0
            agree = float((field_on == pen_on).mean()); agree_d = float((depth_on == pen_on).mean())
            report(f"CONTACT {side}", agree > 0.99 and agree_d > 0.99,
                   f"field!=0 <-> n_pen>0 agree {agree*100:.2f}%; depth>0 <-> n_pen>0 agree {agree_d*100:.2f}%; "
                   f"contact rows {pen_on.mean()*100:.1f}%; depth max {dep.max()*1e3:.3f} mm")

    # ---- DONES
    done_rows = np.nonzero(d["dones"].sum(1))[0]
    gaps = np.diff(np.concatenate([[-1], done_rows]))
    report("DONES", len(done_rows) == m["episodes"] and (len(gaps) == 0 or np.all(gaps == gaps[0])),
           f"episode ends at rows {done_rows.tolist()} (T={T}, episode length {gaps[0] if len(gaps) else 'n/a'})")

    print(f"\n  {'ALL PASS' if all(results) else 'FAILURES PRESENT'}  ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
