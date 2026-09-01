"""Thread C, cue Family 2: INNOVATION against a nominal-pull prediction.

Family 1 (GT-BEHAV-010) taught one fact: the cue is not "more shear", it is "shear UNEXPECTED
GIVEN ORDINARY PULLING". Pre-registered as GT-CDESIGN-010 / GT-CDESIGN-011 before inspection.

  R1  full-field innovation      || T_t - T_hat(t) || / nominal scale
  R2  directional innovation     1 - cos(T_t, T_hat(t))
  R3  frozen tactile-only probe  causal linear probe, privileged labels, CALIBRATION ONLY
  C4  carried UNCHANGED as an existing negative control (not repaired, not retuned)

INFERENCE-TIME INPUTS, binding: past fields and past grip command only. No perturbation
timestamp, no external force, no privileged slip, and NO DRAWER POSITION OR VELOCITY -- the
disturbance changes drawer motion, so conditioning on it would make this a proprioceptive
detector wearing a tactile label.

The nominal predictor is fitted on the 0 N CALIBRATION envs: ordinary pulling, no disturbance at
all. That is what makes R1 different from the failed C3, which used a hand-chosen extrapolator.

The DETECTOR is imported unchanged from threadC_cue.py. It is frozen; only the score changes.
"""
import numpy as np, json, sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, ".")
from threadC_cue import detect_all, episodes_of, scores as fam1_scores, K_GRID, CAL, HELD, D, FORCES

LAG = 3          # causal window: t-1, t-2, t-3
RIDGE = 1e-2

def load_ep(f):
    """-> L,R [M,T,252 split as 2x126], g [M,T], env, tp, tg  for one force."""
    d = np.load(f"{D}/trace_H0F_f{f}.npz", allow_pickle=True)
    C = json.loads(str(d["constants"])); TIGHT, KS = C["GRIP_TIGHT_THRESH"], C["TIGHTEN_SUSTAIN_K"]
    dof, g, th, prog = d["dof"], d["grip_cmd"], d["thresh"], d["prog"]
    FL, FR = d["fieldL"], d["fieldR"]
    N = dof.shape[1]; out = []
    for (a, b) in episodes_of(prog):
        Tl = b - a
        for e in range(N):
            dd, gg = dof[a:b, e], g[a:b, e]
            bel = np.nonzero(dd < th[e])[0]
            if not len(bel): continue
            tp = int(bel[0]); tt = gg < TIGHT; tg = -1
            for t in range(tp, Tl - KS + 1):
                if tt[t:t+KS].all(): tg = t; break
            out.append((np.concatenate([FL[a:b, e].reshape(Tl, -1),
                                        FR[a:b, e].reshape(Tl, -1)], 1).astype(np.float32),
                        gg.astype(np.float32), e, tp, tg))
    n = min(x[0].shape[0] for x in out)
    T = np.stack([x[0][:n] for x in out])            # [M, n, 252]
    G = np.stack([x[1][:n] for x in out])            # [M, n]
    return T, G, np.array([x[2] for x in out]), np.array([x[3] for x in out]), np.array([x[4] for x in out])

def feats(T, G):
    """causal design matrix from t-1..t-LAG. -> X [M, n, 3*252+3], valid from t=LAG."""
    M, n, Dd = T.shape
    X = np.zeros((M, n, LAG*Dd + LAG), np.float32)
    for j in range(1, LAG+1):
        X[:, j:, (j-1)*Dd:j*Dd] = T[:, :n-j]
        X[:, j:, LAG*Dd + (j-1)] = G[:, :n-j]
    return X

def ridge_fit(X, Y, lam=RIDGE):
    XtX = X.T @ X; XtX.flat[::XtX.shape[0]+1] += lam * np.trace(XtX) / XtX.shape[0]
    return np.linalg.solve(XtX, X.T @ Y)

def main():
    print("  loading traces ...", flush=True)
    dat = {f: load_ep(f) for f in FORCES}
    for f in FORCES:
        print(f"    -{f} N: {dat[f][0].shape[0]} env-episodes, T={dat[f][0].shape[1]}", flush=True)

    # ---- nominal-pull predictor: fitted on 0 N CALIBRATION envs only -------------------------
    T0, G0, e0, _, _ = dat["0"]
    m = (e0 >= CAL[0]) & (e0 < CAL[1])
    X0, Y0 = feats(T0[m], G0[m]), T0[m]
    n = X0.shape[1]
    Xf = X0[:, LAG:].reshape(-1, X0.shape[2]); Yf = Y0[:, LAG:].reshape(-1, Y0.shape[2])
    print(f"\n  nominal predictor: {Xf.shape[0]} samples x {Xf.shape[1]} features "
          f"(0 N, calibration envs only)", flush=True)
    W = ridge_fit(Xf, Yf)
    resid0 = np.linalg.norm(Yf - Xf @ W, axis=1)
    SCALE = float(np.sqrt((resid0**2).mean()))
    print(f"  nominal residual scale (RMS on its own training data): {SCALE:.5f}")

    # ---- R3 probe: privileged labels, CALIBRATION envs, all forces --------------------------
    Xs, ys = [], []
    for f in FORCES:
        T, G, env, tp, tg = dat[f]
        mm = (env >= CAL[0]) & (env < CAL[1])
        X = feats(T[mm], G[mm])[:, LAG:]
        tpm = tp[mm]
        lab = np.full(X.shape[:2], -1.0, np.float32)
        if float(f) > 0:                       # a disturbance is ACTIVE only when force != 0
            idx = np.arange(LAG, LAG + X.shape[1])[None, :]
            lab[idx >= tpm[:, None]] = 1.0
        Xs.append(X.reshape(-1, X.shape[2])); ys.append(lab.reshape(-1))
    Xp, yp = np.concatenate(Xs), np.concatenate(ys)
    print(f"  R3 probe: {Xp.shape[0]} samples, {100*np.mean(yp>0):.1f}% positive", flush=True)
    Wp = ridge_fit(Xp, yp[:, None])
    del Xs, ys, Xp, yp

    # ---- score every candidate on every force ----------------------------------------------
    cache = {}
    for f in FORCES:
        T, G, env, tp, tg = dat[f]
        X = feats(T, G)
        P = np.einsum('mtf,fd->mtd', X, W)
        r = np.linalg.norm(T - P, axis=2); r[:, :LAG] = 0
        nt, np_ = np.linalg.norm(T, axis=2), np.linalg.norm(P, axis=2)
        cos = np.einsum('mtd,mtd->mt', T, P) / (nt * np_ + 1e-9)
        c2 = 1.0 - cos; c2[:, :LAG] = 0
        probe = np.einsum('mtf,fd->mt', X, Wp); probe[:, :LAG] = 0
        cache[("R1", f)] = (r / SCALE, env, tp, tg)
        cache[("R2", f)] = (c2, env, tp, tg)
        cache[("R3", f)] = (probe, env, tp, tg)
        L, R = T[..., :126], T[..., 126:]
        cache[("C4", f)] = (fam1_scores("C4", L, R), env, tp, tg)
        del X, P, r, cos, c2, probe

    from threadC_cue import evaluate, crit
    import threadC_cue as F1
    NAMES = ["C4", "R1", "R2", "R3"]
    print("\n  === CALIBRATION (envs 0-63): fit k ===")
    best = {}
    for nm in NAMES:
        rec = sorted(((crit(evaluate(nm, k, cache, *CAL))[3], k) for k in K_GRID),
                     key=lambda x: (-1e9 if not np.isfinite(x[0]) else x[0]), reverse=True)
        best[nm] = rec[0][1]
        print(f"   {nm}  k={best[nm]}  calibration ordering gap {rec[0][0]:+.1f}")
    print("\n  === HELD-OUT (envs 64-127, thresholds -0.06..-0.10 m) ===")
    print("      criteria: A gap>=10   B precedence>=.80   C premature<=.20   E coverage@-2N>=.80")
    c0gap = -1.0    # GT-BEHAV-010: the C0 control's held-out gap
    for nm in NAMES:
        r = evaluate(nm, best[nm], cache, *HELD)
        A, B, Cc, gap = crit(r)
        Dd = bool(np.isfinite(gap) and gap > c0gap)
        E = bool(r["2"]["fired"] >= 0.80)
        v = "ACCEPT" if (A and B and Cc and Dd and E) else "reject"
        print(f"\n   {nm}  k={best[nm]}  gap {gap:+.1f}  A={A} B={B} C={Cc} D={Dd} E={E}  -> {v}")
        for f in FORCES:
            x = r[f]
            print(f"        -{f} N  fired {x['fired']:.2f}  median (t_cue - t_p) {x['med_rel']:+7.1f}  "
                  f"precedence {x['prec']:.2f}  premature {x['prem']:.2f}")

if __name__ == "__main__":
    main()
