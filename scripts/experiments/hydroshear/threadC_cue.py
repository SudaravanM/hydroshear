"""Thread C: the FROZEN cue-candidate family against the FROZEN acceptance criteria.

Definitions written 2026-08-27 BEFORE the field traces existed. Premises:
  GT-CDESIGN-008  candidates C1-C4, with C0 (mean magnitude) as the KNOWN-FAILING control
  GT-CDESIGN-009  calibration envs 0-63, held-out 64-127 -- also a threshold split
                  (-0.02..-0.06 vs -0.06..-0.10 m), so held-out is a real extrapolation
  GT-BEHAV-006    C0 fires identically at 0 N; that is the failure every candidate must avoid

Every candidate is CAUSAL: current and past field only, never t_perturbation. A detector that
knows when the disturbance happened cannot be used to measure reaction to it.

This file is a VECTORISED rewrite of the version shipped before the data landed. The mathematics
of every candidate and of the detector is unchanged; only the loops are gone, because the naive
version recomputed each candidate once per k and would not have finished.
"""
import numpy as np, json, sys
from numpy.lib.stride_tricks import sliding_window_view as swv

BASE_LO, BASE_HI = 40, 10      # trailing baseline window [t-40, t-10]: causal
SUSTAIN = 3
K_GRID = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
CAL, HELD = (0, 64), (64, 128)
D = "/home/sudhirmts/hydroshear/outputs/0_hydroshear/threadC"
FORCES = ["0", "0.5", "1", "2"]

def episodes_of(prog):
    T = prog.shape[0]
    st = [0] + [t for t in range(1, T) if prog[t, 0] < prog[t-1, 0]]
    return [(st[i], (st[i+1] if i+1 < len(st) else T)) for i in range(len(st))]

def scores(name, L, R):
    """L,R: [M, T, 126] flattened fields for M env-episodes. -> [M, T] causal score."""
    M, T, _ = L.shape
    nl, nr = np.linalg.norm(L, axis=2), np.linalg.norm(R, axis=2)
    if name == "C0":
        return np.linalg.norm(L.reshape(M, T, 63, 2), axis=-1).mean(2) + \
               np.linalg.norm(R.reshape(M, T, 63, 2), axis=-1).mean(2)
    if name == "C1":
        o = np.zeros((M, T))
        o[:, 1:] = np.linalg.norm(np.diff(L, axis=1), axis=2) + np.linalg.norm(np.diff(R, axis=1), axis=2)
        return o
    if name == "C2":
        W = 10; o = np.zeros((M, T))
        for A, nA in ((L, nl), (R, nr)):
            dot = np.einsum('mtd,mtd->mt', A[:, W:], A[:, :-W])
            den = nA[:, W:] * nA[:, :-W] + 1e-9
            o[:, W:] += 1.0 - dot / den
        return o
    if name == "C3":
        o = np.zeros((M, T))
        for A in (L, R):
            pred = np.zeros_like(A); pred[:, 2:] = 2*A[:, 1:-1] - A[:, :-2]
            r = np.linalg.norm(A - pred, axis=2); r[:, :2] = 0
            o += r
        return o
    if name == "C4":
        return np.linalg.norm(L - R, axis=2) / (nl + nr + 1e-9)
    raise ValueError(name)

def detect_all(S, k):
    """S: [M, T] -> [M] first index whose value exceeds the trailing-baseline threshold for
    SUSTAIN steps, else -1. Vectorised; identical rule to the pre-registered version."""
    M, T = S.shape
    win = swv(S, BASE_LO - BASE_HI, axis=1)             # [M, T-(BASE_LO-BASE_HI)+1, W]
    med = np.median(win, axis=2)
    mad = np.median(np.abs(win - med[..., None]), axis=2)
    thr = med + k * 1.4826 * mad
    over = np.zeros((M, T), bool)
    valid = thr.shape[1]
    idx = np.arange(BASE_LO, min(T, BASE_LO + valid))
    over[:, idx] = S[:, idx] > thr[:, idx - BASE_LO]
    run = over.copy()
    for j in range(1, SUSTAIN):
        run[:, :T-j] &= over[:, j:]
    run[:, T-SUSTAIN+1:] = False
    first = np.where(run.any(1), run.argmax(1), -1)
    return first

def load(f):
    d = np.load(f"{D}/trace_H0F_f{f}.npz", allow_pickle=True)
    C = json.loads(str(d["constants"])); TIGHT, KS = C["GRIP_TIGHT_THRESH"], C["TIGHTEN_SUSTAIN_K"]
    dof, g, th, prog = d["dof"], d["grip_cmd"], d["thresh"], d["prog"]
    FL, FR = d["fieldL"], d["fieldR"]
    N = dof.shape[1]; segs = episodes_of(prog)
    Ls, Rs, env, tp_, tg_ = [], [], [], [], []
    for (a, b) in segs:
        Tl = b - a
        for e in range(N):
            dd, gg = dof[a:b, e], g[a:b, e]
            bel = np.nonzero(dd < th[e])[0]
            if not len(bel): continue
            tp = int(bel[0]); tt = gg < TIGHT; tg = -1
            for t in range(tp, Tl - KS + 1):
                if tt[t:t+KS].all(): tg = t; break
            Ls.append(FL[a:b, e].reshape(Tl, -1)); Rs.append(FR[a:b, e].reshape(Tl, -1))
            env.append(e); tp_.append(tp); tg_.append(tg)
    n = min(x.shape[0] for x in Ls)
    return (np.stack([x[:n] for x in Ls]), np.stack([x[:n] for x in Rs]),
            np.array(env), np.array(tp_), np.array(tg_))

def evaluate(name, k, cache, lo, hi):
    out = {}
    for f in FORCES:
        S, env, tp, tg = cache[(name, f)]
        tc = detect_all(S, k)
        m = (env >= lo) & (env < hi)
        tc, tp2, tg2 = tc[m], tp[m], tg[m]
        fired = tc >= 0
        rel = (tc - tp2)[fired]
        prec = float(np.mean((tc[fired] < tg2[fired]) & (tg2[fired] >= 0))) if fired.any() else 0.0
        prem = float(np.mean(tc[fired] < tp2[fired])) if fired.any() else 1.0
        out[f] = dict(n=int(m.sum()), fired=float(fired.mean()),
                      med_rel=float(np.median(rel)) if len(rel) else np.nan,
                      prec=prec, prem=prem)
    return out

def crit(r):
    if r is None or not np.isfinite(r["0"]["med_rel"]) or not np.isfinite(r["2"]["med_rel"]):
        return False, False, False, np.nan
    gap = r["0"]["med_rel"] - r["2"]["med_rel"]      # + => -2 N fires EARLIER, as required
    return bool(gap >= 10), all(v["prec"] >= .80 for v in r.values()), \
           all(v["prem"] <= .20 for v in r.values()), gap

if __name__ == "__main__":
    NAMES = ["C0", "C1", "C2", "C3", "C4"]
    print("  loading traces and precomputing candidate scores ...", flush=True)
    cache = {}
    for f in FORCES:
        L, R, env, tp, tg = load(f)
        for nm in NAMES:
            cache[(nm, f)] = (scores(nm, L, R), env, tp, tg)
        print(f"    force -{f} N: {L.shape[0]} env-episodes, T={L.shape[1]}", flush=True)
        del L, R
    print("\n  === CALIBRATION (envs 0-63, thresholds -0.02..-0.06 m): fit k ===")
    best = {}
    for nm in NAMES:
        rec = sorted(((crit(evaluate(nm, k, cache, *CAL))[3], k) for k in K_GRID),
                     key=lambda x: (-1e9 if not np.isfinite(x[0]) else x[0]), reverse=True)
        best[nm] = rec[0][1]
        print(f"   {nm}  k={best[nm]}  best calibration ordering gap {rec[0][0]:+.1f} steps")
    print("\n  === HELD-OUT (envs 64-127, thresholds -0.06..-0.10 m): frozen k ===")
    c0gap = None
    for nm in NAMES:
        r = evaluate(nm, best[nm], cache, *HELD)
        A, B, Cc, gap = crit(r)
        if nm == "C0":
            c0gap = gap; verdict = "control"; Dd = None
        else:
            Dd = bool(np.isfinite(gap) and np.isfinite(c0gap) and gap > c0gap)
            verdict = "ACCEPT" if (A and B and Cc and Dd) else "reject"
        print(f"\n   {nm}  k={best[nm]}   ordering gap {gap:+.1f}   A={A} B={B} C={Cc} D={Dd}  -> {verdict}")
        for f in FORCES:
            v = r[f]
            print(f"        -{f} N  fired {v['fired']:.2f}  median (t_cue - t_p) {v['med_rel']:+7.1f}  "
                  f"precedence {v['prec']:.2f}  premature {v['prem']:.2f}")
