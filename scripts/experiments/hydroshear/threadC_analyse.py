"""Thread C metrics + the UNKNOWN-U010 test, from a trace npz.

Event definitions are the FROZEN ones (GT-CDESIGN-003, GT-TASK-013), read from the trace itself
so they cannot drift between capture and analysis.
"""
import numpy as np, json, sys

def analyse(path):
    d = np.load(path, allow_pickle=True)
    C = json.loads(str(d["constants"]))
    dof, g, fnL, fnR = d["dof"], d["grip_cmd"], d["fnL"], d["fnR"]
    shear, contact, prog = d["shear"], d["contact"].astype(bool), d["prog"]
    thresh, T, N = d["thresh"], dof.shape[0], dof.shape[1]
    TIGHT, K = C["GRIP_TIGHT_THRESH"], C["TIGHTEN_SUSTAIN_K"]
    LATE = C["LATE_CLAMP_DOF"]

    # episode segmentation: progress_buf resets to 0/1 at each episode start
    starts = [0] + [t for t in range(1, T) if prog[t, 0] < prog[t-1, 0]]
    segs = [(starts[i], (starts[i+1] if i+1 < len(starts) else T)) for i in range(len(starts))]
    print(f"\n  {path.split('/')[-1]}   T={T} N={N}  episodes={len(segs)}  force={float(d['force_amount'])} N")

    rows = []
    for (a, b) in segs:
        for e in range(N):
            dd, gg = dof[a:b, e], g[a:b, e]
            # t_p: first step the drawer passes this env's threshold  (GT-TASK-011)
            below = np.nonzero(dd < thresh[e])[0]
            if len(below) == 0:
                continue
            tp = below[0]
            # t_tighten: first t>=tp with grip < TIGHT sustained K steps  (GT-CDESIGN-003)
            tight = gg < TIGHT
            tg = None
            for t in range(tp, len(gg) - K + 1):
                if tight[t:t+K].all():
                    tg = t; break
            # pre/post grip about the perturbation
            pre = gg[max(0, tp-10):tp].mean() if tp > 0 else gg[0]
            post = gg[tp:].min() if tp < len(gg) else np.nan   # to END of episode: response onset reaches +48 steps at 0 N (GT-BEHAV-007)
            rows.append(dict(
                e=e, thresh=float(thresh[e]), tp=int(tp),
                tg=(int(tg) if tg is not None else -1),
                tau_phys=(int(tg - tp) if tg is not None else -1),
                dof_at_tg=(float(dd[tg]) if tg is not None else np.nan),
                dg=float(pre - post),
                Egrip=float(np.trapz(fnL[a:b, e] + fnR[a:b, e])),
                Fpeak=float((fnL[a:b, e] + fnR[a:b, e]).max()),
                lost=bool((~contact[a:b, e]).any()),
            ))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    ok = R["tg"] >= 0
    print(f"  tightened in {ok.sum()}/{len(ok)} env-episodes")
    print(f"  handle-loss rate        {R['lost'].mean():.4f}")
    print(f"  integrated grip effort  {R['Egrip'].mean():.2f}  (sd {R['Egrip'].std():.2f})")
    print(f"  peak grip effort        {R['Fpeak'].mean():.3f} N (sd {R['Fpeak'].std():.3f})")
    print(f"  grip response dg        {R['dg'].mean():+.5f} m (sd {R['dg'].std():.5f})  [+ = tightened]")
    if ok.any():
        print(f"  tau_physical            {R['tau_phys'][ok].mean():.1f} steps "
              f"(sd {R['tau_phys'][ok].std():.1f}, min {R['tau_phys'][ok].min()}, max {R['tau_phys'][ok].max()})")

        # ---- UNKNOWN-U010: reactive, or exploiting the always-legal late-clamp window? ----
        x, y = R["thresh"][ok], R["dof_at_tg"][ok]
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
        late = (y <= LATE).mean()
        print(f"\n  --- UNKNOWN-U010 ---")
        print(f"  corr(perturbation threshold, drawer pos at tighten) = {r:+.3f}")
        print(f"     reactive  -> near +1 (tightens where the perturbation was)")
        print(f"     late-clamp-> near  0 (piles up at the {LATE} m boundary)")
        print(f"  fraction tightening inside the always-legal window (dof <= {LATE}): {late:.3f}")
        print(f"  dof at tighten: mean {y.mean():.4f}  sd {y.std():.4f}  "
              f"range [{y.min():.4f}, {y.max():.4f}]")
        b = np.polyfit(x, y, 1) if len(x) > 2 else [np.nan, np.nan]
        print(f"  slope {b[0]:+.3f}  (1.0 = perfectly tracks the perturbation point)")
    return R

for p in sys.argv[1:]:
    analyse(p)
