"""Thread C evidence figures. The point is that cue -> tighten is VISUALLY INSPECTABLE and that
metric bugs are caught by eye, not that the plots are pretty.

  fig1_story_<force>.png   one episode, every synchronised trace, every event marker
  fig2_dose.png            dose-response across the in-design force range
  fig3_aligned.png         episodes aligned at perturbation onset, median per force
  fig4_gallery.png         representative episodes chosen by FIXED rules
"""
import numpy as np, json, os, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, RULE, GREY, BG = "#1A2230", "#9FBFC4", "#6B7A80", "#FCFBF9"
CUE, GRIP, PERT, GOOD, BAD = "#0E5C6B", "#C2571A", "#7A4E9E", "#2E7D5B", "#A3341F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": RULE,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
    "axes.linewidth": 0.8, "figure.facecolor": BG, "savefig.facecolor": BG})
def sp(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(s in keep)

D = "/home/sudhirmts/hydroshear/outputs/0_hydroshear/threadC"
OUT = D + "/figs"; os.makedirs(OUT, exist_ok=True)

def load(f):
    d = np.load(f"{D}/trace_H0_f{f}.npz", allow_pickle=True)
    C = json.loads(str(d["constants"]))
    prog = d["prog"]; T = prog.shape[0]
    starts = [0] + [t for t in range(1, T) if prog[t, 0] < prog[t-1, 0]]
    segs = [(starts[i], (starts[i+1] if i+1 < len(starts) else T)) for i in range(len(starts))]
    return d, C, segs

def episodes(d, C, segs):
    """per env-episode: events + metrics. One row per (segment, env)."""
    dof, g, shear = d["dof"], d["grip_cmd"], d["shear"]
    fnL, fnR, con, th = d["fnL"], d["fnR"], d["contact"].astype(bool), d["thresh"]
    TIGHT, K = C["GRIP_TIGHT_THRESH"], C["TIGHTEN_SUSTAIN_K"]
    rows = []
    for (a, b) in segs:
        for e in range(dof.shape[1]):
            dd, gg = dof[a:b, e], g[a:b, e]
            below = np.nonzero(dd < th[e])[0]
            tp = int(below[0]) if len(below) else -1
            tg = -1
            if tp >= 0:
                tt = gg < TIGHT
                for t in range(tp, len(gg) - K + 1):
                    if tt[t:t+K].all(): tg = t; break
            pre = gg[max(0, tp-10):tp].mean() if tp > 0 else gg[0]
            post = gg[tp:].min() if 0 <= tp < len(gg) else np.nan   # to END of episode: response onset reaches +48 steps at 0 N (GT-BEHAV-007)
            rows.append(dict(seg=(a, b), e=e, tp=tp, tg=tg,
                             tau=(tg - tp if tg >= 0 and tp >= 0 else np.nan),
                             dg=float(pre - post) if tp >= 0 else np.nan,
                             Fpk=float((fnL[a:b, e] + fnR[a:b, e]).max()),
                             lost=bool((~con[a:b, e]).any()), thresh=float(th[e])))
    return rows

# ------------------------------------------------------------------ fig 1: the story plot
def story(ax_list, d, C, seg, e, title):
    a, b = seg; t = np.arange(b - a)
    dof, g, shear = d["dof"][a:b, e], d["grip_cmd"][a:b, e], d["shear"][a:b, e]
    fn = d["fnL"][a:b, e] + d["fnR"][a:b, e]; con = d["contact"][a:b, e].astype(bool)
    th = float(d["thresh"][e]); TIGHT, K = C["GRIP_TIGHT_THRESH"], C["TIGHTEN_SUSTAIN_K"]
    below = np.nonzero(dof < th)[0]; tp = int(below[0]) if len(below) else None
    tg = None
    if tp is not None:
        tt = g < TIGHT
        for k in range(tp, len(g) - K + 1):
            if tt[k:k+K].all(): tg = k; break

    A, B, Cx, Dx = ax_list
    A.set_title(title, loc="left", fontsize=10, weight="bold", color=INK, pad=6)
    A.plot(t, dof, color=INK, lw=1.4); A.axhline(th, color=PERT, ls="--", lw=1.0)
    A.text(len(t)*0.995, th, " threshold", color=PERT, fontsize=7.5, va="bottom", ha="right")
    A.set_ylabel("drawer\npos (m)")
    B.plot(t, shear, color=CUE, lw=1.4); B.set_ylabel("tactile\nshear")
    Cx.plot(t, g*1000, color=GRIP, lw=1.6)
    Cx.axhline(TIGHT*1000, color=GREY, ls=":", lw=1.0)
    Cx.text(len(t)*0.995, TIGHT*1000, " tightened", color=GREY, fontsize=7.5, va="bottom", ha="right")
    Cx.set_ylabel("grip cmd\n(mm)"); pass  # natural axis: 1 mm at the bottom, so DOWN reads as tighter
    Dx.plot(t, fn, color=GOOD, lw=1.3); Dx.set_ylabel("contact\nforce (N)")
    if (~con).any():
        Dx.fill_between(t, 0, fn.max(), where=~con, color=BAD, alpha=.18, step="mid")
    Dx.set_xlabel("control step")
    for ax in ax_list:
        sp(ax); ax.set_xlim(0, len(t)-1)
        if tp is not None: ax.axvline(tp, color=PERT, lw=1.2, alpha=.9)
        if tg is not None: ax.axvline(tg, color=GRIP, lw=1.2, ls="--", alpha=.9)
    if tp is not None:
        A.annotate("perturbation", (tp, A.get_ylim()[1]), xytext=(4, -2), textcoords="offset points",
                   color=PERT, fontsize=8, va="top", weight="bold")
    if tg is not None:
        Cx.annotate(f"tighten  (+{tg-tp} steps)", (tg, Cx.get_ylim()[0]), xytext=(4, 2),
                    textcoords="offset points", color=GRIP, fontsize=8, va="bottom", weight="bold")

def fig_story(forces):
    for f in forces:
        d, C, segs = load(f)
        rows = [r for r in episodes(d, C, segs) if r["tg"] >= 0]
        if not rows: continue
        taus = np.array([r["tau"] for r in rows]); r = rows[int(np.argsort(taus)[len(taus)//2])]
        fig, axs = plt.subplots(4, 1, figsize=(8.6, 6.4), sharex=True,
                                gridspec_kw=dict(hspace=0.18, height_ratios=[1, 1, 1.15, 1]))
        story(axs, d, C, r["seg"], r["e"],
              f"H0 at {-float(f)} N  ·  median-latency episode  ·  perturbation at drawer {r['thresh']:.3f} m")
        fig.text(0.5, 0.012, "purple = perturbation onset   ·   orange dashed = tightening onset   "
                             "·   grip: LOWER on the axis = tighter",
                 ha="center", fontsize=8, color=GREY)
        fig.savefig(f"{OUT}/fig1_story_f{f}.png", dpi=170, bbox_inches="tight", pad_inches=.25)
        plt.close(fig); print("wrote fig1_story_f%s.png" % f)

# ------------------------------------------------------------------ fig 2: dose-response
def fig_dose(forces):
    F, lat, dg, frac, succ, loss = [], [], [], [], [], []
    for f in forces:
        d, C, segs = load(f); rows = episodes(d, C, segs)
        ok = [r for r in rows if r["tg"] >= 0]
        F.append(-float(f))
        lat.append(np.percentile([r["tau"] for r in ok], [25, 50, 75]) if ok else [np.nan]*3)
        dg.append(np.percentile([r["dg"] for r in rows if np.isfinite(r["dg"])], [25, 50, 75]))
        frac.append(len(ok)/len(rows)); succ.append(float(np.mean(d["rates"])))
        loss.append(np.mean([r["lost"] for r in rows]))
    F = np.array(F); lat = np.array(lat); dg = np.array(dg)
    fig, axs = plt.subplots(1, 4, figsize=(14.4, 3.5), gridspec_kw=dict(wspace=0.42))
    fig.suptitle("H0 modulates HOW FAST and HOW HARD it grips with disturbance size",
                 x=0.055, ha="left", fontsize=12.5, weight="bold", color=INK, y=1.04)
    for ax, y, lo, hi, lab, col in [
            (axs[0], lat[:,1], lat[:,0], lat[:,2], "reaction latency\n$\\tau_{physical}$ (steps)", CUE),
            (axs[1], dg[:,1]*1000, dg[:,0]*1000, dg[:,2]*1000, "grip response\n$\\Delta g$ (mm)", GRIP)]:
        ax.fill_between(F, lo, hi, color=col, alpha=.16)
        ax.plot(F, y, "o-", color=col, lw=2, ms=6)
        ax.set_xlabel("perturbation force (N)"); ax.set_ylabel(lab); sp(ax)
    axs[0].invert_xaxis(); axs[1].invert_xaxis()
    axs[2].plot(F, frac, "o-", color=GOOD, lw=2, ms=6); axs[2].set_ylim(-.04, 1.04)
    axs[2].set_ylabel("fraction of episodes\nthat tightened"); axs[2].set_xlabel("perturbation force (N)")
    axs[3].plot(F, succ, "o-", color=INK, lw=2, ms=6, label="success")
    axs[3].plot(F, loss, "s--", color=BAD, lw=1.6, ms=5, label="handle loss")
    axs[3].set_ylim(-.04, 1.04); axs[3].set_ylabel("rate"); axs[3].set_xlabel("perturbation force (N)")
    axs[3].legend(frameon=False, fontsize=8, loc="center left")
    for a in axs[2:]: sp(a); a.invert_xaxis()
    fig.text(0.055, -0.06, "Bands are the interquartile range over env-episodes. "
             "Latency FALLS and response GROWS as the disturbance grows: graded adaptation, not a fixed clamp.",
             fontsize=8.5, color=GREY, ha="left")
    fig.savefig(f"{OUT}/fig2_dose.png", dpi=170, bbox_inches="tight", pad_inches=.3)
    plt.close(fig); print("wrote fig2_dose.png")

# ------------------------------------------------------------------ fig 3: event-aligned
def fig_aligned(forces, W=70):
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.8))
    cols = plt.cm.viridis(np.linspace(.12, .82, len(forces)))
    for i, f in enumerate(forces):
        d, C, segs = load(f); rows = episodes(d, C, segs)
        G, S = [], []
        for r in rows:
            if r["tp"] < 0: continue
            a, b = r["seg"]; e, tp = r["e"], r["tp"]
            lo, hi = tp - 20, tp + W
            if lo < 0 or hi > (b - a): continue
            G.append(d["grip_cmd"][a:b, e][lo:hi]); S.append(d["shear"][a:b, e][lo:hi])
        if not G: continue
        t = np.arange(-20, W)
        axs[0].plot(t, np.median(np.array(S), 0), color=cols[i], lw=2, label=f"{-float(f)} N")
        axs[1].plot(t, np.median(np.array(G), 0)*1000, color=cols[i], lw=2, label=f"{-float(f)} N")
    for ax, lab in ((axs[0], "tactile shear (median)"), (axs[1], "grip command, mm (median)")):
        ax.axvline(0, color=PERT, lw=1.3); ax.set_xlabel("control steps from perturbation onset")
        ax.set_ylabel(lab); sp(ax); ax.legend(frameon=False, fontsize=8, title="force", title_fontsize=8)
    
    fig.suptitle("Aligned at the perturbation: a bigger disturbance produces an earlier, deeper squeeze",
                 x=0.055, ha="left", fontsize=12.5, weight="bold", color=INK, y=1.06)
    fig.text(0.055, -0.08, "Grip axis: LOWER is tighter (1 mm tight .. 3 mm loose). Each curve is the median over all "
             "env-episodes at that force, aligned on each episode's own perturbation step.",
             fontsize=8.5, color=GREY, ha="left")
    fig.savefig(f"{OUT}/fig3_aligned.png", dpi=170, bbox_inches="tight", pad_inches=.3)
    plt.close(fig); print("wrote fig3_aligned.png")

# ------------------------------------------------------------------ fig 4: gallery, fixed rules
def fig_gallery(f="2"):
    d, C, segs = load(f); rows = episodes(d, C, segs)
    ok = [r for r in rows if r["tg"] >= 0]
    taus = np.array([r["tau"] for r in ok])
    picks = [("median latency", ok[int(np.argsort(taus)[len(taus)//2])]),
             ("fastest reaction", ok[int(np.argmin(taus))]),
             ("slowest reaction", ok[int(np.argmax(taus))]),
             ("largest grip response", max(ok, key=lambda r: r["dg"])),
             ("smallest grip response", min(ok, key=lambda r: r["dg"]))]
    nt = [r for r in rows if r["tg"] < 0]
    if nt: picks.append(("never tightened", nt[0]))
    n = len(picks)
    fig, axs = plt.subplots(4, n, figsize=(3.0*n, 6.6), sharex=False,
                            gridspec_kw=dict(hspace=0.2, wspace=0.42))
    for j, (name, r) in enumerate(picks):
        story([axs[i][j] for i in range(4)], d, C, r["seg"], r["e"], name)
        if j: [axs[i][j].set_ylabel("") for i in range(4)]
    fig.suptitle(f"Representative episodes at {-float(f)} N, chosen by fixed rules (not by eye)",
                 x=0.055, ha="left", fontsize=12.5, weight="bold", color=INK, y=1.02)
    fig.savefig(f"{OUT}/fig4_gallery.png", dpi=150, bbox_inches="tight", pad_inches=.3)
    plt.close(fig); print("wrote fig4_gallery.png")

if __name__ == "__main__":
    have = sorted({os.path.basename(p).split("_f")[1][:-4] for p in glob.glob(f"{D}/trace_H0_f*.npz")},
                  key=float)
    indesign = [f for f in have if float(f) <= 2.0]
    print("traces:", have, " in-design:", indesign)
    fig_story(indesign); fig_dose(indesign); fig_aligned(indesign); fig_gallery("2")
