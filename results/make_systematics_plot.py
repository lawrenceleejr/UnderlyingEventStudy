"""Systematic-uncertainty breakdown (Tufte): shift in corr(y_Z) per source vs the
headline, with the statistical band for reference. Detector/reco effects are small;
charged-only is a separate (PUPPI-model-independent) observable definition."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts
ts.use()
R = "results"
val = json.load(open(f"{R}/validation_opendata.json")); syst = json.load(open(f"{R}/systematics_opendata.json"))
base = val["headline PUPPI>0.5"]["corr"]
# statistical band: train/test-split + training variance from the multi-seed spread
# (std of headline + 4 alternate seeds = 0.0083), consistent with the bootstrap.
stat = 0.0083

# detector / reconstruction systematics (smearings, efficiency, working points)
SRC = [
    ("PUPPI threshold $0.3$",      "PUPPI>0.3"),
    ("PUPPI threshold $0.7$",      "PUPPI>0.7"),
    ("acceptance $|\\eta|<2.4$",   "central |eta|<2.4"),
    ("$Z$-mass window 86–96",      "Z mass 86-96"),
    ("$\\eta$ resolution 0.05",    "eta smear 0.05"),
    ("$p_T$ resolution 10\\%",     "pT smear 10%"),
    ("PF reco eff. $-2\\%$",       "PF reco eff -2%"),
    ("PF reco eff. $-5\\%$",       "PF reco eff -5%"),
    ("PF reco eff. $-10\\%$",      "PF reco eff -10%"),
]
labels = [s[0] for s in SRC]; shifts = [syst[s[1]]["corr"] - base for s in SRC]
order = np.argsort(np.abs(shifts))           # smallest impact at bottom
labels = [labels[i] for i in order]; shifts = [shifts[i] for i in order]
det_env = np.sqrt(np.sum(np.array(shifts)**2))   # quadrature envelope (indicative)

fig, ax = plt.subplots(figsize=(8.8, 5.6))
yy = np.arange(len(shifts))
ax.axvspan(-stat, stat, color=ts.MUTE, alpha=0.16)            # stat band
ax.axvline(0, color=ts.INK, lw=0.8)
ax.hlines(yy, 0, shifts, color=ts.ACCENT, lw=2.0)
ax.plot(shifts, yy, "o", color=ts.ACCENT, ms=6)
for y, s in zip(yy, shifts):
    ax.annotate(f"{s:+.3f}", (s, y), xytext=(6 if s >= 0 else -6, 0), textcoords="offset points",
                va="center", ha="left" if s >= 0 else "right", fontsize=8, color=ts.INK)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel(r"shift in corr$(y_Z)$ vs headline (0.75)")
ax.set_xlim(-0.045, 0.045); ts.minimal(ax); ax.spines["left"].set_visible(False); ax.tick_params(left=False)
ax.annotate(f"stat. $\\pm${stat:.02f}", (stat, len(yy)-0.4), fontsize=8, color=ts.MUTE, ha="left")
ax.set_title(r"Detector/reconstruction systematics: all $\lesssim$0.02 (dominant: PF eff., PUPPI WP); "
             r"grey = stat. band", loc="left", fontsize=10)
# note the charged-only cross-check separately (a different object, not a smearing)
ax.annotate(r"(charged-only, neutral-PUPPI-independent cross-check: corr $0.45$ — a different, conservative object)",
            (0, -1.2), xycoords=("data", "data"), fontsize=8, color=ts.MUTE, ha="center", annotation_clip=False)
fig.tight_layout(); out = f"{R}/opendata_systematics.png"; fig.savefig(out)
print(f"wrote {out}   stat=+/-{stat:.3f}  det-syst envelope=+/-{det_env:.3f}")
