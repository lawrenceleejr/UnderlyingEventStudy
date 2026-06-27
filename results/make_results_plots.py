"""Results & cross-checks figure (Tufte-styled). Pulls validated numbers from the
json outputs + parquet diagnostics (no training). Pythia shown PER-BIN with the
SAME selection (dashed ticks), not a single horizontal line."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts
ts.use()

R = "results"
val = json.load(open(f"{R}/validation_opendata.json"))
abl = json.load(open(f"{R}/ablation_opendata.json")) if os.path.exists(f"{R}/ablation_opendata.json") else {}
pyt = json.load(open(f"{R}/ablation_pythia_matched.json")) if os.path.exists(f"{R}/ablation_pythia_matched.json") else {}
d = ak.from_parquet("data/skim/opendata.parquet")
y = np.asarray(d.y_Z); eta = d.p_eta; pup = d.p_puppi
flatpup = ak.to_numpy(ak.flatten(pup))

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[m], b[m])[0, 1]
def mean_eta(mask):
    n = ak.sum(mask, axis=1)
    return np.asarray(ak.where(n > 0, ak.sum(eta*mask, axis=1)/np.where(np.asarray(n) > 0, np.asarray(n), 1), 0.0))

# data bar value, CI, and matched-Pythia value per selection
BARS = [
    ("nominal\n(pileup)",  abl.get("nominal (all soft, no PU cut)", {}).get("corr", 0.0), None,
     pyt.get("all", {}).get("corr"), ts.MUTE),
    ("PUPPI$>$0.5\n(headline)", val["headline PUPPI>0.5"]["corr"],
     (val["headline PUPPI>0.5"]["corr_lo"], val["headline PUPPI>0.5"]["corr_hi"]),
     pyt.get("all", {}).get("corr"), ts.ACCENT),
    ("shuffled-$y$\nNULL", val["PERMUTED-LABELS null"]["corr"], None, None, ts.ACCENT2),
    ("charged\nonly", val["decomp: charged only"]["corr"], None,
     pyt.get("charged", {}).get("corr"), ts.ACCENT),
    ("neutral\nonly", val["decomp: neutral only"]["corr"], None,
     pyt.get("neutral", {}).get("corr"), ts.ACCENT),
    ("central\n$|\\eta|<2.5$", val["decomp: central |eta|<2.5"]["corr"], None,
     pyt.get("central", {}).get("corr"), ts.ACCENT),
    ("forward\n$|\\eta|>2.5$", val["decomp: forward |eta|>2.5"]["corr"], None,
     pyt.get("forward", {}).get("corr"), ts.ACCENT),
]

fig, ax = plt.subplots(2, 3, figsize=(15.5, 9))

# (0,0) headline + cross-check bars, with per-bin Pythia dashes
a = ax[0,0]
xs = np.arange(len(BARS))
for i, (lab, v, ci, pv, col) in enumerate(BARS):
    a.bar(i, v, width=0.66, color=col, edgecolor="none")
    if ci is not None:
        a.plot([i, i], ci, color=ts.INK, lw=1.1)
    if pv is not None:                      # matched-Pythia dashed tick on this bin
        a.plot([i-0.36, i+0.36], [pv, pv], ls=(0,(4,2)), lw=1.4, color=ts.INK)
    a.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=ts.INK)
a.axhline(0, color=ts.INK, lw=0.7)
a.set_xticks(xs); a.set_xticklabels([b[0] for b in BARS], fontsize=8)
a.set_ylabel(r"corr$(y_Z^{\rm pred},\,y_Z^{\rm true})$"); a.set_ylim(-0.04, 0.85)
a.spines["left"].set_bounds(0, 0.8)
a.set_title("(a) headline & cross-checks (EFN)")
if any(b[3] is not None for b in BARS):
    a.text(0.97, 0.96, "– – Pythia, same selection", transform=a.transAxes, ha="right",
           va="top", fontsize=8, color=ts.INK)

# (0,1) pileup dependence
a = ax[0,1]
pb = val["headline PUPPI>0.5"]["pubins"]
xs2 = [p["mean_pu"] for p in pb]; cs2 = [p["corr"] for p in pb]
a.plot(xs2, cs2, "o-", color=ts.ACCENT, ms=7)
a.axhline(val["headline PUPPI>0.5"]["corr"], ls=(0,(4,2)), lw=1.1, color=ts.MUTE)
ts.label_end(a, xs2[-1], val["headline PUPPI>0.5"]["corr"], "inclusive", ts.MUTE, dx=-130, dy=0.018)
a.set_xlabel(r"pileup proxy: $\langle$nominal soft multiplicity$\rangle$"); a.set_ylabel("corr")
a.set_ylim(0.6, 0.82); ts.rangeframe(a, xs2, cs2)
a.set_title("(b) stable vs pileup")

# (0,2) corr vs PUPPI threshold (training-free) + EFN points
# PUPPI weights are bimodal (93% exactly 0), so the no-cut point (keep ALL) is
# washed out (~0) and removing the zero-weight pileup is a STEP at any cut>0.
a = ax[0,2]
allm = ak.ones_like(eta, dtype=bool)
ths = np.concatenate([[0.0], np.linspace(0.02, 0.7, 14)])
cm = [corr(mean_eta(allm if t <= 0 else (pup > t)), y) for t in ths]
a.plot(ths, cm, "-", color=ts.MUTE, lw=1.4)
a.plot([0.0], [cm[0]], "o", color=ts.MUTE, ms=5)
ts.label_end(a, 0.7, cm[-1], r"mean-$\eta$ (linear)", ts.MUTE, dx=-0.5, dy=-0.05)
ex = [0.0, 0.1, 0.5]; ey = [abl.get("nominal (all soft, no PU cut)", {}).get("corr", 0),
                            abl.get("PUPPI>0.1", {}).get("corr", np.nan), val["headline PUPPI>0.5"]["corr"]]
a.plot(ex, ey, "o", color=ts.ACCENT, ms=8)
ts.label_end(a, 0.5, ey[-1], "EFN (deep set)", ts.ACCENT, dx=-0.42, dy=0.04)
a.annotate("keep all\n$\\to$ washed out", (0.0, cm[0]), xytext=(0.13, 0.12),
           fontsize=7.5, color=ts.INK, va="center",
           arrowprops=dict(arrowstyle="->", lw=0.6, color=ts.INK))
a.set_xlabel("PUPPI weight threshold"); a.set_ylabel("corr"); a.set_ylim(-0.05, 0.82)
a.set_title("(c) pileup removal reveals the boost (bimodal PUPPI $\\Rightarrow$ step)")

# (1,0) decomposition with <n> direct-labeled
a = ax[1,0]
dk = ["decomp: charged only","decomp: neutral only","decomp: central |eta|<2.5","decomp: forward |eta|>2.5","headline PUPPI>0.5"]
dl = ["charged","neutral","central","forward","all"]
dv = [val[k]["corr"] for k in dk]; dn = [val[k]["mean_n"] for k in dk]
a.bar(range(5), dv, width=0.66, color=ts.ACCENT, edgecolor="none")
for i,(v,nn) in enumerate(zip(dv,dn)):
    a.text(i, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=ts.INK)
    a.text(i, 0.02, rf"$\langle n\rangle{{=}}{nn:.0f}$", ha="center", va="bottom", fontsize=7.5, color="white")
a.set_xticks(range(5)); a.set_xticklabels(dl, fontsize=8.5); a.set_ylim(0,0.82)
a.spines["left"].set_bounds(0,0.8); a.set_ylabel("EFN corr"); a.set_title("(d) where the signal lives (PUPPI$>$0.5)")

# (1,1) multiplicity  -- LOG Y
a = ax[1,1]
n_nom = np.asarray(ak.num(eta)); n_pup = np.asarray(ak.sum(pup>0.5, axis=1))
a.hist(n_nom, bins=70, range=(0,1700), histtype="step", lw=1.6, color=ts.MUTE)
a.hist(n_pup, bins=70, range=(0,1700), histtype="step", lw=1.6, color=ts.ACCENT)
a.set_yscale("log")
ts.label_end(a, 760, 600, rf"nominal $\langle n\rangle{{=}}{n_nom.mean():.0f}$", ts.MUTE, dx=120)
ts.label_end(a, 80, 1500, rf"PUPPI$>$0.5 $\langle n\rangle{{=}}{n_pup.mean():.0f}$", ts.ACCENT, dx=120)
a.set_xlabel("soft particles / event"); a.set_ylabel("events (log)")
a.set_title("(e) pileup multiplicity")

# (1,2) puppi weight dist  (log y)
a = ax[1,2]
a.hist(flatpup, bins=50, color=ts.PALE, edgecolor=ts.MUTE, linewidth=0.4)
a.set_yscale("log"); a.set_xlabel("PUPPI weight"); a.set_ylabel("PF candidates (log)")
a.set_title(rf"(f) PUPPI weights ($\sim${100*np.mean(flatpup<0.5):.0f}\% pileup)")

fig.suptitle("CMS Open Data DoubleMuon Run2016G — soft event $\\to$ $Z$ boost: results & cross-checks (29,600 $Z\\to\\mu\\mu$)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0,0,1,0.975])
fig.savefig(f"{R}/opendata_crosschecks.png"); print("wrote results/opendata_crosschecks.png")
