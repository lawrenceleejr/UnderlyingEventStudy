"""Residual-resolution PROFILE vs the PUPPI cut, drawn as filled bands.

For a calibrated estimator the residual y^pred - y^true has width
    sigma_res = sigma_y * sqrt(1 - corr^2)
(exact for the optimal linear predictor; verified R^2 ~ corr^2 for the EFN). So a
band +/- sigma_res(threshold) summarises the residual distribution at each PUPPI
working point. Overlaying bands lets us compare, on one axis:
    EFN (data)  vs  EFN (Pythia MC)  vs  mean-eta (data)  vs  mean-eta (MC).
Pythia has no pileup (puppi=1), so its PUPPI cut is a no-op -> flat bands: the
reference the data approaches once pileup is removed.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts
ts.use()
R = "results"

def jload(p): return json.load(open(p)) if os.path.exists(p) else {}
val = jload(f"{R}/validation_opendata.json")
abl = jload(f"{R}/ablation_opendata.json")
sysj = jload(f"{R}/systematics_opendata.json")
pyt = jload(f"{R}/ablation_pythia_matched.json")

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
def mean_eta(eta, mask):
    n = ak.sum(mask, axis=1)
    return np.asarray(ak.where(n > 0, ak.sum(eta*mask, axis=1)/np.where(np.asarray(n) > 0, np.asarray(n), 1), 0.0))

# ---- samples ----
dd = ak.from_parquet("data/skim/opendata.parquet"); yd = np.asarray(dd.y_Z); syd = yd.std()
pp = ak.from_parquet("data/skim/pythia_demo.parquet"); yp = np.asarray(pp.y_Z); syp = yp.std()
allm_d = ak.ones_like(dd.p_eta, dtype=bool); allm_p = ak.ones_like(pp.p_eta, dtype=bool)
# threshold 0 == "keep all" (no cut). PUPPI weights are bimodal (93% exactly 0),
# so any cut>0 removes the pileup as a step.
THR = np.concatenate([[0.0], np.linspace(0.03, 0.7, 14)])

def sig(sy, c):  # residual width from correlation
    return sy * np.sqrt(max(0.0, 1.0 - c*c))

# ---- mean-eta (training-free) corr vs threshold ----
me_data = np.array([corr(mean_eta(dd.p_eta, allm_d if t <= 0 else (dd.p_puppi > t)), yd) for t in THR])
me_mc_c = corr(mean_eta(pp.p_eta, allm_p), yp)        # puppi=1 -> flat (no pileup)
sig_me_data = np.array([sig(syd, c) for c in me_data])
sig_me_mc = np.full_like(THR, sig(syp, me_mc_c))

# ---- EFN corr vs threshold (data), from the various run jsons ----
efn_pts = {}
if "nominal (all soft, no PU cut)" in abl: efn_pts[0.0] = abl["nominal (all soft, no PU cut)"]["corr"]
if "PUPPI>0.1" in abl: efn_pts[0.1] = abl["PUPPI>0.1"]["corr"]
if "PUPPI>0.3" in sysj: efn_pts[0.3] = sysj["PUPPI>0.3"]["corr"]
if "headline PUPPI>0.5" in val: efn_pts[0.5] = val["headline PUPPI>0.5"]["corr"]
if "PUPPI>0.7" in sysj: efn_pts[0.7] = sysj["PUPPI>0.7"]["corr"]
efn_t = np.array(sorted(efn_pts)); efn_c = np.array([efn_pts[t] for t in efn_t])
# EFN data band: full width at no-cut (t=0, all particles), then the cleaned
# regime for t>0 (interpolated among the measured >0 thresholds). Step at 0.
pos = sorted(t for t in efn_pts if t > 0)
if pos:
    pos_s = [sig(syd, efn_pts[t]) for t in pos]
    s0 = sig(syd, efn_pts.get(0.0, 0.0))
    sig_efn_data = np.array([s0 if t <= 0 else np.interp(t, pos, pos_s) for t in THR])
else:
    sig_efn_data = None
# EFN MC (Pythia, all particles) -> flat
efn_mc_c = pyt.get("PUPPI>0.5 (leading-vertex)", {}).get("corr")
sig_efn_mc = np.full_like(THR, sig(syp, efn_mc_c)) if efn_mc_c is not None else None

# ===================== plot =====================
import matplotlib as mpl
mpl.rcParams["hatch.linewidth"] = 0.6
fig, ax = plt.subplots(figsize=(9.0, 6.2))

# method = colour (curated, distinct, pleasant); sample = fill style
C_EFN = "#2b6a9c"   # steel blue
C_ETA = "#c8772e"   # ochre

def band_data(t, s, color):   # data: solid fill, low opacity
    ax.fill_between(t, -s, s, facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.1, zorder=3)
def band_mc(t, s, color):     # MC: hatched fill, no solid
    ax.fill_between(t, -s, s, facecolor="none", edgecolor=color, hatch="////", linewidth=0.0, zorder=2)

# draw widest (mean-eta) behind, EFN in front; MC behind data within each method
band_mc(THR, sig_me_mc, C_ETA)
band_data(THR, sig_me_data, C_ETA)
if sig_efn_mc is not None: band_mc(THR, sig_efn_mc, C_EFN)
if sig_efn_data is not None: band_data(THR, sig_efn_data, C_EFN)

ax.axhline(0, color=ts.INK, lw=0.7, zorder=4)
# the actual measured EFN-data thresholds, as points on the band edges
if len(efn_t):
    se = [sig(syd, efn_pts[t]) for t in efn_t]
    ax.plot(efn_t, se, "o", color=C_EFN, ms=4.5, zorder=6)
    ax.plot(efn_t, [-v for v in se], "o", color=C_EFN, ms=4.5, zorder=6)

# direct labels (no legend box); colour = method, " (data)/(MC)" notes the fill
xr = THR[-1]
def lab(y, txt, col, **kw): ax.annotate(txt, (xr+0.012, y), color=col, fontsize=9.5, va="center", **kw)
if sig_efn_data is not None: lab(sig_efn_data[-1], "EFN (data)", C_EFN, fontweight="bold")
if sig_efn_mc is not None:   lab(sig_efn_mc[-1], "EFN (MC)", C_EFN)
lab(sig_me_data[-1], r"mean-$\eta$ (data)", C_ETA, fontweight="bold")
if sig_me_mc is not None:    lab(-sig_me_mc[-1]*0.93, r"mean-$\eta$ (MC)", C_ETA)

# headline working point marker + endpoints annotated below the axis
ax.axvline(0.5, color=ts.INK, lw=0.6, ls=(0,(1,2)), zorder=4)
ax.annotate("headline\nPUPPI$>$0.5", (0.5, 1.16), ha="center", fontsize=8, color=ts.INK)
ax.annotate("keep all\n(washed out)", (0.01, 1.16), ha="left", fontsize=8, color=ts.INK)

ax.set_xlabel("PUPPI weight threshold"); ax.set_ylabel(r"residual  $y_Z^{\rm pred}-y_Z^{\rm true}$  ($\pm1\sigma$ band)")
ax.set_xlim(-0.02, 0.82); ax.set_ylim(-1.3, 1.3)
ax.spines["left"].set_bounds(-1.1, 1.1); ax.spines["bottom"].set_bounds(0, 0.7)
ax.set_title("Residual resolution vs pileup suppression — narrower is better; fill = data, hatch = MC", loc="left")
fig.tight_layout()
out = f"{R}/opendata_residual_bands.png"; fig.savefig(out)
print("wrote", out)
print("EFN data thresholds:", dict(zip([round(t,1) for t in efn_t], [round(c,3) for c in efn_c])))
print("mean-eta MC corr=%.3f  EFN MC corr=%s" % (me_mc_c, efn_mc_c))
