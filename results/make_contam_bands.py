"""Residual-resolution PROFILE vs PILEUP CONTAMINATION (the useful 'dig-out' axis).

The PUPPI weight is bimodal (93% exactly 0), so 'residual vs PUPPI threshold' is a
single step and otherwise flat. The continuous knob is pileup *contamination*: start
from the clean leading-vertex set (PUPPI>0.5, <n>~50) and add back a random fraction
of the pileup, sweeping <n> up to the full ~730. This shows how the boost signal is
buried as pileup floods the event (read right->left for 'dug out by removing pileup').

Bands are +/- sigma_res = sigma_y*sqrt(1-corr^2). The dotted line at sigma_y is the
no-information limit (predict-the-mean); bands below it are the improvement. Note the
sqrt compression: corr 0.75 -> residual only 0.66*sigma_y, not ~0.
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
val = jload(f"{R}/validation_opendata.json"); abl = jload(f"{R}/ablation_opendata.json")
contam = jload(f"{R}/contam_opendata.json")     # EFN corr at intermediate contamination

d = ak.from_parquet("data/skim/opendata.parquet"); y = np.asarray(d.y_Z); syd = y.std()
eta = d.p_eta; pup = d.p_puppi
counts = ak.num(eta); flatc = ak.to_numpy(counts)
rng = np.random.default_rng(909)
rflat = ak.unflatten(rng.random(int(ak.sum(counts))), counts)   # one random draw / particle
clean = pup > 0.5

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[m], b[m])[0, 1])
def mean_eta(mask):
    n = ak.sum(mask, axis=1)
    return np.asarray(ak.where(n > 0, ak.sum(eta*mask, axis=1)/np.where(np.asarray(n) > 0, np.asarray(n), 1), 0.0))
def sig(c, sy=syd): return sy*np.sqrt(max(0.0, 1.0-c*c))

# ---- mean-eta vs contamination (training-free, dense) ----
fs = np.linspace(0, 1, 21); me_n, me_c = [], []
for f in fs:
    mask = clean | (rflat < f)
    me_n.append(float(ak.mean(ak.sum(mask, axis=1)))); me_c.append(corr(mean_eta(mask), y))
me_n = np.array(me_n); me_c = np.array(me_c)
me_sig = np.array([sig(c) for c in me_c])

# ---- EFN points: anchors (clean f=0, full f=1) + any scanned intermediates ----
efn = {0.0: val.get("headline PUPPI>0.5", {}).get("corr", None),
       1.0: abl.get("nominal (all soft, no PU cut)", {}).get("corr", None)}
for k, v in contam.items():
    try: efn[float(k)] = v["corr"]
    except Exception: pass
efn = {f: c for f, c in efn.items() if c is not None}
ef_f = np.array(sorted(efn)); ef_c = np.array([efn[f] for f in ef_f])
# map f -> <n> via the mean-eta scan (same contamination construction)
ef_n = np.interp(ef_f, fs, me_n); ef_sig = np.array([sig(c) for c in ef_c])

# ===================== plot =====================
import matplotlib as mpl; mpl.rcParams["hatch.linewidth"] = 0.6
fig, ax = plt.subplots(figsize=(9.2, 6.2))
C_EFN = "#2b6a9c"; C_ETA = "#c8772e"

# no-information limit (predict-the-mean): residual = sigma_y
ax.axhline(syd, ls=(0,(2,2)), color=ts.INK, lw=0.8, zorder=5)
ax.axhline(-syd, ls=(0,(2,2)), color=ts.INK, lw=0.8, zorder=5)
ax.annotate(r"no-information limit ($\pm\sigma_y$, predict-the-mean)", (me_n.max(), syd),
            xytext=(me_n.max(), syd+0.06), ha="right", fontsize=8, color=ts.INK)

# mean-eta band (ochre, data fill)
ax.fill_between(me_n, -me_sig, me_sig, facecolor=C_ETA, alpha=0.22, edgecolor=C_ETA, lw=1.1, zorder=3)
# EFN band (blue, data fill) -- interp across <n>
order = np.argsort(ef_n)
ax.fill_between(ef_n[order], -ef_sig[order], ef_sig[order], facecolor=C_EFN, alpha=0.24, edgecolor=C_EFN, lw=1.3, zorder=4)
ax.plot(ef_n[order], ef_sig[order], "o", color=C_EFN, ms=5, zorder=6)
ax.plot(ef_n[order], -ef_sig[order], "o", color=C_EFN, ms=5, zorder=6)

ax.axhline(0, color=ts.INK, lw=0.7, zorder=4)
# direct labels
ax.annotate("EFN", (ef_n[order][0], ef_sig[order][0]), xytext=(ef_n[order][0]+18, ef_sig[order][0]-0.06),
            color=C_EFN, fontsize=10, fontweight="bold")
ax.annotate(r"mean-$\eta$", (me_n[2], me_sig[2]), xytext=(me_n[2]+15, me_sig[2]+0.05),
            color=C_ETA, fontsize=10, fontweight="bold")
# working-point markers
ax.annotate("clean\n(PUPPI$>$0.5)", (me_n[0], -syd-0.02), xytext=(me_n[0], -syd-0.28),
            ha="center", fontsize=8, color=ts.INK)
ax.annotate("full event\n(pileup)", (me_n[-1], -syd-0.02), xytext=(me_n[-1], -syd-0.28),
            ha="center", fontsize=8, color=ts.INK)

ax.set_xlabel(r"$\langle$soft particles / event$\rangle$  (pileup added back $\to$)")
ax.set_ylabel(r"residual  $y_Z^{\rm pred}-y_Z^{\rm true}$  ($\pm1\sigma$ band)")
ax.set_xlim(0, me_n.max()*1.05); ax.set_ylim(-2.0, 2.0)
ax.spines["left"].set_bounds(-1.3, 1.3); ax.spines["bottom"].set_bounds(me_n.min(), me_n.max())
ax.set_title("Boost information dug out by removing pileup (data; EFN vs mean-$\\eta$)", loc="left")
fig.tight_layout()
out = f"{R}/opendata_contam_bands.png"; fig.savefig(out)
print("wrote", out)
print("EFN contamination points (f -> corr, <n>):", {round(f,2): (round(efn[f],3), int(n)) for f, n in zip(ef_f, ef_n)})
print("mean-eta corr range: %.3f (clean) -> %.3f (full)" % (me_c[0], me_c[-1]))
