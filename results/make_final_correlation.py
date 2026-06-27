"""FINAL correlation figure: how much boost information the soft event carries.
(a) data EFN predicted-vs-true y_Z (2-D density) -- the headline.
(b) information gain + data-vs-MC: corr for predict-the-mean, mean-eta, EFN(MC), EFN(data)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts
from src import config
ts.use()
R = "results"
iy = list(config.TARGETS).index("y_Z")

def jload(p): return json.load(open(p)) if os.path.exists(p) else {}
val = jload(f"{R}/validation_opendata.json"); pyt = jload(f"{R}/ablation_pythia_matched.json")

# data EFN predictions (from the heteroscedastic p_z run)
dz = np.load(f"{R}/pz_opendata_pz_preds.npz") if os.path.exists(f"{R}/pz_opendata_pz_preds.npz") else None

# mean-eta (data) correlation, training-free
d = ak.from_parquet("data/skim/opendata.parquet"); yv = np.asarray(d.y_Z)
eta = d.p_eta; pup = d.p_puppi
n = ak.sum(pup > 0.5, axis=1)
me = np.asarray(ak.where(n > 0, ak.sum(eta*(pup > 0.5), axis=1)/np.where(np.asarray(n) > 0, np.asarray(n), 1), 0.0))
me_corr = float(np.corrcoef(me, yv)[0, 1])

efn_data = val.get("headline PUPPI>0.5", {}).get("corr")
efn_mc = pyt.get("all", {}).get("corr")    # Pythia matched (puppi=1 -> all particles)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) data EFN pred vs true y_Z
if dz is not None:
    yt = dz["yt"][:, iy]; mu = dz["mu"][:, iy]
    cc = np.corrcoef(yt, mu)[0, 1]; r2 = 1 - np.sum((mu-yt)**2)/np.sum((yt-yt.mean())**2)
    lim = np.percentile(np.abs(yt), 99)*1.05
    ax[0].hist2d(yt, mu, bins=70, range=[[-lim, lim], [-lim, lim]], cmap=ts.density_cmap(), cmin=1)
    ax[0].plot([-lim, lim], [-lim, lim], ls=(0,(4,2)), lw=1.1, color=ts.ACCENT2)
    ax[0].set_aspect("equal"); ts.minimal(ax[0])
    ax[0].set_xlabel(r"true $y_Z$ (from $\mu^+\mu^-$)"); ax[0].set_ylabel(r"EFN-predicted $y_Z$ (soft event)")
    ax[0].set_title(rf"(a) data: corr={cc:.2f},  $R^2$={r2:.2f}")
else:
    ax[0].text(0.5, 0.5, "data EFN preds pending", ha="center"); ts.minimal(ax[0])

# (b) information gain + data vs MC
labels = ["predict\nthe mean", r"mean-$\eta$" + "\n(data)", "EFN\n(MC)", "EFN\n(data)"]
vals   = [0.0, me_corr, efn_mc if efn_mc is not None else np.nan, efn_data if efn_data is not None else np.nan]
isdata = [True, True, False, True]
cols   = [ts.MUTE, ts.MUTE, ts.ACCENT, ts.ACCENT]
for i, (v, dat, c) in enumerate(zip(vals, isdata, cols)):
    if not np.isfinite(v): continue
    if dat: ax[1].bar(i, v, width=0.66, color=c, alpha=0.55 if c == ts.MUTE else 1.0, edgecolor="none")
    else:   ax[1].bar(i, v, width=0.66, facecolor="none", edgecolor=c, hatch="////", linewidth=0.0)
    ax[1].text(i, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax[1].axhline(0, color=ts.INK, lw=0.7)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_ylabel(r"corr($y_Z^{\rm pred}$, $y_Z^{\rm true}$)"); ax[1].set_ylim(0, 0.85)
ax[1].spines["left"].set_bounds(0, 0.8); ts.minimal(ax[1])
ax[1].set_title("(b) information gain & data vs MC  (hatch = MC)")
# arrow showing the gain
if efn_data is not None:
    ax[1].annotate("", xy=(3, efn_data), xytext=(0, 0.02),
                   arrowprops=dict(arrowstyle="->", color=ts.GOOD, lw=1.3, connectionstyle="arc3,rad=-0.25"))
    ax[1].text(1.5, 0.62, "new\ninformation", color=ts.GOOD, fontsize=9, ha="center")

fig.suptitle("Soft event $\\to$ $Z$ longitudinal boost: information content (PUPPI$>$0.5, 29,600 $Z\\to\\mu\\mu$)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{R}/opendata_final_correlation.png"; fig.savefig(out); print("wrote", out)
print(f"mean-eta(data)={me_corr:.3f}  EFN(data)={efn_data}  EFN(MC)={efn_mc}")
