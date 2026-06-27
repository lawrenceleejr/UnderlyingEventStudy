"""FINAL residual figure: how well the soft event pins the hard-scatter p_z.
(a) p_z residual: predict-the-mean baseline vs EFN(MC) vs EFN(data) -> information gain + data/MC.
(b) pull of the data EFN (per-event sigma) vs a unit Gaussian -> calibration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts
from src import config
ts.use()
R = "results"
ipz = list(config.TARGETS).index("pz_Z")
C_EFN = "#2b6a9c"; C_MC = "#c8772e"

def load(tag):
    p = f"{R}/pz_{tag}_preds.npz"
    return np.load(p) if os.path.exists(p) else None
dat = load("opendata_pz"); mc = load("mc_pz")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) residual (p_z) overlay
if dat is not None:
    yt = dat["yt"][:, ipz]; mu = dat["mu"][:, ipz]; res = mu - yt
    base = yt.mean() - yt                      # predict-the-mean (no longitudinal info)
    rl = np.percentile(np.abs(np.concatenate([res, base])), 99)
    ax[0].hist(base, bins=80, range=(-rl, rl), histtype="step", lw=1.4, ls=(0,(4,2)), color=ts.MUTE)
    if mc is not None:
        rmc = mc["mu"][:, ipz] - mc["yt"][:, ipz]
        ax[0].hist(rmc, bins=80, range=(-rl, rl), histtype="step", lw=1.5, color=C_MC)
        ts.label_end(ax[0], rl*0.45, ax[0].get_ylim()[1]*0.55, "EFN (MC)", C_MC)
    ax[0].hist(res, bins=80, range=(-rl, rl), histtype="step", lw=1.9, color=C_EFN)
    ax[0].axvline(0, color=ts.INK, lw=0.6)
    ts.label_end(ax[0], rl*0.30, ax[0].get_ylim()[1]*0.85, "predict\nthe mean", ts.MUTE)
    ts.label_end(ax[0], -rl*0.97, ax[0].get_ylim()[1]*0.5, "EFN\n(data)", C_EFN, fontweight="bold")
    ts.minimal(ax[0]); ax[0].set_xlabel(r"$p_z^{\rm pred}-p_z^{\rm true}$  [GeV]"); ax[0].set_ylabel("events")
    base_s = base.std(); efn_s = res.std()
    ax[0].set_title(rf"(a) $p_z$ resolution: {efn_s:.0f} GeV vs $\sigma_{{p_z}}$={base_s:.0f} GeV  ({100*(1-efn_s/base_s):.0f}\% tighter)")
else:
    ax[0].text(0.5, 0.5, "data EFN preds pending", ha="center"); ts.minimal(ax[0])

# (b) pull of the data EFN
if dat is not None:
    sg = np.clip(dat["sg"][:, ipz], 1e-6, None); pull = (mu - yt)/sg
    xb = np.linspace(-5, 5, 160)
    ax[1].hist(pull, bins=70, range=(-5, 5), density=True, histtype="step", lw=1.8, color=ts.GOOD)
    ax[1].plot(xb, np.exp(-xb**2/2)/np.sqrt(2*np.pi), ls=(0,(4,2)), lw=1.2, color=ts.MUTE)
    ts.label_end(ax[1], 2.3, 0.33, "unit\nGaussian", ts.MUTE)
    ax[1].axvline(0, color=ts.INK, lw=0.6); ts.minimal(ax[1])
    ax[1].set_xlabel(r"pull  ($(p_z^{\rm pred}-p_z^{\rm true})/\sigma_{\rm pred}$)"); ax[1].set_ylabel("a.u.")
    ax[1].set_title(rf"(b) pull: mean={pull.mean():.2f}, width={pull.std():.2f}")
else:
    ax[1].text(0.5, 0.5, "pending", ha="center"); ts.minimal(ax[1])

fig.suptitle("Hard-scatter $p_z$ from the soft event: residual & pull (data PUPPI$>$0.5; hatch-free MC overlay)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{R}/opendata_final_residual.png"; fig.savefig(out); print("wrote", out)
