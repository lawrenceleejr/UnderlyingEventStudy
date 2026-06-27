"""Z->mumu mass-reconstruction closure: does the soft-event boost narrow the mass?

Mimic W->l nu in Z->mumu (where the truth is fully known): treat mu1 as the
measured lepton and DISCARD mu2's longitudinal momentum (the 'neutrino'). Then:
  - true mass        : the real dimuon mass (~91 GeV) -- the ceiling
  - transverse mass  : m_T^2 = 2 pT1 pT2 (1 - cos dphi)   (no longitudinal info)
  - soft-corrected   : put dy back using the soft-event-predicted Z p_z:
        pz2_est = pz_Z^pred - pz1,  dy = y1 - y2_est,
        m^2 = 2 pT1 pT2 (cosh dy - cos dphi)
  - truth-corrected  : same with the true pz2 (closure -> recovers the peak)

The EFN p_z prediction comes from the soft event only (never sees the muons), so
this is a faithful, non-circular analog of the W case.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts, config
from src.opendata_validate import split_test_indices
ts.use()
R = "results"
iy = list(config.TARGETS).index("y_Z"); ipz = list(config.TARGETS).index("pz_Z")

d = ak.from_parquet("data/skim/opendata_mumu.parquet")
N = len(d); te = split_test_indices(N, config.SEED)
npz = np.load(f"{R}/pz_opendata_pz_preds.npz")
yt = npz["yt"]; mu = npz["mu"]
assert len(yt) == len(te), f"npz test ({len(yt)}) != reproduced test ({len(te)})"
# alignment sanity: npz y_Z (test) must match the parquet's y_Z[test_idx]
yZ_parq = np.asarray(d["y_Z"])[te]
align = float(np.max(np.abs(yt[:, iy] - yZ_parq)))
print("alignment max|dy_Z| =", align, "(should be ~0)")
assert align < 1e-3, "event order mismatch between parquet and preds!"

def col(name): return np.asarray(d[name])[te]
pt1, phi1, pz1, E1 = col("mu1_pt"), col("mu1_phi"), col("mu1_pz"), col("mu1_E")
pt2, phi2, pz2, E2 = col("mu2_pt"), col("mu2_phi"), col("mu2_pz"), col("mu2_E")
mass_true = col("mass")
pzZ_pred = mu[:, ipz]; pzZ_true = pz1 + pz2

dphi = phi1 - phi2
mT = np.sqrt(np.maximum(2*pt1*pt2*(1 - np.cos(dphi)), 0))
def mass_from_pz(pz2v):
    y1 = np.arcsinh(pz1/pt1); y2 = np.arcsinh(pz2v/pt2)
    return np.sqrt(np.maximum(2*pt1*pt2*(np.cosh(y1 - y2) - np.cos(dphi)), 0))
m_soft = mass_from_pz(pzZ_pred - pz1)        # soft-event-corrected
m_truth = mass_from_pz(pz2)                  # closure (uses true pz2)

def stats(x):
    lo, hi = np.percentile(x, [16, 84]); med = np.median(x)
    core = x[(x > 75) & (x < 107)]                 # core (peak) region
    core_sig = core.std() if len(core) > 10 else np.nan
    frac = np.mean((x > 86) & (x < 96))
    return dict(median=med, half_iqr=(hi-lo)/2, core_sig=core_sig, frac=frac, peak=med)
ST = {}
for nm, x in [("transverse m_T", mT), ("soft-corrected", m_soft),
              ("truth-corrected", m_truth), ("true dimuon", mass_true)]:
    s = stats(x); ST[nm] = s
    print(f"  {nm:16s}: median={s['median']:5.1f}  bias={s['median']-91.2:+5.1f}  "
          f"core-sigma={s['core_sig']:4.1f}  half-IQR={s['half_iqr']:5.1f}  frac[86,96]={s['frac']:.2f}")

# ===================== plot =====================
fig, ax = plt.subplots(figsize=(8.8, 6.0))
b = np.linspace(20, 140, 80)
ax.hist(mass_true, bins=b, histtype="step", lw=1.8, color=ts.GOOD, density=True)
ax.hist(mT, bins=b, histtype="step", lw=1.8, color=ts.MUTE, density=True)
ax.hist(m_soft, bins=b, histtype="step", lw=2.0, color="#2b6a9c", density=True)
ax.hist(m_truth, bins=b, histtype="step", lw=1.2, ls=(0,(3,2)), color=ts.INK, density=True)
ax.axvline(91.2, color=ts.INK, lw=0.6, ls=(0,(1,3)))

ymax = ax.get_ylim()[1]
ts.label_end(ax, 40, ymax*0.32, f"transverse $m_T$\nbias $-${91.2-ST['transverse m_T']['median']:.0f} GeV", ts.MUTE)
ts.label_end(ax, 100, ymax*0.66, f"soft-corrected\npeak$\\to m_Z$, tails", "#2b6a9c", fontweight="bold")
ts.label_end(ax, 96, ymax*0.97, "true dimuon", ts.GOOD)
ts.label_end(ax, 101, ymax*0.45, "truth-corrected\n(closure)", ts.INK)
ts.minimal(ax)
ax.set_xlabel(r"reconstructed mass [GeV]"); ax.set_ylabel("a.u.")
ax.set_title(r"$Z\to\mu\mu$ closure: soft-event boost removes the $m_T$ bias (peak $\to m_Z$); "
             r"per-event tails set by the $p_z$ resolution", loc="left", fontsize=10)
fig.tight_layout(); out = f"{R}/opendata_zmass_closure.png"; fig.savefig(out); print("wrote", out)
