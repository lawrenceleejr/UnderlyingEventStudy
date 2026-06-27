"""Does the soft-event longitudinal correction give more MASS sensitivity than m_T?

Sensitivity to a resonance mass from a template fit is set by how strongly the
observable's SHAPE responds to the mass, not by how narrow it is:
    sigma(m)^-2 = sum_bins (dn_i/dm)^2 / n_i   (Poisson Fisher information).
To leading order a resonance mass is a SCALE on each mass-like observable
(m_full, m_T, m_corr all scale ~ m), so m +/- delta templates are built by
scaling the observable. We compare sigma(m_Z) for:
   full dimuon mass (both leptons measured -- the ceiling),
   transverse mass m_T (one longitudinal d.o.f. discarded -- the W-like case),
   soft-corrected m_corr (m_T + soft-event boost).
This is the proxy 'Z-mass measurement' asked for: how much does the longitudinal
information change the statistical uncertainty?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts, config
from src.opendata_validate import split_test_indices
ts.use()
R = "results"; MZ = 91.19
ipz = list(config.TARGETS).index("pz_Z"); iy = list(config.TARGETS).index("y_Z")

d = ak.from_parquet("data/skim/opendata_mumu.parquet")
N = len(d); te = split_test_indices(N, config.SEED)
npz = np.load(f"{R}/pz_opendata_pz_preds.npz"); mu = npz["mu"]
def col(n): return np.asarray(d[n])[te]
pt1, phi1, pz1, E1 = col("mu1_pt"), col("mu1_phi"), col("mu1_pz"), col("mu1_E")
pt2, phi2, pz2, E2 = col("mu2_pt"), col("mu2_phi"), col("mu2_pz"), col("mu2_E")
mass_full = col("mass"); pzZ_pred = mu[:, ipz]

dphi = phi1 - phi2
mT = np.sqrt(np.maximum(2*pt1*pt2*(1 - np.cos(dphi)), 0))
def mass_from_pz(pz2v):
    y1 = np.arcsinh(pz1/pt1); y2 = np.arcsinh(pz2v/pt2)
    return np.sqrt(np.maximum(2*pt1*pt2*(np.cosh(y1 - y2) - np.cos(dphi)), 0))
m_corr = mass_from_pz(pzZ_pred - pz1)

# ---- Fisher sensitivity via scaling templates ----
WIN = (50, 135); NB = 85; edges = np.linspace(*WIN, NB+1)
delta = 1.0  # GeV mass step
def sigma_m(x, n_scale=None):
    x = x[np.isfinite(x)]
    h0, _ = np.histogram(x, bins=edges)
    hp, _ = np.histogram(x*(1+delta/MZ), bins=edges)
    hm, _ = np.histogram(x*(1-delta/MZ), bins=edges)
    if n_scale:                       # scale counts to a reference N
        f = n_scale/max(len(x), 1); h0 = h0*f; hp = hp*f; hm = hm*f
    dndm = (hp - hm)/(2*delta)
    fisher = np.sum(dndm**2/np.maximum(h0, 1.0))
    return (1/np.sqrt(fisher)) if fisher > 0 else np.inf, h0, (hp-hm)

Nref = 100000   # quote sigma(m_Z) for a 100k-event sample
obs = [("full dimuon", mass_full, ts.GOOD), ("soft-corrected", m_corr, "#2b6a9c"), ("transverse $m_T$", mT, ts.MUTE)]
res = {}
for nm, x, c in obs:
    s, h0, diff = sigma_m(x, n_scale=Nref); res[nm] = (s, h0, diff, c)
    print(f"  {nm:16s}: sigma(m_Z) = {s:.3f} GeV  (per {Nref:,} events)")
sT = res["transverse $m_T$"][0]; sC = res["soft-corrected"][0]; sF = res["full dimuon"][0]
print(f"\n  m_corr vs m_T:  sigma ratio = {sC/sT:.2f}  ({'%.0f'%(100*(1-sC/sT))}%% {'smaller' if sC<sT else 'LARGER'})")
print(f"  fraction of the full-mass ideal recovered: m_T {sF/sT:.2f}, m_corr {sF/sC:.2f}")

# ---- resolution scan: how precise must the boost be for the correction to help? ----
std_pzZ = float(np.std(pz1 + pz2))           # spread of the true Z p_z
sig_grid = np.array([0, 10, 20, 30, 45, 60, 80, 107, 140, 180, 230])
rng = np.random.default_rng(7)
sig_curve, corr_curve = [], []
for sp_ in sig_grid:
    pz2_sm = pz2 + (rng.normal(0, sp_, len(pz2)) if sp_ > 0 else 0.0)
    s, _, _ = sigma_m(mass_from_pz(pz2_sm), n_scale=Nref)
    sig_curve.append(s)
    corr_curve.append(np.sqrt(max(0.0, 1 - (sp_/std_pzZ)**2)))
sig_curve = np.array(sig_curve)
# crossover boost resolution where m_corr sensitivity == m_T
cross = np.interp(sT, sig_curve, sig_grid) if sig_curve[0] < sT < sig_curve[-1] else np.nan
print(f"  crossover: boost p_z resolution must be < ~{cross:.0f} GeV (corr > "
      f"~{np.sqrt(max(0,1-(cross/std_pzZ)**2)):.2f}) to beat m_T; current EFN ~107 GeV (corr 0.72)")

# ===================== plot =====================
ctr = 0.5*(edges[1:]+edges[:-1])
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
# (a) where the mass sensitivity lives
for nm, x, c in obs:
    diff = res[nm][2]
    ax[0].step(ctr, np.abs(diff), where="mid", lw=1.7, color=c)
ts.label_end(ax[0], 100, np.max(np.abs(res["soft-corrected"][2]))*1.4, "soft-corrected", "#2b6a9c", fontweight="bold")
ts.label_end(ax[0], 66, np.max(np.abs(res["transverse $m_T$"][2]))*0.9, r"$m_T$", ts.MUTE)
ts.label_end(ax[0], 99, np.max(np.abs(res["full dimuon"][2]))*0.65, "full dimuon", ts.GOOD)
ax[0].axvline(MZ, color=ts.INK, lw=0.6, ls=(0,(1,3))); ts.minimal(ax[0])
ax[0].set_xlabel("observable [GeV]"); ax[0].set_ylabel(r"$|\partial n/\partial m|$  (mass-sensitivity density)")
ax[0].set_title("(a) where the mass information lives (taller = sharper mass discriminant)")
# (b) sigma(m_Z) vs boost resolution -- the crossover
ax[1].plot(sig_grid, sig_curve, "o-", color="#2b6a9c", ms=5, zorder=4)
ax[1].axhline(sT, ls=(0,(4,2)), color=ts.MUTE, lw=1.3)
ts.label_end(ax[1], sig_grid[-1], sT, r"transverse $m_T$", ts.MUTE)
ax[1].scatter([107], [sC], s=70, color=ts.ACCENT2, zorder=6)
ax[1].annotate("current EFN\n(corr 0.72)", (107, sC), xytext=(120, sC+0.004), fontsize=8.5, color=ts.ACCENT2)
ax[1].scatter([0], [sF], s=55, color=ts.GOOD, zorder=6)
ts.label_end(ax[1], 6, sF, "full dimuon\n(perfect)", ts.GOOD)
if np.isfinite(cross):
    ax[1].axvline(cross, ls=(0,(1,3)), color=ts.INK, lw=0.8)
    ax[1].annotate(f"break-even\n$\\sim${cross:.0f} GeV", (cross, sT*0.5), ha="center", fontsize=8)
ax[1].set_xlabel(r"boost $p_z$ resolution [GeV]  (smaller = better estimator)")
ax[1].set_ylabel(rf"$\sigma(m_Z)$ [GeV]  (per {Nref//1000}k events)"); ts.minimal(ax[1])
ax[1].set_title("(b) the correction helps only left of break-even")
fig.suptitle(r"Mass sensitivity of the proxy $Z\to\mu\mu$ fit: the soft boost must be PRECISE to help (it isn't yet)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0,0,1,0.95]); out = f"{R}/opendata_mass_sensitivity.png"; fig.savefig(out); print("wrote", out)
