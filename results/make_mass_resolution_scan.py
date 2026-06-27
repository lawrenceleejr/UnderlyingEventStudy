"""Can a FIT to the soft-corrected MASS DISTRIBUTION beat m_T -- and how good must
the boost be? Distribution-level (not per-event): template-fit sensitivity sigma(m_Z)
of the soft-corrected mass as a function of the boost (rapidity) resolution.

Faithful proxy: mu2 = neutrino (transverse only, MET-smeared); physical mass morphing;
soft-corrected mass via the rapidity constraint using a boost estimate of rapidity
resolution sigma_y (sigma_y=0 -> perfect boost -> full mass; large -> useless).
m_T is the flat reference. The current EFN sits at sigma_y from its measured corr.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts, config
from src.opendata_validate import split_test_indices
ts.use()
R = "results"; MZ = 91.19; MMU = 0.1057; SIG_MET = 15.0
iy = list(config.TARGETS).index("y_Z")

d = ak.from_parquet("data/skim/opendata_mumu.parquet"); N = len(d); te = split_test_indices(N, config.SEED)
yZ_pred = np.load(f"{R}/pz_opendata_pz_preds.npz")["mu"][:, iy]
def c(n): return np.asarray(d[n])[te].astype(float)
def vec(pt, phi, pz, E): return np.array([pt*np.cos(phi), pt*np.sin(phi), pz, E])
m1 = vec(c("mu1_pt"), c("mu1_phi"), c("mu1_pz"), c("mu1_E"))
m2 = vec(c("mu2_pt"), c("mu2_phi"), c("mu2_pz"), c("mu2_E"))
ZE0 = m1[3]+m2[3]; Zp0 = m1[:3]+m2[:3]; yZ_true = 0.5*np.log((ZE0+Zp0[2])/(ZE0-Zp0[2]))
sig_yZ = float(np.std(yZ_true)); corr_efn = float(np.corrcoef(yZ_true, yZ_pred)[0, 1])

def boost(p, ZE, Zp, sign):
    px, py, pz, E = p; bx, by, bz = sign*Zp[0]/ZE, sign*Zp[1]/ZE, sign*Zp[2]/ZE
    b2 = np.clip(bx*bx+by*by+bz*bz, 1e-12, 1-1e-12); g = 1/np.sqrt(1-b2)
    bp = bx*px+by*py+bz*pz; gf = (g-1)*bp/b2 + g*E
    return np.array([px+gf*bx, py+gf*by, pz+gf*bz, g*(E+bp)])
def morph(m_inj):
    pTZ = np.hypot(Zp0[0], Zp0[1])
    r1 = boost(m1, ZE0, Zp0, -1); r2 = boost(m2, ZE0, Zp0, -1)
    pstar = np.sqrt(np.maximum(r1[0]**2+r1[1]**2+r1[2]**2, 1e-9)); k = np.sqrt(max(m_inj**2/4-MMU**2, 1e-9))/pstar
    r1n = np.array([r1[0]*k, r1[1]*k, r1[2]*k, np.full_like(k, m_inj/2)])
    r2n = np.array([r2[0]*k, r2[1]*k, r2[2]*k, np.full_like(k, m_inj/2)])
    mTZ = np.sqrt(m_inj**2+pTZ**2); Zpn = np.array([Zp0[0], Zp0[1], mTZ*np.sinh(yZ_true)]); ZEn = mTZ*np.cosh(yZ_true)
    return boost(r1n, ZEn, Zpn, +1), boost(r2n, ZEn, Zpn, +1)

def mT_obs(a1, a2, g):
    px2 = a2[0]+g.normal(0, SIG_MET, a2.shape[1]); py2 = a2[1]+g.normal(0, SIG_MET, a2.shape[1])
    pt1 = np.hypot(a1[0], a1[1]); pt2 = np.hypot(px2, py2)
    dphi = np.arctan2(a1[1], a1[0]) - np.arctan2(py2, px2)
    return np.sqrt(np.maximum(2*pt1*pt2*(1-np.cos(dphi)), 0))
def soft_obs(a1, a2, yest, g):
    px2 = a2[0]+g.normal(0, SIG_MET, a2.shape[1]); py2 = a2[1]+g.normal(0, SIG_MET, a2.shape[1])
    pt2 = np.hypot(px2, py2); tau = np.tanh(yest); E1 = a1[3]; pz1 = a1[2]
    C = tau*E1 - pz1; A = 1-tau**2; B = -2*C; Cc = C**2 - tau**2*(pt2**2+MMU**2)
    disc = np.maximum(B**2-4*A*Cc, 0); sq = np.sqrt(disc)
    s1 = (-B+sq)/(2*A); s2 = (-B-sq)/(2*A); pick = np.where(np.abs(s1-a2[2]) < np.abs(s2-a2[2]), s1, s2)
    E2 = np.sqrt(pt2**2+pick**2+MMU**2)
    return np.sqrt(np.maximum((E1+E2)**2-(a1[0]+px2)**2-(a1[1]+py2)**2-(pz1+pick)**2, 0))

edges = np.linspace(40, 140, 201); N_OS = 12; Npe = 3000
GRID = np.round(np.arange(85.0, 97.51, 0.25), 2)
A1A2 = {g: morph(g) for g in GRID}
iMZ = int(np.argmin(np.abs(GRID-MZ)))
def shp(x):
    h, _ = np.histogram(x, bins=edges); s = h.sum(); return h/s if s else h
rng = np.random.default_rng(5); K = 60
def throw_sigma(obs_at_mass):
    # smooth (oversampled) templates on the grid; Poisson pseudo-data at m_Z; fit -> spread
    T = {g: shp(np.concatenate([obs_at_mass(g) for _ in range(N_OS)])) for g in GRID}
    muexp = Npe*T[GRID[iMZ]]; fits = []
    for _ in range(K):
        data = rng.poisson(muexp).astype(float)
        chi = np.array([np.sum((data-Npe*T[g])**2/np.maximum(Npe*T[g], 1.0)) for g in GRID])
        j = int(np.argmin(chi)); mf = GRID[j]
        if 0 < j < len(GRID)-1:
            y0, y1, y2 = chi[j-1], chi[j], chi[j+1]; dm = GRID[1]-GRID[0]; den = y0-2*y1+y2
            if den > 0: mf = GRID[j]+0.5*dm*(y0-y2)/den
        fits.append(mf)
    return float(np.std(fits))

sig_mT = throw_sigma(lambda g: mT_obs(*A1A2[g], rng))
sig_soft_efn = throw_sigma(lambda g: soft_obs(*A1A2[g], yZ_pred, rng))
SY = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1])
sig_scan = []
for sy in SY:
    yest = yZ_true + (rng.normal(0, sy, len(yZ_true)) if sy > 0 else 0.0)
    sig_scan.append(throw_sigma(lambda g, ye=yest: soft_obs(*A1A2[g], ye, rng)))
sig_scan = np.array(sig_scan)
corr_scan = np.sqrt(np.maximum(1 - (SY/sig_yZ)**2, 0))
sy_efn = sig_yZ*np.sqrt(max(1-corr_efn**2, 0))
cross = np.interp(sig_mT, sig_scan, SY) if sig_scan[0] < sig_mT < sig_scan[-1] else np.nan
print(f"  m_T sigma(m_Z) = {sig_mT:.3f} GeV (per {Npe})")
print(f"  soft-corrected @ EFN (corr_y={corr_efn:.2f}, sigma_y={sy_efn:.2f}) = {sig_soft_efn:.3f} GeV")
print(f"  crossover: boost needs sigma_y < ~{cross:.2f} (corr_y > ~{np.sqrt(max(0,1-(cross/sig_yZ)**2)):.2f}) to beat m_T")
print("  scan sigma_y -> sigma(m_Z):", dict(zip(np.round(SY,2), np.round(sig_scan,3))))

# ===================== plot =====================
fig, ax = plt.subplots(figsize=(8.8, 6.2))
ax.plot(SY, sig_scan, "o-", color="#2b6a9c", ms=5, zorder=4)
ax.axhline(sig_mT, ls=(0,(4,2)), color=ts.MUTE, lw=1.4)
ts.label_end(ax, SY[-1], sig_mT, r"transverse $m_T$", ts.MUTE)
ax.scatter([sy_efn], [sig_soft_efn], s=90, color=ts.ACCENT2, zorder=6)
ax.annotate(f"current EFN\n(corr$_y$={corr_efn:.2f})", (sy_efn, sig_soft_efn),
            xytext=(sy_efn+0.06, sig_soft_efn+0.12), fontsize=8.5, color=ts.ACCENT2,
            arrowprops=dict(arrowstyle="->", lw=0.6, color=ts.ACCENT2))
ax.scatter([0], [sig_scan[0]], s=55, color=ts.GOOD, zorder=6)
ts.label_end(ax, 0.02, sig_scan[0], "perfect boost\n($\\to$ full mass)", ts.GOOD)
if np.isfinite(cross):
    ax.axvline(cross, ls=(0,(1,3)), color=ts.INK, lw=0.8)
    ax.annotate(f"break-even\n$\\sigma_y\\!\\approx${cross:.2f}", (cross, sig_mT*1.7), ha="center", fontsize=8)
ax.fill_betweenx([0, sig_mT], 0, cross if np.isfinite(cross) else 0, color=ts.GOOD, alpha=0.07)
ax.set_xlabel(r"boost rapidity resolution $\sigma_y$  (smaller = better estimator)")
ax.set_ylabel(rf"$\sigma(m_Z)$ from the DISTRIBUTION fit [GeV]  (per {Npe} events)")
ax.set_ylim(0, max(sig_scan)*1.1); ts.minimal(ax)
ax.set_title(r"Fitting the soft-corrected mass DISTRIBUTION: beats $m_T$ left of break-even", loc="left", fontsize=10.5)
fig.tight_layout(); out = f"{R}/opendata_mass_distfit_scan.png"; fig.savefig(out); print("wrote", out)
