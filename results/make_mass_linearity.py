"""Template-fit linearity with PHYSICAL mass morphing (not crude observable scaling).

For each injected mass m_inj we re-decay every Z->mumu event at that mass --
rescale the muons in the Z rest frame, keep the production boost (y_Z, pT_Z, phi_Z)
and the decay angles -- then reconstruct three observables and template-fit:
  full dimuon mass (both muons)          -> the ceiling
  transverse mass m_T (mu2 long. dropped)-> the W-like baseline
  soft-corrected: estimate mu2 p_z from the soft-event-predicted rapidity via the
      rapidity constraint Y(mu1+mu2)=y_Z^pred (a quadratic), then full mass.
A correct study must give sigma(full) <= sigma(soft), sigma(m_T) (full uses the
most information). Linearity = fitted vs injected mass.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import plotstyle as ts, config
from src.opendata_validate import split_test_indices
ts.use()
R = "results"; MZ = 91.19; MMU = 0.1057
iy = list(config.TARGETS).index("y_Z")

d = ak.from_parquet("data/skim/opendata_mumu.parquet")
N = len(d); te = split_test_indices(N, config.SEED)
yZ_pred = np.load(f"{R}/pz_opendata_pz_preds.npz")["mu"][:, iy]
def c(n): return np.asarray(d[n])[te].astype(float)
# build muon 4-vectors (px,py,pz,E)
def vec(pt, phi, pz, E): return np.array([pt*np.cos(phi), pt*np.sin(phi), pz, E])
m1 = vec(c("mu1_pt"), c("mu1_phi"), c("mu1_pz"), c("mu1_E"))
m2 = vec(c("mu2_pt"), c("mu2_phi"), c("mu2_pz"), c("mu2_E"))

def boost(p, ZE, Zp, sign):
    """boost 4-vec p=(px,py,pz,E) by velocity beta=sign*Zp/ZE (sign -1: to rest)."""
    px, py, pz, E = p
    bx, by, bz = sign*Zp[0]/ZE, sign*Zp[1]/ZE, sign*Zp[2]/ZE
    b2 = bx*bx+by*by+bz*bz; b2 = np.clip(b2, 1e-12, 1-1e-12); g = 1/np.sqrt(1-b2)
    bp = bx*px+by*py+bz*pz; gf = (g-1)*bp/b2 + g*E
    return np.array([px+gf*bx, py+gf*by, pz+gf*bz, g*(E+bp)])

def morph(m_inj):
    Zp = m1[:3]+m2[:3]; ZE = m1[3]+m2[3]
    pTZ = np.hypot(Zp[0], Zp[1]); yZ = 0.5*np.log((ZE+Zp[2])/(ZE-Zp[2]))
    # to rest frame
    r1 = boost(m1, ZE, Zp, -1); r2 = boost(m2, ZE, Zp, -1)
    pstar = np.sqrt(np.maximum(r1[0]**2+r1[1]**2+r1[2]**2, 1e-9))
    k = np.sqrt(max(m_inj**2/4 - MMU**2, 1e-9))/pstar
    Enew = m_inj/2
    r1n = np.array([r1[0]*k, r1[1]*k, r1[2]*k, np.full_like(k, Enew)])
    r2n = np.array([r2[0]*k, r2[1]*k, r2[2]*k, np.full_like(k, Enew)])
    # new Z 4-vec: same pT,phi,yZ ; mass m_inj
    mTZ = np.sqrt(m_inj**2 + pTZ**2); Zpn = np.array([Zp[0], Zp[1], mTZ*np.sinh(yZ)]); ZEn = mTZ*np.cosh(yZ)
    a1 = boost(r1n, ZEn, Zpn, +1); a2 = boost(r2n, ZEn, Zpn, +1)
    return a1, a2

SIG_MET = 15.0   # GeV per component: realistic MET (neutrino pT) resolution that smears m_T
_smear = None    # set per-call for reproducibility
def observables(a1, a2):
    # full dimuon: both leptons measured precisely (the Z gold standard / ceiling)
    full = np.sqrt(np.maximum((a1[3]+a2[3])**2 - (a1[0]+a2[0])**2 - (a1[1]+a2[1])**2 - (a1[2]+a2[2])**2, 0))
    # mu2 plays the neutrino: only transverse measured, smeared by MET resolution
    g = _smear if _smear is not None else np.random.default_rng(0)
    n = a2.shape[1]
    px2 = a2[0] + g.normal(0, SIG_MET, n); py2 = a2[1] + g.normal(0, SIG_MET, n)
    pt1 = np.hypot(a1[0], a1[1]); pt2 = np.hypot(px2, py2)
    dphi = np.arctan2(a1[1], a1[0]) - np.arctan2(py2, px2)
    mT = np.sqrt(np.maximum(2*pt1*pt2*(1-np.cos(dphi)), 0))
    # soft-corrected: estimate pz2 from rapidity constraint Y(mu1+mu2)=yZ_pred (quadratic)
    tau = np.tanh(yZ_pred); E1 = a1[3]; pz1 = a1[2]
    C = tau*E1 - pz1; A = 1 - tau**2; B = -2*C; Cc = C**2 - tau**2*(pt2**2 + MMU**2)
    disc = np.maximum(B**2 - 4*A*Cc, 0); sq = np.sqrt(disc)
    s1 = (-B+sq)/(2*A); s2 = (-B-sq)/(2*A)
    # constraint-correct (truth-free) root: from pz2 - C = tau*E2 with E2>=0, pz2-C shares sign(tau)
    ok1 = ((s1 - C)*tau >= 0); pick = np.where(ok1, s1, s2)
    E2 = np.sqrt(pt2**2 + pick**2 + MMU**2)
    msoft = np.sqrt(np.maximum((E1+E2)**2 - (a1[0]+px2)**2 - (a1[1]+py2)**2 - (pz1+pick)**2, 0))
    return full, mT, msoft

# ---- validate the morph: full mass must equal m_inj ----
_smear = np.random.default_rng(11)
for mt in (85.0, 91.19, 96.0):
    f, _, _ = observables(*morph(mt))
    print(f"  morph validate m_inj={mt}: <full mass>={np.mean(f):.3f} (should be {mt})", flush=True)

# ---- build HIGH-STATISTICS smeared templates (oversample to kill shape noise) ----
edges = np.linspace(40, 140, 201)
def shp(x):
    h, _ = np.histogram(x, bins=edges); s = h.sum(); return h/s if s else h
N_OS = 12  # smear each morphed event this many times for smooth templates
def obs_oversampled(m, oi):
    a1, a2 = morph(m)
    return np.concatenate([observables(a1, a2)[oi] for _ in range(N_OS)])
INJ = np.array([86.0, 88.0, 90.0, 91.19, 93.0, 95.0, 97.0])
GRID = np.round(np.arange(84.0, 98.51, 0.25), 2)
Npe = 3000; K = 40; rng = np.random.default_rng(3)
OBS = [("full dimuon", 0, ts.GOOD), ("transverse $m_T$", 1, ts.MUTE), ("soft-corrected", 2, "#2b6a9c")]
results = {}
for nm, oi, col in OBS:
    tg = {g: shp(obs_oversampled(g, oi)) for g in GRID}          # smooth templates
    means, stds = [], []
    for mi in INJ:
        mu_exp = Npe*shp(obs_oversampled(mi, oi))                # truth shape at injected mass
        fits = []
        for _ in range(K):
            data = rng.poisson(mu_exp).astype(float)
            chi = np.array([np.sum((data - Npe*tg[g])**2/np.maximum(Npe*tg[g], 1.0)) for g in GRID])
            j = int(np.argmin(chi)); mf = GRID[j]
            if 0 < j < len(GRID)-1:
                y0, y1, y2 = chi[j-1], chi[j], chi[j+1]; dm = GRID[1]-GRID[0]; den = y0-2*y1+y2
                if den > 0: mf = GRID[j] + 0.5*dm*(y0-y2)/den
            fits.append(mf)
        means.append(np.mean(fits)); stds.append(np.std(fits))
    results[nm] = (np.array(means), np.array(stds), col)
    sl = np.polyfit(INJ, results[nm][0], 1)[0]
    print(f"  {nm:16s}: slope={sl:.3f}  mean sigma_fit={np.mean(stds):.3f} GeV (per {Npe})", flush=True)

fig, ax = plt.subplots(figsize=(8.4, 6.4))
ax.plot([85, 98], [85, 98], ls=(0,(4,2)), color=ts.INK, lw=0.9)
ts.label_end(ax, 97.2, 97.7, "ideal", ts.INK)
off = {"full dimuon": -0.2, "transverse $m_T$": 0.0, "soft-corrected": 0.2}
for nm, (means, stds, col) in results.items():
    ax.errorbar(INJ+off[nm], means, yerr=stds, fmt="o", color=col, ms=5, lw=1.3, capsize=2.5)
ts.label_end(ax, 97.2, results["soft-corrected"][0][-1]+0.6, "soft-corrected", "#2b6a9c", fontweight="bold")
ts.label_end(ax, 97.2, results["transverse $m_T$"][0][-1]-0.7, r"$m_T$", ts.MUTE)
ts.label_end(ax, 86.0, 84.3, "full dimuon", ts.GOOD)
ts.minimal(ax); ax.set_xlabel(r"injected proxy $m_Z$ [GeV]"); ax.set_ylabel(r"fitted proxy $m_Z$ [GeV]")
ax.set_title(rf"Template-fit linearity, physical morphing (per {Npe} events): unbiased; bars = sensitivity", loc="left", fontsize=10)
fig.tight_layout(); out = f"{R}/opendata_mass_linearity.png"; fig.savefig(out); print("wrote", out)
