"""Generate W->mu nu events with Pythia8 for the W-boost / W-mass study.

The per-event schema reuses the soft-particle features and target NAMES of the
Z sample (y_Z, pz_Z, beta_z now hold the *W* truth boost) so a model trained on
Z applies directly to W. Extra columns store the muon 4-vector and the neutrino
components needed to reconstruct the W:
  mu_px,mu_py,mu_pz,mu_E      -- the measured charged lepton
  nu_px,nu_py                 -- the *observable* neutrino transverse momentum
  nu_pz_true                  -- truth neutrino p_z (for validation only)

    PYTHONPATH=/opt/pythia8312/lib:. python -m sim.generate_w --nevents 100000 --out data/skim/w.parquet
"""
from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np
import vector

from src import config
from src import softparticles as sp
from src.skim import PART_FEATURES

vector.register_awkward()

NEUTRINOS = {12, 14, 16}
MU_PT_MIN = 25.0      # typical W-analysis lepton cut
MU_ETA_MAX = 2.4
NU_PT_MIN = 25.0      # ~ MET cut


def make_pythia(seed, ecm, mpi, tune):
    import pythia8
    p = pythia8.Pythia("", False)
    p.readString("Print:quiet = on")
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {ecm}")
    p.readString("WeakSingleBoson:ffbar2W = on")
    p.readString("24:onMode = off")
    p.readString("24:onIfAny = 13")          # W -> mu nu
    p.readString("PhaseSpace:mHatMin = 40.")
    p.readString(f"Tune:pp = {tune}")
    p.readString(f"PartonLevel:MPI = {'on' if mpi else 'off'}")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("HadronLevel:Hadronize = on")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")
    p.init()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nevents", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--ecm", type=float, default=13000.0)
    ap.add_argument("--tune", type=int, default=14)
    ap.add_argument("--no-mpi", action="store_true")
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--pt-cap", type=float, default=config.SOFT_PT_CAP)
    ap.add_argument("--out", type=str, default="data/skim/w.parquet")
    ap.add_argument("--batch", type=int, default=12000)
    args = ap.parse_args()

    pythia = make_pythia(args.seed, args.ecm, not args.no_mpi, args.tune)
    base = args.out[:-8] if args.out.endswith(".parquet") else args.out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # accumulators
    A = {k: [] for k in ["mpx", "mpy", "mpz", "me", "mq",
                         "npx", "npy", "npz",
                         "ppt", "peta", "pphi", "pm", "pq", "pid"]}
    cnt_mu, cnt_pf = [], []
    pieces, kept, bidx = [], 0, 0

    def flush():
        nonlocal pieces, kept, bidx
        if not cnt_mu:
            return None
        mu = ak.zip({
            "px": ak.unflatten(np.array(A["mpx"]), cnt_mu),
            "py": ak.unflatten(np.array(A["mpy"]), cnt_mu),
            "pz": ak.unflatten(np.array(A["mpz"]), cnt_mu),
            "E": ak.unflatten(np.array(A["me"]), cnt_mu),
            "charge": ak.unflatten(np.array(A["mq"]), cnt_mu),
        }, with_name="Momentum4D")
        nu = ak.zip({
            "px": ak.unflatten(np.array(A["npx"]), cnt_mu),
            "py": ak.unflatten(np.array(A["npy"]), cnt_mu),
            "pz": ak.unflatten(np.array(A["npz"]), cnt_mu),
            "E": ak.unflatten(np.sqrt(np.array(A["npx"])**2 + np.array(A["npy"])**2
                                      + np.array(A["npz"])**2), cnt_mu),
        }, with_name="Momentum4D")
        pf = ak.zip({
            "pt": ak.unflatten(np.array(A["ppt"]), cnt_pf),
            "eta": ak.unflatten(np.array(A["peta"]), cnt_pf),
            "phi": ak.unflatten(np.array(A["pphi"]), cnt_pf),
            "mass": ak.unflatten(np.array(A["pm"]), cnt_pf),
            "charge": ak.unflatten(np.array(A["pq"]), cnt_pf),
            "pdgId": ak.unflatten(np.array(A["pid"], dtype=np.int64), cnt_pf),
            "puppi": ak.unflatten(np.ones(len(A["ppt"])), cnt_pf),
            "dz": ak.unflatten(np.zeros(len(A["ppt"])), cnt_pf),
            "pvq": ak.unflatten(np.full(len(A["ppt"]), 7, np.int64), cnt_pf),
        }, with_name="Momentum4D")

        has1 = ak.num(mu) >= 1
        mu1 = mu[has1][:, 0]
        nu1 = nu[has1][:, 0]
        pf1 = pf[has1]
        W = mu1 + nu1
        good = (mu1.pt > MU_PT_MIN) & (abs(mu1.eta) < MU_ETA_MAX) & (nu1.pt > NU_PT_MIN)

        m, n, w = mu1[good], nu1[good], W[good]
        pfg = pf1[good]
        soft = sp.select_soft(pfg, m, m, w, include_neutral=not args.no_neutral,
                              use_pv=False, pt_cap=args.pt_cap)
        summ = sp.event_summary(soft)
        E, pz = np.asarray(w.energy), np.asarray(w.pz)
        finite = np.isfinite(pz) & (np.abs(pz) < E)
        rec = ak.Array({
            # boost targets (same names as Z so a Z-trained model applies)
            "y_Z": 0.5 * np.log((E + pz) / (E - pz)),
            "pz_Z": pz,
            "beta_z": pz / E,
            "mass": np.asarray(w.mass),
            "pt_Z": np.asarray(w.pt),
            # W-mass reconstruction inputs
            "mu_px": np.asarray(m.px), "mu_py": np.asarray(m.py),
            "mu_pz": np.asarray(m.pz), "mu_E": np.asarray(m.energy),
            "nu_px": np.asarray(n.px), "nu_py": np.asarray(n.py),
            "nu_pz_true": np.asarray(n.pz),
            # summary + per-particle soft features
            "n_soft": summ["n_soft"], "sum_pt": summ["sum_pt"],
            "eta_ptweighted": summ["eta_ptweighted"], "fb_asym": summ["fb_asym"],
            **{f"p_{f}": soft[f] for f in PART_FEATURES},
        })[finite]

        for k in A:
            A[k] = []
        cnt_mu.clear(); cnt_pf.clear()
        return rec

    n_batch = 0
    for i in range(args.nevents):
        if not pythia.next():
            continue
        ev = pythia.event
        muons, nus, soft = [], [], []
        for j in range(ev.size()):
            prt = ev[j]
            if not prt.isFinal():
                continue
            aid = abs(prt.id())
            if aid == 13:
                muons.append((prt.px(), prt.py(), prt.pz(), prt.e(), prt.charge()))
            elif aid == 14:
                nus.append((prt.px(), prt.py(), prt.pz()))
            elif aid in NEUTRINOS:
                continue
            else:
                soft.append((prt.pT(), prt.eta(), prt.phi(), prt.m(), prt.charge(), prt.id()))
        # keep only the clean W->mu nu_mu topology
        if len(muons) != 1 or len(nus) != 1:
            continue
        mpx, mpy, mpz, me, mq = muons[0]
        npx, npy, npz = nus[0]
        A["mpx"].append(mpx); A["mpy"].append(mpy); A["mpz"].append(mpz)
        A["me"].append(me); A["mq"].append(mq)
        A["npx"].append(npx); A["npy"].append(npy); A["npz"].append(npz)
        for (pt, eta, phi, mm, q, pid) in soft:
            A["ppt"].append(pt); A["peta"].append(eta); A["pphi"].append(phi)
            A["pm"].append(mm); A["pq"].append(q); A["pid"].append(pid)
        cnt_mu.append(1); cnt_pf.append(len(soft))
        n_batch += 1
        if n_batch >= args.batch:
            rec = flush()
            if rec is not None and len(rec):
                ak.to_parquet(rec, f"{base}_b{bidx}.parquet"); bidx += 1
                pieces.append(rec); kept += len(rec)
            print(f"  generated {i+1}/{args.nevents} | kept W {kept}", flush=True)
            n_batch = 0

    rec = flush()
    if rec is not None and len(rec):
        ak.to_parquet(rec, f"{base}_b{bidx}.parquet")
        pieces.append(rec); kept += len(rec)
    data = ak.concatenate(pieces)
    ak.to_parquet(data, args.out)
    print(f"wrote {len(data)} W->munu events -> {args.out}")


if __name__ == "__main__":
    main()
