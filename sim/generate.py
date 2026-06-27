"""Generate Z->mumu events with Pythia8 and skim them to the analysis Parquet.

Truth-level, no pileup, full eta coverage: the cleanest setting in which to ask
whether the soft particles encode the Z longitudinal boost. The output schema is
identical to the open-data skim (src/skim.make_records), so the same dataset /
model / evaluation code runs on both.

Requires the Pythia8 python module (see sim/build_pythia.sh):
    PYTHONPATH=/opt/pythia8312/lib python -m sim.generate --nevents 100000 --out data/skim/pythia.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import awkward as ak
import numpy as np
import vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, select  # noqa: E402
from src import skim  # noqa: E402

vector.register_awkward()

NEUTRINOS = {12, 14, 16}


def make_pythia(seed: int, ecm: float, mpi: bool, tune: int):
    import pythia8
    p = pythia8.Pythia("", False)
    p.readString("Print:quiet = on")
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {ecm}")
    # Drell-Yan Z/gamma* -> mu mu
    p.readString("WeakSingleBoson:ffbar2gmZ = on")
    p.readString("23:onMode = off")
    p.readString("23:onIfAny = 13")
    p.readString("PhaseSpace:mHatMin = 60.")
    p.readString(f"Tune:pp = {tune}")           # underlying-event tune
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
    ap.add_argument("--tune", type=int, default=14)  # 14 = Monash 2013
    ap.add_argument("--no-mpi", action="store_true", help="disable multiparton interactions/UE")
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--pt-cap", type=float, default=config.SOFT_PT_CAP)
    ap.add_argument("--out", type=str, default="data/skim/pythia.parquet")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()

    pythia = make_pythia(args.seed, args.ecm, not args.no_mpi, args.tune)

    # flat accumulators; per-event counts -> unflatten at the end
    mu_pt, mu_eta, mu_phi, mu_q, mu_n = [], [], [], [], []
    pf_pt, pf_eta, pf_phi, pf_m, pf_q, pf_id, pf_n = [], [], [], [], [], [], []
    yboost = []  # per-event partonic-CM boost rapidity y_boost = 1/2 ln(x1/x2)

    pieces = []
    kept = 0
    batch_idx = 0
    base = args.out[:-8] if args.out.endswith(".parquet") else args.out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def flush():
        nonlocal mu_pt, mu_eta, mu_phi, mu_q, mu_n
        nonlocal pf_pt, pf_eta, pf_phi, pf_m, pf_q, pf_id, pf_n
        nonlocal yboost
        if not mu_n:
            return None
        mu = ak.zip(
            {
                "pt": ak.unflatten(np.array(mu_pt, np.float64), mu_n),
                "eta": ak.unflatten(np.array(mu_eta, np.float64), mu_n),
                "phi": ak.unflatten(np.array(mu_phi, np.float64), mu_n),
                "mass": ak.unflatten(np.full(len(mu_pt), 0.1056584, np.float64), mu_n),
                "charge": ak.unflatten(np.array(mu_q, np.float64), mu_n),
            },
            with_name="Momentum4D",
        )
        mu = mu[ak.argsort(mu.pt, ascending=False)]
        has2 = ak.num(mu) >= 2
        mu2 = mu[has2]
        m1, m2 = mu2[:, 0], mu2[:, 1]
        z = m1 + m2
        good = (
            (m1.charge != m2.charge)
            & (m1.pt > config.MU_PT_LEAD) & (m2.pt > config.MU_PT_SUBLEAD)
            & (abs(m1.eta) < config.MU_ETA_MAX) & (abs(m2.eta) < config.MU_ETA_MAX)
            & (z.mass > config.Z_MASS_LO) & (z.mass < config.Z_MASS_HI)
        )
        pf = ak.zip(
            {
                "pt": ak.unflatten(np.array(pf_pt, np.float64), pf_n),
                "eta": ak.unflatten(np.array(pf_eta, np.float64), pf_n),
                "phi": ak.unflatten(np.array(pf_phi, np.float64), pf_n),
                "mass": ak.unflatten(np.array(pf_m, np.float64), pf_n),
                "charge": ak.unflatten(np.array(pf_q, np.float64), pf_n),
                "pdgId": ak.unflatten(np.array(pf_id, np.int64), pf_n),
                "puppi": ak.unflatten(np.ones(len(pf_pt), np.float64), pf_n),  # no pileup
                "dz": ak.unflatten(np.zeros(len(pf_pt), np.float64), pf_n),
                "pvq": ak.unflatten(np.full(len(pf_pt), 7, np.int64), pf_n),
            },
            with_name="Momentum4D",
        )
        pf2 = pf[has2]
        zg = z[good]
        rec = skim.make_records(
            m1[good], m2[good], zg, pf2[good],
            include_neutral=not args.no_neutral, use_pv=False, pt_cap=args.pt_cap,
        )
        # attach y_boost, masked through the SAME has2 -> good -> finite chain that
        # make_records applies (finite = isfinite(y_Z) & isfinite(beta_z)).
        tgt = select.targets(zg)
        finite = np.isfinite(tgt["y_Z"]) & np.isfinite(tgt["beta_z"])
        yb = np.array(yboost, np.float64)[ak.to_numpy(has2)][ak.to_numpy(good)][finite]
        rec = ak.with_field(rec, yb, "y_boost")
        # reset accumulators
        mu_pt, mu_eta, mu_phi, mu_q, mu_n = [], [], [], [], []
        pf_pt, pf_eta, pf_phi, pf_m, pf_q, pf_id, pf_n = [], [], [], [], [], [], []
        yboost = []
        return rec

    n_in_batch = 0
    for i in range(args.nevents):
        if not pythia.next():
            continue
        _info = pythia.infoPython()
        x1, x2 = _info.x1(), _info.x2()
        yboost.append(0.5 * np.log(x1 / x2) if (x1 > 0 and x2 > 0) else np.nan)
        ev = pythia.event
        nm = 0
        npf = 0
        for j in range(ev.size()):
            prt = ev[j]
            if not prt.isFinal():
                continue
            aid = abs(prt.id())
            if aid in NEUTRINOS:
                continue
            pt = prt.pT()
            if abs(prt.id()) == 13:
                mu_pt.append(pt); mu_eta.append(prt.eta()); mu_phi.append(prt.phi())
                mu_q.append(prt.charge()); nm += 1
            pf_pt.append(pt); pf_eta.append(prt.eta()); pf_phi.append(prt.phi())
            pf_m.append(prt.m()); pf_q.append(prt.charge()); pf_id.append(prt.id())
            npf += 1
        mu_n.append(nm); pf_n.append(npf)
        n_in_batch += 1
        if n_in_batch >= args.batch:
            rec = flush()
            if rec is not None and len(rec):
                # write each batch immediately so partial progress survives interruptions
                ak.to_parquet(rec, f"{base}_b{batch_idx}.parquet")
                batch_idx += 1
                pieces.append(rec); kept += len(rec)
            print(f"  generated {i+1}/{args.nevents} | kept Z {kept}", flush=True)
            n_in_batch = 0

    rec = flush()
    if rec is not None and len(rec):
        ak.to_parquet(rec, f"{base}_b{batch_idx}.parquet")
        pieces.append(rec); kept += len(rec)

    data = ak.concatenate(pieces)
    ak.to_parquet(data, args.out)
    print(f"wrote {len(data)} Z->mumu events -> {args.out}")
    print(f"acceptance: {len(data)}/{args.nevents} = {len(data)/args.nevents:.3f}")


if __name__ == "__main__":
    main()
