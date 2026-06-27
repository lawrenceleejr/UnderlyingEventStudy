"""Skim CMS Open Data files into compact per-event Parquet for ML.

For each input file we run the Z->mumu selection, build the soft-particle set,
and store per-event targets (y_Z, pz, beta_z), event-level summary scalars, and
the variable-length per-particle feature arrays. Files are downloaded, processed,
then deleted to respect the disk budget.

Usage:
    python -m src.skim --record 31305 --nfiles 30 --out data/skim/pfnano.parquet
"""
from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np
import uproot

from . import config, io, select
from . import softparticles as sp

# per-particle feature columns, in the order the network consumes them
PART_FEATURES = ["log_pt", "eta", "sin_dphi", "cos_dphi", "charge", "puppi", "dz", "cat"]


def make_records(m1, m2, z, pf, **soft_kw):
    """Assemble the per-event Parquet record from selected Z and PF candidates.

    Shared by the open-data skim and the Pythia generator so both produce an
    identical schema that ``dataset.load_splits`` can read.
    """
    tgt = select.targets(z)
    soft = sp.select_soft(pf, m1, m2, z, **soft_kw)
    summ = sp.event_summary(soft)
    finite = np.isfinite(tgt["y_Z"]) & np.isfinite(tgt["beta_z"])
    out = ak.Array(
        {
            "y_Z": tgt["y_Z"],
            "pz_Z": tgt["pz_Z"],
            "beta_z": tgt["beta_z"],
            "mass": tgt["mass"],
            "pt_Z": tgt["pt_Z"],
            # individual lepton 4-vectors: needed to build the transverse mass and
            # to test the soft-boost mass correction (treat mu2 as the 'neutrino').
            "mu1_pt": np.asarray(m1.pt), "mu1_phi": np.asarray(m1.phi),
            "mu1_pz": np.asarray(m1.pz), "mu1_E": np.asarray(m1.energy),
            "mu2_pt": np.asarray(m2.pt), "mu2_phi": np.asarray(m2.phi),
            "mu2_pz": np.asarray(m2.pz), "mu2_E": np.asarray(m2.energy),
            "n_soft": summ["n_soft"],
            "sum_pt": summ["sum_pt"],
            "eta_ptweighted": summ["eta_ptweighted"],
            "fb_asym": summ["fb_asym"],
            **{f"p_{f}": soft[f] for f in PART_FEATURES},
        }
    )
    return out[finite]


def _open_events(path: str):
    """Locate the 'Events' TTree, whether top-level (Pythia/NanoAOD) or nested
    under the EDAnalyzer module label (CMSSW TFileService, e.g. 'pfnanolite/Events')."""
    f = uproot.open(path)
    if "Events" in f:
        return f["Events"]
    for key in f.keys(recursive=True):
        if key.split(";")[0].rsplit("/", 1)[-1] == "Events":
            return f[key]
    raise KeyError(f"no 'Events' tree in {path}; keys={f.keys()}")


def process_file(path: str, **soft_kw) -> ak.Array:
    """Return a per-event awkward record array for one local ROOT file."""
    ev = _open_events(path)
    arr = ev.arrays(select.MUON_BRANCHES + sp.PF_BRANCHES)
    mask, m1, m2, z = select.zmumu_mask_and_z(arr)
    if mask.sum() == 0:
        return None
    pf = sp.build_pfcands(arr)[mask]
    return make_records(m1, m2, z, pf, **soft_kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=config.RECORD_PFNANO_JETSONLY)
    ap.add_argument("--nfiles", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/skim/pfnano.parquet")
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--no-pv", action="store_true")
    ap.add_argument("--pt-cap", type=float, default=config.SOFT_PT_CAP)
    ap.add_argument("--keep-files", action="store_true", help="do not delete downloads")
    ap.add_argument("--local-glob", type=str, default=None,
                    help="process local files matching this glob instead of a record")
    args = ap.parse_args()

    soft_kw = dict(include_neutral=not args.no_neutral, use_pv=not args.no_pv, pt_cap=args.pt_cap)

    if args.local_glob:
        import glob
        files = sorted(glob.glob(args.local_glob))
        urls = files
        local = True
    else:
        urls = list(io.list_files(args.record))[args.start: args.start + args.nfiles]
        local = False

    print(f"skim: {len(urls)} files | soft={soft_kw}")
    pieces = []
    n_evt = 0
    for i, url in enumerate(urls):
        try:
            path = url if local else io.download(url)
            out = process_file(path, **soft_kw)
            if out is not None and len(out) > 0:
                pieces.append(out)
                n_evt += len(out)
            print(f"  [{i+1}/{len(urls)}] {os.path.basename(path)}: "
                  f"+{0 if out is None else len(out)} evt (total {n_evt})", flush=True)
            if not local and not args.keep_files:
                os.remove(path)
        except Exception as e:
            print(f"  [{i+1}/{len(urls)}] FAILED {url[-50:]}: {type(e).__name__}: {e}", flush=True)

    if not pieces:
        raise SystemExit("no events skimmed")
    data = ak.concatenate(pieces)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ak.to_parquet(data, args.out)
    print(f"wrote {len(data)} events -> {args.out}")


if __name__ == "__main__":
    main()
