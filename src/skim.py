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
            "n_soft": summ["n_soft"],
            "sum_pt": summ["sum_pt"],
            "eta_ptweighted": summ["eta_ptweighted"],
            "fb_asym": summ["fb_asym"],
            **{f"p_{f}": soft[f] for f in PART_FEATURES},
        }
    )
    return out[finite]


def process_file(path: str, step="20000", **soft_kw) -> ak.Array:
    """Return a per-event awkward record array for one local ROOT file.

    Read in chunks (uproot.iterate) so large PFNano files with hundreds of PF
    candidates per event don't blow up memory.
    """
    pieces = []
    for arr in uproot.iterate({path: "Events"}, select.MUON_BRANCHES + sp.PF_BRANCHES,
                              step_size=int(step)):
        mask, m1, m2, z = select.zmumu_mask_and_z(arr)
        if mask.sum() == 0:
            continue
        pf = sp.build_pfcands(arr)[mask]
        rec = make_records(m1, m2, z, pf, **soft_kw)
        if rec is not None and len(rec):
            pieces.append(rec)
    if not pieces:
        return None
    return ak.concatenate(pieces)


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
    ap.add_argument("--url-list", type=str, default=None,
                    help="file with one ROOT URL per line to process")
    args = ap.parse_args()

    soft_kw = dict(include_neutral=not args.no_neutral, use_pv=not args.no_pv, pt_cap=args.pt_cap)

    if args.local_glob:
        import glob
        urls = sorted(glob.glob(args.local_glob))
        local = True
    elif args.url_list:
        urls = [config.to_https(l.strip()) for l in open(args.url_list) if l.strip()]
        local = False
    else:
        urls = list(io.list_files(args.record))[args.start: args.start + args.nfiles]
        local = False

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    base = args.out[:-8] if args.out.endswith(".parquet") else args.out
    print(f"skim: {len(urls)} files | soft={soft_kw}", flush=True)
    pieces = []
    n_evt = 0
    for i, url in enumerate(urls):
        try:
            path = url if local else io.download(url)
            out = process_file(path, **soft_kw)
            if out is not None and len(out) > 0:
                ak.to_parquet(out, f"{base}_b{i}.parquet")  # incremental: survives interruption
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
    ak.to_parquet(data, args.out)
    print(f"wrote {len(data)} events -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
