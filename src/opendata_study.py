"""Open-data pileup study: does the soft event predict the Z boost in REAL data?

Pythia (no pileup) showed the soft, non-muon event predicts the Z longitudinal
boost (EFN corr ~0.26). Real 2016 CMS data has ~20-30 pileup interactions whose
soft particles are uncorrelated with the hard-scatter boost and overwhelm the
signal: the *nominal* soft selection (~730 particles/event, ~93% pileup) gives
corr ~0 (see results/REPORT_opendata.md).

The pileup-robust handle is PUPPI: keeping puppiWeight >= threshold isolates the
leading-vertex (hard-scatter) soft event. This driver trains the same EFN on a
set of PUPPI / eta / charge subsets of the SAME selected Z->mumu events and
reports where the boost information survives.

    python -m src.opendata_study --parquet data/skim/opendata.parquet
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import config, dataset, metrics, train, baselines

# (name, filter-kwargs, max_p): the filter kwargs go straight to load_splits.
CONFIGS = [
    ("nominal (all soft, no PU cut)",      dict(),                                          400),
    ("PUPPI>0.1",                          dict(puppi_min=0.1),                             300),
    ("PUPPI>0.5 (leading-vertex)",         dict(puppi_min=0.5),                             150),
    ("PUPPI>0.5 central |eta|<2.5",        dict(puppi_min=0.5, abs_eta_max=2.5),            150),
    ("PUPPI>0.5 central charged",          dict(puppi_min=0.5, abs_eta_max=2.5,
                                                charged_only=True),                          100),
    ("PUPPI>0.5 NEUTRAL only",             dict(puppi_min=0.5, neutral_only=True),          100),
    ("PUPPI>0.5 forward |eta|>2.5",        dict(puppi_min=0.5, abs_eta_min=2.5),            100),
]


def run_one(parquet, filt, max_p, epochs, device, model="efn"):
    sp = dataset.load_splits(parquet, max_p=max_p, **filt)
    ip = list(config.TARGETS).index(config.PRIMARY_TARGET)
    yte = sp.extra["y_test_raw"][:, ip]
    # linear baseline on hand-built UE summary observables (recomputed on subset)
    b = baselines.run_baselines(sp, ip)
    lin = metrics.regression_metrics(yte, b["pred"]["linear"])
    # deep set (efn) or particle transformer
    _, pred, _ = train.train_model(sp, name=model, epochs=epochs, device=device, verbose=False)
    out = metrics.regression_metrics(yte, pred[:, ip])
    out["sign_acc"] = metrics.sign_accuracy(yte, pred[:, ip])
    out["mean_n"] = float(np.mean(sp.extra["test"]["n_soft"]))
    out["lin_corr"] = float(lin["corr"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/skim/opendata.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="efn", help="efn | transformer")
    ap.add_argument("--only", default=None, help="substring filter on config names")
    ap.add_argument("--out", default=str(config.RESULTS / "ablation_opendata.json"))
    args = ap.parse_args()

    import os
    rows = {}
    if os.path.exists(args.out):           # resume: keep configs already computed
        try:
            rows = json.load(open(args.out))
            print(f"resuming: {len(rows)} configs already present", flush=True)
        except Exception:
            rows = {}
    for name, filt, max_p in CONFIGS:
        if args.only and args.only not in name:
            continue
        if name in rows:
            print(f"\n==== {name}: cached (corr={rows[name]['corr']:+.3f}) ====", flush=True)
            continue
        print(f"\n==== [{args.model}] {name}  ({filt}, max_p={max_p}) ====", flush=True)
        m = run_one(args.parquet, filt, max_p, args.epochs, args.device, model=args.model)
        rows[name] = m
        print(f"   {args.model} corr={m['corr']:+.3f}  R2={m['r2']:+.3f}  sign={m['sign_acc']:.3f}  "
              f"lin corr={m['lin_corr']:+.3f}  <n>={m['mean_n']:.0f}", flush=True)
        json.dump(rows, open(args.out, "w"), indent=2)  # incremental: survive interruption

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}\n")
    print("| subset | EFN corr | EFN R2 | sign acc | linear corr | <n particles> |")
    print("|---|---|---|---|---|---|")
    for n, m in rows.items():
        print(f"| {n} | {m['corr']:+.3f} | {m['r2']:+.3f} | {m['sign_acc']:.3f} | "
              f"{m['lin_corr']:+.3f} | {m['mean_n']:.0f} |")


if __name__ == "__main__":
    main()
