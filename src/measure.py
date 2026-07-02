"""The real-data measurement: soft-event -> y_Z regression on CMS collision data.

Runs the full measurement matrix on the MiniAOD-decoded skim and, with the same
code, the Pythia reference — so the data/simulation comparison is line-by-line:

variants
  charged-PV : charged candidates associated to the leading PV (the clean,
               pileup-suppressed underlying event; the primary result)
  full       : + neutrals (no vertex info -> pileup-diluted; secondary)

per variant
  baselines  : mean / linear / GBDT (4 summary obs) / GBDT (rich engineered set)
  EFN        : the deep set on raw particles
  shuffle    : EFN trained on permuted targets (leakage control, must be ~0)

Usage:
    python -m src.measure --data data/skim/miniaod.parquet \
                          --sim data/skim/pythia.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import baselines, config, dataset, metrics, train

IP = list(config.TARGETS).index(config.PRIMARY_TARGET)


def _run_variant(parquet: str, label: str, max_p: int, epochs: int,
                 device: str, **filt) -> dict:
    sp = dataset.load_splits(parquet, max_p=max_p, **filt)
    yte = sp.extra["y_test_raw"][:, IP]
    out = {"n_train": int(len(sp.Ytr)), "n_test": int(len(sp.Yte)),
           "mean_n_particles": float(np.mean(sp.extra["test"]["n_soft"]))}

    b = baselines.run_baselines(sp, IP)
    for name, pred in b["pred"].items():
        m = metrics.regression_metrics(yte, pred)
        m["sign_acc"] = metrics.sign_accuracy(yte, pred)
        out[name] = m

    _, pred_raw, _ = train.train_model(sp, name="efn", epochs=epochs,
                                       device=device, verbose=False)
    m = metrics.regression_metrics(yte, pred_raw[:, IP])
    m["sign_acc"] = metrics.sign_accuracy(yte, pred_raw[:, IP])
    out["efn"] = m

    # leakage control: same training on permuted targets must give corr ~ 0
    rng = np.random.default_rng(config.SEED)
    sp.Ytr[:] = sp.Ytr[rng.permutation(len(sp.Ytr))]
    _, pred_sh, _ = train.train_model(sp, name="efn", epochs=min(epochs, 20),
                                      device=device, verbose=False)
    out["efn_shuffled_control"] = metrics.regression_metrics(yte, pred_sh[:, IP])

    print(f"[{label}] <n>={out['mean_n_particles']:.0f} | "
          f"linear {out['linear']['corr']:.3f} | rich {out['gbdt_rich']['corr']:.3f} | "
          f"EFN {out['efn']['corr']:.3f} (sign {out['efn']['sign_acc']:.3f}) | "
          f"shuffle {out['efn_shuffled_control']['corr']:.3f}", flush=True)
    return out


VARIANTS = {
    "charged_pv": dict(max_p=200, charged_only=True, abs_eta_max=config.TRK_ETA_MAX),
    "full": dict(max_p=500),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/skim/miniaod.parquet")
    ap.add_argument("--sim", default="data/skim/pythia.parquet")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset like: data/charged_pv sim/full (default: all)")
    ap.add_argument("--out", default=str(config.RESULTS / "metrics_measurement.json"))
    args = ap.parse_args()

    # merge into any existing results so variants can run as separate jobs
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for src, parquet in [("data", args.data), ("sim", args.sim)]:
        if not os.path.exists(parquet):
            print(f"skip {src}: {parquet} not found")
            continue
        for vname, vkw in VARIANTS.items():
            if args.only and f"{src}/{vname}" not in args.only:
                continue
            kw = dict(vkw)
            max_p = kw.pop("max_p")
            results.setdefault(src, {})[vname] = _run_variant(
                parquet, f"{src}/{vname}", max_p=max_p, epochs=args.epochs,
                device=args.device, **kw)
            json.dump(results, open(args.out, "w"), indent=2)  # incremental
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
