"""Ablation: where does the Z-boost information live in the soft event?

Trains the EFN on the same sample but with different particle subsets, to
separate the (more trivial) forward / full-coverage signal from the
(non-trivial, detector-measurable) central charged underlying event.

    python -m src.ablation --parquet data/skim/pythia.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import config, dataset, metrics, train

CONFIGS = {
    "full (|eta|<5, all)":          dict(),
    "central (|eta|<2.5, all)":     dict(abs_eta_max=2.5),
    "central charged (|eta|<2.5)":  dict(abs_eta_max=2.5, charged_only=True),
    "forward only (|eta|>2.5)":     dict(abs_eta_min=2.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/skim/pythia.parquet")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-p", type=int, default=config.MAX_PARTICLES)
    ap.add_argument("--out", default=str(config.RESULTS / "ablation.json"))
    args = ap.parse_args()

    rows = {}
    for name, filt in CONFIGS.items():
        print(f"\n==== ablation: {name} ({filt}) ====", flush=True)
        sp = dataset.load_splits(args.parquet, max_p=args.max_p, **filt)
        _, pred, _ = train.train_model(sp, name="efn", epochs=args.epochs, verbose=False)
        ip = list(config.TARGETS).index(config.PRIMARY_TARGET)
        yte = sp.extra["y_test_raw"][:, ip]
        m = metrics.regression_metrics(yte, pred[:, ip])
        m["sign_acc"] = metrics.sign_accuracy(yte, pred[:, ip])
        m["mean_n_particles"] = float(np.mean(sp.extra["test"]["n_soft"]))
        rows[name] = m
        print(f"   R2={m['r2']:.3f} corr={m['corr']:.3f} sign={m['sign_acc']:.3f} "
              f"<n>={m['mean_n_particles']:.0f}")

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\n| subset | R2 | corr | sign acc | <n particles> |")
    print("|---|---|---|---|---|")
    for n, m in rows.items():
        print(f"| {n} | {m['r2']:.3f} | {m['corr']:.3f} | {m['sign_acc']:.3f} | {m['mean_n_particles']:.0f} |")


if __name__ == "__main__":
    main()
