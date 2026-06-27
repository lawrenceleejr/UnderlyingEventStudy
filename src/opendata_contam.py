"""EFN correlation vs pileup contamination: train at several add-back fractions.

Clean leading-vertex set (PUPPI>0.5) + a random fraction f of the pileup added
back. Records corr(y_Z) and <n> per f so make_contam_bands.py can draw the EFN
residual band with real shape (not just clean/full endpoints). Resumable.

    python -m src.opendata_contam
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from . import config, dataset, train, metrics

FRACS = [0.03, 0.1, 0.25, 0.5]      # intermediate contamination levels
IP = list(config.TARGETS).index(config.PRIMARY_TARGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/skim/opendata.parquet")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--max-p", type=int, default=400)
    ap.add_argument("--out", default=str(config.RESULTS / "contam_opendata.json"))
    args = ap.parse_args()

    rows = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for f in FRACS:
        key = f"{f}"
        if key in rows:
            print(f"  [cached] f={f}: corr={rows[key]['corr']:+.3f}", flush=True); continue
        print(f"\n==== contamination f={f} ====", flush=True)
        sp = dataset.load_splits(args.parquet, max_p=args.max_p, pileup_addback=f)
        _, pred, _ = train.train_model(sp, name="efn", epochs=args.epochs, device="cpu", verbose=False)
        yt = sp.extra["y_test_raw"][:, IP]; yp = pred[:, IP]
        m = metrics.regression_metrics(yt, yp)
        m["sign_acc"] = metrics.sign_accuracy(yt, yp)
        m["mean_n"] = float(np.mean(sp.extra["test"]["n_soft"]))
        rows[key] = m; json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  f={f}: corr={m['corr']:+.3f} R2={m['r2']:+.3f} <n>={m['mean_n']:.0f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
