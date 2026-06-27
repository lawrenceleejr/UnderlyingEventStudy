"""Validation / leakage suite for the open-data soft-event -> Z-boost result.

Gating question: the PUPPI-suppressed EFN reaches corr ~0.74, well above Pythia
truth-level (0.26). Before quoting systematics we must show the number is a
genuine, eta-driven boost measurement, not a pipeline artefact. This driver runs:

  perm        : label-permutation null test (shuffle training y_Z; test corr must ->0)
  decomp:*    : where does the signal live? (charged / neutral / central / forward)
  seed:*      : statistical + training variance over several train/test splits
  pubin:*     : correlation in bins of pileup (nominal multiplicity proxy)

All on the PUPPI>0.5 leading-vertex set unless noted. Results are written
incrementally to results/validation_opendata.json and the run is resumable
(skips labels already present), so a watchdog kill never loses completed work.

    python -m src.opendata_validate
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import awkward as ak

from . import config, dataset, train, metrics

PARQUET = "data/skim/opendata.parquet"
IP = list(config.TARGETS).index(config.PRIMARY_TARGET)
MAXP = 150
EPOCHS = 25


def boot_ci(yt, yp, B=400, seed=0):
    """Bootstrap 68% CI on Pearson corr by resampling test events."""
    rng = np.random.default_rng(seed)
    n = len(yt); cs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        cs[b] = np.corrcoef(yt[idx], yp[idx])[0, 1]
    return float(np.percentile(cs, 16)), float(np.percentile(cs, 84))


def split_test_indices(N, seed=config.SEED):
    """Reproduce dataset.load_splits' test indices to attach external per-event info."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    f_tr, f_va, _ = config.SPLIT
    i_va = int((f_tr + f_va) * N)
    return perm[i_va:]


def train_eval(filt, seed=config.SEED, permute=False, epochs=EPOCHS):
    sp = dataset.load_splits(PARQUET, max_p=MAXP, seed=seed, **filt)
    if permute:                       # break X<->y association in training only
        rng = np.random.default_rng(123)
        sp.Ytr[:] = sp.Ytr[rng.permutation(len(sp.Ytr))]
    _, pred, _ = train.train_model(sp, name="efn", epochs=epochs, device="cpu", verbose=False)
    yt = sp.extra["y_test_raw"][:, IP]; yp = pred[:, IP]
    m = metrics.regression_metrics(yt, yp)
    m["sign_acc"] = metrics.sign_accuracy(yt, yp)
    m["mean_n"] = float(np.mean(sp.extra["test"]["n_soft"]))
    lo, hi = boot_ci(yt, yp)
    m["corr_lo"], m["corr_hi"] = lo, hi
    return m, yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS / "validation_opendata.json"))
    args = ap.parse_args()

    rows = {}
    if os.path.exists(args.out):
        try:
            rows = json.load(open(args.out)); print(f"resuming: {len(rows)} done", flush=True)
        except Exception:
            rows = {}

    def save():
        json.dump(rows, open(args.out, "w"), indent=2)

    def do(label, fn):
        if label in rows:
            print(f"  [cached] {label}: corr={rows[label].get('corr'):+.3f}", flush=True)
            return
        print(f"\n==== {label} ====", flush=True)
        rows[label] = fn(); save()
        r = rows[label]
        print(f"  {label}: corr={r['corr']:+.3f} [{r.get('corr_lo',float('nan')):+.3f},"
              f"{r.get('corr_hi',float('nan')):+.3f}] R2={r['r2']:+.3f} sign={r['sign_acc']:.3f} "
              f"<n>={r.get('mean_n',float('nan')):.0f}", flush=True)

    P = {"puppi_min": 0.5}

    # 0) headline (with bootstrap CI) + pileup-dependence in one trained model
    def headline():
        m, yt, yp = train_eval(P)
        # pileup dependence: bin test events by NOMINAL multiplicity (PU proxy)
        d = ak.from_parquet(PARQUET)
        nom_n = np.asarray(d["n_soft"])               # nominal (all-soft) count, file order
        te = split_test_indices(len(nom_n))
        pu = nom_n[te]
        edges = np.quantile(pu, [0, 1/3, 2/3, 1.0])
        pubins = []
        for k in range(3):
            sel = (pu >= edges[k]) & (pu <= edges[k+1] if k == 2 else pu < edges[k+1])
            if sel.sum() > 50:
                pubins.append({"pu_lo": float(edges[k]), "pu_hi": float(edges[k+1]),
                               "corr": float(np.corrcoef(yt[sel], yp[sel])[0, 1]),
                               "n": int(sel.sum()), "mean_pu": float(pu[sel].mean())})
        m["pubins"] = pubins
        m["empty_frac"] = float(np.mean(np.asarray(ak.sum(d["p_puppi"] > 0.5, axis=1))[te] == 0))
        return m
    do("headline PUPPI>0.5", headline)

    # 1) label-permutation null test
    do("PERMUTED-LABELS null", lambda: train_eval(P, permute=True)[0])

    # 2) charge / region decomposition
    do("decomp: charged only",  lambda: train_eval({**P, "charged_only": True})[0])
    do("decomp: neutral only",  lambda: train_eval({**P, "neutral_only": True})[0])
    do("decomp: central |eta|<2.5", lambda: train_eval({**P, "abs_eta_max": 2.5})[0])
    do("decomp: forward |eta|>2.5", lambda: train_eval({**P, "abs_eta_min": 2.5})[0])

    # 3) multi-seed statistical + training variance
    for s in (1, 2, 3, 4, 5):
        do(f"seed:{s}", (lambda s=s: train_eval(P, seed=1000 + s)[0]))

    save()
    # summary
    print("\n==== SUMMARY ====")
    seeds = [rows[f"seed:{s}"]["corr"] for s in (1,2,3,4,5) if f"seed:{s}" in rows]
    if seeds:
        print(f"seed-corr: {np.mean(seeds):.3f} +/- {np.std(seeds):.3f}  (n={len(seeds)})")
    if "headline PUPPI>0.5" in rows and rows["headline PUPPI>0.5"].get("pubins"):
        print("pileup bins (mean_pu -> corr):",
              ", ".join(f"{b['mean_pu']:.0f}->{b['corr']:.3f}" for b in rows["headline PUPPI>0.5"]["pubins"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
