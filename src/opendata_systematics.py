"""Systematic-uncertainty suite for the open-data soft-event -> Z-boost result.

Headline: PUPPI>0.5 leading-vertex EFN, corr(y_Z) ~ 0.75 (see opendata_validate).
Each entry below varies ONE ingredient; the systematic is the shift in corr vs
the headline. Categories:

  PUPPI threshold      : pileup-suppression working point (0.3 / 0.5 / 0.7)
  acceptance           : central tracker edge (|eta|<2.4 vs 2.5)
  Z-mass window        : 86-96 vs nominal 81-101 GeV
  eta resolution       : neutral eta smearing 0.03 / 0.05
  momentum resolution  : per-particle pT smear 10%
  PUPPI modeling        : charged-only (PV-association, minimal neutral PUPPI model)

Statistical error and training/seed variance come from opendata_validate.json.
Architecture (EFN vs Transformer) is handled by opendata_study --model transformer.

Incremental + resumable -> robust to the background-job watchdog.

    python -m src.opendata_systematics
"""
from __future__ import annotations

import argparse, json, os
import numpy as np

from . import config
from .opendata_validate import train_eval

P = {"puppi_min": 0.5}

VARIATIONS = {
    # label                       : load_splits kwargs (relative to headline PUPPI>0.5)
    "PUPPI>0.3":                    {"puppi_min": 0.3},
    "PUPPI>0.7":                    {"puppi_min": 0.7},
    "central |eta|<2.4":            {**P, "abs_eta_max": 2.4},
    "central |eta|<2.5":            {**P, "abs_eta_max": 2.5},
    "Z mass 86-96":                 {**P, "mass_lo": 86.0, "mass_hi": 96.0},
    "eta smear 0.03":               {**P, "eta_smear": 0.03},
    "eta smear 0.05":               {**P, "eta_smear": 0.05},
    "pT smear 10%":                 {**P, "pt_smear": 0.10},
    "PF reco eff -2%":              {**P, "drop_frac": 0.02},
    "PF reco eff -5%":              {**P, "drop_frac": 0.05},
    "PF reco eff -10%":             {**P, "drop_frac": 0.10},
    "charged-only (PV assoc)":      {**P, "charged_only": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RESULTS / "systematics_opendata.json"))
    args = ap.parse_args()

    rows = {}
    if os.path.exists(args.out):
        try:
            rows = json.load(open(args.out)); print(f"resuming: {len(rows)} done", flush=True)
        except Exception:
            rows = {}

    for label, filt in VARIATIONS.items():
        if label in rows:
            print(f"  [cached] {label}: corr={rows[label]['corr']:+.3f}", flush=True)
            continue
        print(f"\n==== {label}  ({filt}) ====", flush=True)
        m, _, _ = train_eval(filt)
        rows[label] = m
        json.dump(rows, open(args.out, "w"), indent=2)  # incremental
        print(f"  {label}: corr={m['corr']:+.3f} [{m['corr_lo']:+.3f},{m['corr_hi']:+.3f}] "
              f"R2={m['r2']:+.3f} <n>={m['mean_n']:.0f}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
