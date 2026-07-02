"""Reference baselines: the trivial floor the network must beat.

These are NOT the deliverable models -- they quantify how much the deep set adds
over (a) predicting the mean and (b) a couple of hand-built underlying-event
observables (eta-pT-asymmetry etc.) fed to a linear / gradient-boosted fit.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .dataset import ENGINEERED

SIMPLE = ["eta_ptweighted", "fb_asym", "n_soft", "sum_pt"]


def _event_features(extra: dict, cols) -> np.ndarray:
    return np.stack([np.nan_to_num(np.asarray(extra[c], float)) for c in cols], axis=1)


def run_baselines(splits, target_index: int) -> dict:
    """Return baseline predictions/metrics for one target on the test set.

    ``linear``/``gbdt_summary`` use the 4 classic UE summary observables;
    ``gbdt_rich`` uses the full engineered set (dataset.ENGINEERED) built from
    the same per-particle information the network sees — the fair
    features-vs-architecture comparison a referee would ask for.
    """
    ytr = splits.extra["y_train_raw"][:, target_index]
    yte = splits.extra["y_test_raw"][:, target_index]

    out = {}
    out["mean"] = np.full_like(yte, ytr.mean())

    Xtr = _event_features(splits.extra["train"], SIMPLE)
    Xte = _event_features(splits.extra["test"], SIMPLE)
    out["linear"] = LinearRegression().fit(Xtr, ytr).predict(Xte)
    out["gbdt_summary"] = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, subsample=0.7, random_state=0
    ).fit(Xtr, ytr).predict(Xte)

    cols = [c for c in ENGINEERED if c in splits.extra["train"]]
    Xtr_r = _event_features(splits.extra["train"], cols)
    Xte_r = _event_features(splits.extra["test"], cols)
    out["gbdt_rich"] = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, subsample=0.7, random_state=0
    ).fit(Xtr_r, ytr).predict(Xte_r)
    return {"pred": out, "y_true": yte}
