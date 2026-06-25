"""Reference baselines: the trivial floor the network must beat.

These are NOT the deliverable models -- they quantify how much the deep set adds
over (a) predicting the mean and (b) a couple of hand-built underlying-event
observables (eta-pT-asymmetry etc.) fed to a linear / gradient-boosted fit.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def _event_features(extra: dict) -> np.ndarray:
    cols = ["eta_ptweighted", "fb_asym", "n_soft", "sum_pt"]
    X = np.stack([np.nan_to_num(np.asarray(extra[c], float)) for c in cols], axis=1)
    return X


def run_baselines(splits, target_index: int) -> dict:
    """Return baseline predictions/metrics for one target on the test set."""
    ytr = splits.extra["y_train_raw"][:, target_index]
    yte = splits.extra["y_test_raw"][:, target_index]
    Xtr = _event_features(splits.extra["train"])
    Xte = _event_features(splits.extra["test"])

    out = {}
    # constant mean
    out["mean"] = np.full_like(yte, ytr.mean())
    # linear on UE summary features
    lin = LinearRegression().fit(Xtr, ytr)
    out["linear"] = lin.predict(Xte)
    # gradient-boosted on UE summary features
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                   subsample=0.7, random_state=0).fit(Xtr, ytr)
    out["gbdt_summary"] = gb.predict(Xte)
    return {"pred": out, "y_true": yte}
