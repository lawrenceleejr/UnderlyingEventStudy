"""Regression metrics and physics-oriented summaries."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    res = y_pred - y_true
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    m = {
        "rmse": float(np.sqrt(np.mean(res ** 2))),
        "mae": float(np.mean(np.abs(res))),
        "r2": float(1 - ss_res / ss_tot),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "bias": float(res.mean()),
        "resolution": float(res.std()),
    }
    return m


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of events whose boost direction (sign of y_Z) is predicted right."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def resolution_vs_truth(y_true, y_pred, bins=10):
    """Residual std in bins of the true value (the headline resolution curve)."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    edges = np.quantile(y_true, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-6
    idx = np.digitize(y_true, edges) - 1
    centers, std, mean = [], [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum() < 5:
            continue
        centers.append(float(np.median(y_true[sel])))
        std.append(float((y_pred[sel] - y_true[sel]).std()))
        mean.append(float((y_pred[sel] - y_true[sel]).mean()))
    return np.array(centers), np.array(std), np.array(mean)
