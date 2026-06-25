"""Turn the skimmed Parquet into padded tensors and train/val/test splits."""
from __future__ import annotations

import json
from dataclasses import dataclass

import awkward as ak
import numpy as np

from . import config

# continuous per-particle features (standardized) + categorical (one-hot)
CONT = ["log_pt", "eta", "sin_dphi", "cos_dphi", "charge", "puppi", "dz"]
N_CAT = 6
N_FEAT = len(CONT) + N_CAT  # 7 + 6 = 13


def _pad_dense(data: ak.Array, max_p: int):
    """Build (X[N,P,F], mask[N,P]) keeping the highest-pT particles per event."""
    # sort particles by pt (desc) within each event, then truncate/pad to max_p
    order = ak.argsort(data["p_log_pt"], axis=1, ascending=False)
    cols = {}
    for f in CONT:
        c = data[f"p_{f}"][order][:, :max_p]
        cols[f] = ak.to_numpy(ak.fill_none(ak.pad_none(c, max_p, axis=1, clip=True), 0.0))
    cat = data["p_cat"][order][:, :max_p]
    cat = ak.to_numpy(ak.fill_none(ak.pad_none(cat, max_p, axis=1, clip=True), -1)).astype(int)

    n = ak.to_numpy(ak.num(data["p_log_pt"]))
    P = max_p
    mask = (np.arange(P)[None, :] < np.minimum(n, P)[:, None]).astype(np.float32)

    X = np.stack([cols[f] for f in CONT], axis=-1).astype(np.float32)  # [N,P,7]
    onehot = np.zeros((X.shape[0], P, N_CAT), dtype=np.float32)
    valid = cat >= 0
    idx = np.clip(cat, 0, N_CAT - 1)
    np.put_along_axis(onehot, idx[..., None], valid[..., None].astype(np.float32), axis=-1)
    X = np.concatenate([X, onehot], axis=-1)  # [N,P,13]
    X *= mask[..., None]
    return X, mask


@dataclass
class Splits:
    Xtr: np.ndarray; Mtr: np.ndarray; Ytr: np.ndarray
    Xva: np.ndarray; Mva: np.ndarray; Yva: np.ndarray
    Xte: np.ndarray; Mte: np.ndarray; Yte: np.ndarray
    feat_mean: np.ndarray; feat_std: np.ndarray
    y_mean: np.ndarray; y_std: np.ndarray
    extra: dict  # event-level scalars per split (for baselines/plots)


def load_splits(parquet: str, max_p: int = config.MAX_PARTICLES,
                targets=tuple(config.TARGETS), seed: int = config.SEED) -> Splits:
    data = ak.from_parquet(parquet)
    N = len(data)
    Y = np.stack([np.asarray(data[t]) for t in targets], axis=-1).astype(np.float32)
    X, M = _pad_dense(data, max_p)

    extra_all = {k: np.asarray(data[k]) for k in
                 ["n_soft", "sum_pt", "eta_ptweighted", "fb_asym", "mass", "pt_Z"]}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    f_tr, f_va, _ = config.SPLIT
    i_tr = int(f_tr * N); i_va = int((f_tr + f_va) * N)
    tr, va, te = perm[:i_tr], perm[i_tr:i_va], perm[i_va:]

    # standardize continuous features using the training set (cols 0..6)
    flat = X[tr][M[tr].astype(bool)]  # [Ntok,13]
    fmean = np.zeros(N_FEAT, np.float32); fstd = np.ones(N_FEAT, np.float32)
    fmean[:len(CONT)] = flat[:, :len(CONT)].mean(0)
    fstd[:len(CONT)] = flat[:, :len(CONT)].std(0) + 1e-6

    def norm(x, m):
        return ((x - fmean) / fstd) * m[..., None]

    ymean = Y[tr].mean(0); ystd = Y[tr].std(0) + 1e-6
    yn = (Y - ymean) / ystd

    def take(idx):
        return {k: v[idx] for k, v in extra_all.items()}

    return Splits(
        norm(X[tr], M[tr]), M[tr], yn[tr],
        norm(X[va], M[va]), M[va], yn[va],
        norm(X[te], M[te]), M[te], yn[te],
        fmean, fstd, ymean, ystd,
        {"train": take(tr), "val": take(va), "test": take(te),
         "y_test_raw": Y[te], "y_train_raw": Y[tr]},
    )


def save_norm(splits: Splits, path: str):
    json.dump(
        {"feat_mean": splits.feat_mean.tolist(), "feat_std": splits.feat_std.tolist(),
         "y_mean": splits.y_mean.tolist(), "y_std": splits.y_std.tolist(),
         "targets": list(config.TARGETS), "n_feat": N_FEAT},
        open(path, "w"), indent=2,
    )
