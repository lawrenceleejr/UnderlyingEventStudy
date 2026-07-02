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


def _apply_particle_filter(data, abs_eta_max=None, abs_eta_min=None, charged_only=False):
    """Mask particles in-place on the jagged p_* columns (for ablations) and
    recompute the event-level summary scalars so baselines stay consistent."""
    eta = data["p_eta"]
    keep = ak.ones_like(eta, dtype=bool)
    if abs_eta_max is not None:
        keep = keep & (abs(eta) < abs_eta_max)
    if abs_eta_min is not None:
        keep = keep & (abs(eta) >= abs_eta_min)
    if charged_only:
        keep = keep & (data["p_charge"] != 0)
    pcols = [c for c in ak.fields(data) if c.startswith("p_")]
    new = {c: data[c][keep] for c in pcols}
    # recompute summaries from the filtered set
    pt = np.exp(new["p_log_pt"])
    sum_pt = ak.sum(pt, axis=1)
    eta_ptw = ak.where(sum_pt > 0, ak.sum(pt * new["p_eta"], axis=1) / sum_pt, 0.0)
    nf = ak.sum(new["p_eta"] > 0, axis=1); nb = ak.sum(new["p_eta"] < 0, axis=1)
    fb = ak.where((nf + nb) > 0, (nf - nb) / (nf + nb), 0.0)
    keep_cols = {c: data[c] for c in ak.fields(data) if not c.startswith("p_")}
    keep_cols.update(new)
    keep_cols["n_soft"] = ak.num(new["p_log_pt"])
    keep_cols["sum_pt"] = sum_pt
    keep_cols["eta_ptweighted"] = eta_ptw
    keep_cols["fb_asym"] = fb
    return ak.Array(keep_cols)


def engineered_features(data: ak.Array) -> dict:
    """Richer hand-built event observables for a *fair* non-deep baseline.

    Computed from the same stored per-particle arrays the network sees, so the
    engineered baseline and the deep set have access to identical information
    (eta, pT, relative azimuth, charge, species) — the comparison then isolates
    the architecture rather than the feature set.
    """
    pt = np.exp(data["p_log_pt"])
    eta = data["p_eta"]
    ch = data["p_charge"] != 0
    sdp, cdp = data["p_sin_dphi"], data["p_cos_dphi"]
    sum_pt = ak.sum(pt, axis=1)

    def ptw(x, sel=None):
        w = pt if sel is None else pt * sel
        den = ak.sum(w, axis=1)
        return np.asarray(ak.where(den > 0, ak.sum(w * x, axis=1) / den, 0.0), np.float32)

    feats = {
        "eta_ptw_charged": ptw(eta, ch),
        "eta_ptw_neutral": ptw(eta, ~ch),
        "eta_mean": np.asarray(ak.where(ak.num(eta) > 0, ak.mean(eta, axis=1), 0.0), np.float32),
        "eta2_ptw": ptw(eta ** 2),                    # longitudinal width
        "eta3_ptw": ptw(eta ** 3),                    # longitudinal skew
        "eta_cdp_ptw": ptw(eta * cdp),                # recoil-correlated eta
        "cdp_ptw": ptw(cdp),                          # net recoil along Z azimuth
        "sdp_ptw": ptw(sdp),
        "sumpt_fb": np.asarray(ak.where(sum_pt > 0,
                    ak.sum(pt * np.sign(eta), axis=1) / sum_pt, 0.0), np.float32),
        "n_charged": np.asarray(ak.sum(ch, axis=1), np.float32),
        "lead_eta": np.asarray(ak.fill_none(ak.firsts(
                    eta[ak.argsort(pt, axis=1, ascending=False)]), 0.0), np.float32),
    }
    return feats


ENGINEERED = ["eta_ptweighted", "fb_asym", "n_soft", "sum_pt",
              "eta_ptw_charged", "eta_ptw_neutral", "eta_mean", "eta2_ptw",
              "eta3_ptw", "eta_cdp_ptw", "cdp_ptw", "sdp_ptw", "sumpt_fb",
              "n_charged", "lead_eta"]


def load_splits(parquet: str, max_p: int = config.MAX_PARTICLES,
                targets=tuple(config.TARGETS), seed: int = config.SEED,
                abs_eta_max=None, abs_eta_min=None, charged_only=False) -> Splits:
    data = ak.from_parquet(parquet)
    if abs_eta_max is not None or abs_eta_min is not None or charged_only:
        data = _apply_particle_filter(data, abs_eta_max, abs_eta_min, charged_only)
    N = len(data)
    Y = np.stack([np.asarray(data[t]) for t in targets], axis=-1).astype(np.float32)
    X, M = _pad_dense(data, max_p)

    extra_all = {k: np.asarray(data[k]) for k in
                 ["n_soft", "sum_pt", "eta_ptweighted", "fb_asym", "mass", "pt_Z"]}
    extra_all.update(engineered_features(data))

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


def featurize(parquet: str, feat_mean: np.ndarray, feat_std: np.ndarray,
              max_p: int = config.MAX_PARTICLES, abs_eta_max=None, charged_only=False):
    """Pad + normalize a parquet using EXTERNAL stats (e.g. trained on Z).

    Returns (X[N,P,F], mask[N,P], data) so a model trained on one sample can be
    applied to another (the Z -> W transfer).
    """
    data = ak.from_parquet(parquet)
    if abs_eta_max is not None or charged_only:
        data = _apply_particle_filter(data, abs_eta_max=abs_eta_max, charged_only=charged_only)
    X, M = _pad_dense(data, max_p)
    Xn = ((X - feat_mean) / feat_std) * M[..., None]
    return Xn.astype(np.float32), M, data


def save_norm(splits: Splits, path: str):
    json.dump(
        {"feat_mean": splits.feat_mean.tolist(), "feat_std": splits.feat_std.tolist(),
         "y_mean": splits.y_mean.tolist(), "y_std": splits.y_std.tolist(),
         "targets": list(config.TARGETS), "n_feat": N_FEAT},
        open(path, "w"), indent=2,
    )
