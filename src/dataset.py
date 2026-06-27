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


def _apply_particle_filter(data, abs_eta_max=None, abs_eta_min=None, charged_only=False,
                           puppi_min=None, neutral_only=False, pileup_addback=None):
    """Mask particles in-place on the jagged p_* columns (for ablations) and
    recompute the event-level summary scalars so baselines stay consistent."""
    eta = data["p_eta"]
    keep = ak.ones_like(eta, dtype=bool)
    if pileup_addback is not None:
        # pileup-contamination knob: keep the clean leading-vertex set (puppi>0.5)
        # plus a random fraction `pileup_addback` of the pileup (puppi<=0.5).
        prng = np.random.default_rng(909)
        flat = ak.flatten(data["p_puppi"]); counts = ak.num(data["p_puppi"])
        r = ak.unflatten(prng.random(len(flat)), counts)
        keep = keep & ((data["p_puppi"] > 0.5) | (r < pileup_addback))
    if abs_eta_max is not None:
        keep = keep & (abs(eta) < abs_eta_max)
    if abs_eta_min is not None:
        keep = keep & (abs(eta) >= abs_eta_min)
    if charged_only:
        keep = keep & (data["p_charge"] != 0)
    if neutral_only:
        keep = keep & (data["p_charge"] == 0)
    if puppi_min is not None:
        # PUPPI pileup suppression: in real data the soft event is dominated by
        # pileup; keeping puppiWeight >= puppi_min isolates the leading-vertex set.
        keep = keep & (data["p_puppi"] >= puppi_min)
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


def load_splits(parquet: str, max_p: int = config.MAX_PARTICLES,
                targets=tuple(config.TARGETS), seed: int = config.SEED,
                abs_eta_max=None, abs_eta_min=None, charged_only=False,
                puppi_min=None, neutral_only=False,
                mass_lo=None, mass_hi=None,
                eta_smear=None, pt_smear=None, drop_frac=None, pileup_addback=None) -> Splits:
    data = ak.from_parquet(parquet)
    # PF reconstruction-efficiency systematic: randomly drop a fraction of
    # particles (mimics tracking/PF inefficiency). Applied before everything.
    if drop_frac:
        prng = np.random.default_rng(31415 + (seed or 0))
        flat = ak.flatten(data["p_eta"]); counts = ak.num(data["p_eta"])
        keepmask = ak.unflatten(prng.random(len(flat)) > drop_frac, counts)
        pcols = [c for c in ak.fields(data) if c.startswith("p_")]
        newcols = {c: data[c][keepmask] for c in pcols}
        keep_rest = {c: data[c] for c in ak.fields(data) if not c.startswith("p_")}
        keep_rest.update(newcols)
        data = ak.Array(keep_rest)
    # event-level Z-mass window systematic (the skim already applied 81-101 GeV)
    if mass_lo is not None or mass_hi is not None:
        mlo = mass_lo if mass_lo is not None else 0.0
        mhi = mass_hi if mass_hi is not None else 1e9
        data = data[(data["mass"] > mlo) & (data["mass"] < mhi)]
    # detector-resolution systematics: per-particle smearing (uniform scales are
    # degenerate under per-feature standardization, so we smear, not shift).
    if eta_smear or pt_smear:
        prng = np.random.default_rng(20240 + (seed or 0))
        if pt_smear:   # fractional momentum resolution -> log_pt += N(0, sigma)
            flat = ak.flatten(data["p_log_pt"]); counts = ak.num(data["p_log_pt"])
            noise = prng.normal(0.0, pt_smear, size=len(flat))
            data["p_log_pt"] = ak.unflatten(ak.to_numpy(flat) + noise, counts)
        if eta_smear:  # eta resolution -> eta += N(0, sigma)
            flat = ak.flatten(data["p_eta"]); counts = ak.num(data["p_eta"])
            noise = prng.normal(0.0, eta_smear, size=len(flat))
            data["p_eta"] = ak.unflatten(ak.to_numpy(flat) + noise, counts)
    if (abs_eta_max is not None or abs_eta_min is not None or charged_only
            or puppi_min is not None or neutral_only or pileup_addback is not None):
        data = _apply_particle_filter(data, abs_eta_max, abs_eta_min, charged_only,
                                      puppi_min, neutral_only, pileup_addback)
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
