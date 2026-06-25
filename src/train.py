"""Train a particle-cloud regressor (EFN or Transformer) on the soft particles."""
from __future__ import annotations

import os

# let any op without an MPS kernel fall back to CPU (must be set before torch use)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .models.efn import EnergyFlowNetwork
from .models.transformer import ParticleTransformer


def pick_device(pref: str = "auto") -> str:
    """Choose the fastest available backend.

    "auto" -> Apple Metal (mps) on Mac, else CUDA, else CPU. This is what makes
    the training run fast on an M-series Mac with zero extra flags.
    """
    if pref and pref != "auto":
        return pref
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _loaders(splits, batch=256):
    def ds(X, M, Y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y))
    tr = DataLoader(ds(splits.Xtr, splits.Mtr, splits.Ytr), batch_size=batch, shuffle=True)
    va = DataLoader(ds(splits.Xva, splits.Mva, splits.Yva), batch_size=512)
    te = DataLoader(ds(splits.Xte, splits.Mte, splits.Yte), batch_size=512)
    return tr, va, te


def build_model(name, n_feat, n_out):
    if name == "efn":
        return EnergyFlowNetwork(n_feat, n_out)
    if name == "transformer":
        return ParticleTransformer(n_feat, n_out)
    raise ValueError(name)


def train_model(splits, name="efn", epochs=40, lr=1e-3, batch=256,
                patience=8, device="auto", verbose=True):
    torch.manual_seed(config.SEED)
    device = pick_device(device)
    if verbose:
        print(f"  [{name}] device = {device}", flush=True)
    n_feat = splits.Xtr.shape[-1]
    n_out = splits.Ytr.shape[-1]
    model = build_model(name, n_feat, n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    tr, va, te = _loaders(splits, batch)

    best_val = np.inf; best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        for X, M, Y in tr:
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X, M), Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        # validation
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for X, M, Y in va:
                vl += loss_fn(model(X.to(device), M.to(device)), Y.to(device)).item(); nb += 1
        vl /= max(nb, 1)
        if verbose:
            print(f"  [{name}] epoch {ep+1}/{epochs} val_huber={vl:.4f}", flush=True)
        if vl < best_val - 1e-4:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [{name}] early stop at epoch {ep+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # predict on test, return RAW-scale predictions [N,n_out]
    model.eval(); preds = []
    with torch.no_grad():
        for X, M, Y in te:
            preds.append(model(X.to(device), M.to(device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    pred_raw = pred * splits.y_std + splits.y_mean
    return model, pred_raw, best_val
