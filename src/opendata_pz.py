"""Longitudinal-momentum estimator with per-event uncertainty (for residual+pull).

Trains an EFN with a heteroscedastic Gaussian head (outputs mean and log-variance
per target) under a Gaussian negative-log-likelihood loss, so every event gets a
predicted p_z AND an uncertainty sigma -> a meaningful pull (p_z^pred - p_z^true)/sigma.

Two modes:
  (Z)  train+test on one parquet (e.g. data PUPPI>0.5):           --parquet ... --tag opendata_pz
  (W)  train on Z, apply to a W parquet (truth in same columns):  --parquet Zpythia --w Wpythia --tag w_pz

    python -m src.opendata_pz --parquet data/skim/opendata.parquet --puppi 0.5 --tag opendata_pz
    python -m src.opendata_pz --parquet data/skim/pythia.parquet --w data/skim/w.parquet --tag w_pz
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config, dataset, metrics, plotstyle as ts
from .models.efn import EnergyFlowNetwork
ts.use()

T = len(config.TARGETS)


def gaussian_nll(out, y):
    mu, logv = out[:, :T], out[:, T:]
    logv = torch.clamp(logv, -6, 6)
    return (0.5 * torch.exp(-logv) * (y - mu) ** 2 + 0.5 * logv).mean()


def train_hetero(splits, epochs=60, lr=1e-3, batch=256, patience=10, device="cpu"):
    torch.manual_seed(config.SEED)
    model = EnergyFlowNetwork(splits.Xtr.shape[-1], 2 * T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    def ds(X, M, Y): return TensorDataset(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y))
    tr = DataLoader(ds(splits.Xtr, splits.Mtr, splits.Ytr), batch_size=batch, shuffle=True)
    va = DataLoader(ds(splits.Xva, splits.Mva, splits.Yva), batch_size=512)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for X, M, Y in tr:
            opt.zero_grad(); loss = gaussian_nll(model(X.to(device), M.to(device)), Y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for X, M, Y in va:
                vl += gaussian_nll(model(X.to(device), M.to(device)), Y.to(device)).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best - 1e-4: best, best_state, bad = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def predict(model, X, M, ymean, ystd, device="cpu", batch=512):
    model.eval(); mus, sgs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(torch.from_numpy(X[i:i+batch]).to(device), torch.from_numpy(M[i:i+batch]).to(device)).cpu().numpy()
            mus.append(out[:, :T]); sgs.append(np.exp(0.5 * np.clip(out[:, T:], -6, 6)))
    mu = np.concatenate(mus) * ystd + ymean
    sg = np.concatenate(sgs) * ystd
    return mu, sg


def plots(yt, mu, sg, tag, tname, units, outdir):
    ti = list(config.TARGETS).index(tname)
    yt = yt[:, ti]; m = mu[:, ti]; s = np.clip(sg[:, ti], 1e-6, None)
    res = m - yt; pull = res / s
    cc = np.corrcoef(yt, m)[0, 1]; r2 = metrics.regression_metrics(yt, m)["r2"]
    pm, pw = float(pull.mean()), float(pull.std())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (1) pred vs true -- muted density, range-framed, diagonal in accent
    lim = np.percentile(np.abs(yt), 99) * 1.05
    ax[0].hist2d(yt, m, bins=70, range=[[-lim, lim], [-lim, lim]], cmap=ts.density_cmap(), cmin=1)
    ax[0].plot([-lim, lim], [-lim, lim], ls=(0,(4,2)), lw=1.1, color=ts.ACCENT2)
    ax[0].set_xlabel(f"true {tname} {units}"); ax[0].set_ylabel(f"predicted {tname} {units}")
    ax[0].set_aspect("equal"); ts.minimal(ax[0])
    ax[0].set_title(f"(a) {tname}: corr={cc:.2f}, $R^2$={r2:.2f}")
    # (2) residual vs the no-information (predict-the-mean) baseline -> info gain
    base_res = yt.mean() - yt          # predict-the-mean residual (width = sigma_y)
    rl = np.percentile(np.abs(np.concatenate([res, base_res])), 99)
    ax[1].hist(base_res, bins=70, range=(-rl, rl), histtype="step", lw=1.4, color=ts.MUTE, ls=(0,(4,2)))
    ax[1].hist(res, bins=70, range=(-rl, rl), histtype="step", lw=1.7, color=ts.ACCENT)
    ax[1].axvline(0, color=ts.MUTE, lw=0.6)
    ts.label_end(ax[1], rl*0.30, ax[1].get_ylim()[1]*0.85, "predict\nthe mean", ts.MUTE)
    ts.label_end(ax[1], -rl*0.95, ax[1].get_ylim()[1]*0.6, "EFN", ts.ACCENT, fontweight="bold")
    ax[1].set_xlabel(f"residual  ({tname}: pred $-$ true) {units}"); ax[1].set_ylabel("events")
    ts.minimal(ax[1])
    ax[1].set_title(rf"(b) residual: $\sigma$={res.std():.2f} vs $\sigma_y$={base_res.std():.2f} (info gain)")
    # (3) pull vs unit Gaussian (direct-labeled, no legend box)
    xb = np.linspace(-5, 5, 160)
    ax[2].hist(pull, bins=70, range=(-5, 5), density=True, histtype="step", lw=1.6, color=ts.GOOD)
    ax[2].plot(xb, np.exp(-xb**2/2)/np.sqrt(2*np.pi), ls=(0,(4,2)), lw=1.2, color=ts.MUTE)
    ts.label_end(ax[2], 2.4, 0.32, "unit\nGaussian", ts.MUTE)
    ax[2].axvline(0, color=ts.MUTE, lw=0.6)
    ax[2].set_xlabel(r"pull  (residual$/\sigma_{\rm pred}$)"); ax[2].set_ylabel("a.u."); ts.minimal(ax[2])
    ax[2].set_title(rf"(c) pull: $\mu$={pm:.2f}, width={pw:.2f}")
    fig.suptitle(f"{tag} — longitudinal-momentum estimate ({tname})", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(outdir, f"pz_{tag}.png"); fig.savefig(p); plt.close(fig)
    return {"corr": float(cc), "r2": float(r2), "res_std": float(res.std()),
            "pull_mean": pm, "pull_width": pw, "png": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/skim/opendata.parquet")
    ap.add_argument("--w", default=None, help="apply Z-trained model to this W parquet")
    ap.add_argument("--puppi", type=float, default=None)
    ap.add_argument("--max-p", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target", default="pz_Z")
    ap.add_argument("--tag", default="opendata_pz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    outdir = str(config.RESULTS)

    filt = {} if args.puppi is None else {"puppi_min": args.puppi}
    sp = dataset.load_splits(args.parquet, max_p=args.max_p, **filt)
    model = train_hetero(sp, epochs=args.epochs, device=args.device)

    res_all = {}
    if args.w is None:
        mu, sg = predict(model, sp.Xte, sp.Mte, sp.y_mean, sp.y_std, args.device)
        yt = sp.extra["y_test_raw"]
        units = {"y_Z": "", "pz_Z": "[GeV]", "beta_z": ""}.get(args.target, "")
        np.savez(os.path.join(outdir, f"pz_{args.tag}_preds.npz"), yt=yt, mu=mu, sg=sg,
                 targets=np.array(list(config.TARGETS)))
        res_all[args.target] = plots(yt, mu, sg, args.tag, args.target, units, outdir)
        # also y_Z for completeness
        res_all["y_Z"] = plots(yt, mu, sg, args.tag + "_yZ", "y_Z", "", outdir)
    else:
        # transfer: featurize W with the Z-trained normalization, predict, validate on W truth
        Xw, Mw, dataw = dataset.featurize(args.w, sp.feat_mean, sp.feat_std, max_p=args.max_p,
                                          **({"abs_eta_max": None} if False else {}))
        if args.puppi is not None:
            pass  # Pythia W has puppi=1; no-op
        mu, sg = predict(model, Xw, Mw, sp.y_mean, sp.y_std, args.device)
        yt = np.stack([np.asarray(dataw[t]) for t in config.TARGETS], axis=-1).astype(np.float32)
        np.savez(os.path.join(outdir, f"pz_{args.tag}_preds.npz"), yt=yt, mu=mu, sg=sg,
                 targets=np.array(list(config.TARGETS)))
        res_all["pz_Z"] = plots(yt, mu, sg, args.tag, "pz_Z", "[GeV]", outdir)
        res_all["y_Z"] = plots(yt, mu, sg, args.tag + "_yW", "y_Z", "", outdir)

    json.dump(res_all, open(os.path.join(outdir, f"pz_{args.tag}.json"), "w"), indent=2)
    for k, v in res_all.items():
        print(f"{args.tag} [{k}]: corr={v['corr']:.3f} R2={v['r2']:.3f} res_std={v['res_std']:.2f} "
              f"pull={v['pull_mean']:.2f}+/-{v['pull_width']:.2f}  -> {v['png']}", flush=True)


if __name__ == "__main__":
    main()
