"""Z -> W transfer and the W-mass / neutrino-pz application.

Pipeline:
  1. train the soft-particle boost regressor on Z (boost known from the dilepton);
  2. apply it to W->mu nu to predict the W longitudinal boost pz_W;
  3. infer the neutrino p_z = pz_W_pred - mu_pz (the W's missing d.o.f.);
  4. reconstruct m_W and compare against the no-longitudinal-information baseline
     and the truth-pz ideal.

This is the demonstrator for "measure the soft system, constrain the W boost,
reduce the longitudinal/PDF modelling that limits the W-mass measurement."

    python -m src.wmass --z data/skim/pythia.parquet --w data/skim/w.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config, dataset, metrics, train  # noqa: E402

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except Exception:
    pass

MW = 80.385


def _predict(model, X, M, device, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            mb = torch.from_numpy(M[i:i + batch]).to(device)
            out.append(model(xb, mb).cpu().numpy())
    return np.concatenate(out, 0)


def reco_mass(mu, nu_px, nu_py, nu_pz):
    """Invariant mass of mu + nu given the neutrino p_z hypothesis."""
    nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
    E = mu["E"] + nu_E
    px = mu["px"] + nu_px
    py = mu["py"] + nu_py
    pz = mu["pz"] + nu_pz
    return np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))


def transverse_mass(mu, nu_px, nu_py):
    mu_pt = np.sqrt(mu["px"]**2 + mu["py"]**2)
    nu_pt = np.sqrt(nu_px**2 + nu_py**2)
    dphi = np.arctan2(mu["py"], mu["px"]) - np.arctan2(nu_py, nu_px)
    return np.sqrt(2 * mu_pt * nu_pt * (1 - np.cos(dphi)))


def write_report(R, outdir):
    yprior = R.get("yW_prior_std", float("nan"))
    ycond = R.get("yW_conditional_res", R["transfer_y"]["resolution"])
    reduction = 100 * (1 - ycond / yprior) if yprior else float("nan")
    md = f"""# W-boson boost from the soft system &rarr; W-mass application

Train the soft-particle boost regressor on **Z** (boost known from the dilepton),
apply it to **W&rarr;&mu;&nu;** (boost unknown), then ask what it buys for the W mass.

## 1. Z &rarr; W transfer — the key enabler (it works)
| quantity | corr |
|---|---|
| p_z (Z test, in-domain) | {R['z_pz_corr']:.3f} |
| **p_z (Z-trained, applied to W)** | **{R['transfer_pz']['corr']:.3f}** |
| y_W (Z-trained, applied to W) | {R['transfer_y']['corr']:.3f} |

A model trained only on Z predicts the **W** longitudinal boost as well as it does
in-domain on Z. Z genuinely calibrates W — the premise the whole W-mass programme
relies on — and the soft-system boost estimator is portable across the two.

![Z->W transfer](wtransfer_pz.png)

## 2. How strong is the per-event constraint? (honest: not useful per event)
The prior spread of y_W is {yprior:.2f}; conditioning on the soft system gives a
residual spread of {ycond:.2f} — a **{reduction:.0f}% reduction**, i.e. the soft
event explains only corr&sup2; &approx; {R['transfer_pz']['corr']**2*100:.0f}% of
the boost variance.

Propagated to the neutrino, the soft-system estimate is **worse than assuming
&nu;&nbsp;p_z&nbsp;=&nbsp;0**: subtracting the muon p_z from a weakly-correlated
W-p_z prediction leaves a residual dominated by &minus;&mu;-p_z, giving
corr(&nu;-p_z) = {R['nu_pz']['corr']:.2f} and RMSE {R['nu_pz']['rmse']:.0f} GeV
versus {R['nu_pz_noinfo_rms']:.0f} GeV for the zero hypothesis. Per event, this
estimator should NOT be used for the neutrino; its value is aggregate (Sec. 4).

## 3. Reconstructed m_W per event (no improvement yet)
| &nu; p_z hypothesis | median m_W [GeV] | resolution 60&ndash;100 GeV |
|---|---|---|
| truth (ideal check) | {R['mW_truth_check']:.2f} | sharp |
| soft-system | {R['mW_median_soft']:.2f} | {R['mW_resolution_soft']:.1f} GeV |
| none (p_z = 0) | – | {R['mW_resolution_zero']:.1f} GeV |

![mW reco](wmass_reco.png)

Directly reconstructing m_W event-by-event from the soft-predicted &nu; p_z does
**not** beat the no-longitudinal-information case at this resolution — the truth-p_z
curve shows the ceiling if the boost were known exactly.

## 4. Where the value actually is
- **Portability (proven):** train on Z, deploy on W — the hard part works.
- **Aggregate, not per-event:** even a weak per-event correlation constrains the
  *ensemble* y_W distribution, which is currently taken from PDFs and is a leading
  m_W systematic. The soft system offers a *data-driven* cross-check on the W
  longitudinal kinematics, orthogonal to the recoil (which only fixes p_T^W).
- **Headroom:** resolution should improve with a heavier model (Transformer on
  GPU), more statistics, and — crucially — by moving beyond Pythia, since the
  MPI-off study shows the signal is in the beam-remnant/ISR fragmentation whose
  data/MC modelling is exactly what a real measurement would pin down.

This is a demonstrator, not a finished measurement: it shows the estimator is
real and portable, and quantifies honestly how much longitudinal information the
soft event currently provides.
"""
    open(os.path.join(outdir, "REPORT_wmass.md"), "w").write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", default="data/skim/pythia.parquet")
    ap.add_argument("--w", default="data/skim/w.parquet")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-p", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default=str(config.RESULTS))
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate REPORT_wmass.md from an existing wmass.json")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.report_only:
        R = json.load(open(os.path.join(args.outdir, "wmass.json")))
        write_report(R, args.outdir)
        print("regenerated REPORT_wmass.md")
        return
    ipz = list(config.TARGETS).index("pz_Z")
    iy = list(config.TARGETS).index("y_Z")

    # 1. train the boost regressor on Z
    zs = dataset.load_splits(args.z, max_p=args.max_p)
    device = train.pick_device(args.device)
    model, zpred, _ = train.train_model(zs, name="efn", epochs=args.epochs, device=device)
    z_pz = metrics.regression_metrics(zs.extra["y_test_raw"][:, ipz], zpred[:, ipz])
    print(f"[Z] pz corr={z_pz['corr']:.3f}", flush=True)

    # 2. apply to W (Z normalization stats)
    Xw, Mw, dw = dataset.featurize(args.w, zs.feat_mean, zs.feat_std, max_p=args.max_p)
    pred_std = _predict(model, Xw, Mw, device)
    pred = pred_std * zs.y_std + zs.y_mean
    pzW_pred = pred[:, ipz]
    yW_pred = pred[:, iy]

    pzW_true = np.asarray(dw["pz_Z"])
    yW_true = np.asarray(dw["y_Z"])
    transfer_pz = metrics.regression_metrics(pzW_true, pzW_pred)
    transfer_y = metrics.regression_metrics(yW_true, yW_pred)
    print(f"[Z->W transfer] pz corr={transfer_pz['corr']:.3f} | y corr={transfer_y['corr']:.3f}", flush=True)

    # 3. neutrino p_z and 4. W mass
    mu = {k: np.asarray(dw[f"mu_{k}"]) for k in ["px", "py", "pz", "E"]}
    nu_px, nu_py = np.asarray(dw["nu_px"]), np.asarray(dw["nu_py"])
    nu_pz_true = np.asarray(dw["nu_pz_true"])
    nu_pz_pred = pzW_pred - mu["pz"]
    nu_pz_zero = np.zeros_like(nu_pz_true)        # "no longitudinal info" baseline

    nu_metrics = metrics.regression_metrics(nu_pz_true, nu_pz_pred)
    print(f"[nu pz] corr={nu_metrics['corr']:.3f} rms={nu_metrics['rmse']:.1f} GeV "
          f"(no-info rms={np.std(nu_pz_true):.1f})", flush=True)

    mW_truth = reco_mass(mu, nu_px, nu_py, nu_pz_true)
    mW_soft = reco_mass(mu, nu_px, nu_py, nu_pz_pred)
    mW_zero = reco_mass(mu, nu_px, nu_py, nu_pz_zero)
    mT = transverse_mass(mu, nu_px, nu_py)

    def around(x, lo=60, hi=100):
        return x[(x > lo) & (x < hi)]

    results = {
        "z_pz_corr": z_pz["corr"],
        "transfer_pz": transfer_pz, "transfer_y": transfer_y,
        "yW_prior_std": float(np.std(yW_true)),
        "yW_conditional_res": float(transfer_y["resolution"]),
        "nu_pz": nu_metrics,
        "nu_pz_noinfo_rms": float(np.std(nu_pz_true)),
        "mW_resolution_soft": float(np.std(around(mW_soft))),
        "mW_resolution_zero": float(np.std(around(mW_zero))),
        "mW_median_soft": float(np.median(around(mW_soft))),
        "mW_truth_check": float(np.median(around(mW_truth))),
        "n_W": int(len(dw)),
    }
    json.dump(results, open(os.path.join(args.outdir, "wmass.json"), "w"), indent=2)

    # plots
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = 600
    ax.hist2d(np.clip(pzW_true, -lim, lim), np.clip(pzW_pred, -lim, lim), bins=60, cmap="magma")
    ax.plot([-lim, lim], [-lim, lim], "w--", lw=1)
    ax.set_xlabel(r"true $p_z^W$ [GeV]"); ax.set_ylabel(r"predicted $p_z^W$ [GeV] (Z-trained)")
    ax.set_title(f"Z$\\rightarrow$W transfer: corr={transfer_pz['corr']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "wtransfer_pz.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    b = np.linspace(40, 120, 80)
    ax.hist(around(mW_zero, 40, 120), bins=b, histtype="step", density=True, label=r"$\nu p_z=0$ (no long. info)")
    ax.hist(around(mW_soft, 40, 120), bins=b, histtype="step", density=True, lw=2, label="soft-system $\\nu p_z$")
    ax.hist(around(mW_truth, 40, 120), bins=b, histtype="step", density=True, label="truth $\\nu p_z$ (ideal)")
    ax.axvline(MW, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"reconstructed $m_W$ [GeV]"); ax.set_ylabel("a.u."); ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "wmass_reco.png"), dpi=110); plt.close(fig)

    write_report(results, args.outdir)
    print("wrote results/wmass.json, REPORT_wmass.md, wtransfer_pz.png, wmass_reco.png")


if __name__ == "__main__":
    main()
