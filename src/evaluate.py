"""End-to-end evaluation: baselines + deep models -> metrics, plots, report.

Usage:
    python -m src.evaluate --parquet data/skim/pythia.parquet --models efn transformer
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config, dataset, baselines, metrics, train  # noqa: E402

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except Exception:
    pass


def evaluate(parquet, models=("efn",), epochs=40, tag="pythia", outdir=None,
             max_p=config.MAX_PARTICLES, device="auto"):
    outdir = outdir or str(config.RESULTS)
    os.makedirs(outdir, exist_ok=True)
    splits = dataset.load_splits(parquet, max_p=max_p)
    dataset.save_norm(splits, os.path.join(outdir, f"norm_{tag}.json"))
    targets = list(config.TARGETS)
    ip = targets.index(config.PRIMARY_TARGET)

    results = {"tag": tag, "parquet": parquet, "n_test": int(len(splits.Yte)),
               "n_train": int(len(splits.Ytr)), "targets": targets, "models": {}}

    # --- baselines (per target) ---
    base_pred = {}
    for ti, t in enumerate(targets):
        b = baselines.run_baselines(splits, ti)
        for name, pred in b["pred"].items():
            base_pred.setdefault(name, {})[t] = pred
        if ti == ip:
            yte_primary = b["y_true"]
    for name, pert in base_pred.items():
        results["models"][name] = _score(pert, splits, targets)

    # --- deep models ---
    deep_pred = {}
    for mname in models:
        print(f"== training {mname} ==", flush=True)
        _, pred_raw, val = train.train_model(splits, name=mname, epochs=epochs, device=device)
        pert = {t: pred_raw[:, ti] for ti, t in enumerate(targets)}
        deep_pred[mname] = pert
        results["models"][mname] = _score(pert, splits, targets)
        results["models"][mname]["val_huber"] = float(val)

    json.dump(results, open(os.path.join(outdir, f"metrics_{tag}.json"), "w"), indent=2)

    # --- plots (primary target) ---
    yte = splits.extra["y_test_raw"][:, ip]
    best_model = models[-1] if models else "gbdt_summary"
    best = deep_pred.get(best_model, base_pred.get("gbdt_summary"))[config.PRIMARY_TARGET]
    _plots(yte, best, base_pred, deep_pred, results, tag, outdir, best_model)
    _write_report(results, tag, outdir, best_model)
    print(f"\nwrote metrics_{tag}.json and summary_{tag}.md to {outdir}")
    return results


def _score(pert, splits, targets):
    ip = targets.index(config.PRIMARY_TARGET)
    out = {}
    for ti, t in enumerate(targets):
        yt = splits.extra["y_test_raw"][:, ti]
        out[t] = metrics.regression_metrics(yt, pert[t])
        if t == config.PRIMARY_TARGET:
            out[t]["sign_acc"] = metrics.sign_accuracy(yt, pert[t])
    return out


def _plots(yte, best, base_pred, deep_pred, results, tag, outdir, best_model):
    # 1) predicted vs true (headline)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist2d(yte, best, bins=60, cmap="viridis",
              range=[[yte.min(), yte.max()], [yte.min(), yte.max()]])
    lims = [yte.min(), yte.max()]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel(r"true $y_Z$"); ax.set_ylabel(r"predicted $y_Z$ (%s)" % best_model)
    r2 = results["models"][best_model][config.PRIMARY_TARGET]["r2"]
    cc = results["models"][best_model][config.PRIMARY_TARGET]["corr"]
    ax.set_title(f"{tag}: R$^2$={r2:.3f}  corr={cc:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"pred_vs_true_{tag}.png"), dpi=110)
    plt.close(fig)

    # 2) residual distribution: best model vs mean baseline
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(base_pred["mean"][config.PRIMARY_TARGET] - yte, bins=80, histtype="step",
            density=True, label="mean baseline")
    ax.hist(best - yte, bins=80, histtype="step", density=True, label=best_model)
    ax.set_xlabel(r"$y_Z^{pred}-y_Z^{true}$"); ax.set_ylabel("a.u."); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"residual_{tag}.png"), dpi=110)
    plt.close(fig)

    # 3) resolution vs true
    c, s, mn = metrics.resolution_vs_truth(yte, best)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c, s, "o-", label="resolution (std)")
    ax.plot(c, mn, "s--", label="bias (mean)")
    ax.set_xlabel(r"true $y_Z$"); ax.set_ylabel("residual"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"resolution_{tag}.png"), dpi=110)
    plt.close(fig)

    # 4) model comparison bar (R2 on primary)
    names = list(results["models"].keys())
    r2s = [results["models"][n][config.PRIMARY_TARGET]["r2"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, r2s, color="steelblue")
    ax.set_ylabel(r"$R^2$ on $y_Z$"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"model_compare_{tag}.png"), dpi=110)
    plt.close(fig)


def _write_report(results, tag, outdir, best_model):
    lines = [f"# Results ({tag})", ""]
    lines.append(f"- train events: {results['n_train']}, test events: {results['n_test']}")
    lines.append(f"- targets: {', '.join(results['targets'])}")
    lines.append("")
    lines.append("## Primary target: y_Z")
    lines.append("")
    lines.append("| model | R2 | corr | RMSE | resolution | sign acc |")
    lines.append("|---|---|---|---|---|---|")
    for n, m in results["models"].items():
        p = m[config.PRIMARY_TARGET]
        lines.append(f"| {n} | {p['r2']:.3f} | {p['corr']:.3f} | {p['rmse']:.3f} | "
                     f"{p['resolution']:.3f} | {p.get('sign_acc', float('nan')):.3f} |")
    lines.append("")
    for t in results["targets"]:
        if t == config.PRIMARY_TARGET:
            continue
        lines.append(f"## {t}")
        lines.append("| model | R2 | corr | RMSE |")
        lines.append("|---|---|---|---|")
        for n, m in results["models"].items():
            p = m[t]
            lines.append(f"| {n} | {p['r2']:.3f} | {p['corr']:.3f} | {p['rmse']:.3f} |")
        lines.append("")
    lines.append(f"![pred vs true](pred_vs_true_{tag}.png)")
    lines.append(f"![resolution](resolution_{tag}.png)")
    # "summary_" not "REPORT_": REPORT_*.md are curated documents that a
    # reproduce run must never overwrite
    open(os.path.join(outdir, f"summary_{tag}.md"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--models", nargs="+", default=["efn"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tag", default="pythia")
    ap.add_argument("--max-p", type=int, default=config.MAX_PARTICLES)
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    args = ap.parse_args()
    evaluate(args.parquet, models=tuple(args.models), epochs=args.epochs,
             tag=args.tag, max_p=args.max_p, device=args.device)


if __name__ == "__main__":
    main()
