"""Fast matched-Pythia decomposition (same selection as data; puppi=1 so PUPPI
cuts are no-ops). EFN corr per soft-event component, for the data-vs-MC per-bin
reference in the cross-checks figure. max_p=150 (Pythia <n>~74, no truncation)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import dataset, train, metrics, config
ip = list(config.TARGETS).index(config.PRIMARY_TARGET)
PARQUET = "data/skim/pythia_demo.parquet"
CFG = {
    "all":     {},
    "charged": {"charged_only": True, "abs_eta_max": 2.5},
    "neutral": {"neutral_only": True},
    "central": {"abs_eta_max": 2.5},
    "forward": {"abs_eta_min": 2.5},
}
out = "results/ablation_pythia_matched.json"
rows = json.load(open(out)) if os.path.exists(out) else {}
for name, filt in CFG.items():
    if name in rows:
        print(f"  [cached] {name}: {rows[name]['corr']:+.3f}", flush=True); continue
    sp = dataset.load_splits(PARQUET, max_p=150, **filt)
    _, pred, _ = train.train_model(sp, name="efn", epochs=30, device="cpu", verbose=False)
    yt = sp.extra["y_test_raw"][:, ip]; yp = pred[:, ip]
    m = metrics.regression_metrics(yt, yp)
    m["sign_acc"] = metrics.sign_accuracy(yt, yp); m["mean_n"] = float(np.mean(sp.extra["test"]["n_soft"]))
    rows[name] = m; json.dump(rows, open(out, "w"), indent=2)
    print(f"  {name}: corr={m['corr']:+.3f} <n>={m['mean_n']:.0f}", flush=True)
print("wrote", out)
