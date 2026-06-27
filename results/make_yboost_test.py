"""Does the soft event predict the WHOLE hard-scatter boost (y_boost = 1/2 ln x1/x2)
better than the V rapidity (y_Z)? If so, the V+jet recoil (y*_Z = y_Z - y_boost),
which is independently measurable, is the limiting term — and predicting y_boost
then adding the measured recoil would beat predicting y_Z directly.

Pythia Z->mumu with x1,x2 stored (data/skim/pythia_yb.parquet)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, awkward as ak
from src import dataset, train

P = "data/skim/pythia_yb.parquet"
d = ak.from_parquet(P)
yZ = np.asarray(d["y_Z"]); yb = np.asarray(d["y_boost"]); ptZ = np.asarray(d["pt_Z"])
ok = np.isfinite(yb)
print(f"events {len(d)} (finite y_boost {ok.sum()})")
print(f"y_Z std={np.nanstd(yZ):.3f}  y_boost std={np.nanstd(yb):.3f}")
ystar = yZ - yb   # V rapidity in the partonic CM = the recoil-induced shift
print(f"recoil shift y*_Z = y_Z - y_boost: std={np.nanstd(ystar):.3f}, "
      f"corr(|y*_Z|, pT_Z)={np.corrcoef(np.abs(ystar[ok]), ptZ[ok])[0,1]:+.3f}")
print(f"corr(y_Z, y_boost) = {np.corrcoef(yZ[ok], yb[ok])[0,1]:.3f}")

# train ONE EFN to predict both y_Z and y_boost from the same soft event
sp = dataset.load_splits(P, targets=("y_Z", "y_boost"))
_, pred, _ = train.train_model(sp, name="efn", epochs=45, device="cpu", verbose=False)
yt = sp.extra["y_test_raw"]
for i, t in enumerate(["y_Z (the V)", "y_boost (whole system)"]):
    c = np.corrcoef(yt[:, i], pred[:, i])[0, 1]
    print(f"  soft -> {t:24s}: corr = {c:.3f}  R2 = {1-np.sum((pred[:,i]-yt[:,i])**2)/np.sum((yt[:,i]-yt[:,i].mean())**2):.3f}")
cb = np.corrcoef(yt[:,1], pred[:,1])[0,1]; cz = np.corrcoef(yt[:,0], pred[:,0])[0,1]
print(f"\n=> soft predicts y_boost {'BETTER' if cb>cz else 'NOT better'} than y_Z "
      f"(delta corr = {cb-cz:+.3f}). If better, predicting y_boost + measured recoil "
      f"would beat direct y_Z.")
