"""Kinematic-performance plots for the headline PUPPI>0.5 EFN (trains once, saves preds)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import dataset, train, metrics, config

ip = list(config.TARGETS).index(config.PRIMARY_TARGET)
npz = "results/preds_opendata_puppi.npz"
if os.path.exists(npz):
    z = np.load(npz); yte, yp = z["yte"], z["yp"]; ymean_pred = z["ymean"]
    print("loaded cached preds")
else:
    sp = dataset.load_splits("data/skim/opendata.parquet", max_p=150, puppi_min=0.5)
    _, pred, _ = train.train_model(sp, name="efn", epochs=30, device="cpu", verbose=False)
    yte = sp.extra["y_test_raw"][:, ip]; yp = pred[:, ip]
    ymean_pred = np.full_like(yte, sp.extra["y_train_raw"][:, ip].mean())
    np.savez(npz, yte=yte, yp=yp, ymean=ymean_pred)
    print("trained & saved preds")

cc = np.corrcoef(yte, yp)[0,1]; r2 = metrics.regression_metrics(yte, yp)["r2"]
sgn = metrics.sign_accuracy(yte, yp)
print(f"corr={cc:.3f} R2={r2:.3f} sign={sgn:.3f}")

fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

# (0,0) pred vs true 2D
h = ax[0,0].hist2d(yte, yp, bins=70, range=[[-3,3],[-3,3]], cmap="magma")
ax[0,0].plot([-3,3],[-3,3],"w--",lw=1); fig.colorbar(h[3], ax=ax[0,0], fraction=0.046)
ax[0,0].set_xlabel(r"true $y_Z$"); ax[0,0].set_ylabel(r"EFN predicted $y_Z$")
ax[0,0].set_title(rf"pred vs true: corr={cc:.3f}, R$^2$={r2:.3f}")

# (0,1) residual distribution vs mean baseline
ax[0,1].hist(ymean_pred - yte, bins=80, range=(-3,3), histtype="step", lw=2, density=True, label="predict-the-mean")
ax[0,1].hist(yp - yte, bins=80, range=(-3,3), histtype="step", lw=2, density=True, label="EFN")
ax[0,1].set_xlabel(r"$y_Z^{pred}-y_Z^{true}$"); ax[0,1].set_ylabel("a.u."); ax[0,1].legend()
ax[0,1].set_title(rf"residual: $\sigma$={(yp-yte).std():.2f} vs {(ymean_pred-yte).std():.2f}")

# (0,2) resolution & bias vs true y_Z
edges = np.quantile(yte, np.linspace(0,1,11)); edges[-1]+=1e-6
idx = np.digitize(yte, edges)-1
cen, res, bias = [], [], []
for bb in range(10):
    s = idx==bb
    if s.sum()<20: continue
    cen.append(np.median(yte[s])); res.append((yp[s]-yte[s]).std()); bias.append((yp[s]-yte[s]).mean())
ax[0,2].plot(cen, res, "o-", label="resolution (std)")
ax[0,2].plot(cen, bias, "s--", label="bias (mean)")
ax[0,2].axhline(0, color="k", lw=0.6); ax[0,2].set_xlabel(r"true $y_Z$"); ax[0,2].set_ylabel("residual"); ax[0,2].legend()
ax[0,2].set_title("Resolution & bias vs boost")

# (1,0) calibration: mean pred in bins of true
mp = []
for bb in range(10):
    s = idx==bb
    if s.sum()<20: continue
    mp.append(yp[s].mean())
ax[1,0].plot(cen, mp, "o-", label="EFN"); ax[1,0].plot([-2.5,2.5],[-2.5,2.5],"k--",label="ideal")
ax[1,0].set_xlabel(r"true $y_Z$"); ax[1,0].set_ylabel(r"⟨predicted $y_Z$⟩"); ax[1,0].legend()
ax[1,0].set_title("Calibration")

# (1,1) sign accuracy vs |y_Z|
absb = np.abs(yte); be = np.quantile(absb, np.linspace(0,1,9)); be[-1]+=1e-6
ii = np.digitize(absb, be)-1
xc, sa = [], []
for bb in range(8):
    s = ii==bb
    if s.sum()<20: continue
    xc.append(np.median(absb[s])); sa.append(np.mean(np.sign(yte[s])==np.sign(yp[s])))
ax[1,1].plot(xc, sa, "o-", color="#2ca02c"); ax[1,1].axhline(0.5, ls="--", color="gray", label="chance")
ax[1,1].set_xlabel(r"$|y_Z|$ (true)"); ax[1,1].set_ylabel("sign accuracy"); ax[1,1].set_ylim(0.4,1.0); ax[1,1].legend()
ax[1,1].set_title(rf"Boost-direction accuracy (incl. {sgn:.2f})")

# (1,2) profile: predicted vs true with error band
order = np.argsort(yte)
ax[1,2].hexbin(yte, yp, gridsize=45, cmap="Blues", mincnt=1)
ax[1,2].plot([-3,3],[-3,3],"r--",lw=1)
ax[1,2].set_xlabel(r"true $y_Z$"); ax[1,2].set_ylabel(r"EFN predicted $y_Z$")
ax[1,2].set_title("density (hexbin)")

fig.suptitle("Headline EFN (PUPPI>0.5 leading-vertex soft event) — kinematic performance on held-out test", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig("results/opendata_kinematics.png", dpi=115); print("wrote results/opendata_kinematics.png")
