# Results (nompi)

- train events: 31747, test events: 6803
- targets: y_Z, pz_Z, beta_z

## Primary target: y_Z

| model | R2 | corr | RMSE | resolution | sign acc |
|---|---|---|---|---|---|
| mean | -0.000 | 0.000 | 1.089 | 1.089 | 0.511 |
| linear | 0.056 | 0.237 | 1.058 | 1.058 | 0.580 |
| gbdt_summary | 0.070 | 0.265 | 1.050 | 1.050 | 0.583 |
| gbdt_rich | 0.098 | 0.314 | 1.034 | 1.034 | 0.606 |
| efn | 0.115 | 0.344 | 1.024 | 1.023 | 0.617 |

## pz_Z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | 0.000 | 162.055 |
| linear | 0.053 | 0.231 | 157.676 |
| gbdt_summary | 0.074 | 0.273 | 155.908 |
| gbdt_rich | 0.097 | 0.315 | 153.927 |
| efn | 0.120 | 0.350 | 151.952 |

## beta_z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.001 | 0.000 | 0.686 |
| linear | 0.053 | 0.230 | 0.667 |
| gbdt_summary | 0.068 | 0.262 | 0.662 |
| gbdt_rich | 0.095 | 0.309 | 0.652 |
| efn | 0.106 | 0.339 | 0.648 |

![pred vs true](pred_vs_true_nompi.png)
![resolution](resolution_nompi.png)