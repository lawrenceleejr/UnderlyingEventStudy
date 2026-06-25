# Results (smoke)

- train events: 518, test events: 112
- targets: y_Z, pz_Z, beta_z

## Primary target: y_Z

| model | R2 | corr | RMSE | resolution | sign acc |
|---|---|---|---|---|---|
| mean | -0.003 | nan | 1.164 | 1.162 | 0.518 |
| linear | 0.020 | 0.172 | 1.150 | 1.146 | 0.634 |
| gbdt_summary | -0.253 | 0.020 | 1.300 | 1.300 | 0.411 |
| efn | -0.004 | 0.020 | 1.164 | 1.163 | 0.500 |

## pz_Z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.007 | nan | 182.918 |
| linear | 0.013 | 0.166 | 181.153 |
| gbdt_summary | -0.234 | 0.006 | 202.537 |
| efn | -0.007 | 0.034 | 182.968 |

## beta_z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.001 | nan | 0.705 |
| linear | 0.022 | 0.169 | 0.697 |
| gbdt_summary | -0.235 | 0.060 | 0.783 |
| efn | -0.009 | 0.032 | 0.708 |

![pred vs true](pred_vs_true_smoke.png)
![resolution](resolution_smoke.png)