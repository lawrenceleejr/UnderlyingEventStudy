# Results (opendata)

- train events: 44178, test events: 9467
- targets: y_Z, pz_Z, beta_z

## Primary target: y_Z

| model | R2 | corr | RMSE | resolution | sign acc |
|---|---|---|---|---|---|
| mean | -0.000 | nan | 1.068 | 1.068 | 0.500 |
| linear | 0.042 | 0.204 | 1.045 | 1.045 | 0.582 |
| gbdt_summary | 0.042 | 0.206 | 1.045 | 1.045 | 0.577 |
| efn | -0.000 | 0.011 | 1.068 | 1.068 | 0.500 |

## pz_Z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | nan | 155.575 |
| linear | 0.033 | 0.181 | 152.995 |
| gbdt_summary | 0.034 | 0.184 | 152.945 |
| efn | 0.000 | 0.019 | 155.575 |

## beta_z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | nan | 0.680 |
| linear | 0.046 | 0.215 | 0.665 |
| gbdt_summary | 0.048 | 0.220 | 0.664 |
| efn | -0.000 | -0.023 | 0.680 |

![pred vs true](pred_vs_true_opendata.png)
![resolution](resolution_opendata.png)