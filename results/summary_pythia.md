# Results (pythia)

- train events: 53359, test events: 11435
- targets: y_Z, pz_Z, beta_z

## Primary target: y_Z

| model | R2 | corr | RMSE | resolution | sign acc |
|---|---|---|---|---|---|
| mean | -0.000 | 0.000 | 1.098 | 1.098 | 0.500 |
| linear | 0.025 | 0.159 | 1.085 | 1.085 | 0.552 |
| gbdt_summary | 0.026 | 0.161 | 1.084 | 1.084 | 0.550 |
| gbdt_rich | 0.056 | 0.241 | 1.067 | 1.067 | 0.580 |
| efn | 0.068 | 0.269 | 1.061 | 1.061 | 0.586 |

## pz_Z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | 0.000 | 164.413 |
| linear | 0.026 | 0.162 | 162.255 |
| gbdt_summary | 0.025 | 0.158 | 162.371 |
| gbdt_rich | 0.061 | 0.250 | 159.317 |
| efn | 0.082 | 0.288 | 157.487 |

## beta_z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | 0.000 | 0.690 |
| linear | 0.023 | 0.153 | 0.682 |
| gbdt_summary | 0.023 | 0.152 | 0.682 |
| gbdt_rich | 0.054 | 0.236 | 0.671 |
| efn | 0.056 | 0.262 | 0.671 |

![pred vs true](pred_vs_true_pythia.png)
![resolution](resolution_pythia.png)