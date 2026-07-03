# Results (modelcompare)

- train events: 24500, test events: 5250
- targets: y_Z, pz_Z, beta_z

## Primary target: y_Z

| model | R2 | corr | RMSE | resolution | sign acc |
|---|---|---|---|---|---|
| mean | -0.000 | 0.000 | 1.095 | 1.094 | 0.497 |
| linear | 0.019 | 0.139 | 1.084 | 1.084 | 0.550 |
| gbdt_summary | 0.013 | 0.129 | 1.087 | 1.087 | 0.547 |
| gbdt_rich | 0.042 | 0.220 | 1.071 | 1.071 | 0.578 |
| efn | 0.058 | 0.257 | 1.062 | 1.062 | 0.586 |
| transformer | 0.049 | 0.243 | 1.067 | 1.067 | 0.583 |

## pz_Z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.001 | 0.000 | 164.981 |
| linear | 0.017 | 0.136 | 163.488 |
| gbdt_summary | 0.006 | 0.116 | 164.381 |
| gbdt_rich | 0.039 | 0.219 | 161.631 |
| efn | 0.076 | 0.277 | 158.546 |
| transformer | 0.063 | 0.255 | 159.638 |

## beta_z
| model | R2 | corr | RMSE |
|---|---|---|---|
| mean | -0.000 | 0.000 | 0.690 |
| linear | 0.018 | 0.138 | 0.683 |
| gbdt_summary | 0.014 | 0.132 | 0.685 |
| gbdt_rich | 0.041 | 0.216 | 0.676 |
| efn | 0.046 | 0.250 | 0.674 |
| transformer | 0.037 | 0.237 | 0.677 |

![pred vs true](pred_vs_true_modelcompare.png)
![resolution](resolution_modelcompare.png)