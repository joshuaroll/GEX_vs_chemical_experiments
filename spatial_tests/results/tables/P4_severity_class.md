# DILIrank SeverityClass (0-8) prediction: structure vs expression vs both

All arms predict the ordinal severity CLASS. Liver only. chemberta structure; expression = engine predicted DE (near-circular caveat) or measured LINCS DE (overlap, the honest omics test). Drug-disjoint 5-fold x 3 seeds; folds stratified on coarse severity bins. mean +/- std over seeds.

## Table 1 - prediction accuracy

exact = exact 9-class accuracy; within-1 = predicted class within +/-1 of true (ordinal-tolerant).

| cohort | feature | model | n | exact accuracy | within-1 accuracy |
|---|---|---|---|---|---|
| FULL | structure | logreg | 649 | 0.310±0.012 | 0.423±0.021 |
| FULL | structure | rf | 649 | 0.387±0.010 | 0.447±0.008 |
| FULL | expression(pred DE) | logreg | 649 | 0.222±0.003 | 0.356±0.005 |
| FULL | expression(pred DE) | rf | 649 | 0.333±0.007 | 0.404±0.002 |
| FULL | both | logreg | 649 | 0.301±0.012 | 0.412±0.016 |
| FULL | both | rf | 649 | 0.371±0.013 | 0.428±0.018 |
| POS(sev>=1) | structure | logreg | 447 | 0.314±0.006 | 0.509±0.004 |
| POS(sev>=1) | structure | rf | 447 | 0.412±0.012 | 0.577±0.005 |
| POS(sev>=1) | expression(pred DE) | logreg | 447 | 0.243±0.001 | 0.447±0.012 |
| POS(sev>=1) | expression(pred DE) | rf | 447 | 0.345±0.003 | 0.517±0.009 |
| POS(sev>=1) | both | logreg | 447 | 0.305±0.014 | 0.515±0.016 |
| POS(sev>=1) | both | rf | 447 | 0.394±0.018 | 0.563±0.016 |
| FULL/measured-overlap | structure | logreg | 245 | 0.242±0.024 | 0.356±0.016 |
| FULL/measured-overlap | structure | rf | 245 | 0.288±0.008 | 0.376±0.003 |
| FULL/measured-overlap | expression(meas DE) | logreg | 245 | 0.152±0.014 | 0.282±0.012 |
| FULL/measured-overlap | expression(meas DE) | rf | 245 | 0.222±0.008 | 0.318±0.012 |
| FULL/measured-overlap | both | logreg | 245 | 0.200±0.015 | 0.298±0.015 |
| FULL/measured-overlap | both | rf | 245 | 0.261±0.022 | 0.367±0.023 |
| POS/measured-overlap | structure | logreg | 192 | 0.278±0.009 | 0.465±0.002 |
| POS/measured-overlap | structure | rf | 192 | 0.380±0.024 | 0.536±0.019 |
| POS/measured-overlap | expression(meas DE) | logreg | 192 | 0.203±0.007 | 0.370±0.035 |
| POS/measured-overlap | expression(meas DE) | rf | 192 | 0.354±0.000 | 0.512±0.009 |
| POS/measured-overlap | both | logreg | 192 | 0.269±0.016 | 0.431±0.020 |
| POS/measured-overlap | both | rf | 192 | 0.389±0.005 | 0.545±0.006 |

## Table 2 - model-selection metrics

QWK = quadratic-weighted Cohen kappa (ordinal agreement; 0=chance, 1=perfect); rho = Spearman of proba-weighted expected severity vs true; MAE = mean |pred-true| classes (lower better); balanced acc + macro-F1 handle class imbalance. Best predictive model = highest QWK / rho / within-1, lowest MAE.

| cohort | feature | model | balanced acc | macro-F1 | QWK | Spearman rho | MAE |
|---|---|---|---|---|---|---|---|
| FULL | structure | logreg | 0.184±0.007 | 0.181±0.006 | 0.208±0.004 | 0.228±0.010 | 2.630±0.052 |
| FULL | structure | rf | 0.180±0.008 | 0.172±0.010 | 0.239±0.022 | 0.331±0.008 | 2.524±0.061 |
| FULL | expression(pred DE) | logreg | 0.157±0.005 | 0.145±0.003 | 0.133±0.011 | 0.186±0.010 | 2.893±0.036 |
| FULL | expression(pred DE) | rf | 0.158±0.005 | 0.154±0.006 | 0.202±0.017 | 0.283±0.002 | 2.622±0.027 |
| FULL | both | logreg | 0.178±0.005 | 0.174±0.005 | 0.191±0.022 | 0.243±0.022 | 2.658±0.086 |
| FULL | both | rf | 0.176±0.006 | 0.171±0.006 | 0.215±0.024 | 0.340±0.006 | 2.570±0.063 |
| POS(sev>=1) | structure | logreg | 0.190±0.009 | 0.188±0.008 | 0.117±0.007 | 0.157±0.025 | 2.129±0.018 |
| POS(sev>=1) | structure | rf | 0.169±0.009 | 0.153±0.010 | 0.166±0.014 | 0.268±0.015 | 1.921±0.026 |
| POS(sev>=1) | expression(pred DE) | logreg | 0.156±0.003 | 0.149±0.002 | 0.071±0.036 | 0.116±0.008 | 2.311±0.051 |
| POS(sev>=1) | expression(pred DE) | rf | 0.151±0.001 | 0.140±0.004 | 0.048±0.029 | 0.130±0.007 | 2.166±0.060 |
| POS(sev>=1) | both | logreg | 0.174±0.009 | 0.170±0.009 | 0.156±0.024 | 0.197±0.042 | 2.084±0.070 |
| POS(sev>=1) | both | rf | 0.169±0.006 | 0.157±0.005 | 0.122±0.035 | 0.242±0.003 | 1.981±0.080 |
| FULL/measured-overlap | structure | logreg | 0.196±0.029 | 0.190±0.026 | 0.074±0.042 | 0.062±0.054 | 2.909±0.075 |
| FULL/measured-overlap | structure | rf | 0.161±0.007 | 0.146±0.009 | 0.080±0.055 | 0.147±0.028 | 2.535±0.100 |
| FULL/measured-overlap | expression(meas DE) | logreg | 0.102±0.014 | 0.100±0.011 | -0.009±0.039 | -0.014±0.047 | 3.203±0.047 |
| FULL/measured-overlap | expression(meas DE) | rf | 0.097±0.003 | 0.068±0.002 | -0.070±0.035 | -0.032±0.055 | 2.849±0.061 |
| FULL/measured-overlap | both | logreg | 0.142±0.009 | 0.142±0.011 | 0.044±0.032 | 0.032±0.027 | 3.112±0.088 |
| FULL/measured-overlap | both | rf | 0.121±0.010 | 0.092±0.010 | 0.037±0.026 | 0.078±0.033 | 2.528±0.092 |
| POS/measured-overlap | structure | logreg | 0.201±0.024 | 0.202±0.023 | 0.036±0.014 | 0.079±0.039 | 2.262±0.021 |
| POS/measured-overlap | structure | rf | 0.163±0.020 | 0.130±0.023 | 0.029±0.035 | 0.138±0.037 | 2.028±0.086 |
| POS/measured-overlap | expression(meas DE) | logreg | 0.124±0.010 | 0.125±0.009 | -0.065±0.057 | -0.039±0.092 | 2.583±0.110 |
| POS/measured-overlap | expression(meas DE) | rf | 0.135±0.001 | 0.090±0.004 | -0.035±0.022 | -0.056±0.061 | 2.158±0.034 |
| POS/measured-overlap | both | logreg | 0.165±0.018 | 0.164±0.019 | 0.008±0.062 | 0.016±0.039 | 2.391±0.109 |
| POS/measured-overlap | both | rf | 0.150±0.005 | 0.100±0.014 | 0.028±0.010 | 0.054±0.056 | 1.997±0.033 |

## Reading
- Compare the three feature arms within a cohort+model: does expression or both beat structure on QWK / rho / within-1 (the ordinal-quality metrics)?
- FULL includes SeverityClass=0, so structure's edge there partly reflects the binary is-it-toxic split; POS(sev>=1) is the cleaner 'grade how bad among toxicants' test.
- measured-overlap rows are the honest omics test (real biology, smaller n); predicted-DE rows are the deployable-from-SMILES feature (near-circular with structure).

