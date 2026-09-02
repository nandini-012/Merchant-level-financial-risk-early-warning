# OPT-005 Frozen Final Holdout Evaluation

## Frozen candidate

- Feature recipe: existing 15 features plus the eight OPT-004 fraud-history dynamics features.
- Model: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, early_stopping=False)`.
- Validation-derived frozen threshold: `0.812480834427`.
- Validation capacity provenance: 1,220 alerts over 61 days.
- Holdout evaluation period: 2020-05-01 through 2020-06-14.

The candidate feature recipe, configuration, and threshold were frozen before this evaluation. This holdout result was not used for post-hoc tuning or selection.

## Final holdout comparison

| Metric | Existing locked baseline | Frozen OPT-005 | OPT-005 minus baseline |
| --- | ---: | ---: | ---: |
| Threshold | 0.814822766216 | 0.812480834427 | — |
| Alerts | 912 | 937 | +25 |
| Alerts/day | 20.2667 | 20.8222 | +0.5555 |
| TP | 79 | 74 | -5 |
| FP | 833 | 863 | +30 |
| FN | 530 | 535 | +5 |
| TN | 27761 | 27731 | -30 |
| Precision | 0.086623 | 0.078975 | -0.007648 |
| Recall | 0.129721 | 0.121511 | -0.008210 |
| F1 | 0.103879 | 0.095731 | -0.008148 |
| ROC-AUC | 0.820637 | 0.823239 | +0.002602 |
| PR-AUC | 0.071587 | 0.068222 | -0.003365 |

## Safety confirmations

- The holdout was evaluated once using the frozen candidate and threshold.
- No holdout result was used to tune features, configuration, threshold, or capacity.
- `fraudTest.csv` was not read.
- No existing dataset, generated output, or model artifact was modified or saved.
