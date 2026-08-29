# Operational Merchant Risk Report

## Scope

This report summarizes only the threshold-crossing merchant-day alerts in `merchant_risk_output.csv`. It uses the already locked model configuration and threshold; it does not refit, tune, or select anything.

- Holdout period: 2020-05-01 through 2020-06-14 (45 calendar days)
- Locked threshold: `0.814822766216`

## Alert Summary

| Measure | Value |
| --- | --- |
| Alert rows | 912 |
| Alerts per calendar day | 20.2667 |
| Minimum risk score | 0.814868799375 |
| Maximum risk score | 0.911967639132 |
| Mean risk score | 0.826594773330 |
| Median risk score | 0.821962143038 |

## Risk-Score Distribution

| Statistic | Risk score |
| --- | --- |
| Count | 912 |
| Mean | 0.826594773330 |
| Standard deviation | 0.013499186812 |
| Minimum | 0.814868799375 |
| 25th percentile | 0.818149437202 |
| Median | 0.821962143038 |
| 75th percentile | 0.830004450522 |
| 90th percentile | 0.841630711349 |
| 95th percentile | 0.853515920824 |
| 99th percentile | 0.886393572260 |
| Maximum | 0.911967639132 |

## Highest-Risk Merchant Alerts

| merchant | prediction_date | risk_score | alert |
| --- | --- | --- | --- |
| fraud_Kuhn LLC | 2020-06-08 | 0.911967639132 | 1 |
| fraud_Kilback LLC | 2020-05-06 | 0.897958076706 | 1 |
| fraud_Kilback LLC | 2020-05-29 | 0.896111344003 | 1 |
| fraud_Schoen, Kuphal and Nitzsche | 2020-05-14 | 0.894343729192 | 1 |
| fraud_Doyle Ltd | 2020-05-14 | 0.893968554919 | 1 |
| fraud_Doyle Ltd | 2020-05-13 | 0.893231105041 | 1 |
| fraud_Kiehn Inc | 2020-05-14 | 0.893231105041 | 1 |
| fraud_Barton Inc | 2020-06-06 | 0.890273284853 | 1 |
| fraud_Kuhn LLC | 2020-06-14 | 0.888143708370 | 1 |
| fraud_Cormier LLC | 2020-06-09 | 0.886770449153 | 1 |

## Alert Concentration by Merchant

| merchant | alerts | alert_share | maximum_risk_score |
| --- | --- | --- | --- |
| fraud_Padberg-Welch | 34 | 3.73% | 0.849445183843 |
| fraud_McDermott-Weimann | 28 | 3.07% | 0.833385839101 |
| fraud_DuBuque LLC | 27 | 2.96% | 0.852870782549 |
| fraud_Kuhic Inc | 27 | 2.96% | 0.832027674209 |
| fraud_Hackett-Lueilwitz | 25 | 2.74% | 0.851836248431 |
| fraud_Goldner, Kovacek and Abbott | 24 | 2.63% | 0.853629190667 |
| fraud_Barton Inc | 23 | 2.52% | 0.890273284853 |
| fraud_Hudson-Ratke | 23 | 2.52% | 0.848382847050 |
| fraud_Huel, Hammes and Witting | 23 | 2.52% | 0.837001327552 |
| fraud_Bradtke PLC | 21 | 2.30% | 0.872315598700 |

## Alert Concentration by Date

| prediction_date | alerts | alert_share | maximum_risk_score |
| --- | --- | --- | --- |
| 2020-05-11 | 28 | 3.07% | 0.872331410150 |
| 2020-05-02 | 25 | 2.74% | 0.857875858769 |
| 2020-05-03 | 25 | 2.74% | 0.841659478174 |
| 2020-05-25 | 25 | 2.74% | 0.868245105391 |
| 2020-05-23 | 24 | 2.63% | 0.844703295479 |
| 2020-06-09 | 24 | 2.63% | 0.886770449153 |
| 2020-05-15 | 23 | 2.52% | 0.862373523313 |
| 2020-05-18 | 23 | 2.52% | 0.866954974141 |
| 2020-05-19 | 23 | 2.52% | 0.844791802597 |
| 2020-06-12 | 23 | 2.52% | 0.855593517949 |

## Locked Model Configuration

- Model: `HistGradientBoostingClassifier`
- `class_weight="balanced"`
- `random_state=42`
- `early_stopping=False`
- Historical feature schema: the 15 locked historical features used by LOOP 009
- Threshold: `0.814822766216`

## Performance Context

The threshold was derived on validation at the pre-specified 20-alert/day operating point: 1,220 alerts over 61 days, precision 0.092623, recall 0.191851, and F1 0.124931.

At the locked threshold, final holdout evaluation reported 912 alerts over 45 days (20.2667 alerts/day), precision 0.086623, recall 0.129721, F1 0.103879, ROC-AUC 0.820637, PR-AUC 0.071587, and false-positive rate 0.029132. These are evaluation context only; this report does not change the threshold or operating point.

## Limitations

- A risk score is a model ranking output, not a guarantee of fraud, loss, or causal explanation.
- The report does not establish production readiness, long-term stability, calibration quality, or future financial impact.
- The locked operating capacity is a project assumption and not evidence of actual analyst capacity or operational cost.
- Alert concentration describes this evaluated holdout period only; it does not establish persistent merchant behavior or real-world merchant identity semantics.
- No action, intervention, or automatic financial decision is specified by this report.

## Safety Notes

- The report contains no outcome field.
- `fraudTest.csv` was not read.
- No existing dataset was modified.
- No model artifact was saved.
