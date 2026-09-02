# LOOP 011 — Model Optimization Pre-Registration

## Purpose and scope

This document pre-registers a controlled optimization phase for the merchant-level temporal-risk prototype. It does not change the existing locked baseline, datasets, outputs, threshold, code, or holdout result.

The objective is to identify whether a bounded candidate can improve validation PR-AUC and precision at approximately the existing 20-alert/day operating point. The final holdout remains untouched until one candidate is frozen by the validation-only protocol below.

## Locked baseline

- Model: `HistGradientBoostingClassifier`
- `class_weight="balanced"`
- `random_state=42`
- `early_stopping=False`
- Existing baseline threshold: `0.814822766216` — locked for the existing baseline.
- Target: `future_7_fraud >= 2` during `T+1` through `T+7`.

### Existing feature block

The baseline uses exactly 15 historical features:

1. `previous_1d_transaction_count`
2. `previous_3d_transaction_count`
3. `previous_7d_transaction_count`
4. `previous_14d_transaction_count`
5. `previous_7d_fraud_count`
6. `previous_14d_fraud_count`
7. `previous_7d_fraud_rate`
8. `previous_14d_fraud_rate`
9. `preceding_7d_transaction_count`
10. `previous_7d_transaction_count_change`
11. `preceding_7d_fraud_count`
12. `previous_7d_fraud_count_change`
13. `previous_7d_total_transaction_amount`
14. `previous_7d_average_transaction_amount`
15. `previous_7d_maximum_transaction_amount`

### Locked temporal split

| Split | Inclusive dates |
| --- | --- |
| Train | 2019-01-15 to 2020-02-28 |
| Validation | 2020-03-01 to 2020-04-30 |
| Final holdout | 2020-05-01 to 2020-06-14 |

The validation operating budget is 1,220 alerts over 61 calendar days, representing the existing 20-alert/day project assumption.

### Existing locked holdout baseline

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.071587 |
| Precision | 0.086623 |
| Recall | 0.129721 |
| F1 | 0.103879 |
| ROC-AUC | 0.820637 |

## Candidate scope

Candidate experiments may change only:

- an explicitly documented historical feature block; and/or
- a bounded, explicitly documented HGB hyperparameter configuration.

Any new feature must use information strictly before prediction date `T`. Feature computation must apply explicit shifting before rolling or cumulative aggregation. The target remains unchanged.

Candidate feature blocks may include historical activity shape, historical amount shape, fraud-history dynamics, and merchant-history maturity. A feature is eligible only if its input values are available no later than `T-1` and its missing-value and zero-denominator behavior are specified before execution.

### OPT-002 pre-specified activity-shape block

OPT-002 adds the following seven features to the existing 15-feature baseline:

1. `previous_7d_active_day_count`
2. `previous_14d_active_day_count`
3. `previous_7d_maximum_daily_transaction_count`
4. `previous_7d_daily_transaction_count_std`
5. `previous_7d_daily_transaction_count_mean_zero`
6. `previous_7d_daily_transaction_count_cv`
7. `previous_7d_transactions_per_active_day`

All are computed from calendar-day transaction counts shifted by one day before their historical window aggregation. The standard deviation uses population standard deviation (`ddof=0`). For a zero prior-7-day mean, the mean-zero indicator is one and the coefficient of variation is set to zero. For zero prior-7-day active days, transactions per active day is set to zero. These values are deterministic zero-denominator handling, not business thresholds.

### OPT-003 pre-specified amount-shape block

OPT-003 adds the following seven features to the existing 15-feature baseline:

1. `previous_7d_daily_total_transaction_amount_std`
2. `previous_7d_daily_total_transaction_amount_median`
3. `previous_7d_maximum_daily_total_transaction_amount`
4. `previous_7d_daily_total_transaction_amount_mean_zero`
5. `previous_7d_daily_total_transaction_amount_cv`
6. `preceding_7d_total_transaction_amount`
7. `previous_7d_total_transaction_amount_change`

All are calculated over merchant calendar days using daily total transaction amount and an explicit one-day shift before every rolling operation. The standard deviation uses population standard deviation (`ddof=0`). Calendar days with no transactions have daily amount zero. For a zero prior-7-day daily amount mean, the mean-zero indicator is one and the coefficient of variation is set to zero. The preceding amount window is `T-14` through `T-8`; the amount change is prior 7-day total minus preceding 7-day total. These are deterministic calculation rules, not business thresholds.

### OPT-004 pre-specified fraud-history dynamics block

OPT-004 adds the following eight features to the existing 15-feature baseline:

1. `previous_1d_fraud_count`
2. `previous_3d_fraud_count`
3. `previous_3d_fraud_rate`
4. `preceding_7d_fraud_rate`
5. `previous_7d_fraud_rate_change`
6. `previous_7d_fraud_active_day_count`
7. `days_since_prior_fraud_transaction`
8. `no_prior_fraud_transaction_indicator`

All count and rate windows use merchant calendar-day transaction and fraud counts shifted by one day before aggregation. The preceding 7-day window is `T-14` through `T-8`; fraud-rate change is prior 7-day fraud rate minus preceding 7-day fraud rate. A prior 3-day or preceding 7-day rate with zero transaction denominator is set to zero. Fraud-active-day count uses calendar days with at least one fraud in `T-7` through `T-1`. Days since prior fraud uses the most recent fraud date strictly before `T`; where no prior fraud exists, the days value is set to zero and the indicator is one. These are deterministic missing/zero-denominator rules, not business thresholds.

### OPT-005 pre-specified replication block

OPT-005 uses the existing 15-feature baseline plus exactly the eight OPT-004 fraud-history dynamics features above. It uses the unchanged locked HGB configuration with no hyperparameter variation. This is a validation-only reproducibility control for the requested combined feature matrix, not a new feature or configuration search.

### OPT-005 frozen selection

Before final-holdout evaluation, OPT-005 is frozen as the selected candidate. Its fixed recipe is the existing 15-feature baseline plus the eight OPT-004 fraud-history dynamics features; its model is `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, early_stopping=False)`; and its frozen validation-derived threshold is `0.812480834427` at the 1,220-alert validation budget. No feature, parameter, threshold, or selection change may be made after final-holdout evaluation.

Bounded HGB configurations may vary only pre-declared values of `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, and `l2_regularization`. Each candidate must remain deterministic and record its complete configuration.

## Prohibited work

- No random splitting.
- No current-day or future-derived feature, including `T+1` through `T+7` information.
- No target leakage.
- No SMOTE, oversampling, undersampling, or other resampling.
- No customer identity fields, including card number, name, DOB, gender, street, raw ZIP, transaction identifier, or customer identifiers.
- No broad or unbounded model search, package installation, or unrecorded configuration changes.
- No modification of the existing baseline threshold, model configuration, datasets, generated outputs, or locked holdout result.
- No holdout inspection during feature or model selection.

## Validation-only selection protocol

1. Define every candidate and all permitted values in the table below before running it.
2. Build candidate features with the existing temporal safeguards and validate no missing modelling values or future-derived columns.
3. Fit candidates only on the locked train period.
4. Score candidates only on the locked validation period during selection.
5. Derive each candidate’s operating threshold only from validation probabilities using stable descending ranking at the 1,220-alert budget. Record the actual alert count if tied probabilities prevent an exact count.
6. Do not use the existing baseline threshold to select a modified candidate; preserve it as the historical locked-baseline reference.
7. Freeze one candidate’s feature recipe, complete HGB configuration, and validation-derived threshold before any holdout scoring.

## Deterministic selection criteria

The selection rule is fixed before experiments:

1. A candidate is eligible only if it has no temporal-leakage, schema, missing-value, or split-integrity failure.
2. It must improve validation PR-AUC and validation precision at the approximately 20-alert/day operating point relative to the existing locked baseline’s corresponding validation results.
3. If multiple candidates satisfy criterion 2, select higher validation precision at the capacity point.
4. Break a precision tie by higher validation PR-AUC, then higher validation recall, then higher validation F1.
5. If still tied, select the candidate with fewer added features; if still tied, select the candidate with the lexicographically earlier candidate ID.
6. If no candidate satisfies criterion 2, no candidate is selected and the existing locked baseline remains the reference.

## Holdout protocol

After—and only after—selection freezes a candidate, fit it once on the locked train period, derive and lock its threshold from validation only, then evaluate it once on the final holdout period. Holdout results must not change the feature block, HGB configuration, threshold, capacity, or selection decision.

Holdout comparisons will include PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, alert count, alerts/day, false-positive rate, false alerts/day, and captured positive targets against the existing locked baseline.

## Final LOOP 011 outcome

OPT-005 was frozen before a single final-holdout evaluation. Its validation-derived threshold was `0.812480834427`; its feature recipe was the existing 15-feature baseline plus the eight pre-specified OPT-004 fraud-history dynamics features; and its HGB configuration was unchanged.

On the 2020-05-01 through 2020-06-14 holdout, OPT-005 had lower PR-AUC (`0.068222` vs `0.071587`), precision (`0.078975` vs `0.086623`), recall (`0.121511` vs `0.129721`), and F1 (`0.095731` vs `0.103879`) than the original locked baseline. It also generated 937 alerts (20.8222/day), compared with the baseline's 912 alerts (20.2667/day).

Therefore, the original locked 15-feature baseline is formally retained. Its configuration and threshold remain unchanged: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, early_stopping=False)` and threshold `0.814822766216`. The OPT-005 holdout result does not alter any existing dataset, generated output, model artifact, or baseline evaluation.

The candidate was not modified after the holdout result, and no additional threshold or model selection was performed. The complete comparison is recorded in [LOOP_011_OPT_005_HOLDOUT_EVALUATION.md](LOOP_011_OPT_005_HOLDOUT_EVALUATION.md).

## Pre-registered experiment table

Validation result cells remain unpopulated until their corresponding candidate is executed.

| Candidate ID | Feature block | HGB configuration | Validation PR-AUC | Precision at 20 alerts/day | Recall | F1 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-001 | Existing 15-feature baseline | Existing locked configuration | 0.068597 | 0.092623 | 0.191851 | 0.124931 | Validation reference reproduced during OPT-003 |
| OPT-002 | Seven pre-specified historical activity-shape features | Existing locked configuration | Not run — runtime unavailable | Not run — runtime unavailable | Not run — runtime unavailable | Not run — runtime unavailable | Blocked: compatible pandas/scikit-learn runtime unavailable |
| OPT-003 | Seven pre-specified historical amount-shape features | Existing locked configuration | 0.060685 | 0.074590 | 0.154499 | 0.100608 | Not selected: lower validation PR-AUC and precision than OPT-001 |
| OPT-004 | Eight pre-specified fraud-history dynamics features | Existing locked configuration | 0.069232 | 0.094262 | 0.195246 | 0.127142 | Meets both validation metrics vs OPT-001; not selected in this loop |
| OPT-005 | Existing 15 features plus the eight OPT-004 fraud-history dynamics features | Existing locked configuration; no hyperparameter variation | 0.069232 | 0.094262 | 0.195246 | 0.127142 | Final holdout completed once; original locked baseline retained because OPT-005 degraded holdout PR-AUC, precision, recall, and F1 |
