# Merchant Risk Early-Warning & Loss-Exposure Detector

This repository contains a defensive merchant-level temporal-risk prototype. It produces merchant-day alerts from historical merchant behaviour. It does not claim access to, replication of, or deployment within proprietary Razorpay systems or data.

## Current state

- Target: `future_7_fraud >= 2`.
- Feature set: 15 fixed historical merchant-day features.
- Locked model: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, early_stopping=False)`.
- Validation-derived locked threshold: `0.814822766216`.
- Operational modelling assumption: 20 alerts/day.
- Final holdout: 2020-05-01 through 2020-06-14.
- Latest commit at this update: `5ea7841`.

The final holdout was evaluated only after the threshold was derived from validation and frozen.

## Pipeline

1. `tools/build_merchant_features.py` reads `data/raw/fraudTrain.csv` and builds `data/processed/merchant_features_train.parquet`.
2. `tools/build_training_dataset.py` creates `data/processed/training_dataset.parquet` from eligible merchant-day observations.
3. At date `T`, historical features use information strictly before `T`; the target is one when there are at least two fraudulent transactions during `T+1` through `T+7`.
4. The locked model and threshold are evaluated chronologically and used to generate holdout alert outputs.

An eligible saved observation has merchant activity on `T`, complete mandatory historical features, and a complete future seven-day target window. Current-day transactions do not contribute to the target. Merchant, prediction date, target, and future-derived values are not model features.

## Fixed historical feature schema

- `previous_1d_transaction_count`
- `previous_3d_transaction_count`
- `previous_7d_transaction_count`
- `previous_14d_transaction_count`
- `previous_7d_fraud_count`
- `previous_14d_fraud_count`
- `previous_7d_fraud_rate`
- `previous_14d_fraud_rate`
- `preceding_7d_transaction_count`
- `previous_7d_transaction_count_change`
- `preceding_7d_fraud_count`
- `previous_7d_fraud_count_change`
- `previous_7d_total_transaction_amount`
- `previous_7d_average_transaction_amount`
- `previous_7d_maximum_transaction_amount`

## Evaluation methodology and locked result

| Split | Dates |
| --- | --- |
| Train | 2019-01-15 through 2020-02-28 |
| Validation | 2020-03-01 through 2020-04-30 |
| Final holdout | 2020-05-01 through 2020-06-14 |

The 20-alert/day validation capacity corresponds to 1,220 validation alerts over 61 days. Stable descending validation-probability ranking produced threshold `0.814822766216`; it was then frozen before holdout evaluation.

| Final-holdout metric | Value |
| --- | ---: |
| Alerts | 912 |
| Alerts/day | 20.2667 |
| ROC-AUC | 0.820637 |
| PR-AUC | 0.071587 |
| Precision | 0.086623 |
| Recall | 0.129721 |
| F1 | 0.103879 |

PR-AUC is important because the positive class is rare. These results are limited to this repository’s fixed dataset and temporal split; they are not production or financial-impact evidence.

## Outputs

- `data/outputs/merchant_risk_output.csv` — 912 threshold-crossing holdout merchant-days, sorted by risk score descending, prediction date ascending, then merchant ascending. Columns: `merchant`, `prediction_date`, `risk_score`, `alert`.
- `data/outputs/merchant_risk_explanations.csv` — matching alert rows with six existing historical values and deterministic evidence-only text; it contains no target or outcome field.
- `data/outputs/operational_risk_report.md` — descriptive alert-volume, risk-score, concentration, configuration, performance-context, and limitation summary.

## Read-only Merchant Risk Command Center

`tools/merchant_risk_command_center.py` reads only those three generated outputs. It validates schemas and cross-file keys, preserves the existing score-descending alert queue, and does not retrain, score, write, or modify pipeline data.

```bash
python tools/merchant_risk_command_center.py --check
python tools/merchant_risk_command_center.py
```

The local view presents four KPIs and selected-alert details, including the locked threshold, score margin above that threshold, and existing historical evidence.

## Reproducibility

The implementation uses fixed dates, feature columns, model settings, and `random_state=42`. Validation checks cover temporal ordering, duplicate merchant/date keys, missing values, target/future-feature exclusion, threshold provenance, and output contracts.

```bash
python tools/build_merchant_features.py
python tools/build_training_dataset.py
python tools/evaluate_locked_holdout.py
python tools/generate_merchant_risk_output.py
python tools/generate_merchant_risk_explanations.py
python tools/generate_operational_risk_report.py
```

`fraudTest.csv` is not used by the locked feature, target, output, explanation, or Command Center workflows.

## Limitations

- A risk score is a ranking output, not proof of fraud, loss, or causality.
- The 20-alert/day capacity is a project assumption, not a claim about real analyst capacity or cost.
- The work does not establish production readiness, calibration, long-term stability, future business impact, or persistent real-world merchant identity semantics.
- The system specifies no automatic intervention or consequential financial action.
