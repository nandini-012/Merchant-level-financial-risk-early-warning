# Experiments and Evaluation Record

This register summarizes work evidenced by repository scripts, generated outputs, and Git history. It does not add experiments beyond that evidence.

## Locked experiment context

- Target: `future_7_fraud >= 2` for fraudulent transactions during `T+1` through `T+7`.
- Features: the 15 fixed historical merchant-day features listed in `docs/README.md`.
- Model: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, early_stopping=False)`.
- Temporal split: train 2019-01-15–2020-02-28; validation 2020-03-01–2020-04-30; final holdout 2020-05-01–2020-06-14.
- Locked threshold: `0.814822766216`, derived only from the validation 20-alert/day operating point.

## Records

### EXP-001 — Merchant-day data, target, and historical-feature construction

- Evidence: `tools/build_merchant_features.py`, `tools/build_training_dataset.py`, and commit `800d017`.
- Work: constructed merchant-day features from `fraudTrain.csv` and a future seven-calendar-day target.
- Temporal control: historical windows end at `T-1`; the target excludes `T` and uses `T+1` through `T+7`.
- Result: the training dataset persists eligible observations, 15 historical feature columns, and `target`.

### EXP-002 — Temporal baseline and nonlinear model comparison

- Evidence: `tools/train_baseline_models.py`, `tools/train_hist_gradient_boosting.py`, and commit `800d017`.
- Work: evaluated majority-class and Logistic Regression baselines, then the fixed HistGradientBoostingClassifier using the locked temporal split.
- Constraint: the fixed HGB methodology records no random split, resampling, feature selection, or hyperparameter tuning.
- Result: the HGB configuration became the locked methodology for later threshold and holdout scripts.

### EXP-003 — Validation operating-point and cost-sensitivity analysis

- Evidence: `tools/analyze_operational_thresholds.py`, `tools/analyze_operational_cost_sensitivity.py`, and commit `800d017`.
- Work: ranked validation probabilities at pre-specified alert capacities and documented scenario sensitivity without treating costs as real business values.
- Result: the 20-alert/day project assumption gives 1,220 validation alerts over 61 days and threshold `0.814822766216`.

### EXP-004 — Locked final holdout evaluation

- Evidence: `tools/evaluate_locked_holdout.py`, commit `4e7bbf7`, and `docs/PROJECT_CONTEXT.md`.
- Method: threshold derivation is validation-only; holdout scoring is a subsequent separate stage. The script checks temporal ordering, key separation, target exclusion, future-feature exclusion, and missing values.
- Result: 912 holdout alerts (20.2667/day), ROC-AUC 0.820637, PR-AUC 0.071587, precision 0.086623, recall 0.129721, and F1 0.103879.

### EXP-005 — Post-evaluation diagnostics

- Evidence: commits `807d1f9`, `905fef4`, `0f43cfa`, `7343e8d`, `1c81a84`, and `a4d7a86`, with their corresponding `tools/analyze_*.py` scripts.
- Work: added diagnostics for pre-specified holdout operating points and temporal, probability, feature, and model-risk summaries.
- Constraint: these analyses do not change the locked target, threshold, temporal split, or HGB configuration.

### EXP-006 — Alert-output and read-only review artifacts

- Evidence: commits `17115e6`, `663a230`, `1dc018f`, `d9ae639`, and `5ea7841`.
- Work: generated locked holdout alerts, deterministic evidence-only explanations, a Markdown operational report, and a read-only local Command Center.
- Result: these artifacts present existing predictions and historical values; they do not retrain a model or select a threshold.

## Interpretation limits

- The final holdout was not used to choose or revise the threshold.
- The 20-alert/day capacity is a project assumption, not an actual operational cost or capacity claim.
- Evaluation metrics do not establish production readiness, causal feature effects, real-world merchant identity semantics, or Razorpay system performance.
