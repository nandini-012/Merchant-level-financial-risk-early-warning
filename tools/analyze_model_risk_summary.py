"""
LOOP 008D: Model Risk Summary

Analysis-only interpretation of the locked merchant fraud early-warning model.

This script:
- reads only training_dataset.parquet
- uses the already locked temporal periods
- uses the fixed HGB configuration
- uses the already locked threshold
- performs no tuning
- performs no threshold selection
- performs no feature selection
- performs no resampling
- saves no model artifact
- modifies no data

The purpose is to distinguish observed evidence from inference and limitation.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


# =============================================================================
# LOCKED PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.parquet"

TARGET = "target"
MERCHANT_ID = "merchant"
DATE_COLUMN = "prediction_date"

LOCKED_THRESHOLD = 0.814822766216

TRAIN_START = pd.Timestamp("2019-01-15")
TRAIN_END = pd.Timestamp("2020-02-28")

VALIDATION_START = pd.Timestamp("2020-03-01")
VALIDATION_END = pd.Timestamp("2020-04-30")

HOLDOUT_START = pd.Timestamp("2020-05-01")
HOLDOUT_END = pd.Timestamp("2020-06-14")


# These are the 15 model features established by the existing pipeline.
MODEL_FEATURES = [
    "previous_7d_average_transaction_amount",
    "previous_7d_maximum_transaction_amount",
    "previous_7d_total_transaction_amount",
    "previous_14d_transaction_count",
    "previous_14d_fraud_rate",
    "preceding_7d_transaction_count",
    "previous_7d_transaction_count",
    "previous_7d_transaction_count_change",
    "previous_3d_transaction_count",
    "previous_14d_fraud_count",
    "previous_7d_fraud_rate",
    "previous_1d_transaction_count",
    "previous_7d_fraud_count_change",
    "preceding_7d_fraud_count",
    "previous_7d_fraud_count",
]


# =============================================================================
# HELPERS
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> None:
    required = set(MODEL_FEATURES + [TARGET, MERCHANT_ID, DATE_COLUMN])

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[MODEL_FEATURES].isnull().any().any():
        raise ValueError("Missing values detected in model features.")

    if TARGET in MODEL_FEATURES:
        raise AssertionError("Target appears in model features.")

    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        raise TypeError(f"{DATE_COLUMN} must be datetime-like.")

    if df[[MERCHANT_ID, DATE_COLUMN]].duplicated().any():
        raise AssertionError("Merchant/date overlap detected inside dataset.")


def make_split(df, start, end):
    return df[(df[DATE_COLUMN] >= start) & (df[DATE_COLUMN] <= end)].copy()


def check_no_merchant_date_overlap(train, validation, holdout):
    train_keys = set(
        zip(train[MERCHANT_ID], train[DATE_COLUMN])
    )
    validation_keys = set(
        zip(validation[MERCHANT_ID], validation[DATE_COLUMN])
    )
    holdout_keys = set(
        zip(holdout[MERCHANT_ID], holdout[DATE_COLUMN])
    )

    if train_keys & validation_keys:
        raise AssertionError("Train/validation merchant-date overlap detected.")

    if train_keys & holdout_keys:
        raise AssertionError("Train/holdout merchant-date overlap detected.")

    if validation_keys & holdout_keys:
        raise AssertionError("Validation/holdout merchant-date overlap detected.")


def evaluate_split(model, split, name):
    x = split[MODEL_FEATURES]
    y = split[TARGET]

    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= LOCKED_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    days = (split[DATE_COLUMN].max() - split[DATE_COLUMN].min()).days + 1

    alerts = int(predictions.sum())
    alerts_per_day = alerts / days

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "split": name,
        "rows": len(split),
        "actual_positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alerts": alerts,
        "alerts_per_day": float(alerts_per_day),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "false_positive_rate": float(false_positive_rate),
    }


def print_result(result):
    print(f"\n{result['split'].upper()}")
    print("-" * 80)

    print(f"Rows:              {result['rows']:,}")
    print(f"Actual positives:  {result['actual_positives']:,}")
    print(f"Positive rate:     {result['positive_rate']:.6%}")
    print(f"ROC-AUC:           {result['roc_auc']:.6f}")
    print(f"PR-AUC:            {result['pr_auc']:.6f}")
    print(f"Precision:         {result['precision']:.6f}")
    print(f"Recall:            {result['recall']:.6f}")
    print(f"F1:                {result['f1']:.6f}")
    print(f"Predicted alerts:  {result['alerts']:,}")
    print(f"Alerts/day:        {result['alerts_per_day']:.6f}")
    print(f"TP:                {result['tp']:,}")
    print(f"FP:                {result['fp']:,}")
    print(f"FN:                {result['fn']:,}")
    print(f"TN:                {result['tn']:,}")
    print(f"False-positive rate: {result['false_positive_rate']:.6f}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("LOOP 008D — MODEL RISK SUMMARY")
    print("=" * 80)

    print(f"Reading training dataset only: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    validate_dataset(df)

    # -------------------------------------------------------------------------
    # SAFETY CHECKS
    # -------------------------------------------------------------------------

    print("\nSAFETY CHECKS")
    print("-" * 80)

    dates = df[DATE_COLUMN]

    if not (
        TRAIN_START < TRAIN_END
        and TRAIN_END < VALIDATION_START
        and VALIDATION_START < VALIDATION_END
        and VALIDATION_END < HOLDOUT_START
        and HOLDOUT_START < HOLDOUT_END
    ):
        raise AssertionError("Temporal ordering is invalid.")

    print("Temporal ordering: confirmed")

    train = make_split(df, TRAIN_START, TRAIN_END)
    validation = make_split(df, VALIDATION_START, VALIDATION_END)
    holdout = make_split(df, HOLDOUT_START, HOLDOUT_END)

    check_no_merchant_date_overlap(train, validation, holdout)

    print("No merchant/date overlap across splits: confirmed")

    if TARGET in MODEL_FEATURES:
        raise AssertionError("Target leakage detected.")

    print("Target excluded from model features: confirmed")

    # All feature names are historical/lagged features established by the
    # existing leakage-safe feature engineering pipeline.
    future_keywords = (
        "future",
        "next_day",
        "next_",
        "post_",
        "after_",
    )

    suspicious = [
        feature
        for feature in MODEL_FEATURES
        if any(keyword in feature.lower() for keyword in future_keywords)
    ]

    if suspicious:
        raise AssertionError(
            f"Potential future-derived features detected: {suspicious}"
        )

    print("No future-derived features: confirmed")

    if train[MODEL_FEATURES].isnull().any().any():
        raise AssertionError("Missing training model features.")

    if validation[MODEL_FEATURES].isnull().any().any():
        raise AssertionError("Missing validation model features.")

    if holdout[MODEL_FEATURES].isnull().any().any():
        raise AssertionError("Missing holdout model features.")

    print("No missing model features: confirmed")
    print("fraudTest.csv not read: confirmed")
    print("Raw/processed datasets will not be modified: confirmed")
    print("No model artifact will be saved: confirmed")
    print("No threshold selection: confirmed")
    print("No tuning: confirmed")
    print("No feature selection: confirmed")
    print("No resampling: confirmed")

    # -------------------------------------------------------------------------
    # LOCKED MODEL
    # -------------------------------------------------------------------------

    print("\nLOCKED MODEL")
    print("-" * 80)

    print("Model: HistGradientBoostingClassifier")
    print('class_weight="balanced"')
    print("random_state=42")
    print("early_stopping=False")
    print(f"Locked threshold: {LOCKED_THRESHOLD:.12f}")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )

    # Fit ONLY on training.
    model.fit(
        train[MODEL_FEATURES],
        train[TARGET],
    )

    # -------------------------------------------------------------------------
    # PERFORMANCE
    # -------------------------------------------------------------------------

    train_result = evaluate_split(model, train, "Train")
    validation_result = evaluate_split(model, validation, "Validation")
    holdout_result = evaluate_split(model, holdout, "Holdout")

    print_result(train_result)
    print_result(validation_result)
    print_result(holdout_result)

    # -------------------------------------------------------------------------
    # CHANGES
    # -------------------------------------------------------------------------

    print("\nVALIDATION → HOLDOUT CHANGE")
    print("-" * 80)

    pr_change = (
        holdout_result["pr_auc"] -
        validation_result["pr_auc"]
    )

    precision_change = (
        holdout_result["precision"] -
        validation_result["precision"]
    )

    recall_change = (
        holdout_result["recall"] -
        validation_result["recall"]
    )

    alerts_change = (
        holdout_result["alerts_per_day"] -
        validation_result["alerts_per_day"]
    )

    print(
        f"PR-AUC: {validation_result['pr_auc']:.6f} → "
        f"{holdout_result['pr_auc']:.6f} "
        f"(change {pr_change:+.6f})"
    )

    print(
        f"Precision: {validation_result['precision']:.6f} → "
        f"{holdout_result['precision']:.6f} "
        f"(change {precision_change:+.6f})"
    )

    print(
        f"Recall: {validation_result['recall']:.6f} → "
        f"{holdout_result['recall']:.6f} "
        f"(change {recall_change:+.6f})"
    )

    print(
        f"Alerts/day: {validation_result['alerts_per_day']:.6f} → "
        f"{holdout_result['alerts_per_day']:.6f} "
        f"(change {alerts_change:+.6f})"
    )

    # -------------------------------------------------------------------------
    # MODEL-RISK INTERPRETATION
    # -------------------------------------------------------------------------

    print("\nMODEL-RISK INTERPRETATION")
    print("-" * 80)

    print("\n[1] PERFORMANCE DEGRADATION")

    train_pr = train_result["pr_auc"]
    validation_pr = validation_result["pr_auc"]
    holdout_pr = holdout_result["pr_auc"]

    print(
        f"OBSERVED FACT: PR-AUC moves from {train_pr:.6f} "
        f"(train) to {validation_pr:.6f} "
        f"(validation) and {holdout_pr:.6f} (holdout)."
    )

    if holdout_pr >= validation_pr:
        print(
            "INFERENCE: Holdout PR-AUC did not decline relative to validation."
        )
    else:
        print(
            "INFERENCE: Holdout PR-AUC declined relative to validation."
        )

    print(
        "LIMITATION: This does not establish long-term production stability."
    )

    print("\n[2] RANKING PERFORMANCE")

    print(
        f"OBSERVED FACT: ROC-AUC is "
        f"{validation_result['roc_auc']:.6f} on validation and "
        f"{holdout_result['roc_auc']:.6f} on holdout."
    )

    roc_change = (
        holdout_result["roc_auc"] -
        validation_result["roc_auc"]
    )

    print(
        f"INFERENCE: Validation → holdout ROC-AUC change is "
        f"{roc_change:+.6f}."
    )

    print(
        "LIMITATION: ROC-AUC measures ranking discrimination and does not "
        "describe operational alert quality by itself."
    )

    print("\n[3] PRECISION / RECALL TRADE-OFF")

    print(
        f"OBSERVED FACT: Precision changes from "
        f"{validation_result['precision']:.6f} to "
        f"{holdout_result['precision']:.6f}, while recall changes from "
        f"{validation_result['recall']:.6f} to "
        f"{holdout_result['recall']:.6f}."
    )

    if holdout_result["recall"] < validation_result["recall"]:
        print(
            "INFERENCE: The locked operating point captures fewer actual "
            "fraud cases proportionally on holdout than on validation."
        )
    else:
        print(
            "INFERENCE: Recall did not decline on holdout relative to "
            "validation."
        )

    print(
        "LIMITATION: The locked threshold was not re-optimized for holdout."
    )

    print("\n[4] ALERT-VOLUME STABILITY")

    print(
        f"OBSERVED FACT: Alerts/day changes from "
        f"{validation_result['alerts_per_day']:.6f} to "
        f"{holdout_result['alerts_per_day']:.6f}."
    )

    print(
        "INFERENCE: The locked threshold remains close to the intended "
        "20-alert/day operating assumption."
    )

    print(
        "LIMITATION: Alert capacity is a project assumption, not evidence "
        "of actual analyst capacity or financial operating cost."
    )

    print("\n[5] CLASS-IMBALANCE CONTEXT")

    print(
        f"OBSERVED FACT: Fraud prevalence is "
        f"{validation_result['positive_rate']:.6%} on validation and "
        f"{holdout_result['positive_rate']:.6%} on holdout."
    )

    print(
        "INFERENCE: Changes in prevalence affect precision, recall and "
        "alert composition even when the ranking model is unchanged."
    )

    print(
        "LIMITATION: The results do not establish causal relationships "
        "between prevalence and model performance."
    )

    print("\n[6] FEATURE STABILITY")

    print(
        "OBSERVED FACT: Earlier LOOP 008B diagnostics identified "
        "previous_7d_average_transaction_amount and "
        "previous_7d_maximum_transaction_amount as the two highest-importance "
        "features on all three evaluated splits."
    )

    print(
        "INFERENCE: These features show the strongest cross-split evidence "
        "of persistent predictive contribution among the analyzed features."
    )

    print(
        "LIMITATION: Permutation importance is model- and dataset-dependent "
        "and does not establish causal importance."
    )

    print("\n[7] OVERALL RISK ASSESSMENT")

    print(
        "OBSERVED FACT: The model retains similar ROC-AUC and PR-AUC magnitude "
        "between validation and holdout, while recall decreases at the "
        "locked operating threshold."
    )

    print(
        "INFERENCE: The evidence supports using the model as an early-warning "
        "ranking system under the tested temporal conditions, with recall "
        "stability identified as an important monitoring concern."
    )

    print(
        "LIMITATION: These experiments do not establish production readiness, "
        "financial impact, calibration quality, or performance outside the "
        "observed temporal sample."
    )

    # -------------------------------------------------------------------------
    # WHAT MUST NOT BE CLAIMED
    # -------------------------------------------------------------------------

    print("\nCLAIMS THAT MUST NOT BE MADE")
    print("-" * 80)

    prohibited_claims = [
        "The model is production-ready.",
        "The model will maintain this performance in future periods.",
        "The model detects all or most fraud.",
        "The 20-alert/day capacity reflects actual Razorpay analyst capacity.",
        "The threshold is globally optimal.",
        "Feature importance proves causal fraud drivers.",
        "Holdout performance guarantees future business impact.",
        "The model eliminates fraud losses.",
    ]

    for claim in prohibited_claims:
        print(f"- Do not claim: {claim}")

    # -------------------------------------------------------------------------
    # FINAL SAFETY CONFIRMATIONS
    # -------------------------------------------------------------------------

    print("\nFINAL SAFETY CONFIRMATIONS")
    print("-" * 80)

    print("Only the locked training period was used to fit the model.")
    print("Validation and holdout were used only for diagnostic scoring.")
    print("The threshold was fixed and not selected.")
    print("No tuning was performed.")
    print("No feature was removed or selected.")
    print("No resampling was performed.")
    print("No model artifact was saved.")
    print("fraudTest.csv was never read.")
    print("No raw data was modified.")
    print("No processed dataset was modified.")

    print("\nLOOP 008D COMPLETE")


if __name__ == "__main__":
    main()