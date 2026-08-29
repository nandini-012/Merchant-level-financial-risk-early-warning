"""LOOP 007: temporal stability analysis of the locked HGB operating point.

Analysis only:
- Uses the locked training/validation/holdout periods.
- Uses the fixed HGB configuration from LOOP 005C.
- Uses the threshold locked in LOOP 006E.
- Does not select or tune a threshold.
- Does not modify datasets or save model artifacts.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"

FEATURE_COLUMNS = [
    "previous_1d_transaction_count",
    "previous_3d_transaction_count",
    "previous_7d_transaction_count",
    "previous_14d_transaction_count",
    "previous_7d_fraud_count",
    "previous_14d_fraud_count",
    "previous_7d_fraud_rate",
    "previous_14d_fraud_rate",
    "preceding_7d_transaction_count",
    "previous_7d_transaction_count_change",
    "preceding_7d_fraud_count",
    "previous_7d_fraud_count_change",
    "previous_7d_total_transaction_amount",
    "previous_7d_average_transaction_amount",
    "previous_7d_maximum_transaction_amount",
]

TARGET_COLUMN = "target"

TRAIN_START = "2019-01-15"
TRAIN_END = "2020-02-28"

VALIDATION_START = "2020-03-01"
VALIDATION_END = "2020-04-30"

HOLDOUT_START = "2020-05-01"
HOLDOUT_END = "2020-06-14"

LOCKED_THRESHOLD = 0.814822766216


def calculate_metrics(y_true, probabilities, threshold, days):
    predictions = (probabilities >= threshold).astype("int64")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": len(y_true),
        "actual_positives": int(y_true.sum()),
        "predicted_alerts": int(predictions.sum()),
        "alerts_per_day": float(predictions.sum() / days),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "PR-AUC": float(average_precision_score(y_true, probabilities)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 80)

    print(f"Rows: {metrics['rows']:,}")
    print(f"Actual positives: {metrics['actual_positives']:,}")
    print(f"Predicted alerts: {metrics['predicted_alerts']:,}")
    print(f"Alerts/day: {metrics['alerts_per_day']:.4f}")

    print(f"TP: {metrics['TP']:,}")
    print(f"FP: {metrics['FP']:,}")
    print(f"FN: {metrics['FN']:,}")
    print(f"TN: {metrics['TN']:,}")

    print(f"ROC-AUC: {metrics['ROC-AUC']:.6f}")
    print(f"PR-AUC: {metrics['PR-AUC']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['F1']:.6f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.6f}")


def main():
    print("=" * 80)
    print("LOOP 007 — TEMPORAL MODEL STABILITY ANALYSIS")
    print("=" * 80)

    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {
        "merchant",
        "prediction_date",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    if set(dataset.columns) != expected_columns:
        raise RuntimeError(
            "Dataset columns do not match the locked modelling schema."
        )

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise RuntimeError("Target must not be included among model features.")

    future_features = [
        column
        for column in FEATURE_COLUMNS
        if "future" in column.lower()
    ]

    if future_features:
        raise RuntimeError(
            f"Future-derived features are not permitted: {future_features}"
        )

    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("The modelling matrix contains missing values.")

    train = dataset.loc[
        dataset["prediction_date"].between(TRAIN_START, TRAIN_END)
    ].copy()

    validation = dataset.loc[
        dataset["prediction_date"].between(
            VALIDATION_START,
            VALIDATION_END,
        )
    ].copy()

    holdout = dataset.loc[
        dataset["prediction_date"].between(
            HOLDOUT_START,
            HOLDOUT_END,
        )
    ].copy()

    # ------------------------------------------------------------------
    # SAFETY CHECKS
    # ------------------------------------------------------------------

    if len(train) + len(validation) + len(holdout) != len(dataset):
        raise RuntimeError(
            "Temporal split does not cover the dataset exactly once."
        )

    all_keys = pd.concat(
        [
            train[["merchant", "prediction_date"]],
            validation[["merchant", "prediction_date"]],
            holdout[["merchant", "prediction_date"]],
        ],
        ignore_index=True,
    )

    if all_keys.duplicated().any():
        raise RuntimeError(
            "A merchant/date key appears in more than one split."
        )

    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal ordering is invalid.")

    if (
        validation["prediction_date"].max()
        >= holdout["prediction_date"].min()
    ):
        raise RuntimeError("Validation/holdout temporal ordering is invalid.")

    print("\nSAFETY CHECKS")
    print("-" * 80)
    print("Temporal ordering: confirmed")
    print("No merchant/date overlap across splits: confirmed")
    print("Target excluded from model features: confirmed")
    print("No future-derived features: confirmed")
    print("No missing model features: confirmed")
    print("fraudTest.csv not read: confirmed")
    print("Raw/processed datasets will not be modified: confirmed")
    print("No model artifact will be saved: confirmed")
    print("No threshold selection: confirmed")
    print("No tuning: confirmed")

    print("\nLOCKED CONFIGURATION")
    print("-" * 80)
    print("Model: HistGradientBoostingClassifier")
    print('class_weight="balanced"')
    print("random_state=42")
    print("early_stopping=False")
    print(f"Locked threshold: {LOCKED_THRESHOLD:.12f}")

    # ------------------------------------------------------------------
    # FIXED MODEL
    # ------------------------------------------------------------------

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
    )

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    train_probabilities = model.predict_proba(
        train[FEATURE_COLUMNS]
    )[:, 1]

    validation_probabilities = model.predict_proba(
        validation[FEATURE_COLUMNS]
    )[:, 1]

    holdout_probabilities = model.predict_proba(
        holdout[FEATURE_COLUMNS]
    )[:, 1]

    train_metrics = calculate_metrics(
        train[TARGET_COLUMN],
        train_probabilities,
        LOCKED_THRESHOLD,
        train["prediction_date"].nunique(),
    )

    validation_metrics = calculate_metrics(
        validation[TARGET_COLUMN],
        validation_probabilities,
        LOCKED_THRESHOLD,
        validation["prediction_date"].nunique(),
    )

    holdout_metrics = calculate_metrics(
        holdout[TARGET_COLUMN],
        holdout_probabilities,
        LOCKED_THRESHOLD,
        holdout["prediction_date"].nunique(),
    )

    print_metrics("TRAIN", train_metrics)
    print_metrics("VALIDATION", validation_metrics)
    print_metrics("FINAL HOLDOUT", holdout_metrics)

    # ------------------------------------------------------------------
    # VALIDATION -> HOLDOUT CHANGE
    # ------------------------------------------------------------------

    print("\nVALIDATION → HOLDOUT CHANGE")
    print("-" * 80)

    for metric in [
        "PR-AUC",
        "precision",
        "recall",
        "alerts_per_day",
    ]:
        change = holdout_metrics[metric] - validation_metrics[metric]

        print(
            f"{metric}: "
            f"{validation_metrics[metric]:.6f} → "
            f"{holdout_metrics[metric]:.6f} "
            f"(change {change:+.6f})"
        )

    print("\nFINAL SAFETY CONFIRMATIONS")
    print("-" * 80)
    print("No threshold was selected.")
    print("No tuning was performed.")
    print("No feature selection was performed.")
    print("No resampling was performed.")
    print("No model artifact was saved.")
    print("fraudTest.csv was never read.")
    print("No raw data was modified.")
    print("No processed dataset was modified.")

    print("\nLOOP 007 COMPLETE")


if __name__ == "__main__":
    main()