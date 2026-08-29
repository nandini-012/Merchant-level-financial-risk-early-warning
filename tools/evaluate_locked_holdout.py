"""LOOP 006C: one-time holdout evaluation at a validation-locked threshold.

Threshold derivation is deliberately completed before holdout probabilities are
created. The holdout has no code path into threshold selection.
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

TRAIN_START = "2019-01-15"
TRAIN_END = "2020-02-28"
VALIDATION_START = "2020-03-01"
VALIDATION_END = "2020-04-30"
HOLDOUT_START = "2020-05-01"
HOLDOUT_END = "2020-06-14"

VALIDATION_DAYS = 61
LOCKED_CAPACITY_ALERTS_PER_DAY = 20
LOCKED_VALIDATION_ALERT_BUDGET = 1220


def select_validation_threshold(probabilities: pd.Series) -> float:
    """Lock the 20-alert/day threshold solely from validation probabilities."""
    if len(probabilities) < LOCKED_VALIDATION_ALERT_BUDGET:
        raise RuntimeError("Validation data has fewer rows than the locked alert budget.")
    ranked = probabilities.sort_values(ascending=False, kind="mergesort")
    return float(ranked.iloc[LOCKED_VALIDATION_ALERT_BUDGET - 1])


def operating_metrics(y_true: pd.Series, probabilities, threshold: float) -> dict:
    """Calculate metrics for a pre-specified threshold."""
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "rows": len(y_true),
        "actual_positives": int(y_true.sum()),
        "predicted_positives": int(predictions.sum()),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "false_positive_rate": fp / (fp + tn),
    }


def print_validation_provenance(metrics: dict) -> None:
    print("\nVALIDATION OPERATING POINT USED TO LOCK THRESHOLD")
    print(f"Capacity provenance: {LOCKED_CAPACITY_ALERTS_PER_DAY} alerts/day")
    print(f"Validation alert budget: {LOCKED_VALIDATION_ALERT_BUDGET:,}")
    print(f"Locked threshold: {metrics['threshold']:.12f}")
    print(f"Validation rows: {metrics['rows']:,}")
    print(f"Validation actual positives: {metrics['actual_positives']:,}")
    print(f"Validation predicted positives / alerts: {metrics['predicted_positives']:,}")
    print(f"Validation alerts/day: {metrics['predicted_positives'] / VALIDATION_DAYS:.4f}")
    print(
        "Validation confusion matrix [[TN, FP], [FN, TP]]: "
        f"[[{metrics['TN']}, {metrics['FP']}], [{metrics['FN']}, {metrics['TP']}]]"
    )
    print(f"Validation precision: {metrics['precision']:.6f}")
    print(f"Validation recall: {metrics['recall']:.6f}")
    print(f"Validation F1: {metrics['F1']:.6f}")
    print(f"Validation false-positive rate: {metrics['false_positive_rate']:.6f}")
    print(f"Validation positive events captured: {metrics['TP']:,}")


def print_holdout_results(metrics: dict, holdout_days: int) -> None:
    print("\nFINAL HOLDOUT EVALUATION — LOCKED THRESHOLD APPLIED ONCE")
    print(f"Locked threshold: {metrics['threshold']:.12f}")
    print(f"Holdout rows: {metrics['rows']:,}")
    print(f"Actual positives: {metrics['actual_positives']:,}")
    print(f"Predicted positives / total alerts: {metrics['predicted_positives']:,}")
    print(f"Alerts per day: {metrics['predicted_positives'] / holdout_days:.4f}")
    print(f"TP: {metrics['TP']:,}")
    print(f"FP: {metrics['FP']:,}")
    print(f"FN: {metrics['FN']:,}")
    print(f"TN: {metrics['TN']:,}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['F1']:.6f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.6f}")
    print(f"PR-AUC / Average Precision: {metrics['PR-AUC']:.6f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.6f}")
    print(f"False alerts per day: {metrics['FP'] / holdout_days:.4f}")
    print(f"Positive events captured: {metrics['TP']:,}")
    print(
        "Confusion matrix [[TN, FP], [FN, TP]]: "
        f"[[{metrics['TN']}, {metrics['FP']}], [{metrics['FN']}, {metrics['TP']}]]"
    )


def main() -> None:
    print("=" * 80)
    print("LOOP 006C — LOCKED HOLDOUT EVALUATION")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")
    if "target" in FEATURE_COLUMNS:
        raise RuntimeError("Target must not be included among model features.")
    future_features = [column for column in FEATURE_COLUMNS if "future" in column.lower()]
    if future_features:
        raise RuntimeError(f"Future-derived features are not permitted: {future_features}")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("The modelling matrix contains missing values.")

    train = dataset.loc[dataset["prediction_date"].between(TRAIN_START, TRAIN_END)].copy()
    validation = dataset.loc[
        dataset["prediction_date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    holdout = dataset.loc[
        dataset["prediction_date"].between(HOLDOUT_START, HOLDOUT_END)
    ].copy()
    if len(train) + len(validation) + len(holdout) != len(dataset):
        raise RuntimeError("Locked temporal split does not cover the dataset exactly once.")
    all_keys = pd.concat(
        [
            train[["merchant", "prediction_date"]],
            validation[["merchant", "prediction_date"]],
            holdout[["merchant", "prediction_date"]],
        ],
        ignore_index=True,
    )
    if all_keys.duplicated().any():
        raise RuntimeError("A merchant/date key appears in more than one split.")
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal ordering is invalid.")
    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation/holdout temporal ordering is invalid.")
    if validation["prediction_date"].nunique() != VALIDATION_DAYS:
        raise RuntimeError("Validation does not contain the expected 61 calendar days.")

    print("\nVALIDATION CHECKS")
    print("Temporal ordering: confirmed")
    print("No merchant/date overlap across splits: confirmed")
    print("Target excluded from model features: confirmed")
    print("No future-derived feature used: confirmed")
    print("No missing values in modelling matrix: confirmed")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    # STAGE 1: validation-only threshold derivation. No holdout score is made.
    validation_probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    locked_threshold = select_validation_threshold(pd.Series(validation_probabilities))
    validation_metrics = operating_metrics(
        validation["target"], validation_probabilities, locked_threshold
    )
    if validation_metrics["predicted_positives"] < LOCKED_VALIDATION_ALERT_BUDGET:
        raise RuntimeError("Locked threshold yielded fewer alerts than its rank budget.")
    print_validation_provenance(validation_metrics)
    print("Threshold lock complete. Holdout has not been scored during threshold selection.")

    # STAGE 2: one-time holdout evaluation after the threshold is immutable.
    holdout_probabilities = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    holdout_metrics = operating_metrics(holdout["target"], holdout_probabilities, locked_threshold)
    print_holdout_results(holdout_metrics, holdout["prediction_date"].nunique())

    print("\nfraudTest.csv was never read")
    print("raw data was never modified")
    print("training_dataset.parquet was never modified")
    print("no model artifact was saved")
    print("LOOP 006C COMPLETE")
    print(
        "The holdout was evaluated once at the pre-specified validation-derived "
        "threshold and was not used for threshold selection."
    )


if __name__ == "__main__":
    main()
