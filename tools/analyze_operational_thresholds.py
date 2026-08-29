"""LOOP 006 operational threshold analysis on the validation split only."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


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

# Validation has 61 calendar days. These are fixed alert-capacity operating
# points, not an F1 or other metric optimization procedure.
CAPACITY_BUDGETS = {
    5: 305,
    10: 610,
    20: 1220,
    30: 1830,
    50: 3050,
    75: 4575,
    100: 6100,
}


def threshold_for_budget(probabilities: pd.Series, budget: int) -> float:
    """Select the probability at rank budget after stable descending sorting."""
    if not 1 <= budget <= len(probabilities):
        raise ValueError("Alert budget must be between 1 and validation row count.")
    ranked = probabilities.sort_values(ascending=False, kind="mergesort")
    return float(ranked.iloc[budget - 1])


def main() -> None:
    print("=" * 80)
    print("LOOP 006 — OPERATIONAL THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked model schema.")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Feature data contains missing values.")

    train = dataset.loc[
        dataset["prediction_date"].between(TRAIN_START, TRAIN_END)
    ].copy()
    validation = dataset.loc[
        dataset["prediction_date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("The locked temporal train/validation ordering is invalid.")
    if validation["prediction_date"].nunique() != 61:
        raise RuntimeError("Validation does not contain the expected 61 calendar days.")

    # Exact fixed methodology from LOOP 006. No parameters are tuned here.
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1],
        index=validation.index,
        name="probability",
    )

    print(f"Train rows used: {len(train):,}")
    print(f"Validation rows evaluated: {len(validation):,}")
    print(f"Validation calendar days: {validation['prediction_date'].nunique()}")
    print("Threshold selection source: validation predictions only")
    print("Ranking: descending probability, stable mergesort; alert if probability >= threshold")

    results = []
    y_true = validation["target"]
    for target_alerts_per_day, budget in CAPACITY_BUDGETS.items():
        threshold = threshold_for_budget(probabilities, budget)
        predictions = (probabilities >= threshold).astype("int64")
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        actual_alerts = int(predictions.sum())
        results.append(
            {
                "target_alerts_per_day": target_alerts_per_day,
                "validation_alert_budget": budget,
                "threshold": threshold,
                "total_validation_alerts": actual_alerts,
                "average_alerts_per_day": actual_alerts / 61,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "F1": f1_score(y_true, predictions, zero_division=0),
                "false_positive_rate": fp / (fp + tn),
                "captured_positive_targets": int(tp),
                "missed_positive_targets": int(fn),
            }
        )

    result_frame = pd.DataFrame(results)
    print("\nVALIDATION OPERATING POINTS")
    print(
        result_frame.to_string(
            index=False,
            formatters={
                "threshold": "{:.12f}".format,
                "average_alerts_per_day": "{:.4f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "F1": "{:.6f}".format,
                "false_positive_rate": "{:.6f}".format,
            },
        )
    )
    print("\nfraudTest.csv NOT read")
    print("raw data NOT modified")
    print("training dataset NOT modified")
    print("holdout NOT used for threshold selection or evaluation")
    print("no model tuning performed")
    print("no threshold recommendation made")


if __name__ == "__main__":
    main()
