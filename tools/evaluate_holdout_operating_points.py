"""LOOP 006D: holdout operating-point analysis with validation-frozen thresholds."""

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

# These values and the model construction match evaluate_locked_holdout.py.
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

CAPACITY_BUDGETS = {
    5: 305,
    10: 610,
    20: 1220,
    30: 1830,
    50: 3050,
    75: 4575,
    100: 6100,
}


def threshold_at_validation_rank(probabilities: pd.Series, budget: int) -> float:
    """Derive a threshold solely from a stable descending validation ranking."""
    if not 1 <= budget <= len(probabilities):
        raise RuntimeError("Capacity alert budget is outside the validation row count.")
    ranked = probabilities.sort_values(ascending=False, kind="mergesort")
    return float(ranked.iloc[budget - 1])


def threshold_metrics(y_true: pd.Series, probabilities, threshold: float) -> dict:
    """Evaluate probabilities at an already fixed threshold."""
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "actual_positives": int(y_true.sum()),
        "alerts": int(predictions.sum()),
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


def main() -> None:
    print("=" * 80)
    print("LOOP 006D — LOCKED HOLDOUT OPERATING-POINT ANALYSIS")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")
    if "target" in FEATURE_COLUMNS:
        raise RuntimeError("Target must be excluded from model features.")
    future_features = [column for column in FEATURE_COLUMNS if "future" in column.lower()]
    if future_features:
        raise RuntimeError(f"Future-derived features are prohibited: {future_features}")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Model features contain missing values.")

    train = dataset.loc[dataset["prediction_date"].between(TRAIN_START, TRAIN_END)].copy()
    validation = dataset.loc[
        dataset["prediction_date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    holdout = dataset.loc[
        dataset["prediction_date"].between(HOLDOUT_START, HOLDOUT_END)
    ].copy()
    if len(train) + len(validation) + len(holdout) != len(dataset):
        raise RuntimeError("Temporal splits do not cover the dataset exactly once.")
    split_keys = pd.concat(
        [
            train[["merchant", "prediction_date"]],
            validation[["merchant", "prediction_date"]],
            holdout[["merchant", "prediction_date"]],
        ],
        ignore_index=True,
    )
    if split_keys.duplicated().any():
        raise RuntimeError("A merchant/date key appears in multiple splits.")
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train must precede validation.")
    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation must precede holdout.")
    if validation["prediction_date"].nunique() != VALIDATION_DAYS:
        raise RuntimeError("Expected 61 validation calendar days.")

    print("\nVALIDATION CHECKS")
    print("Temporal ordering: confirmed")
    print("No merchant/date overlap across splits: confirmed")
    print("Target excluded from features: confirmed")
    print("No future-derived features: confirmed")
    print("No missing model features: confirmed")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    # STAGE 1: Generate validation probabilities and freeze all pre-specified
    # capacity thresholds. No holdout probability is generated in this stage.
    validation_probabilities = pd.Series(
        model.predict_proba(validation[FEATURE_COLUMNS])[:, 1], index=validation.index
    )
    frozen_thresholds = {
        capacity: threshold_at_validation_rank(validation_probabilities, budget)
        for capacity, budget in CAPACITY_BUDGETS.items()
    }
    validation_results = {
        capacity: threshold_metrics(validation["target"], validation_probabilities, threshold)
        for capacity, threshold in frozen_thresholds.items()
    }
    for capacity, budget in CAPACITY_BUDGETS.items():
        if validation_results[capacity]["alerts"] < budget:
            raise RuntimeError("A validation rank threshold produced fewer alerts than budget.")

    print("\nALL THRESHOLDS LOCKED FROM VALIDATION ONLY")
    print("Ranking: stable descending probability; alert when probability >= threshold")
    for capacity, budget in CAPACITY_BUDGETS.items():
        print(
            f"{capacity:>3} alerts/day | budget={budget:>4,} | "
            f"threshold={frozen_thresholds[capacity]:.12f} | "
            f"validation alerts={validation_results[capacity]['alerts']:,}"
        )
    print("All thresholds are frozen before holdout probabilities are generated.")

    # STAGE 2: Generate holdout probabilities once, then apply each immutable,
    # validation-derived threshold exactly once. No result enters Stage 1.
    holdout_probabilities = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    holdout_days = holdout["prediction_date"].nunique()
    rows = []
    for capacity, budget in CAPACITY_BUDGETS.items():
        validation_metrics = validation_results[capacity]
        holdout_metrics = threshold_metrics(
            holdout["target"], holdout_probabilities, frozen_thresholds[capacity]
        )
        rows.append(
            {
                "capacity_alerts_per_day": capacity,
                "validation_alert_budget": budget,
                "validation_threshold": frozen_thresholds[capacity],
                "validation_actual_positives": validation_metrics["actual_positives"],
                "validation_alerts": validation_metrics["alerts"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "holdout_rows": holdout_metrics["rows"],
                "holdout_actual_positives": holdout_metrics["actual_positives"],
                "holdout_alerts": holdout_metrics["alerts"],
                "holdout_alerts_per_day": holdout_metrics["alerts"] / holdout_days,
                "holdout_TP": holdout_metrics["TP"],
                "holdout_FP": holdout_metrics["FP"],
                "holdout_FN": holdout_metrics["FN"],
                "holdout_TN": holdout_metrics["TN"],
                "holdout_precision": holdout_metrics["precision"],
                "holdout_recall": holdout_metrics["recall"],
                "holdout_F1": holdout_metrics["F1"],
                "holdout_ROC_AUC": holdout_metrics["ROC-AUC"],
                "holdout_PR_AUC": holdout_metrics["PR-AUC"],
                "holdout_false_positive_rate": holdout_metrics["false_positive_rate"],
                "holdout_false_alerts_per_day": holdout_metrics["FP"] / holdout_days,
                "holdout_positive_events_captured": holdout_metrics["TP"],
            }
        )

    results = pd.DataFrame(rows)
    print("\nPRE-SPECIFIED OPERATING-POINT RESULTS")
    print(
        results.to_string(
            index=False,
            formatters={
                "validation_threshold": "{:.12f}".format,
                "validation_precision": "{:.6f}".format,
                "validation_recall": "{:.6f}".format,
                "holdout_alerts_per_day": "{:.4f}".format,
                "holdout_precision": "{:.6f}".format,
                "holdout_recall": "{:.6f}".format,
                "holdout_F1": "{:.6f}".format,
                "holdout_ROC_AUC": "{:.6f}".format,
                "holdout_PR_AUC": "{:.6f}".format,
                "holdout_false_positive_rate": "{:.6f}".format,
                "holdout_false_alerts_per_day": "{:.4f}".format,
            },
        )
    )
    print("\nEvery threshold was derived from validation only and frozen before holdout evaluation.")
    print("No capacity was selected or recommended from holdout results.")
    print("No model tuning, threshold tuning, feature selection, or resampling was performed.")
    print("fraudTest.csv was not read")
    print("raw data was not modified")
    print("training_dataset.parquet was not modified")
    print("no model artifact was saved")
    print("LOOP 006D COMPLETE")


if __name__ == "__main__":
    main()
