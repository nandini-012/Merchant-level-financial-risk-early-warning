"""LOOP 006: fixed HistGradientBoosting temporal baseline comparison."""

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

SPLITS = {
    "TRAIN": ("2019-01-15", "2020-02-29"),
    "VALIDATION": ("2020-03-01", "2020-04-30"),
    "FINAL HOLDOUT": ("2020-05-01", "2020-06-14"),
}

# Metrics copied from the executed fixed Logistic Regression experiment in
# LOOP 005B. They are displayed for comparison only; no Logistic Regression
# is refit and no holdout-driven selection is performed in this loop.
LOOP_005B_LOGISTIC_RESULTS = {
    "TRAIN": (0.746287, 0.051111, 0.047884, 0.615418, 0.088854),
    "VALIDATION": (0.793299, 0.056283, 0.047997, 0.646859, 0.089363),
    "FINAL HOLDOUT": (0.754317, 0.053703, 0.058824, 0.640394, 0.107750),
}


def calculate_metrics(y_true: pd.Series, probabilities) -> tuple[dict[str, float], object]:
    predictions = (probabilities >= 0.5).astype("int64")
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
    }
    return metrics, predictions


def print_metrics(name: str, y_true: pd.Series, metrics: dict[str, float], predictions) -> None:
    print(f"\n{name}")
    for metric_name, value in metrics.items():
        label = "PR-AUC / Average Precision" if metric_name == "PR-AUC" else metric_name
        print(f"{label}: {value:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    print(f"Predicted positives: {int(predictions.sum()):,}")
    print(f"Actual positives: {int(y_true.sum()):,}")


def main() -> None:
    print("=" * 80)
    print("LOOP 006 — NONLINEAR MODEL COMPARISON")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Historical features contain missing values.")

    frames = {
        name: dataset.loc[dataset["prediction_date"].between(start, end)].copy()
        for name, (start, end) in SPLITS.items()
    }
    if sum(len(frame) for frame in frames.values()) != len(dataset):
        raise RuntimeError("The locked temporal split does not cover the dataset exactly once.")
    split_keys = pd.concat(
        [frame[["merchant", "prediction_date"]] for frame in frames.values()],
        ignore_index=True,
    )
    if split_keys.duplicated().any():
        raise RuntimeError("A merchant/date key occurs in more than one split.")

    train = frames["TRAIN"]
    validation = frames["VALIDATION"]
    holdout = frames["FINAL HOLDOUT"]
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal order is invalid.")
    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation/holdout temporal order is invalid.")

    print("\nTEMPORAL SPLIT")
    for name, frame in frames.items():
        positives = int(frame["target"].sum())
        print(
            f"{name}: rows={len(frame):,}, positives={positives:,}, "
            f"positive_rate={positives / len(frame):.4%}, "
            f"dates={frame['prediction_date'].min().date()}.."
            f"{frame['prediction_date'].max().date()}"
        )
    print("Temporal ordering and cross-split key uniqueness: confirmed")

    # Fixed defaults except deterministic random state and disabled internal
    # validation-based early stopping. No tuning is performed.
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    hgb_results = {}
    print("\n" + "=" * 80)
    print("HISTGRADIENTBOOSTINGCLASSIFIER (THRESHOLD 0.5)")
    print("=" * 80)
    for name, frame in frames.items():
        probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        metrics, predictions = calculate_metrics(frame["target"], probabilities)
        hgb_results[name] = metrics
        print_metrics(name, frame["target"], metrics, predictions)

    print("\n" + "=" * 80)
    print("COMPARISON WITH LOOP 005B LOGISTIC REGRESSION")
    print("=" * 80)
    rows = []
    for name in SPLITS:
        logistic = LOOP_005B_LOGISTIC_RESULTS[name]
        nonlinear = hgb_results[name]
        rows.append(
            {
                "split": name,
                "model": "Logistic Regression (LOOP 005B)",
                "ROC-AUC": logistic[0],
                "PR-AUC": logistic[1],
                "Precision": logistic[2],
                "Recall": logistic[3],
                "F1": logistic[4],
            }
        )
        rows.append({"split": name, "model": "HistGradientBoostingClassifier", **nonlinear})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\nTRAINING COMPLETED SUCCESSFULLY")
    print("One additional fixed nonlinear model was trained; no model artifact was saved.")
    print("No tuning, threshold optimization, feature selection, or resampling was performed.")
    print("fraudTest.csv was not read.")
    print("No raw-data or training-dataset files were modified.")


if __name__ == "__main__":
    main()
