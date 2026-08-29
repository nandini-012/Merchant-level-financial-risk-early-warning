"""LOOP 005B: fixed temporal baselines for the locked training dataset."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    "FINAL HOLDOUT TEST": ("2020-05-01", "2020-06-14"),
}


def split_frame(dataset: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return dataset.loc[dataset["prediction_date"].between(start, end)].copy()


def print_split_summary(name: str, frame: pd.DataFrame) -> None:
    positive_count = int(frame["target"].sum())
    print(f"\n{name}")
    print(f"Rows: {len(frame):,}")
    print(f"Positive targets: {positive_count:,}")
    print(f"Positive rate: {positive_count / len(frame):.4%}")
    print(f"Minimum prediction date: {frame['prediction_date'].min().date()}")
    print(f"Maximum prediction date: {frame['prediction_date'].max().date()}")


def print_metrics(name: str, y_true: pd.Series, scores, predictions) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    print(f"\n{name}")
    print(f"ROC-AUC: {roc_auc_score(y_true, scores):.6f}")
    print(f"PR-AUC / Average Precision: {average_precision_score(y_true, scores):.6f}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.6f}")
    print(f"Recall: {recall_score(y_true, predictions, zero_division=0):.6f}")
    print(f"F1: {f1_score(y_true, predictions, zero_division=0):.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    print(f"Predicted positives: {int(predictions.sum()):,}")
    print(f"Actual positives: {int(y_true.sum()):,}")


def main() -> None:
    print("=" * 80)
    print("LOOP 005B — TEMPORAL SPLIT + BASELINE MODELS")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Historical feature columns contain missing values.")
    if not set(dataset["target"].unique()).issubset({0, 1}):
        raise RuntimeError("Target must contain only binary values.")

    split_frames = {
        name: split_frame(dataset, start, end)
        for name, (start, end) in SPLITS.items()
    }
    assigned_rows = sum(len(frame) for frame in split_frames.values())
    if assigned_rows != len(dataset):
        raise RuntimeError("Temporal split does not assign every dataset row exactly once.")
    split_keys = pd.concat(
        [frame[["merchant", "prediction_date"]] for frame in split_frames.values()],
        ignore_index=True,
    )
    if split_keys.duplicated().any():
        raise RuntimeError("A merchant/date key appears in multiple temporal splits.")

    train = split_frames["TRAIN"]
    validation = split_frames["VALIDATION"]
    holdout = split_frames["FINAL HOLDOUT TEST"]
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal order is invalid.")
    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation/holdout temporal order is invalid.")

    print("\nTEMPORAL SPLIT CHECKS")
    for name, frame in split_frames.items():
        print_split_summary(name, frame)
    print("max(train) < min(validation): confirmed")
    print("max(validation) < min(test): confirmed")
    print("No merchant/date key appears in multiple splits: confirmed")
    print("Feature columns:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    logistic_regression = Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )
    logistic_regression.fit(x_train, y_train)

    print("\n" + "=" * 80)
    print("MODEL 0 — MAJORITY-CLASS BASELINE (THRESHOLD 0.5)")
    print("=" * 80)
    for split_name, frame in split_frames.items():
        y_true = frame["target"]
        majority_scores = pd.Series(0.0, index=frame.index)
        majority_predictions = pd.Series(0, index=frame.index, dtype="int64")
        print_metrics(split_name, y_true, majority_scores, majority_predictions)

    print("\n" + "=" * 80)
    print("MODEL 1 — LOGISTIC REGRESSION (THRESHOLD 0.5)")
    print("=" * 80)
    for split_name, frame in split_frames.items():
        scores = logistic_regression.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        predictions = (scores >= 0.5).astype("int64")
        print_metrics(split_name, frame["target"], scores, predictions)

    print("\nTRAINING COMPLETED SUCCESSFULLY")
    print("No hyperparameter tuning, threshold optimization, feature selection, or resampling was performed.")
    print("fraudTest.csv was not read.")
    print("No raw-data or training-dataset files were modified.")
    print("No model artifact was saved; this loop reports baseline metrics only.")


if __name__ == "__main__":
    main()
