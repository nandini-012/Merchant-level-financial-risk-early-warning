"""LOOP 006: fixed HistGradientBoosting temporal comparison."""

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

# Fixed results from LOOP 005B.  They are retained for reporting comparison
# only; Logistic Regression is intentionally not retrained in this experiment.
LOGISTIC_REGRESSION_RESULTS = {
    "TRAIN": {
        "roc_auc": 0.746287, "pr_auc": 0.051111, "precision": 0.047884,
        "recall": 0.615418, "f1": 0.088854,
    },
    "VALIDATION": {
        "roc_auc": 0.793299, "pr_auc": 0.056283, "precision": 0.047997,
        "recall": 0.646859, "f1": 0.089363,
    },
    "FINAL HOLDOUT": {
        "roc_auc": 0.754317, "pr_auc": 0.053703, "precision": 0.058824,
        "recall": 0.640394, "f1": 0.107750,
    },
}


def evaluate(y_true: pd.Series, scores) -> dict[str, object]:
    predictions = (scores >= 0.5).astype("int64")
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "predicted_positives": int(predictions.sum()),
        "actual_positives": int(y_true.sum()),
    }


def main() -> None:
    print("=" * 80)
    print("LOOP 006 — NONLINEAR MODEL COMPARISON")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked schema.")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Historical feature columns contain missing values.")

    split_frames = {
        name: dataset.loc[dataset["prediction_date"].between(start, end)].copy()
        for name, (start, end) in SPLITS.items()
    }
    if sum(len(frame) for frame in split_frames.values()) != len(dataset):
        raise RuntimeError("The locked temporal split does not cover the dataset exactly.")
    train = split_frames["TRAIN"]
    validation = split_frames["VALIDATION"]
    holdout = split_frames["FINAL HOLDOUT"]
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal order is invalid.")
    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation/holdout temporal order is invalid.")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    results = {}
    print("\nHISTGRADIENTBOOSTINGCLASSIFIER — THRESHOLD 0.5")
    for name, frame in split_frames.items():
        scores = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        result = evaluate(frame["target"], scores)
        results[name] = result
        print(f"\n{name}")
        print(f"ROC-AUC: {result['roc_auc']:.6f}")
        print(f"PR-AUC / Average Precision: {result['pr_auc']:.6f}")
        print(f"Precision: {result['precision']:.6f}")
        print(f"Recall: {result['recall']:.6f}")
        print(f"F1: {result['f1']:.6f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(result["confusion_matrix"])
        print(f"Predicted positives: {result['predicted_positives']:,}")
        print(f"Actual positives: {result['actual_positives']:,}")

    print("\nCOMPARISON WITH LOOP 005B LOGISTIC REGRESSION")
    comparison_rows = []
    for split_name in SPLITS:
        hgb = results[split_name]
        logistic = LOGISTIC_REGRESSION_RESULTS[split_name]
        comparison_rows.append(
            {
                "split": split_name,
                "model": "Logistic Regression",
                "ROC-AUC": logistic["roc_auc"],
                "PR-AUC": logistic["pr_auc"],
                "Precision": logistic["precision"],
                "Recall": logistic["recall"],
                "F1": logistic["f1"],
            }
        )
        comparison_rows.append(
            {
                "split": split_name,
                "model": "HistGradientBoosting",
                "ROC-AUC": hgb["roc_auc"],
                "PR-AUC": hgb["pr_auc"],
                "Precision": hgb["precision"],
                "Recall": hgb["recall"],
                "F1": hgb["f1"],
            }
        )
    print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nTRAINING COMPLETED SUCCESSFULLY")
    print("One additional model only; no tuning, threshold optimization, feature selection, or resampling was performed.")
    print("fraudTest.csv was not read.")
    print("No raw-data or training-dataset files were modified.")
    print("No model artifact was saved.")


if __name__ == "__main__":
    main()
