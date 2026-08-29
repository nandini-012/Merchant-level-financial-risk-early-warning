"""LOOP 008C: HGB probability distribution stability analysis.

Diagnostic analysis only.

The locked HGB configuration is fitted on the locked training period.
Predicted probabilities are then examined separately on train, validation,
and holdout periods. No threshold is selected or changed, and no model
selection or tuning is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


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

LOCKED_THRESHOLD = 0.814822766216


def validate_dataset(dataset: pd.DataFrame) -> None:
    expected_columns = {
        "merchant",
        "prediction_date",
        "target",
        *FEATURE_COLUMNS,
    }

    if set(dataset.columns) != expected_columns:
        raise RuntimeError(
            "Dataset columns do not match the locked modelling schema."
        )

    if "target" in FEATURE_COLUMNS:
        raise RuntimeError(
            "Target must not be included among model features."
        )

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
        raise RuntimeError(
            "The modelling matrix contains missing values."
        )


def create_splits(dataset: pd.DataFrame):
    train = dataset.loc[
        dataset["prediction_date"].between(
            TRAIN_START,
            TRAIN_END,
        )
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

    if len(train) + len(validation) + len(holdout) != len(dataset):
        raise RuntimeError(
            "Locked temporal split does not cover the dataset exactly once."
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
        raise RuntimeError(
            "Train/validation temporal ordering is invalid."
        )

    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError(
            "Validation/holdout temporal ordering is invalid."
        )

    return train, validation, holdout


def summarize_probabilities(
    probabilities: np.ndarray,
    targets: pd.Series,
    split_name: str,
) -> dict:
    predictions = probabilities >= LOCKED_THRESHOLD

    positive_probabilities = probabilities[targets.to_numpy() == 1]
    negative_probabilities = probabilities[targets.to_numpy() == 0]

    return {
        "split": split_name,
        "rows": len(probabilities),
        "actual_positives": int(targets.sum()),
        "positive_rate": float(targets.mean()),
        "mean_probability": float(np.mean(probabilities)),
        "median_probability": float(np.median(probabilities)),
        "p90_probability": float(np.percentile(probabilities, 90)),
        "p95_probability": float(np.percentile(probabilities, 95)),
        "p99_probability": float(np.percentile(probabilities, 99)),
        "max_probability": float(np.max(probabilities)),
        "mean_positive_probability": float(
            np.mean(positive_probabilities)
        ),
        "mean_negative_probability": float(
            np.mean(negative_probabilities)
        ),
        "alerts": int(predictions.sum()),
        "alert_rate": float(predictions.mean()),
        "positive_alerts": int(
            ((targets.to_numpy() == 1) & predictions).sum()
        ),
        "negative_alerts": int(
            ((targets.to_numpy() == 0) & predictions).sum()
        ),
    }


def print_summary(summary: dict) -> None:
    print()
    print(summary["split"].upper())
    print("-" * 80)

    print(f"Rows: {summary['rows']:,}")
    print(f"Actual positives: {summary['actual_positives']:,}")
    print(f"Positive rate: {summary['positive_rate']:.6%}")

    print()
    print("PREDICTED PROBABILITY DISTRIBUTION")
    print(
        f"Mean:   {summary['mean_probability']:.6f}"
    )
    print(
        f"Median: {summary['median_probability']:.6f}"
    )
    print(
        f"P90:    {summary['p90_probability']:.6f}"
    )
    print(
        f"P95:    {summary['p95_probability']:.6f}"
    )
    print(
        f"P99:    {summary['p99_probability']:.6f}"
    )
    print(
        f"Maximum:{summary['max_probability']:.6f}"
    )

    print()
    print("CLASS-CONDITIONAL PROBABILITIES")
    print(
        "Mean probability for actual positives: "
        f"{summary['mean_positive_probability']:.6f}"
    )
    print(
        "Mean probability for actual negatives: "
        f"{summary['mean_negative_probability']:.6f}"
    )

    print()
    print("LOCKED THRESHOLD OBSERVATION")
    print(f"Threshold: {LOCKED_THRESHOLD:.12f}")
    print(f"Predicted alerts: {summary['alerts']:,}")
    print(f"Alert rate: {summary['alert_rate']:.6%}")
    print(
        "Alerts among actual positives: "
        f"{summary['positive_alerts']:,}"
    )
    print(
        "Alerts among actual negatives: "
        f"{summary['negative_alerts']:,}"
    )


def main() -> None:
    print("=" * 80)
    print("LOOP 008C — HGB PROBABILITY DISTRIBUTION STABILITY")
    print("=" * 80)

    print(
        f"Reading training dataset only: {DATASET_FILE}"
    )

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(
        dataset["prediction_date"]
    )

    validate_dataset(dataset)

    train, validation, holdout = create_splits(dataset)

    print()
    print("SAFETY CHECKS")
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
    print("No feature selection: confirmed")
    print("No resampling: confirmed")

    print()
    print("LOCKED MODEL")
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

    model.fit(
        train[FEATURE_COLUMNS],
        train["target"],
    )

    train_probabilities = model.predict_proba(
        train[FEATURE_COLUMNS]
    )[:, 1]

    validation_probabilities = model.predict_proba(
        validation[FEATURE_COLUMNS]
    )[:, 1]

    holdout_probabilities = model.predict_proba(
        holdout[FEATURE_COLUMNS]
    )[:, 1]

    train_summary = summarize_probabilities(
        train_probabilities,
        train["target"],
        "Train",
    )

    validation_summary = summarize_probabilities(
        validation_probabilities,
        validation["target"],
        "Validation",
    )

    holdout_summary = summarize_probabilities(
        holdout_probabilities,
        holdout["target"],
        "Holdout",
    )

    print_summary(train_summary)
    print_summary(validation_summary)
    print_summary(holdout_summary)

    summaries = pd.DataFrame(
        [
            train_summary,
            validation_summary,
            holdout_summary,
        ]
    )

    print()
    print("CROSS-SPLIT SUMMARY")
    print("-" * 80)

    columns = [
        "split",
        "mean_probability",
        "median_probability",
        "p90_probability",
        "p95_probability",
        "p99_probability",
        "max_probability",
        "mean_positive_probability",
        "mean_negative_probability",
        "alerts",
        "alert_rate",
    ]

    print(
        summaries[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("FINAL SAFETY CONFIRMATIONS")
    print("-" * 80)
    print("Only the locked training period was used to fit the model.")
    print("Validation and holdout were used only for diagnostic scoring.")
    print("The operational threshold was not selected or changed.")
    print("No tuning was performed.")
    print("No feature was removed or selected.")
    print("No resampling was performed.")
    print("No model artifact was saved.")
    print("fraudTest.csv was never read.")
    print("No raw data was modified.")
    print("No processed dataset was modified.")

    print()
    print("LOOP 008C COMPLETE")


if __name__ == "__main__":
    main()
    