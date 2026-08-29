"""LOOP 008B: HGB feature stability analysis across temporal splits.

This is diagnostic analysis only.

The locked HGB configuration is retrained on the locked training period,
then permutation importance is measured separately on training, validation,
and holdout data. No feature is removed, selected, or modified based on
the results. The operational threshold is not used.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


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

PERMUTATIONS = 10
RANDOM_STATE = 42


def validate_dataset(dataset: pd.DataFrame) -> None:
    expected_columns = {
        "merchant",
        "prediction_date",
        "target",
        *FEATURE_COLUMNS,
    }

    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")

    if "target" in FEATURE_COLUMNS:
        raise RuntimeError("Target must not be included among model features.")

    future_features = [
        column for column in FEATURE_COLUMNS
        if "future" in column.lower()
    ]

    if future_features:
        raise RuntimeError(
            f"Future-derived features are not permitted: {future_features}"
        )

    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("The modelling matrix contains missing values.")


def temporal_splits(dataset: pd.DataFrame):
    train = dataset.loc[
        dataset["prediction_date"].between(TRAIN_START, TRAIN_END)
    ].copy()

    validation = dataset.loc[
        dataset["prediction_date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()

    holdout = dataset.loc[
        dataset["prediction_date"].between(HOLDOUT_START, HOLDOUT_END)
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
        raise RuntimeError("Train/validation temporal ordering is invalid.")

    if validation["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Validation/holdout temporal ordering is invalid.")

    return train, validation, holdout


def calculate_importance(
    model,
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]

    baseline_pr_auc = average_precision_score(
        frame["target"],
        probabilities,
    )

    result = permutation_importance(
        model,
        frame[FEATURE_COLUMNS],
        frame["target"],
        scoring="average_precision",
        n_repeats=PERMUTATIONS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_importance": result.importances_mean,
            "std_importance": result.importances_std,
        }
    )

    importance["split"] = split_name
    importance["baseline_pr_auc"] = baseline_pr_auc

    return importance.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)


def print_ranking(importance: pd.DataFrame, split_name: str) -> None:
    print()
    print(split_name.upper())
    print("-" * 80)

    print(
        f"Baseline PR-AUC: "
        f"{importance['baseline_pr_auc'].iloc[0]:.6f}"
    )

    print("Feature permutation importance:")
    print()

    for rank, row in importance.iterrows():
        print(
            f"{rank + 1:2d}. "
            f"{row['feature']:<50} "
            f"mean={row['mean_importance']:.8f} "
            f"std={row['std_importance']:.8f}"
        )


def main() -> None:
    print("=" * 80)
    print("LOOP 008B — HGB FEATURE STABILITY ANALYSIS")
    print("=" * 80)

    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(
        dataset["prediction_date"]
    )

    validate_dataset(dataset)

    train, validation, holdout = temporal_splits(dataset)

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

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["target"],
    )

    print()
    print("TEMPORAL SPLITS")
    print("-" * 80)
    print(f"Train:      {len(train):,} rows")
    print(f"Validation: {len(validation):,} rows")
    print(f"Holdout:    {len(holdout):,} rows")

    train_importance = calculate_importance(
        model,
        train,
        "train",
    )

    validation_importance = calculate_importance(
        model,
        validation,
        "validation",
    )

    holdout_importance = calculate_importance(
        model,
        holdout,
        "holdout",
    )

    print_ranking(train_importance, "train")
    print_ranking(validation_importance, "validation")
    print_ranking(holdout_importance, "holdout")

    combined = train_importance[
        ["feature", "mean_importance"]
    ].rename(
        columns={"mean_importance": "train_importance"}
    )

    combined = combined.merge(
        validation_importance[
            ["feature", "mean_importance"]
        ].rename(
            columns={"mean_importance": "validation_importance"}
        ),
        on="feature",
    )

    combined = combined.merge(
        holdout_importance[
            ["feature", "mean_importance"]
        ].rename(
            columns={"mean_importance": "holdout_importance"}
        ),
        on="feature",
    )

    combined["validation_minus_train"] = (
        combined["validation_importance"]
        - combined["train_importance"]
    )

    combined["holdout_minus_train"] = (
        combined["holdout_importance"]
        - combined["train_importance"]
    )

    print()
    print("CROSS-SPLIT IMPORTANCE COMPARISON")
    print("-" * 80)

    print(
        combined.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print()
    print("FINAL SAFETY CONFIRMATIONS")
    print("-" * 80)
    print("Only the locked training period was used to fit the model.")
    print("Validation and holdout were used only for diagnostic scoring.")
    print("No threshold was selected.")
    print("No tuning was performed.")
    print("No feature was removed or selected.")
    print("No resampling was performed.")
    print("No model artifact was saved.")
    print("fraudTest.csv was never read.")
    print("No raw data was modified.")
    print("No processed dataset was modified.")

    print()
    print("LOOP 008B COMPLETE")


if __name__ == "__main__":
    main()