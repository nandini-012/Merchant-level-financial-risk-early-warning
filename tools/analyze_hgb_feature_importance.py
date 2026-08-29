"""LOOP 008A: HGB feature-importance analysis.

Analysis only. Uses the locked training period and the fixed HGB configuration.
No validation or holdout data is loaded or evaluated.
No threshold selection, tuning, feature selection, resampling, or artifacts.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


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


def main() -> None:
    print("=" * 80)
    print("LOOP 008A — HGB FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    expected_columns = {"merchant", "prediction_date", "target", *FEATURE_COLUMNS}
    if set(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the locked modelling schema.")

    if "target" in FEATURE_COLUMNS:
        raise RuntimeError("Target must not be included among model features.")

    future_features = [
        column for column in FEATURE_COLUMNS if "future" in column.lower()
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

    if train.empty:
        raise RuntimeError("Locked training period contains no observations.")

    print("\nSAFETY CHECKS")
    print("-" * 80)
    print("Training period only: confirmed")
    print("Validation data not loaded: confirmed")
    print("Holdout data not loaded: confirmed")
    print("Target excluded from model features: confirmed")
    print("No future-derived features: confirmed")
    print("No missing model features: confirmed")
    print("No threshold selection: confirmed")
    print("No tuning: confirmed")
    print("No feature selection: confirmed")
    print("No resampling: confirmed")
    print("No model artifact will be saved: confirmed")
    print("Raw/processed datasets will not be modified: confirmed")
    print("fraudTest.csv not read: confirmed")

    print("\nLOCKED MODEL")
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

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    model.fit(X_train, y_train)

    print("\nTRAINING DATA")
    print("-" * 80)
    print(f"Rows: {len(train):,}")
    print(f"Positive targets: {int(y_train.sum()):,}")
    print(f"Positive rate: {y_train.mean():.6%}")
    print(f"Features analyzed: {len(FEATURE_COLUMNS)}")

    print("\nPERMUTATION IMPORTANCE")
    print("-" * 80)
    print("Scoring metric: average precision (PR-AUC)")
    print("Permutations: 10")
    print("random_state=42")
    print("Importance = decrease in training PR-AUC after permutation")

    result = permutation_importance(
        model,
        X_train,
        y_train,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
        kind="mergesort",
    )

    print("\nFEATURE RANKING")
    print("-" * 80)

    for rank, row in enumerate(importance.itertuples(index=False), start=1):
        print(
            f"{rank:2d}. {row.feature:<45} "
            f"mean={row.importance_mean:.8f} "
            f"std={row.importance_std:.8f}"
        )

    print("\nFINAL SAFETY CONFIRMATIONS")
    print("-" * 80)
    print("Only locked training data was used for importance analysis.")
    print("Validation data was not loaded.")
    print("Holdout data was not loaded.")
    print("No threshold was selected.")
    print("No tuning was performed.")
    print("No feature was removed or selected.")
    print("No resampling was performed.")
    print("No model artifact was saved.")
    print("fraudTest.csv was never read.")
    print("No raw data was modified.")
    print("No processed dataset was modified.")

    print("\nLOOP 008A COMPLETE")


if __name__ == "__main__":
    main()