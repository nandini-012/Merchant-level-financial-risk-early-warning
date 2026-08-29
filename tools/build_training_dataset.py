"""Build the LOOP 004 training dataset from historical features and train data.

The target for merchant M on prediction date T is one when M has at least two
fraudulent transactions on T+1 through T+7.  Future outcome values are used
only internally and are never written to the final dataset.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_FILE = RAW_DIR / "fraudTrain.csv"
FEATURE_FILE = ROOT / "data" / "processed" / "merchant_features_train.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"
KEY_COLUMNS = ["merchant", "prediction_date"]
TARGET_COLUMN = "target"


def raw_file_metadata() -> dict[Path, tuple[int, int]]:
    """Capture raw-file metadata without opening or changing their contents."""
    return {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in RAW_DIR.iterdir()
        if path.is_file()
    }


def future_7_day_fraud_count(outcomes: pd.DataFrame) -> pd.Series:
    """Sum fraud in T+1 through T+7 over an ordered complete daily grid."""
    return outcomes.groupby("merchant", sort=False)["daily_fraud_count"].transform(
        lambda values: values.shift(-1).rolling(window=7, min_periods=7).sum().shift(-6)
    )


def main() -> None:
    raw_before = raw_file_metadata()

    print("=" * 80)
    print("LOOP 004 — BUILD TRAINING DATASET")
    print("=" * 80)
    print(f"Reading historical features: {FEATURE_FILE}")
    features = pd.read_parquet(FEATURE_FILE).sort_values(KEY_COLUMNS).reset_index(drop=True)

    if features.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("Feature data contains duplicate merchant/date rows.")

    feature_columns = [column for column in features if column not in KEY_COLUMNS]
    if not feature_columns:
        raise RuntimeError("Feature data has no historical feature columns.")

    leakage_columns = [
        column
        for column in feature_columns
        if "future" in column.lower() or "target" in column.lower()
    ]
    if leakage_columns:
        raise RuntimeError(f"Future/target feature columns are not permitted: {leakage_columns}")

    merchants = features["merchant"].drop_duplicates().sort_values()
    dates = pd.date_range(
        features["prediction_date"].min(),
        features["prediction_date"].max(),
        freq="D",
    )
    expected_grid_rows = len(merchants) * len(dates)
    if len(features) != expected_grid_rows:
        raise RuntimeError("Feature data is not a complete merchant × calendar-date grid.")

    print(f"Reading training data only for activity and future outcome: {TRAIN_FILE}")
    transactions = pd.read_csv(
        TRAIN_FILE,
        usecols=["trans_date_trans_time", "merchant", "is_fraud"],
        parse_dates=["trans_date_trans_time"],
    )
    transactions["prediction_date"] = transactions["trans_date_trans_time"].dt.floor("D")
    observed_daily = (
        transactions.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            current_day_transaction_count=("is_fraud", "size"),
            daily_fraud_count=("is_fraud", "sum"),
        )
    )

    # The existing feature grid supplies every merchant/calendar date.  Thus,
    # absent raw merchant-days within this grid are genuine zero-activity days.
    outcomes = features[KEY_COLUMNS].merge(
        observed_daily,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    outcomes["current_day_transaction_count"] = (
        outcomes["current_day_transaction_count"].fillna(0).astype("int64")
    )
    outcomes["daily_fraud_count"] = outcomes["daily_fraud_count"].fillna(0).astype("int64")
    outcomes = outcomes.sort_values(KEY_COLUMNS).reset_index(drop=True)
    outcomes["future_7d_fraud_count"] = future_7_day_fraud_count(outcomes)

    complete_future_window = (
        outcomes["prediction_date"] <= dates.max() - pd.Timedelta(days=7)
    )
    if not outcomes.loc[complete_future_window, "future_7d_fraud_count"].notna().all():
        raise RuntimeError("A required future 7-day outcome window is incomplete.")
    if outcomes.loc[~complete_future_window, "future_7d_fraud_count"].notna().any():
        raise RuntimeError("A truncated future window produced an outcome value.")

    sufficient_history = features[feature_columns].notna().all(axis=1)
    outcomes["sufficient_history"] = sufficient_history.to_numpy()
    outcomes["target"] = (
        outcomes["future_7d_fraud_count"] >= 2
    ).astype("Int64")

    eligible = (
        outcomes["current_day_transaction_count"].gt(0)
        & outcomes["sufficient_history"]
        & complete_future_window
    )
    training = features.loc[eligible, [*KEY_COLUMNS, *feature_columns]].copy()
    training[TARGET_COLUMN] = outcomes.loc[eligible, TARGET_COLUMN].astype("int64").to_numpy()

    final_future_columns = [
        column for column in training.columns if "future" in column.lower()
    ]
    if final_future_columns:
        raise RuntimeError(f"Future-derived columns leaked into final data: {final_future_columns}")
    if training[feature_columns].isna().any().any():
        raise RuntimeError("Saved training rows contain insufficient historical features.")
    if not outcomes.loc[eligible, "current_day_transaction_count"].gt(0).all():
        raise RuntimeError("A saved row has no transaction on its prediction date.")
    if not complete_future_window.loc[eligible].all():
        raise RuntimeError("A saved row lacks a complete future outcome window.")

    # Independently validate target values from raw transactions for 10 sampled
    # saved observations.  The current date is intentionally excluded here.
    validation_sample = training.sample(n=10, random_state=20260829)
    validation_passes = 0
    print("\nTarget validation samples:")
    for row in validation_sample.itertuples(index=False):
        future_start = row.prediction_date + pd.Timedelta(days=1)
        future_end = row.prediction_date + pd.Timedelta(days=7)
        fraud_count = int(
            transactions.loc[
                (transactions["merchant"] == row.merchant)
                & transactions["prediction_date"].between(future_start, future_end),
                "is_fraud",
            ].sum()
        )
        expected_target = int(fraud_count >= 2)
        matches = expected_target == row.target
        validation_passes += int(matches)
        print(
            f"merchant={row.merchant!r}, T={row.prediction_date.date()}, "
            f"future window={future_start.date()}..{future_end.date()}, "
            f"future fraud={fraud_count}, target={row.target}, match={matches}"
        )
    if validation_passes != len(validation_sample):
        raise RuntimeError("Independent target validation failed.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    training.to_parquet(OUTPUT_FILE, index=False)

    raw_after = raw_file_metadata()
    if raw_before != raw_after:
        raise RuntimeError("A file under data/raw changed during execution.")

    positives = int(training[TARGET_COLUMN].sum())
    negatives = len(training) - positives
    print("\nVALIDATION")
    print(f"Total feature rows before filtering: {len(features):,}")
    print(
        "Rows with transactions on T: "
        f"{int(outcomes['current_day_transaction_count'].gt(0).sum()):,}"
    )
    print(f"Rows with sufficient historical features: {int(sufficient_history.sum()):,}")
    print(f"Rows with complete future 7-day window: {int(complete_future_window.sum()):,}")
    print(f"Final modelling row count: {len(training):,}")
    print(f"Positive target count: {positives:,}")
    print(f"Negative target count: {negatives:,}")
    print(f"Positive target percentage: {positives / len(training):.4%}")
    print(f"Unique merchant count: {training['merchant'].nunique():,}")
    print(
        "Prediction-date range: "
        f"{training['prediction_date'].min().date()} -> "
        f"{training['prediction_date'].max().date()}"
    )
    print(f"Target column name: {TARGET_COLUMN}")
    print("Final feature columns:")
    for column in feature_columns:
        print(f"- {column}")
    print("No target leakage: confirmed")
    print("Current-day exclusion from target: confirmed (T+1 through T+7 only)")
    print(f"Future-window validation: {validation_passes}/{len(validation_sample)} passed")
    print("Observation-rule validation: confirmed")
    print("fraudTest.csv was not read: confirmed")
    print("No file under data/raw was modified: confirmed")
    print(f"Output written: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
