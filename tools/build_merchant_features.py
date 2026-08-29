"""Build LOOP 003 merchant-day features from the training CSV only.

Every feature for prediction date T is derived from merchant-day values ending
on T-1.  This script deliberately creates no target or future-looking column.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "raw" / "fraudTrain.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "merchant_features_train.parquet"

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


def past_rolling_sum(daily: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Return the calendar-day rolling sum ending at T-1 for each merchant."""
    return daily.groupby("merchant", sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window=window, min_periods=window).sum()
    )


def past_rolling_max(daily: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Return the maximum observed value in the calendar window ending at T-1."""
    return daily.groupby("merchant", sort=False)[column].transform(
        # min_periods=1 permits zero-transaction days, represented by NaN
        # daily maxima, without invalidating a window that has another amount.
        lambda values: values.shift(1).rolling(window=window, min_periods=1).max()
    )


def main() -> None:
    raw_before = (TRAIN_FILE.stat().st_size, TRAIN_FILE.stat().st_mtime_ns)

    print("=" * 80)
    print("LOOP 003 — BUILD HISTORICAL MERCHANT-DAY FEATURES")
    print("=" * 80)
    print(f"Reading training data only: {TRAIN_FILE}")

    transactions = pd.read_csv(
        TRAIN_FILE,
        usecols=["trans_date_trans_time", "merchant", "amt", "is_fraud"],
        parse_dates=["trans_date_trans_time"],
    )
    transactions["prediction_date"] = transactions["trans_date_trans_time"].dt.floor("D")

    observed_daily = (
        transactions.groupby(["merchant", "prediction_date"], as_index=False)
        .agg(
            daily_transaction_count=("is_fraud", "size"),
            daily_fraud_count=("is_fraud", "sum"),
            daily_total_transaction_amount=("amt", "sum"),
            daily_maximum_transaction_amount=("amt", "max"),
        )
    )

    # Reindex to every merchant and every calendar day.  Missing merchant-days
    # have zero transactions/fraud/amount; their daily maximum is undefined.
    merchants = observed_daily["merchant"].drop_duplicates().sort_values()
    dates = pd.date_range(
        observed_daily["prediction_date"].min(),
        observed_daily["prediction_date"].max(),
        freq="D",
    )
    index = pd.MultiIndex.from_product(
        [merchants, dates], names=["merchant", "prediction_date"]
    )
    daily = (
        observed_daily.set_index(["merchant", "prediction_date"])
        .reindex(index)
        .reset_index()
    )
    for column in [
        "daily_transaction_count",
        "daily_fraud_count",
        "daily_total_transaction_amount",
    ]:
        daily[column] = daily[column].fillna(0)

    # The frame is explicitly ordered before group-wise shifting and rolling.
    daily = daily.sort_values(["merchant", "prediction_date"]).reset_index(drop=True)

    for window in (1, 3, 7, 14):
        daily[f"previous_{window}d_transaction_count"] = past_rolling_sum(
            daily, "daily_transaction_count", window
        )

    for window in (7, 14):
        daily[f"previous_{window}d_fraud_count"] = past_rolling_sum(
            daily, "daily_fraud_count", window
        )

    daily["previous_7d_fraud_rate"] = (
        daily["previous_7d_fraud_count"]
        / daily["previous_7d_transaction_count"].where(
            daily["previous_7d_transaction_count"] != 0
        )
    )
    daily["previous_14d_fraud_rate"] = (
        daily["previous_14d_fraud_count"]
        / daily["previous_14d_transaction_count"].where(
            daily["previous_14d_transaction_count"] != 0
        )
    )

    # "Preceding" is T-14 through T-8; "previous" is T-7 through T-1.
    daily["preceding_7d_transaction_count"] = (
        daily.groupby("merchant", sort=False)["previous_7d_transaction_count"]
        .shift(7)
    )
    daily["previous_7d_transaction_count_change"] = (
        daily["previous_7d_transaction_count"]
        - daily["preceding_7d_transaction_count"]
    )
    daily["preceding_7d_fraud_count"] = (
        daily.groupby("merchant", sort=False)["previous_7d_fraud_count"].shift(7)
    )
    daily["previous_7d_fraud_count_change"] = (
        daily["previous_7d_fraud_count"] - daily["preceding_7d_fraud_count"]
    )

    daily["previous_7d_total_transaction_amount"] = past_rolling_sum(
        daily, "daily_total_transaction_amount", 7
    )
    daily["previous_7d_average_transaction_amount"] = (
        daily["previous_7d_total_transaction_amount"]
        / daily["previous_7d_transaction_count"].where(
            daily["previous_7d_transaction_count"] != 0
        )
    )
    daily["previous_7d_maximum_transaction_amount"] = past_rolling_max(
        daily, "daily_maximum_transaction_amount", 7
    ).where(daily["previous_7d_transaction_count"].notna())

    output = daily[["merchant", "prediction_date", *FEATURE_COLUMNS]].copy()
    future_columns = [
        column
        for column in output.columns
        if "future" in column.lower() or "target" in column.lower()
    ]
    if future_columns:
        raise RuntimeError(f"Future/target columns are not permitted: {future_columns}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_FILE, index=False)

    raw_after = (TRAIN_FILE.stat().st_size, TRAIN_FILE.stat().st_mtime_ns)
    if raw_before != raw_after:
        raise RuntimeError("The raw training file metadata changed during execution.")

    print("\nVALIDATION")
    print(f"Row count: {len(output):,}")
    print(f"Merchant count: {output['merchant'].nunique():,}")
    print(
        "Date range: "
        f"{output['prediction_date'].min().date()} -> "
        f"{output['prediction_date'].max().date()}"
    )
    print("Feature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")
    print("Missing-value counts:")
    print(output.isna().sum().to_string())
    print("No future columns/features exist: confirmed")
    print("fraudTest.csv was not read: confirmed")
    print("data/raw was not modified: confirmed")
    print(f"Output written: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
