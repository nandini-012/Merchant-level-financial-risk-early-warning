"""OPT-002: validation-only activity-shape feature comparison.

This analysis preserves the existing baseline and evaluates no holdout rows.
Every added feature is derived from merchant calendar-day counts shifted by one
day, so its latest input for a prediction date T is T-1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN_FILE = ROOT / "data" / "raw" / "fraudTrain.csv"
DATASET_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"

TRAIN_START = pd.Timestamp("2019-01-15")
TRAIN_END = pd.Timestamp("2020-02-28")
VALIDATION_START = pd.Timestamp("2020-03-01")
VALIDATION_END = pd.Timestamp("2020-04-30")
HOLDOUT_START = pd.Timestamp("2020-05-01")
VALIDATION_DAYS = 61
VALIDATION_ALERT_BUDGET = 1_220

BASELINE_FEATURE_COLUMNS = [
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

OPT_002_FEATURE_COLUMNS = [
    "previous_7d_active_day_count",
    "previous_14d_active_day_count",
    "previous_7d_maximum_daily_transaction_count",
    "previous_7d_daily_transaction_count_std",
    "previous_7d_daily_transaction_count_mean_zero",
    "previous_7d_daily_transaction_count_cv",
    "previous_7d_transactions_per_active_day",
]


def raw_metadata() -> tuple[int, int]:
    """Capture raw-file metadata to verify this read-only analysis is safe."""
    return (RAW_TRAIN_FILE.stat().st_size, RAW_TRAIN_FILE.stat().st_mtime_ns)


def past_rolling_sum(daily: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Calendar-day sum ending at T-1, with an explicit one-day shift."""
    return daily.groupby("merchant", sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window=window, min_periods=window).sum()
    )


def past_rolling_stat(
    daily: pd.DataFrame, column: str, window: int, statistic: str
) -> pd.Series:
    """Calendar-day statistic ending at T-1, with an explicit one-day shift."""
    def calculate(values: pd.Series) -> pd.Series:
        windowed = values.shift(1).rolling(window=window, min_periods=window)
        if statistic == "max":
            return windowed.max()
        if statistic == "std":
            return windowed.std(ddof=0)
        raise ValueError(f"Unsupported statistic: {statistic}")

    return daily.groupby("merchant", sort=False)[column].transform(calculate)


def read_observed_daily_through_validation() -> pd.DataFrame:
    """Read and aggregate only transaction inputs eligible through validation."""
    daily_parts = []
    for chunk in pd.read_csv(
        RAW_TRAIN_FILE,
        usecols=["trans_date_trans_time", "merchant"],
        parse_dates=["trans_date_trans_time"],
        chunksize=100_000,
    ):
        chunk["prediction_date"] = chunk["trans_date_trans_time"].dt.floor("D")
        eligible = chunk.loc[
            chunk["prediction_date"] <= VALIDATION_END,
            ["merchant", "prediction_date"],
        ]
        daily_parts.append(
            eligible.groupby(["merchant", "prediction_date"], as_index=False)
            .size()
            .rename(columns={"size": "daily_transaction_count"})
        )
    observed_daily = pd.concat(daily_parts, ignore_index=True)
    observed_daily = (
        observed_daily.groupby(["merchant", "prediction_date"], as_index=False)[
            "daily_transaction_count"
        ]
        .sum()
    )
    if observed_daily.empty:
        raise RuntimeError("No raw transaction records are available through validation.")
    if observed_daily["prediction_date"].max() > VALIDATION_END:
        raise RuntimeError("Holdout transaction rows entered OPT-002 feature construction.")
    return observed_daily


def build_opt_002_features(observed_daily: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the seven pre-registered OPT-002 features through validation."""
    merchants = observed_daily["merchant"].drop_duplicates().sort_values()
    dates = pd.date_range(observed_daily["prediction_date"].min(), VALIDATION_END, freq="D")
    calendar_index = pd.MultiIndex.from_product(
        [merchants, dates], names=["merchant", "prediction_date"]
    )
    daily = (
        observed_daily.set_index(["merchant", "prediction_date"])
        .reindex(calendar_index, fill_value=0)
        .reset_index()
        .sort_values(["merchant", "prediction_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["daily_active_day"] = (daily["daily_transaction_count"] > 0).astype("int64")

    daily["previous_7d_active_day_count"] = past_rolling_sum(
        daily, "daily_active_day", 7
    )
    daily["previous_14d_active_day_count"] = past_rolling_sum(
        daily, "daily_active_day", 14
    )
    daily["previous_7d_maximum_daily_transaction_count"] = past_rolling_stat(
        daily, "daily_transaction_count", 7, "max"
    )
    daily["previous_7d_daily_transaction_count_std"] = past_rolling_stat(
        daily, "daily_transaction_count", 7, "std"
    )
    previous_7d_count = past_rolling_sum(daily, "daily_transaction_count", 7)
    daily["previous_7d_daily_transaction_count_mean_zero"] = (
        previous_7d_count.eq(0).astype("int64")
    )
    daily["previous_7d_daily_transaction_count_cv"] = (
        daily["previous_7d_daily_transaction_count_std"]
        / (previous_7d_count / 7).where(previous_7d_count != 0)
    ).fillna(0.0)
    daily["previous_7d_transactions_per_active_day"] = (
        previous_7d_count
        / daily["previous_7d_active_day_count"].where(
            daily["previous_7d_active_day_count"] != 0
        )
    ).fillna(0.0)

    return daily[["merchant", "prediction_date", *OPT_002_FEATURE_COLUMNS]]


def threshold_for_budget(probabilities: pd.Series) -> float:
    """Use stable validation-only ranking to derive the OPT-002 threshold."""
    if len(probabilities) < VALIDATION_ALERT_BUDGET:
        raise RuntimeError("Validation contains fewer rows than the alert budget.")
    ranked = probabilities.sort_values(ascending=False, kind="mergesort")
    return float(ranked.iloc[VALIDATION_ALERT_BUDGET - 1])


def evaluate(
    name: str, model: HistGradientBoostingClassifier, features: list[str], train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Fit on train and evaluate a validation-only capacity operating point."""
    model.fit(train[features], train["target"])
    probabilities = pd.Series(model.predict_proba(validation[features])[:, 1], index=validation.index)
    threshold = threshold_for_budget(probabilities)
    predictions = (probabilities >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(validation["target"], predictions, labels=[0, 1]).ravel()
    alerts = int(predictions.sum())
    return {
        "candidate": name,
        "threshold": threshold,
        "pr_auc": average_precision_score(validation["target"], probabilities),
        "precision": precision_score(validation["target"], predictions, zero_division=0),
        "recall": recall_score(validation["target"], predictions, zero_division=0),
        "f1": f1_score(validation["target"], predictions, zero_division=0),
        "alerts": alerts,
        "alerts_per_day": alerts / VALIDATION_DAYS,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def fixed_hgb() -> HistGradientBoostingClassifier:
    """Return the unchanged locked HGB configuration for both comparisons."""
    return HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, early_stopping=False
    )


def main() -> None:
    print("=" * 80)
    print("OPT-002 — VALIDATION-ONLY ACTIVITY-SHAPE FEATURE COMPARISON")
    print("=" * 80)
    raw_before = raw_metadata()
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    required_columns = {"merchant", "prediction_date", "target", *BASELINE_FEATURE_COLUMNS}
    if set(dataset.columns) != required_columns:
        raise RuntimeError("Training dataset schema differs from the locked baseline schema.")
    if dataset[BASELINE_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Baseline modelling features contain missing values.")

    observed_daily = read_observed_daily_through_validation()
    activity_features = build_opt_002_features(observed_daily)
    if activity_features["prediction_date"].max() > VALIDATION_END:
        raise RuntimeError("OPT-002 activity features extend into the holdout period.")
    if any("future" in column.lower() or "target" in column.lower() for column in OPT_002_FEATURE_COLUMNS):
        raise RuntimeError("OPT-002 feature names indicate target/future leakage.")

    modelling = dataset.loc[
        dataset["prediction_date"].between(TRAIN_START, VALIDATION_END)
    ].merge(activity_features, on=["merchant", "prediction_date"], how="left", validate="one_to_one")
    if modelling["prediction_date"].max() >= HOLDOUT_START:
        raise RuntimeError("Holdout rows entered the OPT-002 modelling frame.")
    if modelling[OPT_002_FEATURE_COLUMNS].isna().any().any():
        missing = modelling[OPT_002_FEATURE_COLUMNS].isna().sum()
        raise RuntimeError(f"OPT-002 features contain missing values: {missing[missing.gt(0)].to_dict()}")

    train = modelling.loc[modelling["prediction_date"].between(TRAIN_START, TRAIN_END)].copy()
    validation = modelling.loc[
        modelling["prediction_date"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    if train.empty or validation.empty:
        raise RuntimeError("Train or validation split is empty.")
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal ordering is invalid.")
    if validation["prediction_date"].nunique() != VALIDATION_DAYS:
        raise RuntimeError("Validation does not contain exactly 61 calendar days.")
    if set(train[["merchant", "prediction_date"]].itertuples(index=False, name=None)) & set(
        validation[["merchant", "prediction_date"]].itertuples(index=False, name=None)
    ):
        raise RuntimeError("Merchant/date keys overlap across train and validation.")

    baseline = evaluate(
        "OPT-001 baseline", fixed_hgb(), BASELINE_FEATURE_COLUMNS, train, validation
    )
    opt_002 = evaluate(
        "OPT-002 activity shape", fixed_hgb(),
        [*BASELINE_FEATURE_COLUMNS, *OPT_002_FEATURE_COLUMNS], train, validation
    )
    if raw_before != raw_metadata():
        raise RuntimeError("Raw training data metadata changed during analysis.")

    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation days: {validation['prediction_date'].nunique()}")
    print("Added OPT-002 features:")
    for column in OPT_002_FEATURE_COLUMNS:
        print(f"- {column}")
    print("\nVALIDATION RESULTS")
    print(
        pd.DataFrame([baseline, opt_002]).to_string(
            index=False,
            formatters={
                "threshold": "{:.12f}".format,
                "pr_auc": "{:.6f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "f1": "{:.6f}".format,
                "alerts_per_day": "{:.4f}".format,
            },
        )
    )
    print("\nVALIDATION CHECKS")
    print("Explicit shift(1) before every OPT-002 rolling aggregation: confirmed")
    print("Zero-denominator handling: confirmed")
    print("No missing OPT-002 modelling values: confirmed")
    print("No target/future-derived OPT-002 feature: confirmed")
    print("Holdout was not inspected, scored, or used for selection: confirmed")
    print("fraudTest.csv was not read")
    print("Raw training data was not modified")
    print("Existing datasets and outputs were not modified")
    print("No model configuration or threshold was changed")
    print("OPT-002 was not selected")


if __name__ == "__main__":
    main()
