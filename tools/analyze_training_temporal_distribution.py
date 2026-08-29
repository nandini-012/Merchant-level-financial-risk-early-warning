"""Descriptive temporal target-distribution audit for LOOP 005A."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"


def print_table(frame: pd.DataFrame) -> None:
    print(frame.to_string(index=False))


def main() -> None:
    print("=" * 80)
    print("LOOP 005A — TEMPORAL TARGET DISTRIBUTION AUDIT")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")

    dataset = pd.read_parquet(DATASET_FILE, columns=["prediction_date", "target"])
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])

    if not set(dataset["target"].unique()).issubset({0, 1}):
        raise RuntimeError("Target must contain only 0 and 1 values.")

    total = len(dataset)
    positives = int(dataset["target"].sum())
    negatives = total - positives
    overall_rate = positives / total

    print("\nOVERALL")
    print(f"Total observations: {total:,}")
    print(f"Positive targets: {positives:,}")
    print(f"Negative targets: {negatives:,}")
    print(f"Positive rate: {overall_rate:.4%}")
    print(f"Earliest prediction date: {dataset['prediction_date'].min().date()}")
    print(f"Latest prediction date: {dataset['prediction_date'].max().date()}")

    dataset["month"] = dataset["prediction_date"].dt.to_period("M")
    monthly = (
        dataset.groupby("month", as_index=False)
        .agg(
            total_observations=("target", "size"),
            positive_targets=("target", "sum"),
        )
        .sort_values("month")
    )
    monthly["negative_targets"] = (
        monthly["total_observations"] - monthly["positive_targets"]
    )
    monthly["positive_rate"] = (
        monthly["positive_targets"] / monthly["total_observations"]
    )

    print("\nMONTHLY DISTRIBUTION")
    monthly_display = monthly.copy()
    monthly_display["month"] = monthly_display["month"].astype(str)
    monthly_display["positive_rate"] = monthly_display["positive_rate"].map(
        lambda value: f"{value:.4%}"
    )
    print_table(monthly_display)

    dataset["quarter"] = dataset["prediction_date"].dt.to_period("Q")
    quarterly = (
        dataset.groupby("quarter", as_index=False)
        .agg(
            total_observations=("target", "size"),
            positive_targets=("target", "sum"),
        )
        .sort_values("quarter")
    )
    quarterly["positive_rate"] = (
        quarterly["positive_targets"] / quarterly["total_observations"]
    )

    print("\nQUARTERLY DISTRIBUTION")
    quarterly_display = quarterly.copy()
    quarterly_display["quarter"] = quarterly_display["quarter"].astype(str)
    quarterly_display["positive_rate"] = quarterly_display["positive_rate"].map(
        lambda value: f"{value:.4%}"
    )
    print_table(quarterly_display)

    cumulative = monthly[["month", "total_observations", "positive_targets"]].copy()
    cumulative["cumulative_observations"] = cumulative["total_observations"].cumsum()
    cumulative["cumulative_positives"] = cumulative["positive_targets"].cumsum()
    cumulative["cumulative_observation_percentage"] = (
        cumulative["cumulative_observations"] / total
    )
    cumulative["cumulative_positive_percentage"] = (
        cumulative["cumulative_positives"] / positives
    )

    print("\nCUMULATIVE DISTRIBUTION")
    cumulative_display = cumulative.copy()
    cumulative_display["month"] = cumulative_display["month"].astype(str)
    for column in [
        "cumulative_observation_percentage",
        "cumulative_positive_percentage",
    ]:
        cumulative_display[column] = cumulative_display[column].map(
            lambda value: f"{value:.4%}"
        )
    print_table(cumulative_display)

    q1 = monthly["positive_rate"].quantile(0.25)
    q3 = monthly["positive_rate"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    high_months = monthly[monthly["positive_rate"] > upper_bound]
    low_months = monthly[monthly["positive_rate"] < lower_bound]

    print("\nRATE EXTREMES")
    print(
        "Monthly-rate IQR rule: "
        f"Q1={q1:.4%}, Q3={q3:.4%}, "
        f"lower bound={lower_bound:.4%}, upper bound={upper_bound:.4%}"
    )
    print(
        "Highest monthly rate: "
        f"{monthly.loc[monthly['positive_rate'].idxmax(), 'month']} "
        f"({monthly['positive_rate'].max():.4%})"
    )
    print(
        "Lowest monthly rate: "
        f"{monthly.loc[monthly['positive_rate'].idxmin(), 'month']} "
        f"({monthly['positive_rate'].min():.4%})"
    )
    print("High-rate IQR outlier months:")
    print_table(high_months.assign(month=high_months["month"].astype(str)))
    print("Low-rate IQR outlier months:")
    print_table(low_months.assign(month=low_months["month"].astype(str)))

    quarter_changes = quarterly.copy()
    quarter_changes["rate_change_from_prior_quarter"] = quarter_changes[
        "positive_rate"
    ].diff()
    largest_change = quarter_changes.loc[
        quarter_changes["rate_change_from_prior_quarter"].abs().idxmax()
    ]
    print("\nTEMPORAL REGIME CHECK")
    print(
        "Largest consecutive-quarter rate change: "
        f"{largest_change['quarter']} "
        f"({largest_change['rate_change_from_prior_quarter']:+.4%})"
    )
    print(
        "Descriptive regime conclusion: no obvious sustained regime change "
        "is identified from these monthly and quarterly rates."
    )
    print(
        "Interpretation: descriptive only; no formal change-point test or split "
        "decision was performed."
    )
    print("fraudTest.csv was not read.")
    print("No dataset or raw-data files were modified.")


if __name__ == "__main__":
    main()
