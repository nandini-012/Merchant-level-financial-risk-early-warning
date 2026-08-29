from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "raw" / "fraudTrain.csv"


def main():
    print("=" * 80)
    print("LOOP 002D — TARGET CANDIDATE ANALYSIS")
    print("=" * 80)

    usecols = [
        "trans_date_trans_time",
        "merchant",
        "amt",
        "is_fraud",
    ]

    print("\nReading training data...")

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=usecols,
        parse_dates=["trans_date_trans_time"],
    )

    df["date"] = df["trans_date_trans_time"].dt.floor("D")

    # --------------------------------------------------------
    # Merchant × day aggregation
    # --------------------------------------------------------

    daily = (
        df.groupby(["merchant", "date"], as_index=False)
        .agg(
            transactions=("is_fraud", "size"),
            fraud_transactions=("is_fraud", "sum"),
            total_amount=("amt", "sum"),
            fraud_amount=(
                "amt",
                lambda x: x[df.loc[x.index, "is_fraud"].eq(1)].sum()
            ),
        )
    )

    daily = daily.sort_values(
        ["merchant", "date"]
    ).reset_index(drop=True)

    grouped = daily.groupby("merchant", group_keys=False)

    # --------------------------------------------------------
    # PAST windows
    #
    # shift(1) is critical:
    # today's features cannot contain today's outcome.
    # --------------------------------------------------------

    daily["past_7_fraud"] = (
        grouped["fraud_transactions"]
        .transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).sum()
        )
    )

    daily["past_7_transactions"] = (
        grouped["transactions"]
        .transform(
            lambda s: s.shift(1).rolling(7, min_periods=7).sum()
        )
    )

    daily["past_14_fraud"] = (
        grouped["fraud_transactions"]
        .transform(
            lambda s: s.shift(1).rolling(14, min_periods=14).sum()
        )
    )

    daily["past_14_transactions"] = (
        grouped["transactions"]
        .transform(
            lambda s: s.shift(1).rolling(14, min_periods=14).sum()
        )
    )

    daily["past_7_rate"] = (
        daily["past_7_fraud"]
        / daily["past_7_transactions"]
    )

    daily["past_14_rate"] = (
        daily["past_14_fraud"]
        / daily["past_14_transactions"]
    )

    # --------------------------------------------------------
    # FUTURE 7-DAY OUTCOME
    #
    # This is the outcome we are trying to predict.
    # It is intentionally NOT used as a model feature.
    # --------------------------------------------------------

    daily["future_7_fraud"] = (
        grouped["fraud_transactions"]
        .transform(
            lambda s: s.shift(-1)
            .rolling(7, min_periods=7)
            .sum()
            .shift(-6)
        )
    )

    daily["future_7_amount"] = (
        grouped["fraud_amount"]
        .transform(
            lambda s: s.shift(-1)
            .rolling(7, min_periods=7)
            .sum()
            .shift(-6)
        )
    )

    daily["future_7_transactions"] = (
        grouped["transactions"]
        .transform(
            lambda s: s.shift(-1)
            .rolling(7, min_periods=7)
            .sum()
            .shift(-6)
        )
    )

    daily["future_7_rate"] = (
        daily["future_7_fraud"]
        / daily["future_7_transactions"]
    )

    analysis = daily.dropna(
        subset=[
            "past_7_fraud",
            "past_14_fraud",
            "future_7_fraud",
            "future_7_amount",
            "future_7_rate",
        ]
    ).copy()

    print(f"\nUsable observations: {len(analysis):,}")

    # ========================================================
    # CANDIDATE 1 — FUTURE FRAUD COUNT
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 1 — FUTURE FRAUD COUNT")
    print("=" * 80)

    print(
        analysis["future_7_fraud"]
        .describe()
        .to_string()
    )

    for threshold in [1, 2, 3, 4, 5]:
        positive = (
            analysis["future_7_fraud"] >= threshold
        )

        print(
            f"\nTarget: future_7_fraud >= {threshold}"
        )
        print(f"Positive: {positive.sum():,}")
        print(f"Negative: {(~positive).sum():,}")
        print(f"Positive rate: {positive.mean():.4%}")

    # ========================================================
    # CANDIDATE 2 — FUTURE FRAUD AMOUNT
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 2 — FUTURE FRAUD AMOUNT")
    print("=" * 80)

    print(
        analysis["future_7_amount"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    amount_thresholds = [
        0,
        250,
        500,
        750,
        1000,
        1500,
        2000,
    ]

    for threshold in amount_thresholds:
        positive = (
            analysis["future_7_amount"] >= threshold
        )

        print(
            f"\nTarget: future_7_fraud_amount >= "
            f"{threshold}"
        )
        print(f"Positive: {positive.sum():,}")
        print(f"Positive rate: {positive.mean():.4%}")

    # ========================================================
    # CANDIDATE 3 — MERCHANT-RELATIVE FUTURE FRAUD RATE
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 3 — FUTURE RATE VS PAST BASELINE")
    print("=" * 80)

    analysis["rate_ratio"] = (
        analysis["future_7_rate"]
        / analysis["past_14_rate"].replace(0, np.nan)
    )

    valid_ratio = analysis["rate_ratio"].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    print("\nFuture 7-day rate / past 14-day rate:")
    print(
        valid_ratio.describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    for multiplier in [1.5, 2, 3, 4]:
        positive = (
            valid_ratio >= multiplier
        )

        print(
            f"\nTarget: future rate >= "
            f"{multiplier}x past 14-day rate"
        )
        print(f"Positive: {positive.sum():,}")
        print(f"Positive rate: {positive.mean():.4%}")

    # ========================================================
    # CANDIDATE 4 — FUTURE RATE ABOVE FIXED LEVEL
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 4 — ABSOLUTE FUTURE FRAUD RATE")
    print("=" * 80)

    for threshold in [
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.05,
    ]:
        positive = (
            analysis["future_7_rate"] >= threshold
        )

        print(
            f"\nTarget: future fraud rate >= "
            f"{threshold:.2%}"
        )
        print(f"Positive: {positive.sum():,}")
        print(f"Positive rate: {positive.mean():.4%}")

    # ========================================================
    # CANDIDATE 5 — FUTURE FRAUD COUNT VS PAST ACTIVITY
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 5 — FUTURE COUNT CONDITIONED ON PAST")
    print("=" * 80)

    analysis["past_7_ge_1"] = (
        analysis["past_7_fraud"] >= 1
    )

    analysis["past_7_ge_2"] = (
        analysis["past_7_fraud"] >= 2
    )

    analysis["past_14_ge_2"] = (
        analysis["past_14_fraud"] >= 2
    )

    conditions = [
        ("past_7_ge_1", "Past 7-day fraud >= 1"),
        ("past_7_ge_2", "Past 7-day fraud >= 2"),
        ("past_14_ge_2", "Past 14-day fraud >= 2"),
    ]

    for column, label in conditions:

        yes = analysis[analysis[column]]
        no = analysis[~analysis[column]]

        print(f"\n{label}")

        print(
            f"YES observations: {len(yes):,}"
        )
        print(
            f"NO observations:  {len(no):,}"
        )

        if len(yes):
            print(
                f"Mean future fraud — YES: "
                f"{yes['future_7_fraud'].mean():.4f}"
            )

        if len(no):
            print(
                f"Mean future fraud — NO:  "
                f"{no['future_7_fraud'].mean():.4f}"
            )

        if len(yes):
            print(
                f"Future >=2 fraud — YES: "
                f"{(yes['future_7_fraud'] >= 2).mean():.4%}"
            )

        if len(no):
            print(
                f"Future >=2 fraud — NO:  "
                f"{(no['future_7_fraud'] >= 2).mean():.4%}"
            )

    # ========================================================
    # CANDIDATE 6 — LOSS EXPOSURE
    # ========================================================

    print("\n" + "=" * 80)
    print("CANDIDATE 6 — FUTURE FRAUD LOSS EXPOSURE")
    print("=" * 80)

    print(
        "\nFuture 7-day fraud amount distribution:"
    )

    print(
        analysis["future_7_amount"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    # ========================================================
    # FINAL SAFETY CHECK
    # ========================================================

    print("\n" + "=" * 80)
    print("METHODOLOGY CHECK")
    print("=" * 80)

    print(
        "\nPast features use shift(1): "
        "current-day/future fraud is excluded."
    )

    print(
        "Future variables are outcomes only."
    )

    print(
        "Test data was NOT used."
    )

    print(
        "No model was trained."
    )

    print(
        "No target was permanently created."
    )

    print(
        "Raw data was NOT modified."
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()