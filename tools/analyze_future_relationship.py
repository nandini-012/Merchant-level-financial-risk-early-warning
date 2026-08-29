from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "raw" / "fraudTrain.csv"


def main():
    print("=" * 80)
    print("LOOP 002C — PAST VS FUTURE MERCHANT FRAUD")
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
    # Past 7-day behaviour
    # --------------------------------------------------------

    daily["past_7_fraud"] = (
        grouped["fraud_transactions"]
        .rolling(7, min_periods=7)
        .sum()
        .reset_index(level=0, drop=True)
    )

    daily["past_7_transactions"] = (
        grouped["transactions"]
        .rolling(7, min_periods=7)
        .sum()
        .reset_index(level=0, drop=True)
    )

    daily["past_7_rate"] = (
        daily["past_7_fraud"]
        / daily["past_7_transactions"]
    )

    # --------------------------------------------------------
    # Past 14-day behaviour
    # --------------------------------------------------------

    daily["past_14_fraud"] = (
        grouped["fraud_transactions"]
        .rolling(14, min_periods=14)
        .sum()
        .reset_index(level=0, drop=True)
    )

    daily["past_14_transactions"] = (
        grouped["transactions"]
        .rolling(14, min_periods=14)
        .sum()
        .reset_index(level=0, drop=True)
    )

    daily["past_14_rate"] = (
        daily["past_14_fraud"]
        / daily["past_14_transactions"]
    )

    # --------------------------------------------------------
    # FUTURE 7-day behaviour
    #
    # IMPORTANT:
    # This uses ONLY days AFTER the current observation.
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
            "past_7_rate",
            "past_14_rate",
            "future_7_rate",
        ]
    ).copy()

    print(f"\nUsable merchant-time observations: {len(analysis):,}")

    # --------------------------------------------------------
    # Basic future fraud distribution
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("FUTURE 7-DAY FRAUD DISTRIBUTION")
    print("-" * 80)

    print(
        analysis["future_7_fraud"]
        .describe()
        .to_string()
    )

    for threshold in [1, 2, 3, 4, 5]:
        count = (
            analysis["future_7_fraud"] >= threshold
        ).sum()

        percentage = count / len(analysis)

        print(
            f"Future 7-day fraud >= {threshold}: "
            f"{count:,} ({percentage:.2%})"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("PAST → FUTURE RELATIONSHIP")
    print("-" * 80)

    print(
        "\nCorrelation: past 7-day fraud count vs "
        "future 7-day fraud count:"
    )

    print(
        analysis["past_7_fraud"]
        .corr(analysis["future_7_fraud"])
    )

    print(
        "\nCorrelation: past 14-day fraud count vs "
        "future 7-day fraud count:"
    )

    print(
        analysis["past_14_fraud"]
        .corr(analysis["future_7_fraud"])
    )

    print(
        "\nCorrelation: past 7-day fraud rate vs "
        "future 7-day fraud rate:"
    )

    print(
        analysis["past_7_rate"]
        .corr(analysis["future_7_rate"])
    )

    print(
        "\nCorrelation: past 14-day fraud rate vs "
        "future 7-day fraud rate:"
    )

    print(
        analysis["past_14_rate"]
        .corr(analysis["future_7_rate"])
    )

    # --------------------------------------------------------
    # Compare future fraud based on past fraud activity
    # --------------------------------------------------------

    analysis["past_7_has_fraud"] = (
        analysis["past_7_fraud"] > 0
    )

    analysis["past_7_high_activity"] = (
        analysis["past_7_fraud"] >= 2
    )

    print("\n" + "-" * 80)
    print("CONDITIONAL FUTURE FRAUD")
    print("-" * 80)

    for condition, label in [
        ("past_7_has_fraud", "Past 7 days had >=1 fraud"),
        ("past_7_high_activity", "Past 7 days had >=2 fraud"),
    ]:

        subset_yes = analysis[analysis[condition]]
        subset_no = analysis[~analysis[condition]]

        print(f"\n{label}")

        print(
            f"Observations YES: {len(subset_yes):,}"
        )

        print(
            f"Observations NO:  {len(subset_no):,}"
        )

        if len(subset_yes):
            print(
                "Mean future 7-day fraud (YES): "
                f"{subset_yes['future_7_fraud'].mean():.4f}"
            )

        if len(subset_no):
            print(
                "Mean future 7-day fraud (NO):  "
                f"{subset_no['future_7_fraud'].mean():.4f}"
            )

        if len(subset_yes):
            print(
                "Future >=2 fraud rate (YES): "
                f"{(subset_yes['future_7_fraud'] >= 2).mean():.2%}"
            )

        if len(subset_no):
            print(
                "Future >=2 fraud rate (NO):  "
                f"{(subset_no['future_7_fraud'] >= 2).mean():.2%}"
            )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nNo model was trained.")
    print("No target was created.")
    print("Test data was NOT used.")
    print("Raw data was NOT modified.")


if __name__ == "__main__":
    main()