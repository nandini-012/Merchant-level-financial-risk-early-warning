from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TRAIN_FILE = RAW / "fraudTrain.csv"


def main():
    print("=" * 80)
    print("LOOP 002B — FRAUD CLUSTERING ANALYSIS")
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

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["merchant", "trans_date_trans_time"]
    ).reset_index(drop=True)

    df["date"] = df["trans_date_trans_time"].dt.floor("D")

    # --------------------------------------------------------
    # Aggregate merchant × day
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

    daily["fraud_rate"] = (
        daily["fraud_transactions"]
        / daily["transactions"]
    )

    print(f"Merchant-day rows: {len(daily):,}")

    # --------------------------------------------------------
    # Rolling fraud behaviour
    # --------------------------------------------------------

    daily = daily.sort_values(
        ["merchant", "date"]
    ).reset_index(drop=True)

    grouped = daily.groupby("merchant", group_keys=False)

    for window in [3, 7, 14]:

        daily[f"fraud_{window}d"] = (
            grouped["fraud_transactions"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        daily[f"transactions_{window}d"] = (
            grouped["transactions"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        daily[f"fraud_rate_{window}d"] = (
            daily[f"fraud_{window}d"]
            / daily[f"transactions_{window}d"]
        )

        print("\n" + "-" * 80)
        print(f"{window}-DAY WINDOW")
        print("-" * 80)

        print(
            f"Total merchant-days with >=1 fraud in "
            f"previous/current {window} days: "
            f"{(daily[f'fraud_{window}d'] >= 1).sum():,}"
        )

        print(
            f"Merchant-days with >=2 fraud in "
            f"previous/current {window} days: "
            f"{(daily[f'fraud_{window}d'] >= 2).sum():,}"
        )

        print(
            f"Merchant-days with >=3 fraud in "
            f"previous/current {window} days: "
            f"{(daily[f'fraud_{window}d'] >= 3).sum():,}"
        )

        print(
            f"Merchant-days with >=5 fraud in "
            f"previous/current {window} days: "
            f"{(daily[f'fraud_{window}d'] >= 5).sum():,}"
        )

        print(
            f"Maximum fraud count in {window}-day window: "
            f"{daily[f'fraud_{window}d'].max():.0f}"
        )

        print(
            f"Maximum fraud rate in {window}-day window: "
            f"{daily[f'fraud_rate_{window}d'].max():.4%}"
        )

    # --------------------------------------------------------
    # Fraud amount concentration
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FRAUD AMOUNT DISTRIBUTION")
    print("=" * 80)

    fraud_only = df[df["is_fraud"] == 1]

    print(f"\nFraud transactions: {len(fraud_only):,}")

    print("\nFraud transaction amount:")
    print(
        fraud_only["amt"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Merchant concentration
    # --------------------------------------------------------

    merchant = (
        df.groupby("merchant")
        .agg(
            transactions=("is_fraud", "size"),
            fraud_transactions=("is_fraud", "sum"),
            total_amount=("amt", "sum"),
        )
        .reset_index()
    )

    merchant["fraud_rate"] = (
        merchant["fraud_transactions"]
        / merchant["transactions"]
    )

    merchant = merchant.sort_values(
        "fraud_transactions",
        ascending=False
    )

    print("\n" + "=" * 80)
    print("TOP MERCHANTS BY FRAUD COUNT")
    print("=" * 80)

    print(
        merchant[
            [
                "merchant",
                "transactions",
                "fraud_transactions",
                "fraud_rate",
            ]
        ]
        .head(20)
        .to_string(index=False)
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