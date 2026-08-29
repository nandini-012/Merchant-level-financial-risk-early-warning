from pathlib import Path
import pandas as pd


# ============================================================
# LOOP 002A — MERCHANT/FRAUD TEMPORAL ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TRAIN_FILE = RAW / "fraudTrain.csv"
TEST_FILE = RAW / "fraudTest.csv"


def inspect_file(path: Path, name: str):
    print("\n" + "=" * 80)
    print(f"{name}: {path.name}")
    print("=" * 80)

    usecols = [
        "trans_date_trans_time",
        "merchant",
        "amt",
        "is_fraud",
    ]

    # Read only columns needed for this analysis.
    df = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=["trans_date_trans_time"],
    )

    print(f"\nRows: {len(df):,}")
    print(f"Date range: {df['trans_date_trans_time'].min()} -> "
          f"{df['trans_date_trans_time'].max()}")

    print(f"Unique merchants: {df['merchant'].nunique():,}")

    # --------------------------------------------------------
    # Overall fraud statistics
    # --------------------------------------------------------

    fraud_count = int(df["is_fraud"].sum())
    total_count = len(df)

    print("\n--- Overall Fraud ---")
    print(f"Fraud transactions: {fraud_count:,}")
    print(f"Legitimate transactions: {total_count - fraud_count:,}")
    print(f"Fraud rate: {fraud_count / total_count:.4%}")

    # --------------------------------------------------------
    # Merchant-level aggregate statistics
    # --------------------------------------------------------

    merchant_stats = (
        df.groupby("merchant")
        .agg(
            transactions=("is_fraud", "size"),
            fraud_transactions=("is_fraud", "sum"),
            total_amount=("amt", "sum"),
            fraud_amount=("amt", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()),
        )
        .reset_index()
    )

    merchant_stats["fraud_rate"] = (
        merchant_stats["fraud_transactions"]
        / merchant_stats["transactions"]
    )

    print("\n--- Merchant-Level Fraud Distribution ---")

    print("\nFraud transactions per merchant:")
    print(merchant_stats["fraud_transactions"].describe().to_string())

    print("\nMerchant fraud-rate distribution:")
    print(merchant_stats["fraud_rate"].describe().to_string())

    affected = merchant_stats[
        merchant_stats["fraud_transactions"] > 0
    ]

    print("\nMerchants with at least one fraud transaction:",
          len(affected))

    print("Merchants with zero fraud transactions:",
          len(merchant_stats) - len(affected))

    # --------------------------------------------------------
    # Daily merchant behaviour
    # --------------------------------------------------------

    df["date"] = df["trans_date_trans_time"].dt.date

    daily = (
        df.groupby(["merchant", "date"])
        .agg(
            transactions=("is_fraud", "size"),
            fraud_transactions=("is_fraud", "sum"),
            total_amount=("amt", "sum"),
            fraud_amount=("amt", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()),
        )
        .reset_index()
    )

    daily["fraud_rate"] = (
        daily["fraud_transactions"]
        / daily["transactions"]
    )

    print("\n--- Merchant × Day ---")

    print(f"Merchant-day observations: {len(daily):,}")

    print("\nTransactions per merchant-day:")
    print(daily["transactions"].describe().to_string())

    print("\nFraud transactions per merchant-day:")
    print(daily["fraud_transactions"].describe().to_string())

    print("\nMerchant-days with >= 1 fraud:")
    print(
        (daily["fraud_transactions"] >= 1).sum()
    )

    print("\nMerchant-days with >= 2 fraud:")
    print(
        (daily["fraud_transactions"] >= 2).sum()
    )

    print("\nMerchant-days with >= 3 fraud:")
    print(
        (daily["fraud_transactions"] >= 3).sum()
    )

    print("\nMerchant-days with >= 5 fraud:")
    print(
        (daily["fraud_transactions"] >= 5).sum()
    )

    # --------------------------------------------------------
    # Distribution of fraud count in merchant-days
    # --------------------------------------------------------

    fraud_days = daily[daily["fraud_transactions"] > 0].copy()

    print("\n--- Fraud-Positive Merchant-Days ---")

    if len(fraud_days) > 0:
        print(
            fraud_days["fraud_transactions"]
            .value_counts()
            .sort_index()
            .head(20)
            .to_string()
        )

    # --------------------------------------------------------
    # Monthly merchant behaviour
    # --------------------------------------------------------

    df["month"] = (
        df["trans_date_trans_time"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        df.groupby(["merchant", "month"])
        .agg(
            transactions=("is_fraud", "size"),
            fraud_transactions=("is_fraud", "sum"),
            total_amount=("amt", "sum"),
            fraud_amount=("amt", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()),
        )
        .reset_index()
    )

    monthly["fraud_rate"] = (
        monthly["fraud_transactions"]
        / monthly["transactions"]
    )

    print("\n--- Merchant × Month ---")
    print(f"Merchant-month observations: {len(monthly):,}")

    print("\nFraud transactions per merchant-month:")
    print(
        monthly["fraud_transactions"]
        .describe()
        .to_string()
    )

    print("\nMerchant-months with >= 1 fraud:")
    print(
        (monthly["fraud_transactions"] >= 1).sum()
    )

    print("\nMerchant-months with >= 2 fraud:")
    print(
        (monthly["fraud_transactions"] >= 2).sum()
    )

    print("\nMerchant-months with >= 3 fraud:")
    print(
        (monthly["fraud_transactions"] >= 3).sum()
    )

    print("\nMerchant-months with >= 5 fraud:")
    print(
        (monthly["fraud_transactions"] >= 5).sum()
    )

    # --------------------------------------------------------
    # Train/test merchant overlap
    # --------------------------------------------------------

    return {
        "df": df,
        "merchant_stats": merchant_stats,
        "daily": daily,
        "monthly": monthly,
    }


def main():
    train = inspect_file(TRAIN_FILE, "TRAIN")
    test = inspect_file(TEST_FILE, "TEST")

    train_merchants = set(train["df"]["merchant"].unique())
    test_merchants = set(test["df"]["merchant"].unique())

    overlap = train_merchants & test_merchants

    print("\n" + "=" * 80)
    print("TRAIN / TEST MERCHANT OVERLAP")
    print("=" * 80)

    print(f"Train merchants: {len(train_merchants):,}")
    print(f"Test merchants: {len(test_merchants):,}")
    print(f"Shared merchants: {len(overlap):,}")

    print("\nNO MODEL WAS TRAINED.")
    print("NO TARGET WAS CREATED.")
    print("NO TRAIN/TEST DATA WAS MODIFIED.")


if __name__ == "__main__":
    main()