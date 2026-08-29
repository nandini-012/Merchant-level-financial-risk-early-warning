"""LOOP 009: generate locked-threshold merchant risk alerts for holdout days."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"
OUTPUT_FILE = ROOT / "data" / "outputs" / "merchant_risk_output.csv"

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

TARGET_COLUMN = "target"
TRAIN_START = "2019-01-15"
TRAIN_END = "2020-02-28"
HOLDOUT_START = pd.Timestamp("2020-05-01")
HOLDOUT_END = pd.Timestamp("2020-06-14")
LOCKED_THRESHOLD = 0.814822766216
OUTPUT_COLUMNS = ["merchant", "prediction_date", "risk_score", "alert"]


def validate_input(dataset: pd.DataFrame) -> None:
    """Validate the locked model input schema before fitting or scoring."""
    required_columns = {"merchant", "prediction_date", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise RuntimeError(f"Required input columns are missing: {missing_columns}")
    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise RuntimeError("Target must not be included among model features.")
    future_features = [column for column in FEATURE_COLUMNS if "future" in column.lower()]
    if future_features:
        raise RuntimeError(f"Future-derived features are prohibited: {future_features}")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Model features contain missing values.")
    if dataset[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Input contains duplicate merchant/date pairs.")


def validate_output(output: pd.DataFrame) -> None:
    """Validate the specified alert-only output schema and values."""
    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Output columns do not match the locked output schema.")
    if TARGET_COLUMN in output.columns:
        raise RuntimeError("Output must not contain target or outcome-derived fields.")
    if not output["risk_score"].ge(LOCKED_THRESHOLD).all():
        raise RuntimeError("Output contains a score below the locked threshold.")
    if not output["alert"].eq(1).all():
        raise RuntimeError("Every emitted row must have alert=1.")
    if output[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Output contains duplicate merchant/date pairs.")
    expected_sort = output.sort_values(
        ["risk_score", "prediction_date", "merchant"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if not output.reset_index(drop=True).equals(expected_sort):
        raise RuntimeError("Output is not sorted by score, date, then merchant as required.")


def main() -> None:
    print("=" * 80)
    print("LOOP 009 — GENERATE MERCHANT RISK OUTPUT")
    print("=" * 80)
    print(f"Reading training dataset only: {DATASET_FILE}")
    dataset = pd.read_parquet(DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    validate_input(dataset)

    train = dataset.loc[
        dataset["prediction_date"].between(TRAIN_START, TRAIN_END)
    ].copy()
    holdout = dataset.loc[
        dataset["prediction_date"].between(HOLDOUT_START, HOLDOUT_END)
    ].copy()
    if train.empty or holdout.empty:
        raise RuntimeError("Locked train or holdout period is empty.")
    if holdout["prediction_date"].min() != HOLDOUT_START:
        raise RuntimeError("Holdout minimum prediction date does not match the locked period.")
    if holdout["prediction_date"].max() != HOLDOUT_END:
        raise RuntimeError("Holdout maximum prediction date does not match the locked period.")
    if train["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Train period must precede the holdout period.")

    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=42,
        early_stopping=False,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    holdout_scores = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    scored_holdout = holdout[["merchant", "prediction_date"]].copy()
    scored_holdout["risk_score"] = holdout_scores
    output = scored_holdout.loc[
        scored_holdout["risk_score"] >= LOCKED_THRESHOLD
    ].copy()
    output["alert"] = 1
    output = output[OUTPUT_COLUMNS].sort_values(
        ["risk_score", "prediction_date", "merchant"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    validate_output(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    # Re-read the written CSV to verify the persisted output contract.
    persisted = pd.read_csv(OUTPUT_FILE, parse_dates=["prediction_date"])
    validate_output(persisted)

    holdout_days = holdout["prediction_date"].nunique()
    print("\nLOOP 009 SUMMARY")
    print(f"Holdout rows scored: {len(holdout):,}")
    print(f"Alerts generated: {len(output):,}")
    print(f"Alerts/day: {len(output) / holdout_days:.4f}")
    print(f"Minimum risk score among alerts: {output['risk_score'].min():.12f}")
    print(f"Maximum risk score among alerts: {output['risk_score'].max():.12f}")
    print(f"Output path: {OUTPUT_FILE}")
    print("\nSAFETY CONFIRMATIONS")
    print("Locked HGB configuration reproduced: confirmed")
    print("Locked threshold used without selection or modification: confirmed")
    print("Only final holdout dates were scored for output: confirmed")
    print("fraudTest.csv was not read")
    print("No existing dataset was modified")
    print("No model artifact was saved")
    print("No tuning, feature selection, or resampling was performed")
    print("LOOP 009 COMPLETE")


if __name__ == "__main__":
    main()
