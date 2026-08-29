"""LOOP 011: generate evidence-only explanations for locked merchant alerts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ALERT_FILE = ROOT / "data" / "outputs" / "merchant_risk_output.csv"
TRAINING_DATASET_FILE = ROOT / "data" / "processed" / "training_dataset.parquet"
OUTPUT_FILE = ROOT / "data" / "outputs" / "merchant_risk_explanations.csv"

LOCKED_THRESHOLD = 0.814822766216
HOLDOUT_START = pd.Timestamp("2020-05-01")
HOLDOUT_END = pd.Timestamp("2020-06-14")

ALERT_COLUMNS = ["merchant", "prediction_date", "risk_score", "alert"]
HISTORICAL_FEATURE_COLUMNS = [
    "previous_7d_average_transaction_amount",
    "previous_7d_maximum_transaction_amount",
    "previous_7d_total_transaction_amount",
    "previous_14d_transaction_count",
    "previous_14d_fraud_rate",
    "previous_7d_transaction_count_change",
]
OUTPUT_COLUMNS = [*ALERT_COLUMNS, *HISTORICAL_FEATURE_COLUMNS, "explanation"]


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest for an input file safety check."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alerts(alerts: pd.DataFrame) -> None:
    """Validate the locked alert-only input contract."""
    if list(alerts.columns) != ALERT_COLUMNS:
        raise RuntimeError("Alert CSV columns do not match the locked output schema.")
    if alerts.empty:
        raise RuntimeError("Alert CSV must contain at least one alert row.")
    if alerts.isna().any().any():
        raise RuntimeError("Alert CSV contains missing values.")
    if not alerts["risk_score"].ge(LOCKED_THRESHOLD).all():
        raise RuntimeError("Alert CSV contains a score below the locked threshold.")
    if not alerts["alert"].eq(1).all():
        raise RuntimeError("Alert CSV must contain alert=1 for every row.")
    if alerts[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Alert CSV contains duplicate merchant/date pairs.")
    if not alerts["prediction_date"].between(HOLDOUT_START, HOLDOUT_END).all():
        raise RuntimeError("Alert CSV contains a date outside the locked holdout period.")


def build_explanation(row: pd.Series) -> str:
    """Describe the requested historical values without causal interpretation."""
    return (
        "Historical evidence only: prior 7-day average transaction amount "
        f"{row['previous_7d_average_transaction_amount']:.2f}; prior 7-day "
        f"maximum transaction amount {row['previous_7d_maximum_transaction_amount']:.2f}; "
        f"prior 7-day total transaction amount {row['previous_7d_total_transaction_amount']:.2f}; "
        f"prior 14-day transaction count {row['previous_14d_transaction_count']:.0f}; "
        f"prior 14-day fraud rate {row['previous_14d_fraud_rate']:.6f}; "
        f"recent 7-day transaction-count change versus the preceding 7 days "
        f"{row['previous_7d_transaction_count_change']:.0f}."
    )


def validate_output(output: pd.DataFrame, alerts: pd.DataFrame) -> None:
    """Validate the explanation output against the locked alert input."""
    if list(output.columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Output columns do not match the required LOOP 011 schema.")
    if "target" in output.columns or any("future" in column.lower() for column in output.columns):
        raise RuntimeError("Output must not contain target or future-derived fields.")
    if len(output) != len(alerts):
        raise RuntimeError("Output row count does not match the alert input row count.")
    if output[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Output contains duplicate merchant/date pairs.")
    if not output["prediction_date"].between(HOLDOUT_START, HOLDOUT_END).all():
        raise RuntimeError("Output contains a date outside the locked holdout period.")
    if output["explanation"].isna().any() or output["explanation"].str.strip().eq("").any():
        raise RuntimeError("Every output row must contain a non-empty explanation.")

    expected_alert_values = alerts.set_index(["merchant", "prediction_date"])[
        ["risk_score", "alert"]
    ].sort_index()
    actual_alert_values = output.set_index(["merchant", "prediction_date"])[
        ["risk_score", "alert"]
    ].sort_index()
    if not actual_alert_values.index.equals(expected_alert_values.index):
        raise RuntimeError("Output merchant/date pairs do not exactly match the alert input.")
    if not actual_alert_values["risk_score"].equals(expected_alert_values["risk_score"]):
        raise RuntimeError("Output risk scores do not exactly match the alert input.")
    if not actual_alert_values["alert"].equals(expected_alert_values["alert"]):
        raise RuntimeError("Output alert values do not exactly match the alert input.")
    if output[HISTORICAL_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Output contains missing historical feature values.")


def main() -> None:
    print("=" * 80)
    print("LOOP 011 — GENERATE MERCHANT RISK EXPLANATIONS")
    print("=" * 80)
    print(f"Reading locked alert output: {ALERT_FILE}")
    print(f"Reading historical feature values only: {TRAINING_DATASET_FILE}")

    input_digests_before = {
        ALERT_FILE: file_digest(ALERT_FILE),
        TRAINING_DATASET_FILE: file_digest(TRAINING_DATASET_FILE),
    }
    alerts = pd.read_csv(ALERT_FILE, parse_dates=["prediction_date"])
    validate_alerts(alerts)

    source_columns = ["merchant", "prediction_date", *HISTORICAL_FEATURE_COLUMNS]
    historical = pd.read_parquet(TRAINING_DATASET_FILE, columns=source_columns)
    historical["prediction_date"] = pd.to_datetime(historical["prediction_date"])
    if historical[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Training dataset contains duplicate merchant/date pairs.")
    if historical[HISTORICAL_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Training dataset contains missing requested historical features.")

    output = alerts.merge(
        historical,
        on=["merchant", "prediction_date"],
        how="left",
        validate="one_to_one",
    )
    if len(output) != len(alerts):
        raise RuntimeError("Joining historical features changed the alert row count.")
    if output[HISTORICAL_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("One or more alert rows have no historical feature match.")

    output["explanation"] = output.apply(build_explanation, axis=1)
    output = output[OUTPUT_COLUMNS]
    validate_output(output, alerts)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    persisted = pd.read_csv(OUTPUT_FILE, parse_dates=["prediction_date"])
    validate_output(persisted, alerts)
    input_digests_after = {path: file_digest(path) for path in input_digests_before}
    if input_digests_before != input_digests_after:
        raise RuntimeError("An existing input file changed while this script ran.")

    print("\nLOOP 011 SUMMARY")
    print(f"Explanation rows generated: {len(output):,}")
    print(f"Output path: {OUTPUT_FILE}")
    print("Validation: passed")
    print("Alert merchant/date pairs: exact match")
    print("Risk scores and alert values: exact match")
    print("No target or outcome-derived field in output: confirmed")
    print("All explanations are non-empty: confirmed")
    print("fraudTest.csv was not read")
    print("Existing input files were not modified")
    print("No model was trained, tuned, or saved")
    print("LOOP 011 COMPLETE")


if __name__ == "__main__":
    main()
