"""LOOP 010: generate an operational report from the locked alert output."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ALERT_FILE = ROOT / "data" / "outputs" / "merchant_risk_output.csv"
REPORT_FILE = ROOT / "data" / "outputs" / "operational_risk_report.md"

LOCKED_THRESHOLD = 0.814822766216
EXPECTED_COLUMNS = ["merchant", "prediction_date", "risk_score", "alert"]
HOLDOUT_START = pd.Timestamp("2020-05-01")
HOLDOUT_END = pd.Timestamp("2020-06-14")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a small DataFrame as a dependency-free Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def validate_alert_output(alerts: pd.DataFrame) -> None:
    """Validate the existing locked merchant-risk output contract."""
    if list(alerts.columns) != EXPECTED_COLUMNS:
        raise RuntimeError("Alert CSV columns do not match the locked output schema.")
    if alerts.empty:
        raise RuntimeError("Alert CSV contains no emitted alert rows.")
    if alerts["risk_score"].isna().any() or alerts["alert"].isna().any():
        raise RuntimeError("Alert CSV contains missing scores or alert flags.")
    if not alerts["risk_score"].ge(LOCKED_THRESHOLD).all():
        raise RuntimeError("Alert CSV contains a score below the locked threshold.")
    if not alerts["alert"].eq(1).all():
        raise RuntimeError("Alert CSV must contain alert=1 on every emitted row.")
    if alerts[["merchant", "prediction_date"]].duplicated().any():
        raise RuntimeError("Alert CSV contains duplicate merchant/date pairs.")
    if alerts["prediction_date"].min() < HOLDOUT_START:
        raise RuntimeError("Alert CSV includes a date before the locked holdout period.")
    if alerts["prediction_date"].max() > HOLDOUT_END:
        raise RuntimeError("Alert CSV includes a date after the locked holdout period.")
    expected_sort = alerts.sort_values(
        ["risk_score", "prediction_date", "merchant"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if not alerts.reset_index(drop=True).equals(expected_sort):
        raise RuntimeError("Alert CSV is not sorted by score, date, then merchant.")


def main() -> None:
    print("=" * 80)
    print("LOOP 010 — GENERATE OPERATIONAL RISK REPORT")
    print("=" * 80)
    print(f"Reading locked merchant-risk output only: {ALERT_FILE}")
    alerts = pd.read_csv(ALERT_FILE, parse_dates=["prediction_date"])
    validate_alert_output(alerts)

    holdout_days = (HOLDOUT_END - HOLDOUT_START).days + 1
    score_summary = alerts["risk_score"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    top_alerts = alerts.head(10).copy()
    top_alerts["prediction_date"] = top_alerts["prediction_date"].dt.date.astype(str)
    top_alerts["risk_score"] = top_alerts["risk_score"].map(lambda value: f"{value:.12f}")

    merchant_concentration = (
        alerts.groupby("merchant", as_index=False)
        .agg(alerts=("alert", "size"), maximum_risk_score=("risk_score", "max"))
        .sort_values(["alerts", "merchant"], ascending=[False, True], kind="mergesort")
        .head(10)
        .copy()
    )
    merchant_concentration["alert_share"] = merchant_concentration["alerts"] / len(alerts)
    merchant_concentration["maximum_risk_score"] = merchant_concentration[
        "maximum_risk_score"
    ].map(lambda value: f"{value:.12f}")
    merchant_concentration["alert_share"] = merchant_concentration["alert_share"].map(
        lambda value: f"{value:.2%}"
    )

    date_concentration = (
        alerts.groupby("prediction_date", as_index=False)
        .agg(alerts=("alert", "size"), maximum_risk_score=("risk_score", "max"))
        .sort_values(["alerts", "prediction_date"], ascending=[False, True], kind="mergesort")
        .head(10)
        .copy()
    )
    date_concentration["alert_share"] = date_concentration["alerts"] / len(alerts)
    date_concentration["prediction_date"] = date_concentration["prediction_date"].dt.date.astype(str)
    date_concentration["maximum_risk_score"] = date_concentration[
        "maximum_risk_score"
    ].map(lambda value: f"{value:.12f}")
    date_concentration["alert_share"] = date_concentration["alert_share"].map(
        lambda value: f"{value:.2%}"
    )

    summary_rows = pd.DataFrame(
        [
            ("Alert rows", f"{len(alerts):,}"),
            ("Alerts per calendar day", f"{len(alerts) / holdout_days:.4f}"),
            ("Minimum risk score", f"{alerts['risk_score'].min():.12f}"),
            ("Maximum risk score", f"{alerts['risk_score'].max():.12f}"),
            ("Mean risk score", f"{alerts['risk_score'].mean():.12f}"),
            ("Median risk score", f"{alerts['risk_score'].median():.12f}"),
        ],
        columns=["Measure", "Value"],
    )
    distribution_rows = pd.DataFrame(
        [
            ("Count", f"{int(score_summary['count']):,}"),
            ("Mean", f"{score_summary['mean']:.12f}"),
            ("Standard deviation", f"{score_summary['std']:.12f}"),
            ("Minimum", f"{score_summary['min']:.12f}"),
            ("25th percentile", f"{score_summary['25%']:.12f}"),
            ("Median", f"{score_summary['50%']:.12f}"),
            ("75th percentile", f"{score_summary['75%']:.12f}"),
            ("90th percentile", f"{score_summary['90%']:.12f}"),
            ("95th percentile", f"{score_summary['95%']:.12f}"),
            ("99th percentile", f"{score_summary['99%']:.12f}"),
            ("Maximum", f"{score_summary['max']:.12f}"),
        ],
        columns=["Statistic", "Risk score"],
    )

    report = f"""# Operational Merchant Risk Report

## Scope

This report summarizes only the threshold-crossing merchant-day alerts in `merchant_risk_output.csv`. It uses the already locked model configuration and threshold; it does not refit, tune, or select anything.

- Holdout period: {HOLDOUT_START.date()} through {HOLDOUT_END.date()} ({holdout_days} calendar days)
- Locked threshold: `{LOCKED_THRESHOLD:.12f}`

## Alert Summary

{markdown_table(summary_rows, ["Measure", "Value"])}

## Risk-Score Distribution

{markdown_table(distribution_rows, ["Statistic", "Risk score"])}

## Highest-Risk Merchant Alerts

{markdown_table(top_alerts, ["merchant", "prediction_date", "risk_score", "alert"])}

## Alert Concentration by Merchant

{markdown_table(merchant_concentration, ["merchant", "alerts", "alert_share", "maximum_risk_score"])}

## Alert Concentration by Date

{markdown_table(date_concentration, ["prediction_date", "alerts", "alert_share", "maximum_risk_score"])}

## Locked Model Configuration

- Model: `HistGradientBoostingClassifier`
- `class_weight="balanced"`
- `random_state=42`
- `early_stopping=False`
- Historical feature schema: the 15 locked historical features used by LOOP 009
- Threshold: `{LOCKED_THRESHOLD:.12f}`

## Performance Context

The threshold was derived on validation at the pre-specified 20-alert/day operating point: 1,220 alerts over 61 days, precision 0.092623, recall 0.191851, and F1 0.124931.

At the locked threshold, final holdout evaluation reported 912 alerts over 45 days (20.2667 alerts/day), precision 0.086623, recall 0.129721, F1 0.103879, ROC-AUC 0.820637, PR-AUC 0.071587, and false-positive rate 0.029132. These are evaluation context only; this report does not change the threshold or operating point.

## Limitations

- A risk score is a model ranking output, not a guarantee of fraud, loss, or causal explanation.
- The report does not establish production readiness, long-term stability, calibration quality, or future financial impact.
- The locked operating capacity is a project assumption and not evidence of actual analyst capacity or operational cost.
- Alert concentration describes this evaluated holdout period only; it does not establish persistent merchant behavior or real-world merchant identity semantics.
- No action, intervention, or automatic financial decision is specified by this report.

## Safety Notes

- The report contains no outcome field.
- `fraudTest.csv` was not read.
- No existing dataset was modified.
- No model artifact was saved.
"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    persisted = REPORT_FILE.read_text(encoding="utf-8")
    required_sections = [
        "# Operational Merchant Risk Report",
        "## Alert Summary",
        "## Risk-Score Distribution",
        "## Highest-Risk Merchant Alerts",
        "## Alert Concentration by Merchant",
        "## Alert Concentration by Date",
        "## Locked Model Configuration",
        "## Performance Context",
        "## Limitations",
    ]
    if not all(section in persisted for section in required_sections):
        raise RuntimeError("Generated report is missing a required section.")
    if "target" in persisted.lower():
        raise RuntimeError("Generated report must not contain a target field.")

    print("\nLOOP 010 SUMMARY")
    print(f"Holdout period: {HOLDOUT_START.date()} -> {HOLDOUT_END.date()}")
    print(f"Total alerts: {len(alerts):,}")
    print(f"Alerts/day: {len(alerts) / holdout_days:.4f}")
    print(f"Risk-score range: {alerts['risk_score'].min():.12f} -> {alerts['risk_score'].max():.12f}")
    print(f"Report path: {REPORT_FILE}")
    print("Report validation: passed")
    print("fraudTest.csv was not read")
    print("No dataset was modified")
    print("No model artifact was saved")
    print("No threshold selection, tuning, feature selection, or resampling was performed")
    print("LOOP 010 COMPLETE")


if __name__ == "__main__":
    main()
