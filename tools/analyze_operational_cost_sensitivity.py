"""LOOP 006B validation-only operational cost sensitivity analysis.

All rupee values in this script are scenario assumptions, not observed costs,
financial losses, or claims about any real payment provider's operations.
"""

import pandas as pd


VALIDATION_DAYS = 61
VALIDATION_POSITIVES = 589

# Fixed validation results from LOOP 006 operational threshold analysis.
# No model is loaded, trained, changed, or scored in this sensitivity analysis.
OPERATING_POINTS = [
    {"alerts_per_day": 5, "total_alerts": 305, "TP": 38, "FP": 267},
    {"alerts_per_day": 10, "total_alerts": 610, "TP": 66, "FP": 544},
    {"alerts_per_day": 20, "total_alerts": 1220, "TP": 113, "FP": 1107},
    {"alerts_per_day": 30, "total_alerts": 1830, "TP": 148, "FP": 1682},
    {"alerts_per_day": 50, "total_alerts": 3050, "TP": 219, "FP": 2831},
    {"alerts_per_day": 75, "total_alerts": 4575, "TP": 299, "FP": 4276},
    {"alerts_per_day": 100, "total_alerts": 6100, "TP": 346, "FP": 5754},
]

INVESTIGATION_COSTS_PER_ALERT = [100, 250, 500, 1000]
MISSED_EVENT_COSTS_PER_POSITIVE = [1000, 2500, 5000, 10000]


def format_rupees(value: int) -> str:
    return f"₹{value:,.0f}"


def main() -> None:
    operating = pd.DataFrame(OPERATING_POINTS)
    operating["FN"] = VALIDATION_POSITIVES - operating["TP"]
    operating["false_alerts_per_day"] = operating["FP"] / VALIDATION_DAYS
    # This percentage is explicitly FP / total alerts, i.e. alert-level false rate.
    operating["false_positive_percentage"] = operating["FP"] / operating["total_alerts"]
    operating["true_positives_per_day"] = operating["TP"] / VALIDATION_DAYS
    operating["fraction_validation_positives_captured"] = (
        operating["TP"] / VALIDATION_POSITIVES
    )

    scenario_rows = []
    for point in operating.itertuples(index=False):
        for investigation_cost in INVESTIGATION_COSTS_PER_ALERT:
            for missed_event_cost in MISSED_EVENT_COSTS_PER_POSITIVE:
                investigation_total = point.total_alerts * investigation_cost
                missed_total = point.FN * missed_event_cost
                scenario_rows.append(
                    {
                        "alerts_per_day": point.alerts_per_day,
                        "total_alerts": point.total_alerts,
                        "TP": point.TP,
                        "FP": point.FP,
                        "FN": point.FN,
                        "scenario_investigation_cost_per_alert": investigation_cost,
                        "scenario_missed_event_cost_per_positive": missed_event_cost,
                        "scenario_investigation_cost": investigation_total,
                        "scenario_missed_event_cost": missed_total,
                        "scenario_total_cost": investigation_total + missed_total,
                    }
                )
    scenarios = pd.DataFrame(scenario_rows)

    print("=" * 80)
    print("LOOP 006B — OPERATIONAL COST SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("VALIDATION-ONLY ANALYSIS")
    print(f"Validation calendar days: {VALIDATION_DAYS}")
    print(f"Validation positive targets: {VALIDATION_POSITIVES}")
    print(
        "SCENARIO ASSUMPTIONS ONLY: values below are not actual financial losses, "
        "Razorpay costs, or operational costs."
    )

    print("\nALERT-BURDEN ANALYSIS")
    burden = operating[
        [
            "alerts_per_day",
            "total_alerts",
            "TP",
            "FP",
            "FN",
            "false_alerts_per_day",
            "false_positive_percentage",
            "true_positives_per_day",
            "fraction_validation_positives_captured",
        ]
    ].copy()
    print(
        burden.to_string(
            index=False,
            formatters={
                "false_alerts_per_day": "{:.4f}".format,
                "false_positive_percentage": "{:.4%}".format,
                "true_positives_per_day": "{:.4f}".format,
                "fraction_validation_positives_captured": "{:.4%}".format,
            },
        )
    )
    print("false_positive_percentage is defined here as FP / total alerts.")

    print("\nSCENARIO COST SENSITIVITY")
    print(
        "Columns: assumed investigation cost/alert; assumed missed-event cost/positive; "
        "investigation cost; missed-event cost; total scenario cost."
    )
    scenario_display = scenarios.copy()
    for column in [
        "scenario_investigation_cost_per_alert",
        "scenario_missed_event_cost_per_positive",
        "scenario_investigation_cost",
        "scenario_missed_event_cost",
        "scenario_total_cost",
    ]:
        scenario_display[column] = scenario_display[column].map(format_rupees)
    print(scenario_display.to_string(index=False))

    print("\nvalidation-only analysis")
    print("holdout not accessed")
    print("no threshold selected")
    print("no model modified")
    print("no raw data modified")
    print("no project decision made")


if __name__ == "__main__":
    main()
