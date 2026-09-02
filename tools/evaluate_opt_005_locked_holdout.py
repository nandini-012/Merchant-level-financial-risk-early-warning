"""One-time final-holdout evaluation for the frozen OPT-005 candidate."""

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
    roc_auc_score,
)

import analyze_opt_004_fraud_history_dynamics as opt_004


ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = ROOT / "docs" / "LOOP_011_OPT_005_HOLDOUT_EVALUATION.md"
HOLDOUT_END = pd.Timestamp("2020-06-14")
FROZEN_THRESHOLD = 0.812480834427
FEATURE_COLUMNS = [
    *opt_004.BASELINE_FEATURE_COLUMNS,
    *opt_004.OPT_004_FEATURE_COLUMNS,
]
BASELINE_HOLDOUT = {
    "threshold": 0.814822766216,
    "alerts": 912,
    "alerts_per_day": 20.2667,
    "tp": 79,
    "fp": 833,
    "fn": 530,
    "tn": 27761,
    "precision": 0.086623,
    "recall": 0.129721,
    "f1": 0.103879,
    "roc_auc": 0.820637,
    "pr_auc": 0.071587,
}


def read_observed_daily_through_holdout() -> pd.DataFrame:
    """Aggregate raw merchant-day counts only through the fixed holdout end."""
    daily_parts = []
    for chunk in pd.read_csv(
        opt_004.RAW_TRAIN_FILE,
        usecols=["trans_date_trans_time", "merchant", "is_fraud"],
        parse_dates=["trans_date_trans_time"],
        chunksize=100_000,
    ):
        chunk["prediction_date"] = chunk["trans_date_trans_time"].dt.floor("D")
        eligible = chunk.loc[
            chunk["prediction_date"] <= HOLDOUT_END,
            ["merchant", "prediction_date", "is_fraud"],
        ]
        daily_parts.append(
            eligible.groupby(["merchant", "prediction_date"], as_index=False).agg(
                daily_transaction_count=("is_fraud", "size"),
                daily_fraud_count=("is_fraud", "sum"),
            )
        )
    observed_daily = pd.concat(daily_parts, ignore_index=True).groupby(
        ["merchant", "prediction_date"], as_index=False
    )[["daily_transaction_count", "daily_fraud_count"]].sum()
    if observed_daily["prediction_date"].max() != HOLDOUT_END:
        raise RuntimeError("Raw feature input does not end at the fixed holdout end.")
    return observed_daily


def build_features_through_holdout(observed_daily: pd.DataFrame) -> pd.DataFrame:
    """Reproduce OPT-004 features with its calendar end fixed to holdout end."""
    merchants = observed_daily["merchant"].drop_duplicates().sort_values()
    dates = pd.date_range(observed_daily["prediction_date"].min(), HOLDOUT_END, freq="D")
    index = pd.MultiIndex.from_product(
        [merchants, dates], names=["merchant", "prediction_date"]
    )
    daily = (
        observed_daily.set_index(["merchant", "prediction_date"])
        .reindex(index, fill_value=0)
        .reset_index()
        .sort_values(["merchant", "prediction_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["previous_1d_fraud_count"] = opt_004.past_rolling_sum(
        daily, "daily_fraud_count", 1
    )
    daily["previous_3d_fraud_count"] = opt_004.past_rolling_sum(
        daily, "daily_fraud_count", 3
    )
    previous_3d_transaction_count = opt_004.past_rolling_sum(
        daily, "daily_transaction_count", 3
    )
    daily["previous_3d_fraud_rate"] = (
        daily["previous_3d_fraud_count"]
        / previous_3d_transaction_count.where(previous_3d_transaction_count != 0)
    ).fillna(0.0)
    previous_7d_fraud_count = opt_004.past_rolling_sum(daily, "daily_fraud_count", 7)
    previous_7d_transaction_count = opt_004.past_rolling_sum(
        daily, "daily_transaction_count", 7
    )
    daily["prior_7d_fraud_rate"] = (
        previous_7d_fraud_count
        / previous_7d_transaction_count.where(previous_7d_transaction_count != 0)
    ).fillna(0.0)
    daily["preceding_7d_fraud_rate"] = (
        daily.groupby("merchant", sort=False)["prior_7d_fraud_rate"].shift(7)
    ).fillna(0.0)
    daily["previous_7d_fraud_rate_change"] = (
        daily["prior_7d_fraud_rate"] - daily["preceding_7d_fraud_rate"]
    )
    daily["daily_fraud_active_day"] = (daily["daily_fraud_count"] > 0).astype("int64")
    daily["previous_7d_fraud_active_day_count"] = opt_004.past_rolling_sum(
        daily, "daily_fraud_active_day", 7
    )
    daily["current_day_fraud_date"] = daily["prediction_date"].where(
        daily["daily_fraud_count"] > 0
    )
    prior_fraud_date = daily.groupby("merchant", sort=False)["current_day_fraud_date"].transform(
        lambda values: values.shift(1).ffill()
    )
    daily["no_prior_fraud_transaction_indicator"] = prior_fraud_date.isna().astype("int64")
    daily["days_since_prior_fraud_transaction"] = (
        (daily["prediction_date"] - prior_fraud_date).dt.days.fillna(0).astype("int64")
    )
    return daily[["merchant", "prediction_date", *opt_004.OPT_004_FEATURE_COLUMNS]]


def metrics(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float | int]:
    """Evaluate the already frozen threshold without any selection logic."""
    predictions = (probabilities >= FROZEN_THRESHOLD).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    alerts = int(predictions.sum())
    return {
        "alerts": alerts,
        "alerts_per_day": alerts / 45,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def main() -> None:
    print("=" * 80)
    print("OPT-005 — ONE-TIME FROZEN FINAL HOLDOUT EVALUATION")
    print("=" * 80)
    raw_before = opt_004.raw_metadata()
    dataset = pd.read_parquet(opt_004.DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    expected = {"merchant", "prediction_date", "target", *opt_004.BASELINE_FEATURE_COLUMNS}
    if set(dataset.columns) != expected:
        raise RuntimeError("Training dataset schema differs from the locked baseline schema.")
    if dataset[opt_004.BASELINE_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Baseline modelling features contain missing values.")

    candidate_features = build_features_through_holdout(
        read_observed_daily_through_holdout()
    )
    modelling = dataset.merge(
        candidate_features, on=["merchant", "prediction_date"], how="left", validate="one_to_one"
    )
    train = modelling.loc[
        modelling["prediction_date"].between(opt_004.TRAIN_START, opt_004.TRAIN_END)
    ].copy()
    holdout = modelling.loc[
        modelling["prediction_date"].between(opt_004.HOLDOUT_START, HOLDOUT_END)
    ].copy()
    if train[FEATURE_COLUMNS].isna().any().any() or holdout[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Frozen OPT-005 feature matrix contains missing values.")
    if holdout["prediction_date"].min() != opt_004.HOLDOUT_START or holdout["prediction_date"].max() != HOLDOUT_END:
        raise RuntimeError("Holdout dates do not match the frozen evaluation period.")
    if train["prediction_date"].max() >= holdout["prediction_date"].min():
        raise RuntimeError("Frozen train period must precede holdout.")

    print(f"Frozen threshold: {FROZEN_THRESHOLD:.12f}")
    print("Threshold source: OPT-005 validation-only stable rank at 1,220 alerts")
    print(f"Train rows: {len(train):,}; holdout rows: {len(holdout):,}")
    model = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, early_stopping=False
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    holdout_probabilities = pd.Series(
        model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1], index=holdout.index
    )
    result = metrics(holdout["target"], holdout_probabilities)
    if raw_before != opt_004.raw_metadata():
        raise RuntimeError("Raw training data metadata changed during evaluation.")

    report = f"""# OPT-005 Frozen Final Holdout Evaluation

## Frozen candidate

- Feature recipe: existing 15 features plus the eight OPT-004 fraud-history dynamics features.
- Model: `HistGradientBoostingClassifier(class_weight=\"balanced\", random_state=42, early_stopping=False)`.
- Validation-derived frozen threshold: `{FROZEN_THRESHOLD:.12f}`.
- Validation capacity provenance: 1,220 alerts over 61 days.
- Holdout evaluation period: 2020-05-01 through 2020-06-14.

The candidate feature recipe, configuration, and threshold were frozen before this evaluation. This holdout result was not used for post-hoc tuning or selection.

## Final holdout comparison

| Metric | Existing locked baseline | Frozen OPT-005 | OPT-005 minus baseline |
| --- | ---: | ---: | ---: |
| Threshold | {BASELINE_HOLDOUT['threshold']:.12f} | {FROZEN_THRESHOLD:.12f} | — |
| Alerts | {BASELINE_HOLDOUT['alerts']} | {result['alerts']} | {result['alerts'] - BASELINE_HOLDOUT['alerts']:+d} |
| Alerts/day | {BASELINE_HOLDOUT['alerts_per_day']:.4f} | {result['alerts_per_day']:.4f} | {result['alerts_per_day'] - BASELINE_HOLDOUT['alerts_per_day']:+.4f} |
| TP | {BASELINE_HOLDOUT['tp']} | {result['tp']} | {result['tp'] - BASELINE_HOLDOUT['tp']:+d} |
| FP | {BASELINE_HOLDOUT['fp']} | {result['fp']} | {result['fp'] - BASELINE_HOLDOUT['fp']:+d} |
| FN | {BASELINE_HOLDOUT['fn']} | {result['fn']} | {result['fn'] - BASELINE_HOLDOUT['fn']:+d} |
| TN | {BASELINE_HOLDOUT['tn']} | {result['tn']} | {result['tn'] - BASELINE_HOLDOUT['tn']:+d} |
| Precision | {BASELINE_HOLDOUT['precision']:.6f} | {result['precision']:.6f} | {result['precision'] - BASELINE_HOLDOUT['precision']:+.6f} |
| Recall | {BASELINE_HOLDOUT['recall']:.6f} | {result['recall']:.6f} | {result['recall'] - BASELINE_HOLDOUT['recall']:+.6f} |
| F1 | {BASELINE_HOLDOUT['f1']:.6f} | {result['f1']:.6f} | {result['f1'] - BASELINE_HOLDOUT['f1']:+.6f} |
| ROC-AUC | {BASELINE_HOLDOUT['roc_auc']:.6f} | {result['roc_auc']:.6f} | {result['roc_auc'] - BASELINE_HOLDOUT['roc_auc']:+.6f} |
| PR-AUC | {BASELINE_HOLDOUT['pr_auc']:.6f} | {result['pr_auc']:.6f} | {result['pr_auc'] - BASELINE_HOLDOUT['pr_auc']:+.6f} |

## Safety confirmations

- The holdout was evaluated once using the frozen candidate and threshold.
- No holdout result was used to tune features, configuration, threshold, or capacity.
- `fraudTest.csv` was not read.
- No existing dataset, generated output, or model artifact was modified or saved.
"""
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("\nFINAL HOLDOUT RESULT")
    for name in ("alerts", "alerts_per_day", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"{name}: {result[name]}")
    print(f"Report: {REPORT_FILE}")
    print("OPT-005 HOLDOUT EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
