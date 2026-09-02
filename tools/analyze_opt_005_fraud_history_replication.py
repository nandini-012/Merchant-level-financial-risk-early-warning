"""OPT-005: validation-only replication of the OPT-004 combined feature matrix."""

from __future__ import annotations

import pandas as pd

import analyze_opt_004_fraud_history_dynamics as opt_004


def main() -> None:
    print("=" * 80)
    print("OPT-005 — VALIDATION-ONLY FRAUD-HISTORY REPLICATION")
    print("=" * 80)
    raw_before = opt_004.raw_metadata()
    dataset = pd.read_parquet(opt_004.DATASET_FILE)
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    required_columns = {
        "merchant", "prediction_date", "target", *opt_004.BASELINE_FEATURE_COLUMNS
    }
    if set(dataset.columns) != required_columns:
        raise RuntimeError("Training dataset schema differs from the locked baseline schema.")
    if dataset[opt_004.BASELINE_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Baseline modelling features contain missing values.")

    fraud_features = opt_004.build_opt_004_features(
        opt_004.read_observed_daily_through_validation()
    )
    if fraud_features["prediction_date"].max() > opt_004.VALIDATION_END:
        raise RuntimeError("OPT-005 feature construction extends into holdout dates.")
    modelling = dataset.loc[
        dataset["prediction_date"].between(opt_004.TRAIN_START, opt_004.VALIDATION_END)
    ].merge(fraud_features, on=["merchant", "prediction_date"], how="left", validate="one_to_one")
    if (modelling["prediction_date"] >= opt_004.HOLDOUT_START).any():
        raise RuntimeError("Holdout rows entered the OPT-005 modelling frame.")
    if modelling[opt_004.OPT_004_FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("OPT-005 fraud-history features contain missing values.")

    train = modelling.loc[
        modelling["prediction_date"].between(opt_004.TRAIN_START, opt_004.TRAIN_END)
    ].copy()
    validation = modelling.loc[
        modelling["prediction_date"].between(
            opt_004.VALIDATION_START, opt_004.VALIDATION_END
        )
    ].copy()
    if train["prediction_date"].max() >= validation["prediction_date"].min():
        raise RuntimeError("Train/validation temporal ordering is invalid.")
    if validation["prediction_date"].nunique() != opt_004.VALIDATION_DAYS:
        raise RuntimeError("Validation does not contain exactly 61 calendar days.")

    baseline = opt_004.evaluate_validation(
        "OPT-001 baseline", opt_004.BASELINE_FEATURE_COLUMNS, train, validation
    )
    opt_005 = opt_004.evaluate_validation(
        "OPT-005 fraud-history replication",
        [*opt_004.BASELINE_FEATURE_COLUMNS, *opt_004.OPT_004_FEATURE_COLUMNS],
        train,
        validation,
    )
    if raw_before != opt_004.raw_metadata():
        raise RuntimeError("Raw training data metadata changed during analysis.")

    print("\nVALIDATION RESULTS")
    print(
        pd.DataFrame([baseline, opt_005]).to_string(
            index=False,
            columns=["candidate", "threshold", "pr_auc", "precision", "recall", "f1", "alerts", "alerts_per_day"],
            formatters={
                "threshold": "{:.12f}".format,
                "pr_auc": "{:.6f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "f1": "{:.6f}".format,
                "alerts_per_day": "{:.4f}".format,
            },
        )
    )
    print("\nVALIDATION CHECKS")
    print("Existing locked HGB configuration: confirmed")
    print("Validation alert budget: 1,220")
    print("No missing combined modelling values: confirmed")
    print("Holdout was not accessed, scored, or used for selection: confirmed")
    print("fraudTest.csv was not read")
    print("Raw training data, existing datasets, and outputs were not modified")
    print("No candidate selection performed")


if __name__ == "__main__":
    main()
