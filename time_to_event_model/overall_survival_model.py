"""
In this file we evaluate the primary endpoint of the study, overall survival (OS),
as a time-to-event outcome. For the primary, the metastatic and the pooled cohorts we
report the discrimination of a Random Survival Forest (Harrell C with a bootstrap 95%
confidence interval, Uno/IPCW C, time-dependent AUC at 1, 2 and 3 years) together with
an integrated Brier score. An XGBoost accelerated failure time model is fitted on the
pooled cohort as a sensitivity analysis. The numbers written here populate Table 2.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the results are written

Returns:
    None

Example:
    python model_training_review/overall_survival_model.py --merged-data data/patient_survival.csv --output-dir results/

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import numpy as np
from loguru import logger

import survival_utils as su


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Evaluate overall survival models (Table 2)")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def evaluate_cohort(data, cohort):
    """
    This function evaluates the Random Survival Forest on one cohort and returns the
    discrimination metrics with their bootstrap confidence intervals.
    """
    # We restrict the data to the cohort and keep the pooled-cohort indicator only when pooling.
    cohort_data = su.select_cohort(data, cohort)
    add_origin = cohort == "pooled"
    n_features = 8 if cohort == "pooled" else 5
    columns = su.feature_columns(cohort_data, "all")

    # We obtain an out-of-fold risk score with the leakage-free nested cross-validation.
    risk, y, event, frequency = su.out_of_fold_risk(
        cohort_data, "os", columns, n_features, add_origin
    )

    # We score discrimination on the out-of-fold predictions and bootstrap the intervals.
    all_idx = np.arange(len(event))
    harrell = su.harrell_c(y, risk)
    result = {
        "n": int(len(cohort_data)),
        "events": int(event.sum()),
        "harrell": round(float(harrell), 3),
        "harrell_ci": su.bootstrap_ci(lambda idx: su.harrell_c(y[idx], risk[idx]), event),
        "uno": round(float(su.uno_c(y, y, risk)), 3),
        "ibs": round(float(su.integrated_brier(cohort_data, "os", risk)), 3),
        "lasso_frequency": frequency,
    }
    for label, horizon in zip(["1y", "2y", "3y"], su.HORIZONS):
        result[f"auc_{label}"] = round(float(su.time_dependent_auc(y, y, risk, horizon)), 3)
        result[f"auc_{label}_ci"] = su.bootstrap_ci(
            lambda idx, h=horizon: su.time_dependent_auc(y, y[idx], risk[idx], h), event, n_boot=400
        )
    return result


def aft_sensitivity(data):
    """
    This function runs the XGBoost AFT sensitivity analysis on the pooled cohort with the
    same nested design (LASSO selection inside the folds, AFT trained on the training fold).
    """
    cohort_data = su.select_cohort(data, "pooled")
    columns = su.feature_columns(cohort_data, "all")
    time = cohort_data["os_time"].values.astype(float)
    event = cohort_data["os_event"].values.astype(int)
    origin = cohort_data["origin_meta"].values
    y = su.survival_target(cohort_data, "os")

    from sklearn.model_selection import RepeatedStratifiedKFold

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=su.RANDOM_STATE)
    risk = np.zeros(len(cohort_data))
    folds = np.zeros(len(cohort_data))
    for train_idx, test_idx in splitter.split(cohort_data, event):
        x_train, x_test = su.preprocess_fold(cohort_data.iloc[train_idx], cohort_data.iloc[test_idx], columns)
        selected = su.lasso_cox_select(x_train, y[train_idx], 8)
        x_train, x_test = x_train[selected].copy(), x_test[selected].copy()
        x_train["origin_meta"] = origin[train_idx]
        x_test["origin_meta"] = origin[test_idx]
        model = su.fit_aft(x_train, time[train_idx], event[train_idx])
        risk[test_idx] += su.aft_risk(model, x_test)
        folds[test_idx] += 1
    risk = risk / np.maximum(folds, 1)
    return {
        "harrell": round(float(su.harrell_c(y, risk)), 3),
        "harrell_ci": su.bootstrap_ci(lambda idx: su.harrell_c(y[idx], risk[idx]), event),
        "auc_1y": round(float(su.time_dependent_auc(y, y, risk, su.HORIZONS[0])), 3),
    }


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # We load the patient-level dataset.
    data = su.load_dataset(args.merged_data)
    logger.info(f"Loaded {len(data)} patients ({(data.primitif == 1).sum()} primary, {(data.primitif != 1).sum()} metastatic)")

    # We evaluate the Random Survival Forest on each cohort.
    results = {}
    for cohort in ("primary", "metastatic", "pooled"):
        results[cohort] = evaluate_cohort(data, cohort)
        r = results[cohort]
        logger.info(f"OS {cohort:11s} | C {r['harrell']} {r['harrell_ci']} | AUC1y {r['auc_1y']} | IBS {r['ibs']}")

    # We add the AFT sensitivity analysis on the pooled cohort.
    results["pooled_aft"] = aft_sensitivity(data)
    logger.info(f"OS pooled (AFT) | C {results['pooled_aft']['harrell']} {results['pooled_aft']['harrell_ci']}")

    # We save every number so that Table 2 can be rebuilt from disk.
    output_file = os.path.join(args.output_dir, "overall_survival.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file}")


if __name__ == "__main__":
    main()
