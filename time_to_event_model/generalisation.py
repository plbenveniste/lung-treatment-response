"""
In this file we assess how well the pooled overall-survival model generalises. Two
external-like validation schemes are used, as requested by the reviewers:
    - a leave-one-centre-out cross-validation, where the centre is read from the first
      letter of the anonymised patient code, so that each centre is held out in turn;
    - a temporal split, where the model is trained on the earlier-treated patients and
      tested on the later-treated ones.

The contrast between the two (near chance across centres, but retained over time) is the
main limitation reported in the discussion.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the results are written

Returns:
    None

Example:
    python model_training_review/generalisation.py --merged-data data/patient_survival.csv --output-dir results/

Author: Pierre-Louis Benveniste
"""

import argparse
import datetime
import json
import os

import numpy as np
from loguru import logger

import survival_utils as su


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Generalisation of the overall-survival model")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def fit_pooled(data, train_idx, test_idx, model="rsf"):
    """
    This function trains the pooled model on a training split and returns the risk score on
    the held-out split. The LASSO selection is refitted on the training split only.
    """
    columns = su.feature_columns(data, "all")
    time = data["os_time"].values.astype(float)
    event = data["os_event"].values.astype(int)
    origin = data["origin_meta"].values
    y = su.survival_target(data, "os")

    x_train, x_test = su.preprocess_fold(data.iloc[train_idx], data.iloc[test_idx], columns)
    selected = su.lasso_cox_select(x_train, y[train_idx], 8)
    x_train, x_test = x_train[selected].copy(), x_test[selected].copy()
    x_train["origin_meta"] = origin[train_idx]
    x_test["origin_meta"] = origin[test_idx]
    if model == "aft":
        fitted = su.fit_aft(x_train, time[train_idx], event[train_idx])
        return su.aft_risk(fitted, x_test)
    fitted = su.build_rsf().fit(x_train, y[train_idx])
    return fitted.predict(x_test)


def leave_one_centre_out(data):
    """
    This function runs the leave-one-centre-out cross-validation and returns the per-centre
    and the mean Harrell C for both the RSF and the AFT models.
    """
    centre = data["base"].str[0].values
    y = su.survival_target(data, "os")
    per_centre = {}
    for held_out in sorted(set(centre)):
        test_idx = np.where(centre == held_out)[0]
        train_idx = np.where(centre != held_out)[0]
        if y["event"][test_idx].sum() < 3 or len(test_idx) < 8:
            continue
        for model in ("rsf", "aft"):
            risk = fit_pooled(data, train_idx, test_idx, model)
            c = su.harrell_c(y[test_idx], risk)
            per_centre.setdefault(model, {})[held_out] = round(float(c), 3)
    summary = {
        model: round(float(np.mean(list(scores.values()))), 3)
        for model, scores in per_centre.items()
    }
    return {"per_centre": per_centre, "mean": summary}


def temporal_split(data):
    """
    This function trains on the first 65% of the patients by treatment date and tests on
    the remaining, later-treated 35%.
    """
    dated = data[data["treat_ord"].notna()].reset_index(drop=True)
    order = np.argsort(dated["treat_ord"].values)
    cut = int(0.65 * len(dated))
    train_idx, test_idx = order[:cut], order[cut:]
    y = su.survival_target(dated, "os")
    cutoff = datetime.date.fromordinal(int(dated["treat_ord"].values[order[cut]]))
    result = {
        "cutoff_date": str(cutoff),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    for model in ("rsf", "aft"):
        risk = fit_pooled(dated, train_idx, test_idx, model)
        result[f"harrell_{model}"] = round(float(su.harrell_c(y[test_idx], risk)), 3)
    return result


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = su.select_cohort(su.load_dataset(args.merged_data), "pooled")

    results = {"leave_one_centre_out": leave_one_centre_out(data), "temporal_split": temporal_split(data)}
    logger.info(f"Leave-one-centre-out mean C: {results['leave_one_centre_out']['mean']}")
    logger.info(f"Temporal split: {results['temporal_split']}")

    output_file = os.path.join(args.output_dir, "generalisation.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file}")


if __name__ == "__main__":
    main()
