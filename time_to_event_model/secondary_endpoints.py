"""
In this file we evaluate the secondary, exploratory endpoints of the study:
recurrence-free survival (RFS) and metastasis-free survival (MFS). Following the
reviewers we keep these analyses at the cohort level only (primary and metastatic,
no pooling) and report the Random Survival Forest discrimination (Harrell C with a
bootstrap 95% confidence interval). These endpoints are reported as hypothesis
generating because their performance stays close to chance.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the results are written

Returns:
    None

Example:
    python model_training_review/secondary_endpoints.py --merged-data data/patient_survival.csv --output-dir results/

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
    parser = argparse.ArgumentParser(description="Evaluate the secondary endpoints RFS and MFS")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def evaluate(data, endpoint, cohort):
    """
    This function evaluates a Random Survival Forest for one secondary endpoint and one
    cohort, keeping only patients with a valid time for that endpoint.
    """
    cohort_data = su.select_cohort(data, cohort)
    cohort_data = cohort_data[cohort_data[f"{endpoint}_time"] > 0].dropna(
        subset=[f"{endpoint}_event"]
    ).reset_index(drop=True)
    columns = su.feature_columns(cohort_data, "all")

    risk, y, event, frequency = su.out_of_fold_risk(
        cohort_data, endpoint, columns, n_features=5, add_origin=False
    )
    return {
        "n": int(len(cohort_data)),
        "events": int(event.sum()),
        "harrell": round(float(su.harrell_c(y, risk)), 3),
        "harrell_ci": su.bootstrap_ci(lambda idx: su.harrell_c(y[idx], risk[idx]), event),
        "lasso_frequency": frequency,
    }


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = su.load_dataset(args.merged_data)

    # We evaluate both secondary endpoints on the primary and the metastatic cohorts.
    results = {}
    for endpoint in ("rfs", "mfs"):
        for cohort in ("primary", "metastatic"):
            key = f"{endpoint}_{cohort}"
            results[key] = evaluate(data, endpoint, cohort)
            r = results[key]
            logger.info(f"{endpoint.upper()} {cohort:11s} | C {r['harrell']} {r['harrell_ci']} ({r['events']} events)")

    output_file = os.path.join(args.output_dir, "secondary_endpoints.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file}")


if __name__ == "__main__":
    main()
