"""
In this file we interpret the final overall-survival models. Two complementary views
are produced, matching the reviewers' request to separate selection from interpretation:
    - the cross-fold selection frequency of the LASSO Cox model, i.e. how often each
      feature is retained across the resampled training folds (this feeds Table 3);
    - the SHAP values of the final Random Survival Forest refitted on the whole cohort,
      restricted to the consensus features (this feeds Figure 4).

The LASSO does the selection inside the folds; SHAP only explains the already-fitted
model, so no feature is chosen on the basis of the full data.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the results and the SHAP figure are written

Returns:
    None

Example:
    python model_training_review/feature_selection_and_shap.py --merged-data data/patient_survival.csv --output-dir results/

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

import survival_utils as su


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Feature selection frequency and SHAP interpretation")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def interpret_cohort(data, cohort):
    """
    This function returns the LASSO selection frequency and the mean absolute SHAP value
    of the final Random Survival Forest for one cohort.
    """
    cohort_data = su.select_cohort(data, cohort)
    add_origin = cohort == "pooled"
    n_features = 8 if cohort == "pooled" else 5
    columns = su.feature_columns(cohort_data, "all")
    y = su.survival_target(cohort_data, "os")
    origin = cohort_data["origin_meta"].values

    # We recover the cross-fold selection frequency from the nested cross-validation.
    _, _, _, frequency = su.out_of_fold_risk(cohort_data, "os", columns, n_features, add_origin)

    # We keep the consensus features (selected in at least half of the folds).
    consensus = [f for f, freq in frequency.items() if freq >= 0.5][:n_features]
    consensus = consensus or list(frequency.keys())[:n_features]

    # We refit the forest on the whole cohort with the consensus features and explain it.
    x_all, _ = su.preprocess_fold(cohort_data, cohort_data, consensus)
    model_columns = list(consensus)
    if add_origin:
        x_all = x_all.copy()
        x_all["origin_meta"] = origin
        model_columns = model_columns + ["origin_meta"]
    forest = su.build_rsf().fit(x_all, y)

    background = shap.sample(x_all, min(25, len(x_all)), random_state=0)
    explainer = shap.PermutationExplainer(
        lambda values: forest.predict(pd.DataFrame(values, columns=model_columns)), background
    )
    n_explained = min(50, len(x_all))
    shap_values = explainer(x_all.iloc[:n_explained], max_evals=150, silent=True)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    shap_importance = {
        c: round(float(v), 4) for c, v in sorted(zip(model_columns, mean_abs), key=lambda kv: -kv[1])
    }
    return {
        "harrell_selected": consensus,
        "lasso_frequency": frequency,
        "shap": shap_importance,
    }, shap_values, x_all.iloc[:n_explained], model_columns


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = su.load_dataset(args.merged_data)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cohort in zip(axes, ("primary", "metastatic", "pooled")):
        summary, shap_values, x_used, model_columns = interpret_cohort(data, cohort)
        results[cohort] = summary
        logger.info(f"OS {cohort:11s} | consensus {summary['harrell_selected']}")

        # We draw a compact bar plot of the mean absolute SHAP value per feature.
        order = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1][:8][::-1]
        names = [model_columns[i] for i in order]
        values = np.abs(shap_values.values).mean(axis=0)[order]
        ax.barh(names, values, color="#4C72B0")
        ax.set_title(f"{cohort.capitalize()} cohort")
        ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    figure_file = os.path.join(args.output_dir, "shap_overall_survival.png")
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)

    output_file = os.path.join(args.output_dir, "feature_importance.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file} and {figure_file}")


if __name__ == "__main__":
    main()
