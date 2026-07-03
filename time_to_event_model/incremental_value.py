"""
In this file we quantify how much the radiomic features add to the overall-survival model
on the pooled cohort. We build three nested feature sets, clinical only, clinical plus
dosimetric, and clinical plus dosimetric plus radiomic, and compare their out-of-fold
Harrell C. The gain brought by the radiomic block is reported with a paired bootstrap 95%
confidence interval. These numbers and the accompanying plot feed Figure 5.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the results and the figure are written

Returns:
    None

Example:
    python model_training_review/incremental_value.py --merged-data data/patient_survival.csv --output-dir results/

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.model_selection import RepeatedStratifiedKFold

import survival_utils as su


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Incremental value of radiomics for overall survival")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def out_of_fold_by_set(data, feature_sets):
    """
    This function computes an out-of-fold risk score for each nested feature set using the
    same folds, so that the concordance indices can be compared on identical resamples.
    """
    event = data["os_event"].values.astype(int)
    origin = data["origin_meta"].values
    y = su.survival_target(data, "os")
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=su.RANDOM_STATE)
    risk = {name: np.zeros(len(data)) for name in feature_sets}
    folds = np.zeros(len(data))
    for train_idx, test_idx in splitter.split(data, event):
        folds[test_idx] += 1
        for name, columns in feature_sets.items():
            n_features = 5 if name == "clinical" else 8
            x_train, x_test = su.preprocess_fold(data.iloc[train_idx], data.iloc[test_idx], columns)
            selected = su.lasso_cox_select(x_train, y[train_idx], n_features)
            x_train, x_test = x_train[selected].copy(), x_test[selected].copy()
            x_train["origin_meta"] = origin[train_idx]
            x_test["origin_meta"] = origin[test_idx]
            model = su.build_rsf().fit(x_train, y[train_idx])
            risk[name][test_idx] += model.predict(x_test)
    for name in risk:
        risk[name] /= np.maximum(folds, 1)
    return risk, y, event


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = su.select_cohort(su.load_dataset(args.merged_data), "pooled")

    # We assemble the three nested feature sets.
    clinical = su.feature_columns(data, "clinical")
    dosimetric = su.feature_columns(data, "dosimetric")
    radiomic = su.feature_columns(data, "radiomic")
    feature_sets = {
        "clinical": clinical,
        "clinical+dosimetric": clinical + dosimetric,
        "clinical+dosimetric+radiomic": clinical + dosimetric + radiomic,
    }

    risk, y, event = out_of_fold_by_set(data, feature_sets)

    # We report the concordance index of each set and the radiomic gain with its interval.
    concordance = {name: round(float(su.harrell_c(y, risk[name])), 3) for name in feature_sets}
    full, without_radiomic = "clinical+dosimetric+radiomic", "clinical+dosimetric"
    delta = su.bootstrap_ci(
        lambda idx: su.harrell_c(y[idx], risk[full][idx]) - su.harrell_c(y[idx], risk[without_radiomic][idx]),
        event,
    )
    results = {"concordance": concordance, "radiomic_gain_ci": delta,
               "radiomic_gain": round(concordance[full] - concordance[without_radiomic], 3)}
    logger.info(f"Concordance by set: {concordance}")
    logger.info(f"Radiomic gain: {results['radiomic_gain']} {delta}")

    # We draw the incremental-value bar plot.
    labels = ["Clinical", "+ Dosimetric", "+ Radiomic"]
    values = [concordance["clinical"], concordance[without_radiomic], concordance[full]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, values, color=["#B0B0B0", "#7FA6C9", "#4C72B0"])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, max(values) + 0.08)
    ax.set_ylabel("Out-of-fold Harrell C")
    ax.set_title("Incremental value of radiomics (overall survival)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.tight_layout()
    figure_file = os.path.join(args.output_dir, "incremental_value.png")
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)

    output_file = os.path.join(args.output_dir, "incremental_value.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file} and {figure_file}")


if __name__ == "__main__":
    main()
