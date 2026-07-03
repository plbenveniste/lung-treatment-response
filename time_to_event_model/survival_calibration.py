"""
In this file we assess the calibration of the overall-survival model on the pooled cohort,
using the XGBoost AFT model that provides a full predicted survival function. For each of
the 1, 2 and 3-year horizons we bin the patients by predicted survival, and compare the mean
predicted survival of each bin with the Kaplan-Meier estimate observed in that bin. The
resulting reliability curves feed Figure 3.

Args:
    --merged-data: Path to the patient-level survival dataset (csv)
    --output-dir: Path to the directory where the figure and the numbers are written

Returns:
    None

Example:
    python model_training_review/survival_calibration.py --merged-data data/patient_survival.csv --output-dir results/

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from loguru import logger
from scipy.stats import norm
from sklearn.model_selection import RepeatedStratifiedKFold

import survival_utils as su

HORIZON_LABELS = ["1 year", "2 years", "3 years"]


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Survival calibration of the overall-survival model")
    parser.add_argument("--merged-data", type=str, help="Path to the patient-level survival dataset")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the results")
    return parser.parse_args()


def out_of_fold_survival(data, sigma=1.0):
    """
    This function returns, for every patient, the AFT-predicted survival probability at each
    horizon, obtained out-of-fold so that predictions are not made on training patients.
    """
    columns = su.feature_columns(data, "all")
    time = data["os_time"].values.astype(float)
    event = data["os_event"].values.astype(int)
    origin = data["origin_meta"].values
    y = su.survival_target(data, "os")

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=su.RANDOM_STATE)
    predicted = np.zeros((len(data), len(su.HORIZONS)))
    folds = np.zeros(len(data))
    for train_idx, test_idx in splitter.split(data, event):
        x_train, x_test = su.preprocess_fold(data.iloc[train_idx], data.iloc[test_idx], columns)
        selected = su.lasso_cox_select(x_train, y[train_idx], 8)
        x_train, x_test = x_train[selected].copy(), x_test[selected].copy()
        x_train["origin_meta"] = origin[train_idx]
        x_test["origin_meta"] = origin[test_idx]
        model = su.fit_aft(x_train, time[train_idx], event[train_idx])
        import xgboost as xgb

        mu = np.log(np.clip(model.predict(xgb.DMatrix(x_test)), 1e-6, None))
        # Under a log-normal AFT, S(t) = 1 - Phi((log t - mu) / sigma).
        for j, horizon in enumerate(su.HORIZONS):
            predicted[test_idx, j] += 1 - norm.cdf((np.log(horizon) - mu) / sigma)
        folds[test_idx] += 1
    predicted /= np.maximum(folds, 1)[:, None]
    return predicted, time, event


def reliability(predicted, time, event, horizon, n_bins=4):
    """
    This function compares mean predicted survival with the Kaplan-Meier estimate inside bins
    of predicted survival, returning the (predicted, observed) pairs.
    """
    order = np.argsort(predicted)
    bins = np.array_split(order, n_bins)
    pairs = []
    for group in bins:
        if len(group) < 5:
            continue
        km = KaplanMeierFitter().fit(time[group], event[group])
        observed = float(km.predict(horizon))
        pairs.append((float(predicted[group].mean()), observed, len(group)))
    return pairs


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = su.select_cohort(su.load_dataset(args.merged_data), "pooled")
    predicted, time, event = out_of_fold_survival(data)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, label, horizon, column in zip(axes, HORIZON_LABELS, su.HORIZONS, range(len(su.HORIZONS))):
        pairs = reliability(predicted[:, column], time, event, horizon)
        results[label] = pairs
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.plot([p for p, _, _ in pairs], [o for _, o, _ in pairs], marker="o", color="#4C72B0")
        ax.set_title(label)
        ax.set_xlabel("Predicted survival")
        ax.set_ylabel("Observed survival (KM)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Overall-survival calibration (pooled cohort)")
    fig.tight_layout()
    figure_file = os.path.join(args.output_dir, "calibration.png")
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)

    output_file = os.path.join(args.output_dir, "calibration.json")
    with open(output_file, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Saved {output_file} and {figure_file}")


if __name__ == "__main__":
    main()
