"""
In this file we draw the two figures that are assembled from the saved results rather than
from a single model fit: the study flow diagram (Figure 1) and the time-dependent AUC of the
overall-survival model at 1, 2 and 3 years with its bootstrap confidence intervals (Figure 2).
The calibration, SHAP and incremental-value figures are produced by their own scripts.

Args:
    --results-dir: Path to the directory holding overall_survival.json
    --output-dir: Path to the directory where the figures are written

Returns:
    None

Example:
    python model_training_review/plot_figures.py --results-dir results/ --output-dir results/

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


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Draw the flow diagram and the time-dependent AUC figure")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with overall_survival.json")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for the figures")
    return parser.parse_args()


def plot_flow_diagram(output_dir):
    """
    This function draws a simple TRIPOD-style flow diagram of the cohort assembly. The counts
    are the ones reported in the manuscript.
    """
    steps = [
        "181 lesions in 163 patients with radiomics",
        "Double segmentation, features with ICC >= 0.80 retained",
        "Patient-level aggregation (one row per patient)",
        "158 patients analysed (106 primary, 52 metastatic)",
        "Overall survival (primary) + RFS / MFS (secondary)",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.axis("off")
    y = 0.9
    for i, text in enumerate(steps):
        ax.add_patch(plt.Rectangle((0.05, y - 0.06), 0.9, 0.11, fill=True, facecolor="#EDF2F7", edgecolor="#4C72B0"))
        ax.text(0.5, y, text, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.14), arrowprops=dict(arrowstyle="->", color="#4C72B0"))
        y -= 0.18
    figure_file = os.path.join(output_dir, "flow_diagram.png")
    fig.savefig(figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_file


def plot_time_auc(results_dir, output_dir):
    """
    This function plots the time-dependent AUC at 1, 2 and 3 years for each cohort, with the
    bootstrap confidence intervals read from overall_survival.json.
    """
    with open(os.path.join(results_dir, "overall_survival.json")) as handle:
        results = json.load(handle)
    horizons = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colours = {"primary": "#4C72B0", "metastatic": "#DD8452", "pooled": "#55A868"}
    for cohort, colour in colours.items():
        values = [results[cohort][f"auc_{h}y"] for h in horizons]
        lower = [results[cohort][f"auc_{h}y"] - results[cohort][f"auc_{h}y_ci"][0] for h in horizons]
        upper = [results[cohort][f"auc_{h}y_ci"][1] - results[cohort][f"auc_{h}y"] for h in horizons]
        ax.errorbar(horizons, values, yerr=[lower, upper], marker="o", capsize=3, color=colour,
                    label=f"{cohort.capitalize()} (C {results[cohort]['harrell']})")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(horizons)
    ax.set_xlabel("Years since end of radiotherapy")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title("Overall-survival discrimination over time")
    ax.legend()
    figure_file = os.path.join(output_dir, "time_dependent_auc.png")
    fig.tight_layout()
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)
    return figure_file


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    flow = plot_flow_diagram(args.output_dir)
    auc = plot_time_auc(args.results_dir, args.output_dir)
    logger.info(f"Saved {flow} and {auc}")


if __name__ == "__main__":
    main()
