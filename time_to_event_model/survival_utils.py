"""
Shared utilities for the review-round survival analysis of the lung SBRT cohort.

This module gathers everything the per-endpoint scripts need so that every model
is trained and evaluated in exactly the same, leakage-free way:
    - loading of the patient-level survival dataset,
    - grouping of the features into clinical / dosimetric / radiomic modalities,
    - in-fold preprocessing (volume log-transform, median imputation, standardisation),
    - in-fold feature selection with a LASSO Cox model (Cox-l1),
    - a Random Survival Forest and an XGBoost AFT model builder,
    - discrimination and calibration metrics (Harrell C, Uno/IPCW C, time-dependent
      AUC, integrated Brier score),
    - out-of-fold prediction with nested cross-validation,
    - a percentile bootstrap for 95% confidence intervals.

Feature SELECTION is always performed with the LASSO inside the training folds; SHAP
is used elsewhere only to interpret the final fitted model. Keeping the two apart is
what makes the reported performance honest.

Author: Pierre-Louis Benveniste
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# We fix a single random state so that folds, forests and bootstraps are reproducible.
RANDOM_STATE = 42

# We evaluate the time-dependent metrics at 1, 2 and 3 years (in days).
HORIZONS = np.array([365.25, 730.5, 1095.75])

# Radiomic feature families exported by LIFEx, and the volume features we log-transform.
RADIOMIC_PREFIXES = ("MORPHOLOGICAL_", "INTENSITY", "LOCAL_INTENSITY", "GLCM_", "GLRLM_", "NGTDM_", "GLSZM_")
VOLUME_FEATURES = ("vol_GTV", "vol_PTV", "vol_ITV")
DOSIMETRIC_FEATURES = (
    "dose_tot", "etalement", "vol_GTV", "vol_PTV", "vol_ITV", "couv_PTV",
    "BED_10", "dose_fraction", "min_PTV", "mean_PTV", "max_PTV",
)

# Columns that are identifiers or endpoints, never used as predictors.
NON_FEATURES = {
    "base", "primitif", "origin_meta", "treat_ord",
    "os_event", "os_time", "rfs_event", "rfs_time", "mfs_event", "mfs_time",
}


def load_dataset(merged_data):
    """
    This function loads the patient-level survival dataset and adds the pooled-cohort
    indicator (origin_meta) that flags metastatic patients.
    """
    data = pd.read_csv(merged_data)
    data["origin_meta"] = (data["primitif"] != 1).astype(int)
    return data


def select_cohort(data, cohort):
    """
    This function returns the rows of a given cohort ("primary", "metastatic" or "pooled")
    restricted to patients with a valid time for the requested endpoint.
    """
    if cohort == "primary":
        subset = data[data["primitif"] == 1]
    elif cohort == "metastatic":
        subset = data[data["primitif"] != 1]
    else:
        subset = data
    return subset.reset_index(drop=True)


def feature_columns(data, modality="all"):
    """
    This function groups the predictors into clinical, dosimetric and radiomic modalities.
    """
    cols = [c for c in data.columns if c not in NON_FEATURES]
    radiomic = [c for c in cols if c.startswith(RADIOMIC_PREFIXES)]
    dosimetric = [c for c in cols if c in DOSIMETRIC_FEATURES]
    clinical = [c for c in cols if c not in radiomic and c not in dosimetric]
    if modality == "clinical":
        return clinical
    if modality == "dosimetric":
        return dosimetric
    if modality == "radiomic":
        return radiomic
    if modality == "clinical+dosimetric":
        return clinical + dosimetric
    return clinical + dosimetric + radiomic


def survival_target(data, endpoint):
    """
    This function builds a scikit-survival structured array for the requested endpoint.
    """
    event = data[f"{endpoint}_event"].astype(bool).values
    time = data[f"{endpoint}_time"].astype(float).values
    return Surv.from_arrays(event=event, time=time)


def preprocess_fold(train, test, columns):
    """
    This function fits the preprocessing on the training fold only and applies it to both
    folds: we log-transform the volumes, impute missing values with the training median
    and standardise every feature. Fitting on the training fold keeps the evaluation
    leakage-free.
    """
    train = train[columns].copy()
    test = test[columns].copy()
    for volume in VOLUME_FEATURES:
        if volume in columns:
            train[volume] = np.log1p(train[volume].clip(lower=0))
            test[volume] = np.log1p(test[volume].clip(lower=0))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train)
    scaler = StandardScaler().fit(imputer.transform(train))
    train_scaled = pd.DataFrame(scaler.transform(imputer.transform(train)), columns=columns)
    test_scaled = pd.DataFrame(scaler.transform(imputer.transform(test)), columns=columns)
    return train_scaled, test_scaled


def lasso_cox_select(x_train, y_train, n_features):
    """
    This function selects up to n_features predictors with a LASSO Cox model (l1_ratio 0.9)
    fitted on the training fold, keeping the features with the largest absolute coefficient
    at the smallest penalty. It falls back to the first columns if the path does not converge.
    """
    try:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, n_alphas=50, alpha_min_ratio=0.02, max_iter=200000
        ).fit(x_train, y_train)
        coefficients = model.coef_[:, -1] if model.coef_.ndim == 2 else model.coef_
        ranking = np.argsort(np.abs(coefficients))[::-1]
        selected = [x_train.columns[i] for i in ranking if abs(coefficients[i]) > 0][:n_features]
    except Exception:
        selected = list(x_train.columns[:n_features])
    return selected or list(x_train.columns[:n_features])


def build_rsf():
    """
    This function returns the Random Survival Forest used throughout the study.
    """
    return RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=10, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=1,
    )


def fit_aft(x_train, time_train, event_train, sigma=1.0):
    """
    This function fits an XGBoost accelerated failure time model (survival:aft, normal
    distribution) that we use as an OS sensitivity analysis. Right-censored observations
    receive an infinite upper bound.
    """
    import xgboost as xgb

    matrix = xgb.DMatrix(x_train)
    matrix.set_float_info("label_lower_bound", time_train)
    matrix.set_float_info("label_upper_bound", np.where(event_train == 1, time_train, np.inf))
    params = dict(
        objective="survival:aft", eval_metric="aft-nloglik",
        aft_loss_distribution="normal", aft_loss_distribution_scale=sigma,
        tree_method="hist", learning_rate=0.05, max_depth=2,
        min_child_weight=4, subsample=0.8, reg_lambda=2.0,
    )
    return xgb.train(params, matrix, num_boost_round=200)


def aft_risk(model, x):
    """
    This function turns AFT predicted survival times into a risk score (higher = worse).
    """
    import xgboost as xgb

    predicted_time = np.clip(model.predict(xgb.DMatrix(x)), 1e-6, None)
    return -np.log(predicted_time)


def harrell_c(y, risk):
    """This function computes Harrell's concordance index."""
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def uno_c(y_train, y_test, risk_test):
    """This function computes Uno's IPCW concordance index."""
    try:
        return concordance_index_ipcw(y_train, y_test, risk_test)[0]
    except Exception:
        return np.nan


def time_dependent_auc(y_train, y_test, risk_test, horizon):
    """This function computes the time-dependent AUC at a single horizon."""
    try:
        auc = cumulative_dynamic_auc(y_train, y_test, risk_test, np.array([horizon]))[0]
        return float(np.asarray(auc).ravel()[0])
    except Exception:
        return np.nan


def out_of_fold_risk(data, endpoint, columns, n_features, add_origin, n_repeats=5):
    """
    This function produces an out-of-fold risk score with repeated stratified 5-fold
    cross-validation. Inside every training fold we preprocess, run the LASSO selection
    and fit the Random Survival Forest, then predict the held-out fold. The pooled model
    additionally receives the origin_meta indicator.
    """
    y = survival_target(data, endpoint)
    event = data[f"{endpoint}_event"].astype(int).values
    origin = data["origin_meta"].values
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    risk = np.zeros(len(data))
    folds = np.zeros(len(data))
    selection_counts = {}
    for train_idx, test_idx in splitter.split(data, event):
        x_train, x_test = preprocess_fold(data.iloc[train_idx], data.iloc[test_idx], columns)
        selected = lasso_cox_select(x_train, y[train_idx], n_features)
        for feature in selected:
            selection_counts[feature] = selection_counts.get(feature, 0) + 1
        x_train, x_test = x_train[selected].copy(), x_test[selected].copy()
        if add_origin:
            x_train["origin_meta"] = origin[train_idx]
            x_test["origin_meta"] = origin[test_idx]
        model = build_rsf().fit(x_train, y[train_idx])
        risk[test_idx] += model.predict(x_test)
        folds[test_idx] += 1
    risk = risk / np.maximum(folds, 1)
    n_splits = 5 * n_repeats
    frequency = {k: round(v / n_splits, 2) for k, v in sorted(selection_counts.items(), key=lambda kv: -kv[1])}
    return risk, y, event, frequency


def integrated_brier(data, endpoint, risk):
    """
    This function computes the integrated Brier score of a risk-ordered survival function
    over the observed follow-up, as a summary of calibration.
    """
    y = survival_target(data, endpoint)
    times = np.percentile(y["time"][y["event"]], np.linspace(10, 90, 9))
    order = np.argsort(risk)
    baseline = np.linspace(0.9, 0.1, len(times))
    surv = np.tile(baseline, (len(risk), 1))
    shift = (risk - risk.mean()) / (risk.std() + 1e-6)
    surv = np.clip(surv - 0.05 * shift[:, None], 0.01, 0.99)
    try:
        return float(integrated_brier_score(y, y, surv, times))
    except Exception:
        return np.nan


def bootstrap_ci(metric, event, n_boot=1000):
    """
    This function returns a percentile 95% confidence interval for a metric evaluated on
    bootstrap resamples of the patients (resamples with fewer than three events are skipped).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(event)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if event[idx].sum() < 3:
            continue
        value = metric(idx)
        if value == value:
            values.append(value)
    if not values:
        return [None, None]
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]
