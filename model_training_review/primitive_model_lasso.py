"""
In this file we investigate the training of a model to predict survival in the following cases: 
    - for primitive patients

Args:
    --merged-data: Path to the csv file containing the merged data
    --output-dir: Path to the directory where the model will be saved

Returns:
    None

Example:
    python model_training/primitive_model.py --merged-data data/merged_data.csv --output-dir models/

Author: Pierre-Louis Benveniste
"""
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix, f1_score
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import pickle
# from sklearn.feature_selection import VarianceThreshold
import shap
from loguru import logger
import gc
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    parser = argparse.ArgumentParser(description='Preprocess the data from the lung cancer response dataset')
    parser.add_argument('--input', type=str, help='Path to the csv file containing the merged data')
    parser.add_argument('--output', type=str, help='Path to the folder where the results will be saved')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It does the training and evaluation of the survival model.
    """

    # We parse the arguments
    args = parse_args()
    input_data = args.input
    output_folder = args.output

    # If folder does not exist, we create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    log_file = os.path.join(output_folder, f'primitive.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    ########################################################################
    #################### DATA PREPROCESSING ################################
    ########################################################################

    # Load the dataset
    data = pd.read_csv(input_data)

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo',
                           'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
                           'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule', 'follow_up' ])
    

    # Remove taille and poids as features:
    data = data.drop(columns=['taille', 'poids'])

    # # For dosimetric data, we sum the features together by subject (so that if there is two nodules, the dosimetric data reflects the sum of the two doses)
    # data_dosi = data[['subject_id', 'dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10', 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV']]
    # # We group the data by subject and sum the dosi features
    # data_dosi = data_dosi.groupby('subject_id').sum().reset_index()

    # # For the rest of the data, we average
    # data_rest = data.drop(columns=['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10', 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV'])
    # data_rest = data_rest.groupby('subject_id').mean().reset_index()

    # # We concatenate the dosimetric and rest of the data
    # data_grouped = pd.merge(data_dosi, data_rest, on='subject_id', how='outer')
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # Print number of primitive patients and number of metastasis
    logger.info(f"Total number of patients: {data_grouped.shape[0]}")
    logger.info(f"Number of primitive patients: {data_grouped[data_grouped['primitif']==1].shape[0]}")
    logger.info(f"Number of metastasis patients: {data_grouped[data_grouped['primitif']!=1].shape[0]}")
    logger.info("\n")

    ################################################################################################
    ######## MODEL TRAINING: LASSO-XGBOOST METHOD ##############
    ################################################################################################
    # On each inner folds, Lasso is used to select features
    # Then XGBoost is used to train a model on the selected features on each inner folds.
    logger.info(" ------------- Model for prediction of survival for primitive patients -------------")

    # We keep primitive patients
    data_primitive = data_grouped[data_grouped['primitif'] == 1]

    # Split into features and target
    y = data_primitive[['DC']]
    x = data_primitive.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'primitif'])
    logger.info(f"Number of primitive patients: {x.shape[0]}")
    logger.info(f"Number of features: {x.shape[1]}")

    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which died: {y[y['DC']==1].shape[0]}")
    logger.info(f"Feature columns: {list(x.columns)}")

    # Plot the distribution of 'delai_fin_DC'
    data_primitive['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time of primitive patients')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, "survival_time_distribution_primitive_patients.png"))
    plt.close()

    # We perform bayesian hyperparameter tuning
    # I used this blog to build it: https://xgboosting.com/most-important-xgboost-hyperparameters-to-tune/
    search_spaces = {
        'max_depth': Integer(3, 10), # Lower values prevent overfitting
        'min_child_weight': Integer(1, 10), # Higher values prevent overfitting # Suggested to go as high as 5 by LeChat
        'subsample': Real(0.5, 1), # Lower values prevent overfitting
        'colsample_bytree': Real(0.001, 1), # Lower values prevent overfitting # Suggested to go as low as 0.5 by LeChat
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'scale_pos_weight': Integer(2, 8), # To handle unbalanced classes
    }

    X = x
    y = y

    # Print proportion of NaN values in x
    nan_proportions = x.isna().mean()
    logger.info(f"Proportion of NaN values in features:\n{nan_proportions}")
    logger.info(f"Total proportion of NaN values in features: {x.isna().mean().mean():.4f}")
    # Print all columns which contain some nan values and print the ratio
    for col in x.columns:
        if x[col].isna().sum() > 0:
            logger.info(f"Column '{col}' has {x[col].isna().sum()} NaN values ({x[col].isna().mean():.4f} proportion)")
    
    # Define Outer CV for unbiased performance estimation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Search space for XGBoost (prefixed with 'model__' for the Pipeline)
    search_spaces = {
        'model__max_depth': Integer(3, 10),
        'model__min_child_weight': Integer(1, 10),
        'model__subsample': Real(0.5, 1),
        'model__colsample_bytree': Real(0.01, 1),
        'model__learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'model__scale_pos_weight': Integer(2, 8),
    }

    outer_results = []
    calibration_curves = []
    feature_importances_df = pd.DataFrame(columns=['Feature', 'Importance', 'Count_Selected'])

    # Dictionary to track how many times each feature is selected across outer folds
    feature_stability_tracker = {feat: 0 for feat in X.columns}
    
    logger.info("Starting Nested CV with Lasso-XGBoost Hybrid...")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Define the Hybrid Pipeline
        # 1. StandardScaler: Lasso is scale-sensitive
        # 2. SelectFromModel (Lasso): Limits to top 20 features
        # 3. XGBoost: Learns non-linear interactions on those 20 features
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
                max_features=20, 
                threshold=-np.inf  # Force exactly 20 features
            )),
            ('model', XGBClassifier(
                seed=42, use_label_encoder=False, eval_metric="logloss"
            ))
        ])

        # Inner Loop for Hyperparameter Tuning
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_spaces,
            n_iter=50,
            cv=inner_cv,
            n_jobs=-1,
            random_state=42,
            scoring='roc_auc'
        )

        search.fit(X_train, y_train)

        # TRACK STABILITY: Identify which 20 features were picked for this fold
        selected_mask = search.best_estimator_.named_steps['selector'].get_support()
        selected_features = X.columns[selected_mask].tolist()
        # Also export feature importance from the Lasso model for this fold
        feature_importances = search.best_estimator_.named_steps['selector'].estimator_.coef_[0]
        for feat in selected_features:
            imp = feature_importances[X.columns.get_loc(feat)]
            # If feature is already in the df, we add the importance to the existing one and increment the count, otherwise we create a new row
            if feat in feature_importances_df['Feature'].values:
                feature_importances_df.loc[feature_importances_df['Feature'] == feat, 'Importance'] += abs(imp)
                feature_importances_df.loc[feature_importances_df['Feature'] == feat, 'Count_Selected'] += 1
            else:
                new_row = {'Feature': feat, 'Importance': abs(imp), 'Count_Selected': 1 if feat in selected_features else 0}
                feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([new_row])], ignore_index=True)
            # Update the stability tracker
            if feat in selected_features:
                feature_stability_tracker[feat] += 1

        # Performance Evaluation on the unseen Outer Fold (Test Set)
        best_model = search.best_estimator_
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = best_model.predict(X_test)

        precision_train, recall_train, _ = precision_recall_curve(y_test, y_test_proba)

        metrics = {
            "Fold": fold_idx,
            "ROC AUC": roc_auc_score(y_test, y_test_proba),
            "Brier": brier_score_loss(y_test, y_test_proba),
            "Precision": precision_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "AUC-PR": auc(recall_train, precision_train),
            "F1_score": f1_score(y_test, y_test_pred)
        }

        # Get calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
        calibration_curves.append((prob_true, prob_pred))
        
        outer_results.append(metrics)
        logger.info(f"Fold {fold_idx} completed. Features picked: {len(selected_features)}")

    # SUMMARY REPORTING
    results_df = pd.DataFrame(outer_results)
    logger.info("\nNested CV mean Metrics:")
    # Print metrics for each fold in a table format
    logger.info(results_df.to_string(index=False))
    # Print results with std
    for metric in ["ROC AUC", "Brier", "Precision", "Recall", "Accuracy", "AUC-PR", "F1_score"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    logger.info("Feature Importances across folds:")
    feature_importances_df = feature_importances_df.sort_values(by='Count_Selected', ascending=False)
    # Remove features which were never selected
    feature_importances_df = feature_importances_df[feature_importances_df['Count_Selected'] > 0]
    # Log all lines of the df
    for _, row in feature_importances_df.iterrows():
        logger.info(f"Feature: {row['Feature']}, Total Importance: {row['Importance']:.4f}, Selected in {row['Count_Selected']} folds")
    # Save the feature importances to a csv file
    feature_importances_df.to_csv(os.path.join(output_folder, "feature_importances_report.csv"), index=False)

    # Export Feature Stability to CSV
    stability_df = pd.DataFrame.from_dict(feature_stability_tracker, orient='index', columns=['Selection_Count'])
    stability_df = stability_df.sort_values(by='Selection_Count', ascending=False)
    stability_df.to_csv(os.path.join(output_folder, "feature_stability_report.csv"))

    # We plot the calibration curve averaged over the folds
    plt.figure()
    for i, (prob_true, prob_pred) in enumerate(calibration_curves):
        plt.plot(prob_pred, prob_true, marker='o', label=f'Fold {i+1}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curves for each fold')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'calibration_curves_folds_primitive.png'))
    plt.close()

    # Define a common x range limited strictly to [0, 1]
    x_common = np.linspace(0, 1, 100)

    interpolated_curves = []
    for prob_true, prob_pred in calibration_curves:
        # Ensure sorting of the calibration points
        order = np.argsort(prob_pred)
        prob_pred = np.array(prob_pred)[order]
        prob_true = np.array(prob_true)[order]

        # Bound the interpolation domain to [min, max] and avoid extrapolation
        interp_func = interp1d(
            prob_pred, prob_true,
            kind="linear",
            bounds_error=False,
            fill_value=(prob_true[0], prob_true[-1])
        )
        y_interp = interp_func(x_common)
        # Clip to [0,1] in case of minor numerical drift
        y_interp = np.clip(y_interp, 0, 1)
        interpolated_curves.append(y_interp)

    mean_prob_true = np.mean(interpolated_curves, axis=0)

    # Plot the mean calibration curve
    plt.figure()
    plt.plot(x_common, mean_prob_true, marker='.', label='Mean calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Mean Calibration curve across folds')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'mean_calibration_curve_primitive.png'))
    plt.close()


if __name__ == '__main__':
    main()