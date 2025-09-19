""""
This script is used to train and evaluate the model for the survival prediction of lung cancer patients.
It uses nested cross-validation to evaluate the model and bayesian optimisation to tune the hyperparameters.

Args:
    --input: path to the merged dataset
    --output: path to the output folder
    --nb-years: number of years to consider for the survival prediction (default: 3)

Returns:
    None

Example run:
    python survival_model.py --input /path/to/merged.csv --output /path/to/output_folder --nb-years 3

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
from sklearn.feature_selection import VarianceThreshold
import shap
from loguru import logger
import gc
from sklearn.calibration import calibration_curve
from scipy.interpolate import interp1d

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
    parser.add_argument('--nb-years', type=int, default=3, help='Number of years to consider for the survival prediction')
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
    nb_years = args.nb_years

    # If the output_folder doesn't exist create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    log_file = os.path.join(output_folder, f'survival_model_{nb_years}_years.txt')
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
    
    # For dosimetric data, we sum the features together by subject (so that if there is two nodules, the dosimetric data reflects the sum of the two doses)
    data_dosi = data[['subject_id', 'dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']]
    # We group the data by subject and sum the dosi features
    data_dosi = data_dosi.groupby('subject_id').sum().reset_index()

    # For the rest of the data, we average
    data_rest = data.drop(columns=['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10'])
    data_rest = data_rest.groupby('subject_id').mean().reset_index()

    # We concatenate the dosimetric and rest of the data
    data_grouped = pd.merge(data_dosi, data_rest, on='subject_id', how='outer')

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, 'delai_fin_DC_distribution.png'))
    plt.close()

    # Split into features and target
    y = data_grouped[['DC', 'delai_fin_DC']]
    x = data_grouped.drop(columns=['DC', 'delai_fin_DC', 'subject_id'])
    logger.info(f"Final features in the dataset: {str(list(x.columns))}")
    logger.info("\n")

    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which died: {y[y['DC']==1].shape[0]}")
    logger.info(f"Total number of subjects: {y.shape[0]}")
    logger.info("\n")

    # Now we consider that every person that die after X year is considered as not dead (using the delai_fin_DC column)
    y_deadline_X_year = data_grouped[['delai_fin_DC']]
    y_deadline_X_year.fillna(1e6, inplace=True)
    y_deadline_X_year['delai_fin_DC'] = y_deadline_X_year['delai_fin_DC'].apply(lambda x: 0 if x > round(nb_years*365) else 1)
    logger.info(f"Number of subjects that died within {nb_years} year: {y_deadline_X_year[y_deadline_X_year['delai_fin_DC'] == 1].shape[0]}")

    # Final data used
    X = x
    y = y_deadline_X_year['delai_fin_DC']

    ##########################################
    # Training of the model with all features:
    ##########################################
    
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
    
    # === Définition des folds externes (évaluation) ===
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Stockage des résultats
    outer_results = []
    feature_importances = []
    best_params_list = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # === Recherche Bayésienne interne (hyperparamètres) ===
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base_model = XGBClassifier(seed=42, use_label_encoder=False, eval_metric="logloss", objective='binary:logistic')

        search = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=50,                # Ajustable
            n_jobs=1,
            cv=inner_cv,
            random_state=42,
            scoring='roc_auc'
        )
        search.fit(X_train, y_train)

        # Meilleur modèle trouvé
        best_model = search.best_estimator_

        # === Évaluation sur le fold externe ===
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        precision_train, recall_train, _ = precision_recall_curve(y_test, y_test_proba)

        # Calcul des métriques
        metrics = {
            "ROC AUC": roc_auc_score(y_test, y_test_proba),
            "Brier": brier_score_loss(y_test, y_test_proba),
            "Precision": precision_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "AUC-PR": auc(recall_train, precision_train),
            "F1_score": f1_score(y_test, y_test_pred)
        }
        outer_results.append(metrics)
        best_params_list.append(search.best_params_)
        logger.info("=== Fold results ===")
        logger.info(f"Fold results: {metrics}")
        logger.info(f"Best hyperparameters: {search.best_params_}")

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # For the best model, we get the feature importance using SHAP
    best_overall_model_idx = np.argmax([res['ROC AUC'] for res in outer_results])
    best_overall_params = best_params_list[best_overall_model_idx]
    best_overall_model = XGBClassifier(seed=42, use_label_encoder=False, eval_metric="logloss", objective='binary:logistic', **best_overall_params)
    best_overall_model.fit(X, y)
    explainer = shap.Explainer(best_overall_model, seed=42)
    importances = np.abs(explainer.shap_values(X)).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    # Print the top 20 feature performances
    logger.info('Top feature importances from the best overall model: Importance, feature')
    for i, feature in enumerate(importance_df['Feature'][:20]):
        # Print feature and importance
        logger.info(f"{i+1}: {importance_df['Importance'].iloc[i]:.4f}: {feature}")

    # free memory 
    del best_overall_model, explainer, importances, results_df, outer_results, best_params_list, search, base_model
    gc.collect()

    #########################################################
    # Training of the final model which uses only 20 features:
    #########################################################
    # We select the 20 most important features
    top_n = 20
    selected_features = importance_df['Feature'][:top_n].values
    X_selected = X[selected_features]
    logger.info(f"Features selected for the final model: {str(list(selected_features))}")

    # We keep the same search space and outer_cv
    search_spaces = search_spaces
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Stockage des résultats
    outer_results = []
    feature_importances = []
    calibration_curves = []
    best_params_list = []
    for train_idx, test_idx in outer_cv.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # === Recherche Bayésienne interne (hyperparamètres) ===
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base_model = XGBClassifier(seed=42, use_label_encoder=False, eval_metric="logloss", objective='binary:logistic')

        search = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=50,                # Ajustable
            n_jobs=1,
            cv=inner_cv,
            random_state=42,
            scoring='roc_auc'
        )
        search.fit(X_train, y_train)

        # Meilleur modèle trouvé
        best_model = search.best_estimator_

        # === Évaluation sur le fold externe ===
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        precision_train, recall_train, _ = precision_recall_curve(y_test, y_test_proba)

        # Calcul des métriques
        metrics = {
            "ROC AUC": roc_auc_score(y_test, y_test_proba),
            "Brier": brier_score_loss(y_test, y_test_proba),
            "Precision": precision_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "AUC-PR": auc(recall_train, precision_train),
            "F1_score": f1_score(y_test, y_test_pred)
        }
        outer_results.append(metrics)
        best_params_list.append(search.best_params_)
        logger.info("=== Fold results ===")
        logger.info(f"Fold results: {metrics}")
        logger.info(f"Best hyperparameters: {search.best_params_}")

        # Get calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
        calibration_curves.append((prob_true, prob_pred))

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # We plot the calibration curve averaged over the folds
    plt.figure()
    for i, (prob_true, prob_pred) in enumerate(calibration_curves):
        plt.plot(prob_pred, prob_true, marker='o', label=f'Fold {i+1}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curves for each fold')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'calibration_curves_folds.png'))
    plt.close()

    # Define a common set of x-values (predicted probabilities) for interpolation
    x_common = np.linspace(0, 1, 100)

    # Interpolate each curve to the common x-values
    interpolated_curves = []
    for prob_pred, prob_true in calibration_curves:
        interp_func = interp1d(prob_pred, prob_true, bounds_error=False, fill_value="extrapolate")
        y_interp = interp_func(x_common)
        interpolated_curves.append(y_interp)

    # Average the interpolated curves
    mean_prob_true = np.mean(interpolated_curves, axis=0)

    # Plot the mean calibration curve
    plt.figure()
    plt.plot(x_common, mean_prob_true, marker='.', label='Mean calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Mean Calibration curve across folds')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'mean_calibration_curve.png'))
    plt.close()

    # Need to remove some stuff to free memory
    del base_model, search, best_model, X_train, X_test, y_train, y_test, X_selected
    gc.collect()    

    ####################################################
    # Final model training on the whole dataset with the selected features
    ####################################################
    logger.info("\n=== Final model training on the whole dataset with the selected features ===")
    # Select features
    X_final = X[selected_features]
    y_final = y

    # Identify the best hyperparameters based on previous tuning
    best_hyperparams = best_params_list[np.argmax([res['ROC AUC'] for res in outer_results])]
    logger.info(f"Best hyperparameters for the final model: {best_hyperparams}")

    # Train the final model
    final_model = XGBClassifier(seed=42, use_label_encoder=False, eval_metric="logloss", objective='binary:logistic', **best_hyperparams)
    final_model.fit(X_final, y_final)

    # Save the final model
    model_path = os.path.join(output_folder, 'final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"Final model saved to {model_path}")   

    # Eval the final model on the whole dataset
    y_final_pred = final_model.predict(X_final)
    y_final_proba = final_model.predict_proba(X_final)[:, 1]

    # Calcul des métriques
    precision_final, recall_final, _ = precision_recall_curve(y_final, y_final_proba)
    metrics = {
        "ROC AUC": roc_auc_score(y_final, y_final_proba),
        "Brier": brier_score_loss(y_final, y_final_proba),
        "Precision": precision_score(y_final, y_final_pred),
        "Recall": recall_score(y_final, y_final_pred),
        "Accuracy": accuracy_score(y_final, y_final_pred),
        "AUC-PR": auc(recall_final, precision_final),
        "F1_score": f1_score(y_final, y_final_pred)
    }
    logger.info("=== Fold results ===")
    logger.info(f"Fold results: {metrics}")

    return None


if __name__ == '__main__':
    main()