""""
This script is used to train and evaluate the model for local relapse.
It also demonstrates the results of the model on the test set.

Args:
    --input: path to the merged dataset
    --output: path to the output folder

Returns:
    None

Example run:
    python local_relapse_model.py --input /path/to/merged.csv --output /path/to/output_folder

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
from loguru import logger


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
    It does the training and evaluation of the local relapse model.
    """

    # We parse the arguments
    args = parse_args()
    input_data = args.input
    output_folder = args.output

    # If the output folder does not exist, we create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    log_file = os.path.join(output_folder, f'local_relapse.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    ########################################################################
    ############## DATA PRE-PROCESSING #####################################
    ########################################################################

    # Load the dataset
    data_load = pd.read_csv(input_data)

    # We remove data which is note useful to make the averaging easier
    data = data_load.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse','rechute_homo',
                           'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
                           'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule', 'follow_up','subject_id', 'nodule'])
    # For the local relapse, there is no need to average per subject -> we focus on nodules specifically
    y = data[['rechute_PTV']]
    x = data.drop(columns=['rechute_PTV'])

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which had a local relapse: {y[y['rechute_PTV']==1].shape[0]}")

    # Plot the distribution of 'delai_fin_DC'
    data_load['delai_fin_rechutePTV'].hist()
    plt.title('Distribution of time between end of treatment and local \nrelapse (in days) for those that had a local relapse')
    plt.xlabel('Time')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, 'delai_fin_rechutePTV_histogram.png'))

    # Box plot of the 'delai_fin_rechutePTV' column
    data_load.boxplot(column='delai_fin_rechutePTV')
    plt.title('Box plot of the time between end of treatment and local relapse')
    plt.ylabel('Time')
    plt.savefig(os.path.join(output_folder, 'delai_fin_rechutePTV_boxplot.png'))

    ########################################################################
    ############## MODEL TRAINING ##########################################
    ########################################################################

    # Final data used
    X = x
    y = y
    
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
            n_jobs=-1,
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

        # Get feature importances for this fold
        importances = best_model.feature_importances_
        feature_importances.append(importances)

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # Plot feature importance across folds
    ## Average feature importances across all folds
    avg_importances = np.mean(feature_importances, axis=0)
    ## Create a DataFrame for visualization
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_importances
    }).sort_values('Importance', ascending=False)
    # I also want to print the top 20 feature performances
    logger.info('Top feature importances: Importance, feature')
    for i, feature in enumerate(importance_df['Feature'][:top_n]):
        # Print feature and importance
        logger.info(f"{i+1}: {importance_df['Importance'].iloc[i]:.4f}: {feature}")



    return None


if __name__ == '__main__':
    main()