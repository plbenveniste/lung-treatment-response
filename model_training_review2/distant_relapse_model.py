""""
This script is used to train and evaluate the model for the prediction of distant (meaning non-local) relapse of lung cancer.
It also demonstrates the results of the model on the test set.

Args:
    --input: path to the merged dataset
    --output: path to the output folder

Returns:
    None

Example run:
    python distant_relpase_model.py --input /path/to/merged.csv --output /path/to/output_folder

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, f1_score
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
    It does the training and evaluation of the distant relapse model.
    """

    # We parse the arguments
    args = parse_args()
    input_data = args.input
    output_folder = args.output

    # if the output folder does not exist, we create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    log_file = os.path.join(output_folder, f'distant_relapse.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    # Load the dataset
    data = pd.read_csv(input_data)

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DC', 'delai_fin_DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro',
                              'Date_R_horspoum','subject_nodule', 'nodule', 'Reponse', 'rechute_PTV','delai_fin_rechutePTV'])
    
    # For dosimetric data, we sum the features together by subject (so that if there is two nodules, the dosimetric data reflects the sum of the two doses)
    data_dosi = data[['subject_id', 'dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10', 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV']]
    # We group the data by subject and sum the dosi features
    data_dosi = data_dosi.groupby('subject_id').sum().reset_index()

    # For the rest of the data, we average
    data_rest = data.drop(columns=['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10', 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV'])
    data_rest = data_rest.groupby('subject_id').mean().reset_index()

    # We concatenate the dosimetric and rest of the data
    data_grouped = pd.merge(data_dosi, data_rest, on='subject_id', how='outer')

    # For all values in the 'rechute' columns, we replace values above 0 by 1 and the rest is 0
    data_grouped['rechute_homo'] = data_grouped['rechute_homo'].apply(lambda x: 1 if x > 0 else 0)
    data_grouped['rechute_med'] = data_grouped['rechute_med'].apply(lambda x: 1 if x > 0 else 0)
    data_grouped['rechute_contro'] = data_grouped['rechute_contro'].apply(lambda x: 1 if x > 0 else 0)
    data_grouped['rechute_horspoum'] = data_grouped['rechute_horspoum'].apply(lambda x: 1 if x > 0 else 0)

    # First we build a column which is 1 if the patient had a distant relapse and 0 otherwise
    # To do so we can sum the colums consisting of relapse (rechute)
    data_grouped['rechute_dist'] = data_grouped['rechute_homo'] + data_grouped['rechute_med'] + data_grouped['rechute_contro'] + data_grouped['rechute_horspoum']
    data_grouped['rechute_dist'] = data_grouped['rechute_dist'].apply(lambda x: 1 if x > 0 else 0)

    # Then we build a distant relapse delay which is the average of the delays of the different types of relapse
    data_grouped['rechute_dist_moy_delai'] = data_grouped[['delai_fin_rechuteHomo','delai_fin_rechuteMed','delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum']].mean(axis=1)

    # Split into features and target
    y = data_grouped[['rechute_dist','rechute_dist_moy_delai']]
    x = data_grouped.drop(columns=['rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechuteHomo', 'follow_up',
                                   'delai_fin_rechuteMed','delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum', 'subject_id', 'rechute_dist','rechute_dist_moy_delai'])
    
    # In this case, because we are only interested in the prediction of distant relapse, we extract only the 'DC'
    y = y[['rechute_dist']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which had a distant relapse: {y[y['rechute_dist']==1].shape[0]}")

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['rechute_dist_moy_delai'].hist()
    plt.title('Distribution of distant relapse time between end of treatment and\n occurence of a relapse (in days) for those that relapsed')
    plt.xlabel('Time')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, 'delai_fin_rechute_dist_histogram.png'))


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

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    return None


if __name__ == '__main__':
    main()