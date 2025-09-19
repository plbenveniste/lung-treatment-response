"""
In this file we investigate the training of a model to predict survival in the following cases: 
    - for primitive patients
    - for metastasis patients
    - for all patients but with a feature indicating whether the patient is primitive or metastasis

Args:
    --merged-data: Path to the csv file containing the merged data
    --output-dir: Path to the directory where the model will be saved

Returns:
    None

Example:
    python model_training/primitif_vs_metastase_model.py --merged-data data/merged_data.csv --output-dir models/

Author: Pierre-Louis Benveniste
"""
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.feature_selection import VarianceThreshold
import os
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer


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
    log_file = os.path.join(output_folder, f'primitive_vs_metastasis.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    ########################################################################
    #################### DATA PREPROCESSING ################################
    ########################################################################

    # Load the dataset
    data = pd.read_csv(input_data)

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo',
                                'delai_fin_rechuteMed', 'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule' ])
    

    # We average the columns for the same patients across the different nodules
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # We build the column relapse which takes 1 if either of the relapse occur ('rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum')
    data_grouped['relapse'] = data_grouped['rechute_PTV'] + data_grouped['rechute_homo'] + data_grouped['rechute_med'] + data_grouped['rechute_contro'] + data_grouped['rechute_horspoum']
    # Modify the 'relapse' column to take value 1 if it is above 0
    data_grouped['relapse'] = data_grouped['relapse'].apply(lambda x: 1 if x > 0 else 0)

    # Print number of primitive patients and number of metastasis
    logger.info(f"Total number of patients: {data_grouped.shape[0]}")
    logger.info(f"Number of primitive patients: {data_grouped[data_grouped['relapse']==0].shape[0]}")
    logger.info(f"Number of metastasis patients: {data_grouped[data_grouped['relapse']!=0].shape[0]}")
    logger.info("\n")

    ################################################################################################
    ######## FIRST SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR PRIMITIVE PATIENTS ##############
    ################################################################################################

    logger.info(" ------------- Model for prediction of survival for primitive patients -------------")

    # We remove patients that are metastasis (have relapse different from 0)
    data_primitive = data_grouped[data_grouped['relapse'] == 0]

    # Split into features and target
    y = data_primitive[['DC']]
    x = data_primitive.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'relapse', 'follow_up', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum'])
    
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

        # Feature importances
        feature_importances.append(best_model.feature_importances_)

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # Print the feature importances
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns)
    feature_importances_df = feature_importances_df.describe().T[['mean', 'std']].sort_values(by='mean', ascending=False)
    logger.info("\n=== Feature importances (mean ± std) ===")
    for column in feature_importances_df.columns:
        logger.info(f"{column}: {feature_importances_df[column]['mean']:.4f} ± {feature_importances_df[column]['std']:.4f}")

    ################################################################################################
    ######## SECOND SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR METASTASIS PATIENTS ############
    ################################################################################################

    logger.info(" ------------- Model for prediction of survival for metastasis patients -------------")

    # We keep patients that are metastasis (have relapse different from 0)
    data_metastasis = data_grouped[data_grouped['relapse'] == 1]

    # Split into features and target
    y = data_metastasis[['DC']]
    x = data_metastasis.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'relapse', 'follow_up'])
    
    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which died: {y[y['DC']==1].shape[0]}")
    logger.info(f"Feature columns: {list(x.columns)}")
    logger.info("As we can see in the features, the model predicts survival for metastasis patients knowing which metastasis they have")

    # Plot the distribution of 'delai_fin_DC'
    data_metastasis['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time of metastasis patients')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, "survival_time_distribution_metastasis_patients.png"))
    plt.close()

    X = x
    y = y
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

        # Feature importances
        feature_importances.append(best_model.feature_importances_)

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # Print the feature importances
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns)
    feature_importances_df = feature_importances_df.describe().T[['mean', 'std']].sort_values(by='mean', ascending=False)
    logger.info("\n=== Feature importances (mean ± std) ===")
    for column in feature_importances_df.columns:
        logger.info(f"{column}: {feature_importances_df[column]['mean']:.4f} ± {feature_importances_df[column]['std']:.4f}")


    ################################################################################################
    ######## THIRD SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR ALL PATIENTS ####################
    ################################################################################################

    logger.info(" ------------- Model for prediction of survival for all patients -------------")

    # We don't remove any patient
    data_all = data_grouped

    # Split into features and target but keep 'relapse' as a feature
    y = data_all[['DC']]
    x = data_all.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum', 'follow_up'])
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    logger.info(f'Feature data shape: {x.shape}')
    logger.info(f'Target data shape: {y.shape}')
    logger.info(f"Number of subjects which died: {y[y['DC']==1].shape[0]}")
    logger.info(f"Feature columns: {list(x.columns)}")
    logger.info("As we can see in the features, the model predicts survival for all patients knowing whether they are primitive or metastasis (relapse feature)")

    # Plot the distribution of 'delai_fin_DC'
    plt.figure()
    data_all['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, "survival_time_distribution_all_patients.png"))
    plt.close()

    X = x
    y = y
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

        # Feature importances
        feature_importances.append(best_model.feature_importances_)

    # === Résultats globaux (moyenne ± std sur les folds externes) ===
    results_df = pd.DataFrame(outer_results)
    logger.info("\n=== Résultats Nested CV ===")
    for element in results_df.columns:
        logger.info(f"{element}: {results_df[element].mean():.4f} ± {results_df[element].std():.4f}")

    # Print the feature importances
    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns)
    feature_importances_df = feature_importances_df.describe().T[['mean', 'std']].sort_values(by='mean', ascending=False)
    logger.info("\n=== Feature importances (mean ± std) ===")
    for column in feature_importances_df.columns:
        logger.info(f"{column}: {feature_importances_df[column]['mean']:.4f} ± {feature_importances_df[column]['std']:.4f}")
    
    return None

   

if __name__ == '__main__':
    main()