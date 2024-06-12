""""
This script is used to train and evaluate the model for the survival prediction of lung cancer patients.
It also demonstrates the results of the model on the test set.

Args:
    --input: path to the merged dataset
    --output: path to the output folder

Returns:
    None

Example run:
    python survival_model.py --input /path/to/merged.csv --output /path/to/output_folder

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score
from skopt.space import Real, Integer
from skopt import BayesSearchCV


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

    # Load the dataset
    data = pd.read_csv(input_data)

    # Split into features and target
    y = data[['DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum']]
    x = data.drop(columns=['DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum'])
    
    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print('Description of the values by "DC"')
    print(y.describe())

    # For xgboost, we replace the column names by feature_1,...

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing ", x_test.shape[0])
    print("\n")

    # Initialise the model
    model = XGBClassifier(seed=40)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score ", brier_score_loss(y_test, y_test_proba))
    print("Average precision", precision_score(y_test, y_test_pred))
    print("Average Recall", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score  ", auc(recall_test, precision_test))

    # Before any optimisation
    # ROC AUC Score:  0.8066666666666666
    # Brier score  0.13951960733034136
    # Average precision 0.7272727272727273
    # Average Recall 0.6666666666666666
    # Accuracy Score:  0.8108108108108109
    # AUC-PR score   0.751023751023751

    # Train and optimise a model
    # Initialize the XGBClassifier fixed parameters
    fixed_params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'exact',
        'random_state': 0
    }
    search_space_xgb = {
        'learning_rate': Real(0.01, 1.0, 'uniform'),
        'max_depth': Integer(2, 10),
        'subsample': Real(0.1, 1.0, 'uniform'),
        'colsample_bytree': Real(0.1, 1.0, 'uniform'),
        'n_estimators': Integer(50, 1000),
        'reg_alpha': Real(0.001, 100., 'uniform'), 
    }

    print("\nProceding to a Bayesian optimisation of the model...")
    xgb_model = XGBClassifier(**fixed_params_xgb)
    # Perform Bayesian hyperparameter optimization
    bayes_search_xgb = BayesSearchCV(estimator=xgb_model, search_spaces=search_space_xgb, cv=5, random_state=0)
    bayes_search_xgb.fit(x_train, y_train)

    # Save the best model
    xgb_model = bayes_search_xgb.best_estimator_

    # Performance on the test set
    y_test_pred = xgb_model.predict(x_test)
    y_test_proba = xgb_model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print(" #### On the test set ")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score ", brier_score_loss(y_test, y_test_proba))
    print("Average precision", precision_score(y_test, y_test_pred))
    print("Average Recall", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score  ", auc(recall_test, precision_test))


    return None


if __name__ == '__main__':
    main()