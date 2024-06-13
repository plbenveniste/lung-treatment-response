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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import pickle


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
    y = data[['DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo',
              'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_DC', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
              'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum']]
    x = data.drop(columns=['DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo',
                           'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_DC', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
                           'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum'])
    
    # In this case, because we are only interested in the prediction of relapse in the PTV (Planning Target Volume) survival, we extract only the 'rechute_PTV'
    y = y[['rechute_PTV']]    

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print("Number of subjects which had a local relapse", y[y['rechute_PTV']==1].shape[0])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing ", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data['delai_fin_rechutePTV'].hist()
    plt.title('Distribution of time between end of treatment and local \nrelapse (in days) for those that had a local relapse')
    plt.xlabel('Time')
    plt.ylabel('Number of subjects')
    plt.show()

    # Box plot of the 'delai_fin_rechutePTV' column
    data.boxplot(column='delai_fin_rechutePTV')
    plt.title('Box plot of the time between end of treatment and local relapse')
    plt.ylabel('Time')
    plt.show()

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance without any occurence deadline")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score ", brier_score_loss(y_test, y_test_proba))
    print("Average precision", precision_score(y_test, y_test_pred))
    print("Average Recall", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score  ", auc(recall_test, precision_test))
    print("\n")

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'local_relapse_model'), 'wb'))

    #############################################
    #############################################
    # Let's inspect the data:
    print("Number of subjects which had a local relapse date", data['delai_fin_rechutePTV'].unique().shape[0]-1)
    # Moving on because it doesn't make sense to look into adding a deadline since only 7 out of 24 have a local relapse date
    print("Moving on because it doesn't make sense to look into adding a deadline since only 7 out of 24 have a local relapse date")

    return None


if __name__ == '__main__':
    main()