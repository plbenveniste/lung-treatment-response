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

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DC', 'delai_fin_DC', 'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro',
                              'Date_R_horspoum','subject_nodule', 'nodule', 'Reponse', 'rechute_PTV','delai_fin_rechutePTV'])
    

    # We average the columns for the same patients across the different nodules
    data_grouped = data.groupby('subject_id').mean().reset_index()

    ## The code commented below was written to see if there were any incoherence in the input data
    # selected_columns = [
    #     'sexe',
    #     'age',
    #     'BMI',
    #     'score_charlson',
    #     'OMS',
    #     'tabac',
    #     'tabac_PA',
    #     'tabac_sevre',
    #     'DC',
    # ]
    # column_problems = []
    # for patient in data['subject_id'].unique():
    #     data_patient = data[data['subject_id']==patient]
    #     for column in selected_columns:
    #         if len(data_patient[column].unique())!=1:
    #             column_problems.append(column)
    #             print(patient)
    #             print(column)
    #             # print(data_patient[column].unique())
    #             print("\n")

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
    x = data_grouped.drop(columns=['rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechuteHomo',
                                   'delai_fin_rechuteMed','delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum', 'subject_id', 'rechute_dist','rechute_dist_moy_delai'])
    
    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['rechute_dist']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print("Number of subjects which had a distant relapse:", y[y['rechute_dist']==1].shape[0])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['rechute_dist_moy_delai'].hist()
    plt.title('Distribution of distant relapse time between end of treatment and\n occurence of a relapse (in days) for those that relapsed')
    plt.xlabel('Time')
    plt.ylabel('Number of subjects')
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
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model'), 'wb'))

    #############################################
    #############################################
    # Now we consider that every person that had a distant relapse after 1 year is considered as not having a distant relapse
    y_deadline_1_year = data_grouped[['rechute_dist_moy_delai']]
    y_deadline_1_year.fillna(366, inplace=True)
    y_deadline_1_year['rechute_dist_moy_delai'] = y_deadline_1_year['rechute_dist_moy_delai'].apply(lambda x: 0 if x > 365 else 1)
    print("Number of subjects that had a distant relapse within 1 year:", y_deadline_1_year[y_deadline_1_year['rechute_dist_moy_delai'] == 1].shape[0])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y_deadline_1_year, test_size=0.2, random_state=1)
    print("Number of subjects that had a distant relapse within 1 year (train):", y_train[y_train['rechute_dist_moy_delai'] == 1].shape[0])
    print("Number of subjects that had a distant relapse within 1 year (test):", y_test[y_test['rechute_dist_moy_delai'] == 1].shape[0])  

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance on the deadline of 1 year")
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score:",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model_1_year'), 'wb'))

    #############################################
    #############################################
    # Now we consider that every person that had a distant relapse after 3 years is considered as not having a distant relapse
    y_deadline_3_year = data_grouped[['rechute_dist_moy_delai']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['rechute_dist_moy_delai'] = y_deadline_3_year['rechute_dist_moy_delai'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that had a distant relapse within 3 year:", y_deadline_3_year[y_deadline_3_year['rechute_dist_moy_delai'] == 1].shape[0])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y_deadline_3_year, test_size=0.2, random_state=1)
    print("\nNumber of subjects that had a distant relapse within 3 year (train):", y_train[y_train['rechute_dist_moy_delai'] == 1].shape[0])
    print("Number of subjects that had a distant relapse within 3 year (test):", y_test[y_test['rechute_dist_moy_delai'] == 1].shape[0])   

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance on the deadline of 3 year")
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score:",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # Evaluate the model on the train set
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]

    print("Model performance on the deadline of 3 year on the train set")
    print("ROC AUC Score:", roc_auc_score(y_train, y_train_proba))
    print("Brier score:", brier_score_loss(y_train, y_train_proba))
    print("Average precision:", precision_score(y_train, y_train_pred))
    print("Average Recall:", recall_score(y_train, y_train_pred))
    print("Accuracy Score:",accuracy_score(y_train, y_train_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_train, y_train_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model_3_year'), 'wb'))

    return None


if __name__ == '__main__':
    main()