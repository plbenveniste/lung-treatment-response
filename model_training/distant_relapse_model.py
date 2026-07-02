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
from sklearn.feature_selection import VarianceThreshold


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
    data_grouped['train_test'] = data_grouped['subject_id'].apply(lambda x: 'train' if x[0] != 'V' else 'test')

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
    y = data_grouped[['rechute_dist','rechute_dist_moy_delai', 'train_test']]
    x = data_grouped.drop(columns=['rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechuteHomo',
                                   'delai_fin_rechuteMed','delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum', 'subject_id', 'rechute_dist','rechute_dist_moy_delai'])
    
    # In this case, because we are only interested in the prediction of distant relapse, we extract only the 'DC'
    y = y[['rechute_dist', 'train_test']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print("Number of subjects which had a distant relapse:", y[y['rechute_dist']==1].shape[0])

    # Split the data into training and testing sets
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['rechute_dist_moy_delai'].hist()
    plt.title('Distribution of distant relapse time between end of treatment and\n occurence of a relapse (in days) for those that relapsed')
    plt.xlabel('Time')
    plt.ylabel('Number of subjects')
    # plt.show()

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
    y_deadline_1_year = data_grouped[['rechute_dist_moy_delai', 'train_test']]
    y_deadline_1_year.fillna(366, inplace=True)
    y_deadline_1_year['rechute_dist_moy_delai'] = y_deadline_1_year['rechute_dist_moy_delai'].apply(lambda x: 0 if x > 365 else 1)
    print("Number of subjects that had a distant relapse within 1 year:", y_deadline_1_year[y_deadline_1_year['rechute_dist_moy_delai'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_1_year[y_deadline_1_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_1_year[y_deadline_1_year['train_test'] == 'test'].drop(columns=['train_test'])
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
    y_deadline_3_year = data_grouped[['rechute_dist_moy_delai', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['rechute_dist_moy_delai'] = y_deadline_3_year['rechute_dist_moy_delai'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that had a distant relapse within 3 year:", y_deadline_3_year[y_deadline_3_year['rechute_dist_moy_delai'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
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

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model_3_year'), 'wb'))

    #############################################
    #############################################
    # Now we move on to training the model with no deadline with fewer features based on what we explored in file `data_preprocessing/5_eliminating_radiomics_features.py` 
    # We keep the clinical features
    clinical_features = ['sexe', 'age', 'BMI', 'score_charlson', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'dose_tot', 'etalement',
                         'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']
    # We keep some radiomics features: selected here: 
    radiomics_features = ['MORPHOLOGICAL_Compacity', 'MORPHOLOGICAL_CentreOfMassShift', 'INTENSITY-BASED_IntensityInterquartileRange', 'INTENSITY-BASED_AreaUnderCurveCIVH',
                          'GLCM_SumVariance', 'GLCM_ClusterShade', 'GLCM_ClusterProminence', 'GLRLM_RunLengthNonUniformity', 'NGTDM_Contrast', 'GLSZM_LargeZoneLowGreyLevelEmphasis']
    # Join the two lists
    x = data_grouped[clinical_features + radiomics_features + ['train_test']]
    y = data_grouped[['rechute_dist', 'train_test']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Split the data into training and testing sets
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_train = y_train['rechute_dist']
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    y_test = y_test['rechute_dist']
    print("\nInitial number of features:", x_train.shape[1])
    print("Number of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

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

    #Now we perform variance thresholding
    shape_ini = x_train.shape
    selector = VarianceThreshold(threshold=0.2)
    selector.fit(x_train)
    # Extract the names of the columns to remove
    columns_to_remove_var_thresh = x_train.columns[~selector.get_support()]
    # We remove these columns from the data
    x_train = x_train.drop(columns=columns_to_remove_var_thresh)
    x_test = x_test.drop(columns=columns_to_remove_var_thresh)
    print("Number of features after variance thresholding:", x_train.shape[1])
    print("Number of features removed by variance thresholding:", shape_ini[1]-x_train.shape[1])
    print("\n")

    # We evaluate the model after variance thresholding
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance after variance thresholding")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # We now perform feature selection based on the correlation with other features
    corr_matrix = x_train.corr()
    columns = corr_matrix.columns
    columns_to_drop_corr_feat = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if corr_matrix.loc[columns[i], columns[j]] > 0.85:
                columns_to_drop_corr_feat.append(columns[j])
    columns_to_drop_corr_feat = set(columns_to_drop_corr_feat)
    print("Number of features after correlation thresholding:", x_train.shape[1]-len(columns_to_drop_corr_feat))
    print("Number of features removed by correlation thresholding:", len(columns_to_drop_corr_feat))
    print("\n")

    # # Remove the columns
    x_train = x_train.drop(columns = list(columns_to_drop_corr_feat))
    x_test = x_test.drop(columns = list(columns_to_drop_corr_feat))

    # We evaluate the model after feature selection based on correlation
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance after feature selection based on correlation")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # We now perform feature selection based on the correlation with the target variable
    corr_matrix = x_train.corrwith(y_train)
    columns = corr_matrix.index
    columns_to_drop_corr_target = []
    for i in range(len(columns)):
        if abs(corr_matrix.iloc[i]) < 0.05:
            columns_to_drop_corr_target.append(columns[i])
    columns_to_drop_corr_target = set(columns_to_drop_corr_target)
    print("Number of features after correlation with target thresholding:", x_train.shape[1]-len(columns_to_drop_corr_target))
    print("Number of features removed by correlation with target thresholding:", len(columns_to_drop_corr_target))
    print("\n")

    # Remove the columns
    x_train = x_train.drop(columns = list(columns_to_drop_corr_target))
    x_test = x_test.drop(columns = list(columns_to_drop_corr_target))

    # Print the final features of the model
    print("Final features of the model:")
    print(list(x_train.columns))
    print("\n")

    # We evaluate the model after feature selection based on correlation with the target variable
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance after feature selection based on correlation with target")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    ################################################################
    # Section removed because the best hyperparameters were already found
    ################################################################
    # # Then we do hyperparameter tuning with Bayesian optimization
    # # We define the hyperparameters to tune
    # search_spaces = {
    #     'learning_rate': Real(0.01, 0.5),
    #     'n_estimators': Integer(50, 1000),
    #     'max_depth': Integer(3, 10),
    #     'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.5, 1),
    #     'colsample_bytree': Real(0.01, 1),
    #     'gamma': Real(0, 1),
    #     'reg_alpha': Real(0, 1),
    #     'reg_lambda': Real(0, 1),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42)
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=40, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train, y_train)
    # # We print the best parameters
    # print("Model parameters after hyperparameter tuning:")
    # print(search.best_params_)
    # # We evaluate the model
    # model = search.best_estimator_
    # y_test_pred = model.predict(x_test)
    # y_test_proba = model.predict_proba(x_test)[:, 1]

    # # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    # print("Model performance after hyperparameter tuning")
    # print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    # print("Brier score:", brier_score_loss(y_test, y_test_proba))
    # print("Average precision:", precision_score(y_test, y_test_pred))
    # print("Average Recall:", recall_score(y_test, y_test_pred))
    # print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    # precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    # print("AUC-PR score:", auc(recall_test, precision_test))
    # print("\n")

    # # Save the model
    # pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model_no_deadline_fewer_features'), 'wb'))

    # We now train the best model with the best hyperparameters:
    #[('colsample_bytree', 1.0), ('gamma', 0.0), ('learning_rate', 0.5), ('max_depth', 4), ('min_child_weight', 3), ('n_estimators', 369),
    #  ('reg_alpha', 0.0), ('reg_lambda', 0.0), ('subsample', 0.5)]
    model = XGBClassifier(colsample_bytree=1.0, gamma=0.0, learning_rate=0.5, max_depth=4, min_child_weight=3, n_estimators=369,
                          reg_alpha=0.0, reg_lambda=0.0, subsample=0.5, seed=42)
    model.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance after hyperparameter tuning")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'distant_relapse_model_no_deadline_fewer_features'), 'wb'))

    return None


if __name__ == '__main__':
    main()