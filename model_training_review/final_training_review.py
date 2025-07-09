""""
This script is used to train and evaluate the model for the survival prediction of lung cancer patients.
It also demonstrates the results of the model on the test set.
It uses D as the test set.
It performs feature selection only using Shapley values.

Args:
    --input: path to the merged dataset
    --output: path to the output folder
    --test-set: letter corresponding to the test set (default: 'L') (possible values are ['D' 'H' 'L' 'R' 'T' 'V'])
    --optim: Whether to perform bayesian optimisation or not

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
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import VarianceThreshold
import shap

import warnings
warnings.filterwarnings('ignore')



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

    # If the output_folder doesn't exist create it
    os.makedirs(output_folder, exist_ok=True)

    # We define the letter of the test set
    test_set_letter = 'D'  # Default test set letter

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

    # We extract the site where the subjects are from which is the first letter of the subject_id
    data_grouped['site'] = data_grouped['subject_id'].apply(lambda x: x[0])

    # Print the different sites in the dataset and the repartition
    print("Sites in the dataset and their number of subjects:")
    print(data_grouped['site'].value_counts())
    print("\n")
    # We build a column which indicates if it is used as training or testing data (testing if the subject is from site 'V')
    data_grouped['train_test'] = data_grouped['site'].apply(lambda x: 'train' if x != test_set_letter else 'test')

    # Split into features and target
    y = data_grouped[['DC', 'delai_fin_DC', 'train_test']]
    x = data_grouped.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'site'])
    print(" Final features in the dataset:", list(x.columns))
    print("\n")

    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC', 'train_test']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print("Number of subjects which died:", y[y['DC']==1].shape[0])
    print("Total number of subjects:", y.shape[0])
    print("\n")

    # Split the data into training and testing sets based on column 'train_test'
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])

    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    plt.savefig(os.path.join(output_folder, 'delai_fin_DC_distribution.png'))

    ########################################################################
    #################### 3 YEAR MODEL TRAINING #############################
    ########################################################################

    # Now we consider that every person that die after 3 year is considered as not dead (using the delai_fin_DC column)
    y_deadline_3_year = data_grouped[['delai_fin_DC', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['delai_fin_DC'] = y_deadline_3_year['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that died within 3 year:", y_deadline_3_year[y_deadline_3_year['delai_fin_DC'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
    print("\nNumber of subjects that died within 3 year (train):", y_train[y_train['delai_fin_DC'] == 1].shape[0])
    print("Number of subjects that died within 3 year (test):", y_test[y_test['delai_fin_DC'] == 1].shape[0])   

    # # If we want to perform bayesian optimisation, we can do it here
    # # We perform bayesian hyperparameter tuning
    # search_spaces = {
    #     'learning_rate': Real(0.001, 0.5),
    #     'n_estimators': Integer(10, 1000),
    #     'max_depth': Integer(3, 10),
    #     # 'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.1, 1),
    #     'colsample_bytree': Real(0.001, 1),
    #     # 'gamma': Real(0, 1),
    #     # 'reg_alpha': Real(0, 1),
    #     # 'reg_lambda': Real(0, 1),

    # }
    # # We define the model
    # model = XGBClassifier(seed=42, scale_pos_weight=5)  # We set scale_pos_weight to the ratio of negative to positive samples
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train, y_train)
    # # We print the best parameters
    # print("Model parameters after hyperparameter tuning:")
    # print(search.best_params_)
    # # We evaluate the model
    # model = search.best_estimator_

    # If we don't want to perform bayesian optimisation, we can directly define the model
    model = XGBClassifier(seed=46, scale_pos_weight=5, colsample_bytree=0.001, learning_rate=0.001, max_depth=8, n_estimators=285, subsample=1.0)
    model.fit(x_train, y_train)
    
    # Show train performances
    print("\n")
    print("Model performance on the training set")
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]
    print("ROC AUC Score:", roc_auc_score(y_train, y_train_proba))
    print("Brier score:", brier_score_loss(y_train, y_train_proba))
    print("Average precision:", precision_score(y_train, y_train_pred))
    print("Average Recall:", recall_score(y_train, y_train_pred))
    print("Accuracy Score:",accuracy_score(y_train, y_train_pred))
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred)
    print("AUC-PR score:", auc(recall_train, precision_train))
    print("\n")

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


    ########################################################################
    #################### FEATURE SELECTION #################################
    ########################################################################
    
    # We use Shapley values to perform feature selection
    ## This is done solely on the train dataset
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)
    # shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
    # plt.title('Feature importance of the final model')
    # plt.show()
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model.png'))
    # shap.summary_plot(shap_values, x_train, show=False)
    # plt.title('Feature importance of the final model')
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model_shap.png'))
    # plt.show()

    # Print table of feature importance with feature names
    feature_importance = pd.DataFrame(list(zip(x_train.columns, np.abs(shap_values).mean(axis=0))), columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    #print(feature_importance)
    # Print the top 20 features
    print("Top 20 features:")
    print(feature_importance.head(20))

    # Now we train the model only on the 20 most important features
    twenty_top_features = feature_importance['feature'][:20]
    x_train_copy = x_train.copy()
    x_test_copy = x_test.copy()

    ########################################################################
    #################### TRAINING OF 20 FEATURE MODEL #################################
    ########################################################################

    print("-------------------------- 20 FEATURES -----------------------------")

    x_train_20_feat = x_train_copy[twenty_top_features]
    x_test_20_feat = x_test_copy[twenty_top_features]

    # Print the top 20 features
    print("Top 20 features used for training:")
    print(list(twenty_top_features))

    # # We perform bayesian hyperparameter tuning
    # search_spaces = {
    #     'learning_rate': Real(0.001, 0.5),
    #     'n_estimators': Integer(10, 1000),
    #     'max_depth': Integer(3, 10),
    #     # 'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.1, 1),
    #     'colsample_bytree': Real(0.001, 1),
    #     # 'gamma': Real(0, 1),
    #     # 'reg_alpha': Real(0, 1),
    #     # 'reg_lambda': Real(0, 1),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42, scale_pos_weight=5)  # We set scale_pos_weight to the ratio of negative to positive samples
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train_20_feat, y_train)
    # # We print the best parameters
    # print("Model parameters after hyperparameter tuning:")
    # print(search.best_params_)
    # # We evaluate the model
    # model = search.best_estimator_

    # If we don't want to perform bayesian optimisation, we can directly define the model
    model = XGBClassifier(seed=44, scale_pos_weight=5, colsample_bytree=0.030034360975174177, learning_rate=0.31946438373196623, max_depth=3, n_estimators=10, subsample=0.7158824754556723)
    model.fit(x_train_20_feat, y_train)

    # Show train performances
    print("\n")
    print("Model performance on the training set")
    y_train_pred = model.predict(x_train_20_feat)
    y_train_proba = model.predict_proba(x_train_20_feat)[:, 1]
    print("ROC AUC Score:", roc_auc_score(y_train, y_train_proba))
    print("Brier score:", brier_score_loss(y_train, y_train_proba))
    print("Average precision:", precision_score(y_train, y_train_pred))
    print("Average Recall:", recall_score(y_train, y_train_pred))
    print("Accuracy Score:",accuracy_score(y_train, y_train_pred))
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred)
    print("AUC-PR score:", auc(recall_train, precision_train))
    print("\n")

    # Performance on the training set
    y_test_pred = model.predict(x_test_20_feat)
    y_test_proba = model.predict_proba(x_test_20_feat)[:, 1]

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

    ########################################################################
    #################### FEATURE SELECTION #################################
    ########################################################################
    
    # We use Shapley values to perform feature selection
    ## This is done solely on the train dataset
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train_20_feat)
    # shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
    # plt.title('Feature importance of the final model')
    # plt.show()
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model.png'))
    # shap.summary_plot(shap_values, x_train, show=False)
    # plt.title('Feature importance of the final model')
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model_shap.png'))
    # plt.show()

    # Print table of feature importance with feature names
    feature_importance = pd.DataFrame(list(zip(x_train_20_feat.columns, np.abs(shap_values).mean(axis=0))), columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    #print(feature_importance)
    # Print the top 20 features
    print("Top 20 features:")
    print(feature_importance.head(20))

    ########################################################################
    #################### TRAINING OF 15 FEATURE MODEL #################################
    ########################################################################

    print("-------------------------- 15 FEATURES -----------------------------")

    x_train_15_feat = x_train_copy[feature_importance['feature'][:15]]
    x_test_15_feat = x_test_copy[feature_importance['feature'][:15]]

    # Print the top 15 features
    print("Top 15 features used for training:")
    print(list(feature_importance['feature'][:15]))

    # # We perform bayesian hyperparameter tuning
    # search_spaces = {
    #     'learning_rate': Real(0.001, 0.5),
    #     'n_estimators': Integer(10, 1000),
    #     'max_depth': Integer(3, 10),
    #     # 'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.1, 1),
    #     'colsample_bytree': Real(0.001, 1),
    #     # 'gamma': Real(0, 1),
    #     # 'reg_alpha': Real(0, 1),
    #     # 'reg_lambda': Real(0, 1),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42, scale_pos_weight=5)  # We set scale_pos_weight to the ratio of negative to positive samples
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train_15_feat, y_train)
    # # We print the best parameters
    # print("Model parameters after hyperparameter tuning:")
    # print(search.best_params_)
    # # We evaluate the model
    # model = search.best_estimator_

    # If we don't want to perform bayesian optimisation, we can directly define the model
    model = XGBClassifier(seed=45, scale_pos_weight=5, colsample_bytree=0.001, learning_rate=0.001, max_depth=10, n_estimators=1000, subsample=0.898757960596005)
    model.fit(x_train_15_feat, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test_15_feat)
    y_test_proba = model.predict_proba(x_test_15_feat)[:, 1]

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

    ########################################################################
    #################### FEATURE SELECTION #################################
    ########################################################################
    
    # We use Shapley values to perform feature selection
    ## This is done solely on the train dataset
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train_15_feat)
    # shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
    # plt.title('Feature importance of the final model')
    # plt.show()
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model.png'))
    # shap.summary_plot(shap_values, x_train, show=False)
    # plt.title('Feature importance of the final model')
    # plt.savefig(os.path.join(output_folder, 'feature_importance_final_model_shap.png'))
    # plt.show()

    # Print table of feature importance with feature names
    feature_importance = pd.DataFrame(list(zip(x_train_15_feat.columns, np.abs(shap_values).mean(axis=0))), columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    #print(feature_importance)
    # Print the top 20 features
    print("Top 15 features:")
    print(feature_importance.head(15))

    ########################################################################
    #################### TRAINING OF 10 FEATURE MODEL #################################
    ########################################################################

    print("-------------------------- 10 FEATURES -----------------------------")    

    x_train_10_feat = x_train_copy[feature_importance['feature'][:10]]
    x_test_10_feat = x_test_copy[feature_importance['feature'][:10]]

    # Print the top 10 features
    print("Top 10 features used for training:")
    print(list(feature_importance['feature'][:10]))

    # # We perform bayesian hyperparameter tuning
    # search_spaces = {
    #     'learning_rate': Real(0.001, 0.5),
    #     'n_estimators': Integer(10, 1000),
    #     'max_depth': Integer(3, 10),
    #     # 'min_child_weight': Integer(1, 10),
    #     'subsample': Real(0.1, 1),
    #     'colsample_bytree': Real(0.001, 1),
    #     # 'gamma': Real(0, 1),
    #     # 'reg_alpha': Real(0, 1),
    #     # 'reg_lambda': Real(0, 1),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42, scale_pos_weight=5)  # We set scale_pos_weight to the ratio of negative to positive samples
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(x_train_10_feat, y_train)
    # # We print the best parameters
    # print("Model parameters after hyperparameter tuning:")
    # print(search.best_params_)
    # # We evaluate the model
    # model = search.best_estimator_

    # We evaluate the model after feature selection based on correlation with the target variable
    model = XGBClassifier(seed=43, scale_pos_weight=5, colsample_bytree=0.16911997834516218, learning_rate=0.001, max_depth=10, n_estimators=144, subsample=0.9307758693653829)
    model.fit(x_train_10_feat, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test_10_feat)
    y_test_proba = model.predict_proba(x_test_10_feat)[:, 1]

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

    return None


if __name__ == '__main__':
    main()