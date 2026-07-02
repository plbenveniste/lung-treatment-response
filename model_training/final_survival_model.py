"""
This script is used to display the performances of the final model used in the article which predicts the 3-year survival of patients after lung SBRT.

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
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix
from skopt.space import Real, Integer
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import VarianceThreshold
import shap
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


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

    ########################################################################
    #################### DATA PREPROCESSING ################################
    ########################################################################

    # Load the dataset
    data = pd.read_csv(input_data)

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo',
                           'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
                           'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule' ])
    

    # We average the columns for the same patients across the different nodules
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # We extract the site where the subjects are from which is the first letter of the subject_id
    data_grouped['site'] = data_grouped['subject_id'].apply(lambda x: x[0])
    
    # We build a column which indicates if it is used as training or testing data (testing if the subject is from site 'V')
    data_grouped['train_test'] = data_grouped['site'].apply(lambda x: 'train' if x != 'L' else 'test')

    # Split into features and target
    y = data_grouped[['DC', 'delai_fin_DC', 'train_test']]
    x = data_grouped.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'site'])
    
    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC', 'train_test']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # Describe x and y
    print('Feature data shape:', x.shape)
    print('Target data shape:', y.shape)
    print("Number of subjects which died:", y[y['DC']==1].shape[0])

    # Split the data into training and testing sets based on column 'train_test'
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # We only keep the 14 final features
    x_train = x_train[['sexe', 'BMI', 'score_charlson', 'tabac_sevre', 'dose_tot', 'BED_10', 'INTENSITY-BASED_MeanIntensity', 'INTENSITY-BASED_IntensitySkewness',
           'INTENSITY-BASED_IntensityKurtosis', 'INTENSITY-BASED_AreaUnderCurveCIVH', 'INTENSITY-BASED_RootMeanSquareIntensity', 'INTENSITY-HISTOGRAM_IntensityHistogramMean',
           'INTENSITY-HISTOGRAM_IntensityHistogramVariance', 'NGTDM_Strength']]
    x_test = x_test[['sexe', 'BMI', 'score_charlson', 'tabac_sevre', 'dose_tot', 'BED_10', 'INTENSITY-BASED_MeanIntensity', 'INTENSITY-BASED_IntensitySkewness',
              'INTENSITY-BASED_IntensityKurtosis', 'INTENSITY-BASED_AreaUnderCurveCIVH', 'INTENSITY-BASED_RootMeanSquareIntensity', 'INTENSITY-HISTOGRAM_IntensityHistogramMean',
                'INTENSITY-HISTOGRAM_IntensityHistogramVariance', 'NGTDM_Strength']]
    
    print("Number of features:", x_train.shape[1])
    print("Selected features:", x_train.columns)
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_grouped['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    plt.show()

    # Now we consider that every person that die after 3 year is considered as not dead (using the delai_fin_DC column)
    y_deadline_3_year = data_grouped[['delai_fin_DC', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['delai_fin_DC'] = y_deadline_3_year['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that died within 3 year:", y_deadline_3_year[y_deadline_3_year['delai_fin_DC'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
    print("Number of subjects that died within 3 year (train):", y_train[y_train['delai_fin_DC'] == 1].shape[0])
    print("Number of subjects that died within 3 year (test):", y_test[y_test['delai_fin_DC'] == 1].shape[0])
    print("\n") 

    # We now test a model with the following parameters (found with Bayesian optimization): 
    # [('colsample_bytree', 1.0), ('learning_rate', 0.08995772980827092), ('max_depth', 3), ('n_estimators', 344), ('subsample', 0.1)])
    model = XGBClassifier(colsample_bytree=1.0, learning_rate=0.08995772980827092, max_depth=3, n_estimators=344, subsample=0.1, seed=42)
    model.fit(x_train, y_train)
    print("------------------------------------")
    print("Training the model to predict survival within 3 years with only 14 features")
    print("------------------------------------")
    print("\n")

    # Performance on the training set
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Final model performance on the training set")
    print("ROC AUC Score: ", roc_auc_score(y_train, y_train_proba))
    print("Brier score:", brier_score_loss(y_train, y_train_proba))
    print("Average precision:", precision_score(y_train, y_train_pred))
    print("Average Recall:", recall_score(y_train, y_train_pred))
    print("Accuracy Score: ",accuracy_score(y_train, y_train_pred))
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred)
    print("AUC-PR score:", auc(recall_train, precision_train))
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")

    # Performance on the test set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Final model performance on the testing set")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")

    # Plot the ROC curve comparing the training and testing set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    plt.plot(fpr_train, tpr_train, label='Training set')
    plt.plot(fpr_test, tpr_test, label='Testing set')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of the final model')
    plt.legend()
    plt.show()

    # Save the model
    pickle.dump(model, open(os.path.join(output_folder, 'final_survival_model'), 'wb'))

    ############### CALIBRATION OF THE MODEL     ############################
    # We calibrate the model using the isotonic method
    model_calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    model_calibrated.fit(x_train, y_train)

    # Performance on the test set
    y_test_pred_calibrated = model_calibrated.predict(x_test)
    y_test_proba_calibrated = model_calibrated.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Final model performance on the testing set after calibration")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba_calibrated))
    print("Brier score:", brier_score_loss(y_test, y_test_proba_calibrated))
    print("Average precision:", precision_score(y_test, y_test_pred_calibrated))
    print("Average Recall:", recall_score(y_test, y_test_pred_calibrated))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred_calibrated))
    precision_test_calibrated, recall_test_calibrated, _ = precision_recall_curve(y_test, y_test_pred_calibrated)
    print("AUC-PR score:", auc(recall_test_calibrated, precision_test_calibrated))
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_calibrated).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")

    # Plot the calibration before and after calibration
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_test_proba_calibrated, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Uncalibrated')
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve of the final model')
    plt.legend()
    plt.show()

    # Plot the calibration curve only of the calibrated model
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_test_proba_calibrated, n_bins=10)
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated XGBoost model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve of the final model')
    plt.legend()
    plt.show()

    # Save the calibrated model
    pickle.dump(model_calibrated, open(os.path.join(output_folder, 'final_survival_model_calibrated'), 'wb'))

    # Using shapley libray we showcase the importance of the features and how they impact the model
    # We first rename the columns to have a more readable name in the plot
    column_renaming = {
        'sexe': 'Sex',
        'BMI': 'BMI',
        'score_charlson': 'Charlson Comorbidity Index (i.e. score_charlson)',
        'tabac_sevre': 'Smoking cessation (i.e. tabac_sevre)',
        'dose_tot': 'Total Dose (i.e. dose_tot)',
        'BED_10': 'BED 10'
    }
    x_train.rename(columns=column_renaming, inplace=True)
    x_test.rename(columns=column_renaming, inplace=True)
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
    plt.title('Feature importance of the final model')
    plt.show()
    shap.summary_plot(shap_values, x_train, show=False)
    plt.title('Feature importance of the final model')
    plt.show()

    # We now do the same for the calibrated model
    shap.initjs()
    explainer = shap.KernelExplainer(model_calibrated.predict_proba, x_train)
    shap_values = explainer.shap_values(x_train)
    # We only select the values for the positive class (meaning having a survival time less than 3 years)
    shap_values = shap_values[:,:,1]
    shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
    plt.title('Feature importance of the calibrated model')
    plt.show()
    shap.summary_plot(shap_values, x_train, show=False)
    plt.title('Feature importance of the calibrated model')
    plt.show()

    return None


if __name__ == '__main__':
    main()