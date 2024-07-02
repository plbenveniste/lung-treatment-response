"""
In this script, we will look at the radiomics features and eliminate the ones that are not correlated with the target variable.
Then we will cluster the festures which are correlated together and keep only one feature from each cluster which is the most correlated with the target variable.

Input:
    --merged-data: Path to the csv file containing the merged data
    --output: Path to the folder where the results will be saved

Output:
    None

Example:
    python data_preprocessing/5_eliminating_radiomics_features.py --merged-data data/merged_data.csv --output data/

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, precision_score, recall_score
from sklearn.feature_selection import VarianceThreshold



def parse_arguments():
    parser = argparse.ArgumentParser(description='Eliminating radiomics features')
    parser.add_argument('--merged-data', type=str, help='Path to the csv file containing the merged data')
    parser.add_argument('--output', type=str, help='Path to the folder where the results will be saved')
    return parser.parse_args()


def remove_features(x_train, y_train, x_test, y_test, threshold_var=0.05, threshold_corr_feat=0.90, threshold_corr_target=0.2):
    """
    This script is used to remove features which don't play a major role in the prediction of the target variable.
    It contains the following steps:
      - variance thresholding: remove features with low variance
      - feature selection based on correlation with other features: remove features which are highly correlated with other features
      - feature selection based on correlation with the target variable: remove features which are not correlated with the target variable
    
    Args:
        x_train: training data
        y_train: training target
        x_test: test data
        y_test: test target
    
    Returns:
        final_features: list of the final features
    """
    print("Initial number of features: ", x_train.shape[1])
    print("\n")

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print("Model performance without any feature removal")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # First we perform variance thresholding
    shape_ini = x_train.shape
    selector = VarianceThreshold(threshold=threshold_var)
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
            if corr_matrix.loc[columns[i], columns[j]] > threshold_corr_feat:
                columns_to_drop_corr_feat.append(columns[j])
    columns_to_drop_corr_feat = set(columns_to_drop_corr_feat)
    print("Number of features after correlation thresholding:", x_train.shape[1]-len(columns_to_drop_corr_feat))
    print("Number of features removed by correlation thresholding:", len(columns_to_drop_corr_feat))
    print("\n")
    
    # Remove the columns
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
        if abs(corr_matrix.iloc[i]) < threshold_corr_target:
            columns_to_drop_corr_target.append(columns[i])
    columns_to_drop_corr_target = set(columns_to_drop_corr_target)
    print("Number of features after correlation with target thresholding:", x_train.shape[1]-len(columns_to_drop_corr_target))
    print("Number of features removed by correlation with target thresholding:", len(columns_to_drop_corr_target))
    print("\n")

    # Remove the columns
    x_train = x_train.drop(columns = list(columns_to_drop_corr_target))
    x_test = x_test.drop(columns = list(columns_to_drop_corr_target))

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

    # We print the correlation matrix of those remaining features with the target variable
    corr_matrix = x_train.corrwith(y_train)
    print("Correlation of remaining features with target variable:")
    print(corr_matrix)
    print("\n")

    # Final features
    final_features = x_train.columns
    
    return final_features
    

def main():
    """
    This is the main function of the script.

    Args:
        None
    
    Returns:
        None
    """
    # Parsing the arguments
    args = parse_arguments()

    # Load the data 
    data_ini = pd.read_csv(args.merged_data)

    # We remove the clinical features
    clinical_features = ['sexe', 'age', 'BMI', 'score_charlson', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'dose_tot',
                         'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10',  'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo',
                         'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse',  'delai_fin_DC', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo',
                         'delai_fin_rechuteMed', 'delai_fin_rechuteContro','delai_fin_rechuteHorspoum', 'subject_nodule', 'subject_id', 'nodule']
    output_features = ['DC','rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro','rechute_horspoum']

    # We remove the clinical features
    output = data_ini[output_features]
    data = data_ini.drop(columns=clinical_features)
    data = data.drop(columns=output_features)

    ############################################################################################################################################################################
    ###################### DEATH PREDICTION ####################################################################################################################################
    ############################################################################################################################################################################
    # First let's focus on death
    ## We look at the impact of features on predicting death
    x= data
    y= output['DC']
    y.fillna(0, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Perform feature selection
    final_features = remove_features(x_train, y_train, x_test, y_test, threshold_var=0.05, threshold_corr_feat=0.90, threshold_corr_target=0.25)

    # ############################################################################################################################################################################
    # ###################### LOCAL RELAPSE #######################################################################################################################################
    # ############################################################################################################################################################################
    # Now let's focus on local relapse
    ## We look at the impact of features on predicting local relapse
    data = data_ini.drop(columns=clinical_features)
    data = data.drop(columns=output_features)
    x = data
    y = output['rechute_PTV']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Perform feature selection
    final_features = remove_features(x_train, y_train, x_test, y_test, threshold_var=0.05, threshold_corr_feat=0.90, threshold_corr_target=0.07)

    # ############################################################################################################################################################################
    # ###################### REGIONAL RELAPSE ####################################################################################################################################
    # ############################################################################################################################################################################
    # Now let's focus on regional relapse
    ## We look at the impact of features on predicting regional relapse
    data = data_ini.drop(columns=clinical_features)
    data = data.drop(columns=output_features)
    x = data
    y = output
    y['rechute_dist'] = output['rechute_homo'] + output['rechute_med'] + output['rechute_contro'] + output['rechute_horspoum']
    y['rechute_dist'] = output['rechute_dist'].apply(lambda x: 1 if x > 0 else 0)
    y = y['rechute_dist']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Perform feature selection
    final_features = remove_features(x_train, y_train, x_test, y_test, threshold_var=0.05, threshold_corr_feat=0.90, threshold_corr_target=0.055)

    return None

if __name__ == '__main__':
    main()