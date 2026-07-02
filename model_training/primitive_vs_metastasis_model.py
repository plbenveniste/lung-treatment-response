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
from sklearn.metrics import roc_auc_score, auc, brier_score_loss, precision_recall_curve, accuracy_score, roc_curve, precision_score, recall_score, confusion_matrix
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
    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo',
                                'delai_fin_rechuteMed', 'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule' ])
    

    # We average the columns for the same patients across the different nodules
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # We build the column relapse which takes 1 if either of the relapse occur ('rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum')
    data_grouped['relapse'] = data_grouped['rechute_PTV'] + data_grouped['rechute_homo'] + data_grouped['rechute_med'] + data_grouped['rechute_contro'] + data_grouped['rechute_horspoum']
    # Modify the 'relapse' column to take value 1 if it is above 0
    data_grouped['relapse'] = data_grouped['relapse'].apply(lambda x: 1 if x > 0 else 0)

    # Print number of primitive patients and number of metastasis
    print("Total number of patients", data_grouped.shape[0])
    print("Number of primitive patients", data_grouped[data_grouped['relapse']==0].shape[0])
    print("Number of metastasis patients", data_grouped[data_grouped['relapse']!=0].shape[0])
    print("\n")
    
    # We extract the site where the subjects are from which is the first letter of the subject_id
    data_grouped['site'] = data_grouped['subject_id'].apply(lambda x: x[0])
    
    # We build a column which indicates if it is used as training or testing data (testing if the subject is from site 'V')
    data_grouped['train_test'] = data_grouped['site'].apply(lambda x: 'train' if x != 'V' else 'test')

    ################################################################################################
    ######## FIRST SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR PRIMITIVE PATIENTS ##############
    ################################################################################################

    print(" ------------- Model for prediction of survival for primitive patients -------------")

    # We remove patients that are metastasis (have relapse different from 0)
    data_primitive = data_grouped[data_grouped['relapse'] == 0]

    # Split into features and target
    y = data_primitive[['DC', 'train_test']]
    x = data_primitive.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'site', 'relapse'])
    
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
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_primitive['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    # plt.show()

    # Now we consider that every person that die after 3 year is considered as not dead (using the delai_fin_DC column)
    y_deadline_3_year = data_primitive[['delai_fin_DC', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['delai_fin_DC'] = y_deadline_3_year['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that died within 3 year:", y_deadline_3_year[y_deadline_3_year['delai_fin_DC'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
    print("\nNumber of subjects that died within 3 year (train):", y_train[y_train['delai_fin_DC'] == 1].shape[0])
    print("Number of subjects that died within 3 year (test):", y_test[y_test['delai_fin_DC'] == 1].shape[0])   

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print(f"\nModel performance on the deadline of 3 year with {x.shape[1]} features")
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score:",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # We now perform feature selection
    
    # Now we move on to training the model with no deadline with fewer features based on what we explored in file `data_preprocessing/5_eliminating_radiomics_features.py` 
    # We keep the clinical features
    clinical_features = ['sexe', 'age', 'BMI', 'score_charlson', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'dose_tot', 'etalement',
                         'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']
    # We keep some radiomics features: selected here: 
    radiomics_features = ['INTENSITY-BASED_MeanIntensity', 'INTENSITY-BASED_IntensitySkewness', 'INTENSITY-BASED_IntensityKurtosis', 'INTENSITY-BASED_10thIntensityPercentile',          
                         'INTENSITY-BASED_AreaUnderCurveCIVH', 'INTENSITY-BASED_RootMeanSquareIntensity', 'INTENSITY-HISTOGRAM_IntensityHistogramMean',       
                         'INTENSITY-HISTOGRAM_IntensityHistogramVariance', 'NGTDM_Complexity', 'NGTDM_Strength']
    # Join the two lists
    x = data_primitive[clinical_features + radiomics_features + ['train_test']]
    y = data_primitive[['train_test', 'delai_fin_DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # In column 'delai_fin_DC', we replace all nan values by 1096 (3 years)
    y['delai_fin_DC'].fillna(1096, inplace=True)
    y['delai_fin_DC'] = y['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)

    # Renane column 'delai_fin_DC' to 'DC'
    y = y.rename(columns={'delai_fin_DC': 'DC'})

    # Split the data into training and testing sets
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_train = y_train['DC']
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    y_test = y_test['DC']
    print("\n --- Feature selection ---")
    print("Initial number of features:", x_train.shape[1])
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
    print(f"Model performance with 3 year deadline with {x_train.shape[1]} features")
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
    selector = VarianceThreshold(threshold=0.3)
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
    print(f"Model performance after variance thresholding with 3 year deadline and {x_train.shape[1]} features")
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
            if corr_matrix.loc[columns[i], columns[j]] > 0.80:
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
    print(f"Model performance after feature selection based on correlation (3-year deadline) with {x_train.shape[1]} features")
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
        if abs(corr_matrix.iloc[i]) < 0.014:
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
    print(f"Model performance after feature selection based on correlation with target (3-year deadline) with {x_train.shape[1]} features")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    # Print confusion matrix
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")


    ################################################################################################
    ######## SECOND SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR METASTASIS PATIENTS ############
    ################################################################################################

    print(" ------------- Model for prediction of survival for metastasis patients -------------")

    # We remove patients that are metastasis (have relapse different from 0)
    data_metastasis = data_grouped[data_grouped['relapse'] == 1]

    # Split into features and target
    y = data_metastasis[['DC', 'train_test']]
    x = data_metastasis.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'site', 'relapse'])
    
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
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_metastasis['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    # plt.show()

    # Now we consider that every person that die after 3 year is considered as not dead (using the delai_fin_DC column)
    y_deadline_3_year = data_metastasis[['delai_fin_DC', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['delai_fin_DC'] = y_deadline_3_year['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that died within 3 year:", y_deadline_3_year[y_deadline_3_year['delai_fin_DC'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
    print("\nNumber of subjects that died within 3 year (train):", y_train[y_train['delai_fin_DC'] == 1].shape[0])
    print("Number of subjects that died within 3 year (test):", y_test[y_test['delai_fin_DC'] == 1].shape[0])   

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print(f"\nModel performance on the deadline of 3 year with {x.shape[1]} features")
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score:",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # We now perform feature selection
    
    # Now we move on to training the model with no deadline with fewer features based on what we explored in file `data_preprocessing/5_eliminating_radiomics_features.py` 
    # We keep the clinical features
    clinical_features = ['sexe', 'age', 'BMI', 'score_charlson', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'dose_tot', 'etalement',
                         'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']
    # We keep some radiomics features: selected here: 
    radiomics_features = ['INTENSITY-BASED_MeanIntensity', 'INTENSITY-BASED_IntensitySkewness', 'INTENSITY-BASED_IntensityKurtosis', 'INTENSITY-BASED_10thIntensityPercentile',          
                         'INTENSITY-BASED_AreaUnderCurveCIVH', 'INTENSITY-BASED_RootMeanSquareIntensity', 'INTENSITY-HISTOGRAM_IntensityHistogramMean',       
                         'INTENSITY-HISTOGRAM_IntensityHistogramVariance', 'NGTDM_Complexity', 'NGTDM_Strength']
    # Join the two lists
    x = data_metastasis[clinical_features + radiomics_features + ['train_test']]
    y = data_metastasis[['train_test', 'delai_fin_DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # In column 'delai_fin_DC', we replace all nan values by 1096 (3 years)
    y['delai_fin_DC'].fillna(1096, inplace=True)
    y['delai_fin_DC'] = y['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)

    # Renane column 'delai_fin_DC' to 'DC'
    y = y.rename(columns={'delai_fin_DC': 'DC'})

    # Split the data into training and testing sets
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_train = y_train['DC']
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    y_test = y_test['DC']
    print("\n --- Feature selection ---")
    print("Initial number of features:", x_train.shape[1])
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
    print(f"Model performance with 3 year deadline with {x_train.shape[1]} features")
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
    selector = VarianceThreshold(threshold=0.3)
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
    print(f"Model performance after variance thresholding with 3 year deadline and {x_train.shape[1]} features")
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
            if corr_matrix.loc[columns[i], columns[j]] > 0.80:
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
    print(f"Model performance after feature selection based on correlation (3-year deadline) with {x_train.shape[1]} features")
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
        if abs(corr_matrix.iloc[i]) < 0.1:
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
    print(f"Model performance after feature selection based on correlation with target (3-year deadline) with {x_train.shape[1]} features")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    # Print confusion matrix
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")


    ################################################################################################
    ######## THIRD SCENARIO : MODEL FOR PREDICTION OF SURVIVAL FOR ALL PATIENTS ####################
    ################################################################################################

    print(" ------------- Model for prediction of survival for all patients -------------")

    # We don't remove any patient
    data_all = data_grouped

    # Split into features and target but keep 'relapse' as a feature
    y = data_all[['DC', 'train_test']]
    x = data_all.drop(columns=['DC', 'delai_fin_DC', 'subject_id', 'site'])
    
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
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print("\nNumber of subject for training:", x_train.shape[0])
    print("Number of subject for testing:", x_test.shape[0])
    print("\n")

    # Plot the distribution of 'delai_fin_DC'
    data_all['delai_fin_DC'].hist()
    plt.title('Distribution of survival time between end of treatment and\n death (in days) for those that died')
    plt.xlabel('Survival time')
    plt.ylabel('Number of subjects')
    # plt.show()

    # Now we consider that every person that die after 3 year is considered as not dead (using the delai_fin_DC column)
    y_deadline_3_year = data_all[['delai_fin_DC', 'train_test']]
    y_deadline_3_year.fillna(1096, inplace=True)
    y_deadline_3_year['delai_fin_DC'] = y_deadline_3_year['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)
    print("Number of subjects that died within 3 year:", y_deadline_3_year[y_deadline_3_year['delai_fin_DC'] == 1].shape[0])

    # Split the data into training and testing sets
    y_train = y_deadline_3_year[y_deadline_3_year['train_test'] == 'train'].drop(columns=['train_test'])
    y_test = y_deadline_3_year[y_deadline_3_year['train_test'] == 'test'].drop(columns=['train_test'])
    print("\nNumber of subjects that died within 3 year (train):", y_train[y_train['delai_fin_DC'] == 1].shape[0])
    print("Number of subjects that died within 3 year (test):", y_test[y_test['delai_fin_DC'] == 1].shape[0])   

    # Initialise the model
    model = XGBClassifier(seed=42)
    model.fit(x_train, y_train)

    # Performance on the training set
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC-AUC, accuracy score, Brier score and PR-AUC score
    print(f"\nModel performance on the deadline of 3 year with {x.shape[1]} features")
    print("ROC AUC Score:", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score:",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    print("\n")

    # We now perform feature selection
    
    # Now we move on to training the model with no deadline with fewer features based on what we explored in file `data_preprocessing/5_eliminating_radiomics_features.py` 
    # We keep the clinical features
    clinical_features = ['sexe', 'age', 'BMI', 'score_charlson', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'dose_tot', 'etalement',
                         'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10', 'relapse']
    # We keep some radiomics features: selected here: 
    radiomics_features = ['INTENSITY-BASED_MeanIntensity', 'INTENSITY-BASED_IntensitySkewness', 'INTENSITY-BASED_IntensityKurtosis', 'INTENSITY-BASED_10thIntensityPercentile',          
                         'INTENSITY-BASED_AreaUnderCurveCIVH', 'INTENSITY-BASED_RootMeanSquareIntensity', 'INTENSITY-HISTOGRAM_IntensityHistogramMean',       
                         'INTENSITY-HISTOGRAM_IntensityHistogramVariance', 'NGTDM_Complexity', 'NGTDM_Strength']
    # Join the two lists
    x = data_all[clinical_features + radiomics_features + ['train_test']]
    y = data_all[['train_test', 'delai_fin_DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    # In column 'delai_fin_DC', we replace all nan values by 1096 (3 years)
    y['delai_fin_DC'].fillna(1096, inplace=True)
    y['delai_fin_DC'] = y['delai_fin_DC'].apply(lambda x: 0 if x > 1095 else 1)

    # Renane column 'delai_fin_DC' to 'DC'
    y = y.rename(columns={'delai_fin_DC': 'DC'})

    # Split the data into training and testing sets
    x_train = x[x['train_test'] == 'train'].drop(columns=['train_test'])
    x_test = x[x['train_test'] == 'test'].drop(columns=['train_test'])
    y_train = y[y['train_test'] == 'train'].drop(columns=['train_test'])
    y_train = y_train['DC']
    y_test = y[y['train_test'] == 'test'].drop(columns=['train_test'])
    y_test = y_test['DC']
    print("\n --- Feature selection ---")
    print("Initial number of features:", x_train.shape[1])
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
    print(f"Model performance with 3 year deadline with {x_train.shape[1]} features")
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
    selector = VarianceThreshold(threshold=0.3)
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
    print(f"Model performance after variance thresholding with 3 year deadline and {x_train.shape[1]} features")
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
            if corr_matrix.loc[columns[i], columns[j]] > 0.80:
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
    print(f"Model performance after feature selection based on correlation (3-year deadline) with {x_train.shape[1]} features")
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
        if abs(corr_matrix.iloc[i]) < 0.06:
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
    print(f"Model performance after feature selection based on correlation with target (3-year deadline) with {x_train.shape[1]} features")
    print("ROC AUC Score: ", roc_auc_score(y_test, y_test_proba))
    print("Brier score:", brier_score_loss(y_test, y_test_proba))
    print("Average precision:", precision_score(y_test, y_test_pred))
    print("Average Recall:", recall_score(y_test, y_test_pred))
    print("Accuracy Score: ",accuracy_score(y_test, y_test_pred))
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    print("AUC-PR score:", auc(recall_test, precision_test))
    # Print confusion matrix
    print("Confusion matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("\n")
    
    return None


if __name__ == '__main__':
    main()