"""
In this script we perform student T test between the training and testing sets.
We compare the features of the training and testing sets to ensure they are similar.

Args:
    --input: path to the merged dataset
    --output: path to the output folder

Output:
    None

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import pandas as pd
from scipy.stats import ttest_ind


def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    import pandas as pd
    import numpy as np

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

    #################################################################################
    ################## STUDENT T TEST BETWEEN TRAINING AND TESTING SETS #############
    #################################################################################

    # Initilize a list to store the results which will be saved in a text file
    amount_nans = []
    results = []
    
    # For each feature, we perform a student t-test between the training and testing sets
    for column in x_train.columns:
        # For both train and test sets, we compute the amount of NaN values
        train_nan_count = x_train[column].isna().sum()
        test_nan_count = x_test[column].isna().sum()
        # Append the amount of NaN values to the list
        amount_nans.append(f"{column}: train NaN count = {train_nan_count} (% {train_nan_count / x_train.shape[0] * 100:.2f}), test NaN count = {test_nan_count} (% {test_nan_count / x_test.shape[0] * 100:.2f})")
        # Remove the NaN values from both sets for the t-test
        x_train[column] = x_train[column].dropna()
        x_test[column] = x_test[column].dropna()
        # Perform the t-test
        t_stat, p_value = ttest_ind(x_train.dropna(subset=[column])[column], x_test.dropna(subset=[column])[column], equal_var=False)
        # Append the results to the list
        results.append(f"{column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    # Save the results in a text file
    output_file = os.path.join(output_folder, 'student_t_test_results.txt')
    with open(output_file, 'w') as f:
        f.write("Student T-test results between training and testing sets:\n")
        # For each feature, we write the amount of NaN values and then the t-test results
        for nan_info, result in zip(amount_nans, results):
            f.write(f"{nan_info}\n")
            f.write(f"{result}\n")
        f.write("\n\nSummary of the results:\n")
        f.write(f"Total number of features compared: {len(x_train.columns)}\n")

    print(f"Results saved to {output_file}")

    return None


if __name__ == "__main__":
    main()