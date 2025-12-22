"""
In this file we check whether there is a difference in survival between primitive and metastasis patients using the log-rank test.

Args:
    --merged-data: Path to the csv file containing the merged data

Returns:
    None

Example:
    python model_training/6_survival_primitive_vs_metastasis.py --merged-data data/merged_data.csv

Author: Pierre-Louis Benveniste
"""
import pandas as pd
import argparse
from lifelines.statistics import logrank_test


def parse_args():
    parser = argparse.ArgumentParser(description='Check whether there is a difference in survival between primitive and metastasis patients using the log-rank test')
    parser.add_argument('--merged-data', type=str, help='Path to the csv file containing the merged data', required=True)
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()

    # Load data
    data = pd.read_csv(args.merged_data)

    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'delai_fin_rechutePTV',
                            'delai_fin_rechuteHomo','delai_fin_rechuteMed','delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule' ])

    # We average the columns for the same patients across the different nodules
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # We build the column relapse which takes 1 if either of the relapse occur ('rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum')
    data_grouped['relapse'] = data_grouped['rechute_PTV'] + data_grouped['rechute_homo'] + data_grouped['rechute_med'] + data_grouped['rechute_contro'] + data_grouped['rechute_horspoum']
    # Modify the 'relapse' column to take value 1 if it is above 0
    data_grouped['relapse'] = data_grouped['relapse'].apply(lambda x: 1 if x > 0 else 0)

    # we only keep the columns for relapse (RC), death (DC), and time before death (delai_fin_DC)
    data_logrank = data_grouped[['relapse', 'delai_fin_DC']]

    # Null Hypothesis (H₀): There is no difference in survival distributions between the groups (e.g., subjects with relapse vs. without relapse).
    # Alternative Hypothesis (H₁): There is a significant difference in survival distributions between the groups.

    ###########################
    ###### FIRST STRATEGY #####
    ###########################
    # In this first strategy we consider that if 'delai_fin_DC' is nan then death occurs at time = + infinite
    # Replace nan in 'delai_fin_DC' by infinite positive (That means that the person is not based on the analysis)
    data_first_strategy = data_logrank.copy()
    data_first_strategy['delai_fin_DC'].fillna(float('inf'), inplace=True)

    # Extract first group : no-relapse
    no_relapse = data_first_strategy[data_first_strategy['relapse'] == 0]['delai_fin_DC']
    # Extract the second group: relapse
    relapse = data_first_strategy[data_first_strategy['relapse'] == 1]['delai_fin_DC']

    # Perform log-rank test
    results = logrank_test(no_relapse, relapse)

    # Print the test statistic and p-value
    print("First Strategy: if 'delai_fin_DC' is nan then death occurs at time = + infinite")
    print("Log-Rank Test Statistic:", results.test_statistic)
    print("Log-Rank Test p-value:", results.p_value)

    ###########################
    ##### SECOND STRATEGY #####
    ###########################
    # In this second strategy we remove the patients for which 'delai_fin_DC' is nan
    data_second_strategy = data_logrank.dropna(subset=['delai_fin_DC'])

    # Extract first group : no-relapse
    no_relapse = data_second_strategy[data_second_strategy['relapse'] == 0]['delai_fin_DC']
    # Extract the second group: relapse
    relapse = data_second_strategy[data_second_strategy['relapse'] == 1]['delai_fin_DC']

    # Perform log-rank test
    results = logrank_test(no_relapse, relapse)

    # Print the test statistic and p-value
    print("Second Strategy: remove the patients for which 'delai_fin_DC' is nan")
    print("Log-Rank Test Statistic:", results.test_statistic)
    print("Log-Rank Test p-value:", results.p_value)

    return None


if __name__ == '__main__':
    main()