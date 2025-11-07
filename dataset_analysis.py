"""
Dataset analysis of either primitive or metastasis data

Input:
    --input: path to the merged dataset
    --output: path to the output folder
    --primitif: value 0 and 1 indicating if we want to analyze primitive data (1) or metastasis data (0)

Output:
    None

Author: Pierre-Louis Benveniste
"""

import argparse
import os
from loguru import logger
import pandas as pd


def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    parser = argparse.ArgumentParser(description='Analyze the dataset for either primitive or metastasis data')
    parser.add_argument('--input', type=str, help='Path to the csv file containing the merged data')
    parser.add_argument('--output', type=str, help='Path to the folder where the results will be saved')
    parser.add_argument('--primitif', type=int, choices=[0, 1], help='Value 0 and 1 indicating if we want to analyze primitive data (1) or metastasis data (0)')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It does the analysis of the dataset.
    """

    # We parse the arguments
    args = parse_args()
    input_data = args.input
    output_folder = args.output
    primitif = args.primitif

    # If the output_folder doesn't exist create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    if primitif:
        log_file = os.path.join(output_folder, f'data_analysis_primitif.txt')
    else:
        log_file = os.path.join(output_folder, f'data_analysis_metas.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    # Load the dataset
    data = pd.read_csv(input_data)

    # In the primitif
    if primitif:
        # We keep primitive patients
        data_all = data[data['primitif'] == 1]
    elif not primitif:
        # We keep metastasis patients
        data_all = data[data['primitif'] != 1]

    # First we focus on the clinical columns:
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Clinical columns:")
    clinical_columns = ['sexe', 'age', 'BMI', 'score_charlson', 'dyspnee_NYHA', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'poids', 'taille', 'score_age_50', 'score_IDM', 'score_ICC', 'score_vasc_periph', 'score_AIT_AVC', 'score_demence', 'score_BPCO', 'score_systeme', 'score_ulcere', 'score_hepatique_mod', 'score_diabete', 'score_diabete_comp', 'score_hemiplegie', 'score_IRC', 'score_kc_local', 'score_leucemie', 'score_lymphome', 'score_hepatique_HTP', 'score_kc_meta', 'score_sida', 'SAS', 'HTA', 'N', 'M', 'nbre_cibles', 'loc', 'ATCD_neo_NP', 'ATCD_neo_pulm', 'ATCD_loc_1', 'ATCD_loc_2', 'ATCD_loc_3']
    data_clinical = data_all[['subject_id'] + clinical_columns]
    # remove L803_2 and L810_2 from data_clinical if present
    data_clinical = data_clinical[~data_clinical['subject_id'].isin(['L803_2', 'L810_2'])]
    logger.info(f"Total number of subject IDs after removing L803_2 and L810_2: {data_clinical['subject_id'].nunique()}")
    # For clinical columns, we group the data by subject_id and take the first value
    data_grouped = data_clinical.groupby('subject_id').mean().reset_index()
    logger.info(f"Total number of patients: {data_grouped.shape[0]}")
    ## sort them by alphabetical order
    clinical_columns = sorted(clinical_columns)
    logger.info(f"Clinical columns: {clinical_columns}")
    logger.info(f"Number of clinical columns: {len(clinical_columns)}")

    # For each column, we print the the mean, the std,
    logger.info("Clinical columns analysis (per patient): we report mean +- std. median [range] and number of nans")
    for col in clinical_columns:
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_grouped[col].mean()} +- {data_grouped[col].std()}")
        logger.info(f"  Median: {data_grouped[col].median()}, [{data_grouped[col].min()} - {data_grouped[col].max()}]")
        logger.info(f"  Number of NaNs: {data_grouped[col].isna().sum()}")
        # If there are less than 10 unique values, we also print the counts for each value
        if data_grouped[col].nunique() <= 10:
            value_counts = data_grouped[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"    Value {value}: {count}")

    logger.info("\n")
    # Delete the data_grouped to free memory
    del data_grouped

    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Dosimetric columns:")
    logger.info(f"Total number of nodules: {data_all.shape[0]}")
    # Then we focus on the dosimetric columns:
    dosimetric_columns = ['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10',  'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV',]
    ## sort them by alphabetical order
    dosimetric_columns = sorted(dosimetric_columns)
    logger.info(f"Dosimetric columns: {dosimetric_columns}")
    logger.info(f"Number of dosimetric columns: {len(dosimetric_columns)}")

    # For each column, we print the the mean, the std, the median, and the number of NaNs
    logger.info("Dosimetric columns analysis (per nodule): we report mean +- std. median [range] and number of nans")
    for col in dosimetric_columns:
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_all[col].mean()} +- {data_all[col].std()}")
        logger.info(f"  Median: {data_all[col].median()}, [{data_all[col].min()} - {data_all[col].max()}]")
        logger.info(f"  Number of NaNs: {data_all[col].isna().sum()}")

        # If there are less than 10 unique values, we also print the counts for each value
        if data_all[col].nunique() <= 10:
            value_counts = data_all[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"    Value {value}: {count}")

    # Then we focus on the other columns:
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Other columns:")
    logger.info(f"Total number of nodules: {data_all.shape[0]}")
    cols_others = [
        'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum',
        'Reponse', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum',
        'delai_fin_rechutePTV', 'delai_fin_rechuteHomo', 'delai_fin_rechuteMed',
        'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum',
        'subject_nodule', 'nodule', 'follow_up', 'DC', 'delai_fin_DC'
    ]
    for col in cols_others:
        # We don't want to analyze date columns
        if 'DDD' in col or 'Date' in col :
            continue
        # We don't want to analyze subject_nodule or nodule columns
        if 'subject_nodule' in col or 'nodule' in col:
            continue
        # We print mean and std, median and range, number of nans
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_all[col].mean()} +- {data_all[col].std()}")
        logger.info(f"  Median: {data_all[col].median()}, [{data_all[col].min()} - {data_all[col].max()}]")
        logger.info(f"  Number of NaNs: {data_all[col].isna().sum()}")
        # If there are less than 10 unique values, we also print the counts for each value
        if data_all[col].nunique() <= 10:
            value_counts = data_all[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"    Value {value}: {count}")
    

    # Finally we focus on the radiomic columns:
    radiomic_columns = [col for col in data_all.columns if col not in clinical_columns + dosimetric_columns + cols_others]
    ## remove 'primitif', 'subject_id' from radiomic columns if present
    if 'primitif' in radiomic_columns:
        radiomic_columns.remove('primitif')
    if 'subject_id' in radiomic_columns:
        radiomic_columns.remove('subject_id')
    ## sort them by alphabetical order
    radiomic_columns = sorted(radiomic_columns)
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Radiomic columns:")
    logger.info(f"Total number of nodules: {data_all.shape[0]}")
    logger.info(f"Radiomic columns: {radiomic_columns}")
    logger.info(f"Number of radiomic columns: {len(radiomic_columns)}")

    for col in radiomic_columns:
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_all[col].mean()} +- {data_all[col].std()}")
        logger.info(f"  Median: {data_all[col].median()}, [{data_all[col].min()} - {data_all[col].max()}]")
        logger.info(f"  Number of NaNs: {data_all[col].isna().sum()}")
    
    return None


if __name__ == '__main__':
    main()