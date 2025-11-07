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

    # Load the dataset
    data_all = pd.read_csv(input_data)

    # Remove columns that are not useful for averaging
    cols_others = [
        'DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum',
        'Reponse', 'rechute_PTV', 'rechute_homo', 'rechute_med', 'rechute_contro', 'rechute_horspoum',
        'delai_fin_rechutePTV', 'delai_fin_rechuteHomo', 'delai_fin_rechuteMed',
        'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum',
        'subject_nodule', 'nodule', 'follow_up', 'DC', 'delai_fin_DC'
    ]
    logger.info(f"Columns removed for averaging: {cols_others}")
    logger.info(f"Number of columns removed for averaging: {len(cols_others)}")
    data = data_all.drop(columns=cols_others)

    # We average all features
    data_grouped = data.groupby('subject_id').mean().reset_index()

    # We re-add the removed features
    data_removed = data_all[['subject_id'] + cols_others].drop_duplicates(subset='subject_id')
    # We merge the data
    data_grouped = pd.merge(data_grouped, data_removed, on='subject_id',  how='outer')
    
    # Print number of primitive patients and number of metastasis
    logger.info(f"Total number of patients: {data_grouped.shape[0]}")
    logger.info(f"Number of primitive patients: {data_grouped[data_grouped['primitif']==1].shape[0]}")
    logger.info(f"Number of metastasis patients: {data_grouped[data_grouped['primitif']!=1].shape[0]}")
    logger.info("\n")

    if primitif:
        # We keep primitive patients
        data_grouped = data_grouped[data_grouped['primitif'] == 1]
    elif not primitif:
        # We keep metastasis patients
        data_grouped = data_grouped[data_grouped['primitif'] != 1]

    # Remove the 'primitif' column
    data_grouped = data_grouped.drop(columns=['primitif', 'subject_id'])
    
    logger.info("Analysis of the dataset:")
    logger.info(f"Number of columns: {data_grouped.shape[1]}")
    logger.info(f"Number of patients: {data_grouped.shape[0]}")
    logger.info(f"Columns in the dataset: {data_grouped.columns.tolist()}")

    # First we focus on the clinical columns:
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    clinical_columns = ['sexe', 'age', 'BMI', 'score_charlson', 'dyspnee_NYHA', 'OMS', 'tabac', 'tabac_PA', 'tabac_sevre', 'histo', 'T', 'centrale', 'poids', 'taille', 'score_age_50', 'score_IDM', 'score_ICC', 'score_vasc_periph', 'score_AIT_AVC', 'score_demence', 'score_BPCO', 'score_systeme', 'score_ulcere', 'score_hepatique_mod', 'score_diabete', 'score_diabete_comp', 'score_hemiplegie', 'score_IRC', 'score_kc_local', 'score_leucemie', 'score_lymphome', 'score_hepatique_HTP', 'score_kc_meta', 'score_sida', 'SAS', 'HTA', 'N', 'M', 'nbre_cibles', 'loc', 'ATCD_neo_NP', 'ATCD_neo_pulm', 'ATCD_loc_1', 'ATCD_loc_2', 'ATCD_loc_3']
    ## sort them by alphabetical order
    clinical_columns = sorted(clinical_columns)
    logger.info(f"Clinical columns: {clinical_columns}")
    logger.info(f"Number of clinical columns: {len(clinical_columns)}")

    # For each column, we print the the mean, the std,
    logger.info("Clinical columns analysis: we report mean +- std. median [range] and number of nans")
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

    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    # Then we focus on the dosimetric columns:
    dosimetric_columns = ['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10',  'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV',]
    ## sort them by alphabetical order
    dosimetric_columns = sorted(dosimetric_columns)
    logger.info(f"Dosimetric columns: {dosimetric_columns}")
    logger.info(f"Number of dosimetric columns: {len(dosimetric_columns)}")

    # For each column, we print the the mean, the std, the median, and the number of NaNs
    logger.info("Dosimetric columns analysis: we report mean +- std. median [range] and number of nans")
    for col in dosimetric_columns:
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_grouped[col].mean()} +- {data_grouped[col].std()}")
        logger.info(f"  Median: {data_grouped[col].median()}, [{data_grouped[col].min()} - {data_grouped[col].max()}]")
        logger.info(f"  Number of NaNs: {data_grouped[col].isna().sum()}")

        # If there are less than 10 unique values, we also print the counts for each value
        if data_grouped[col].nunique() <= 10:
            value_counts = data_grouped[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"    Value {value}: {count}")

    # Then we focus on the other columns:
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Other columns:")
    for col in cols_others:
        # We don't want to analyze date columns
        if 'DDD' in col or 'Date' in col :
            continue
        # We don't want to analyze subject_nodule or nodule columns
        if 'subject_nodule' in col or 'nodule' in col:
            continue
        # We print mean and std, median and range, number of nans
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_grouped[col].mean()} +- {data_grouped[col].std()}")
        logger.info(f"  Median: {data_grouped[col].median()}, [{data_grouped[col].min()} - {data_grouped[col].max()}]")
        logger.info(f"  Number of NaNs: {data_grouped[col].isna().sum()}")
        # If there are less than 10 unique values, we also print the counts for each value
        if data_grouped[col].nunique() <= 10:
            value_counts = data_grouped[col].value_counts().sort_index()
            for value, count in value_counts.items():
                logger.info(f"    Value {value}: {count}")
    

    # Finally we focus on the radiomic columns:
    radiomic_columns = [col for col in data_grouped.columns if col not in clinical_columns + dosimetric_columns + cols_others]
    ## sort them by alphabetical order
    radiomic_columns = sorted(radiomic_columns)
    logger.info("\n ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info(f"Radiomic columns: {radiomic_columns}")
    logger.info(f"Number of radiomic columns: {len(radiomic_columns)}")

    for col in radiomic_columns:
        logger.info(f"Column: {col}")
        logger.info(f"  Mean: {data_grouped[col].mean()} +- {data_grouped[col].std()}")
        logger.info(f"  Median: {data_grouped[col].median()}, [{data_grouped[col].min()} - {data_grouped[col].max()}]")
        logger.info(f"  Number of NaNs: {data_grouped[col].isna().sum()}")
    



if __name__ == '__main__':
    main()