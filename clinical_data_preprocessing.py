"""
This file is created to pre-process the data from the lung cancer response dataset put in place by Camille Invernizzi.

Args:
    input: Path to the input csv file
    output: Path to the output folder

Returns:
    None

Example:
    python clinical_data_preprocessing.py --input /path/to/csv_file.csv --output /output/folder/

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    parser = argparse.ArgumentParser(description='Preprocess the data from the lung cancer response dataset')
    parser.add_argument('--input', type=str, help='Path to the input csv file')
    parser.add_argument('--output', type=str, help='Path to the output folder')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It preprocesses the data from the lung cancer response csv dataset
    """

    # We parse the arguments
    args = parse_args()
    input_path = args.input
    output_folder = args.output

    # We first load the dataset (in csv)
    data = pd.read_csv(input_path)

    # From our conversations we selected the following features to keep:
    ## sex, age, BMI, Charlson's score, OMS, tabac, tabac_PA, tabac_sevre, histo, T, centrale, dose_tot, etalement, vol_GTV, vol_PTV, vol_ITV, couv-PTV, BED_10
    # The outputs are the following:
    ## deces, date de mort (DDD), cause_deces, toutes les dates de rechute, reponse (le nodule a repondu)
    feature_columns = ['sexe', 'age', 'BMI', 'score\n_charlson', 'OMS', 'tabac', 'tabac\n_PA', 'tabac\n_sevre', 
                'histo', 'T', 'centrale', 'dose\n_tot', 'etalement', 'vol\n_GTV', 'vol\n_PTV', 'vol\n_ITV', 'couv\n_PTV', 'BED\n_10']
    outputs_columnns = ['DC', 'DDD', 'cause_DC', 'Date_R\n_PTV', 'Date_R\n_homo','Date_R\n_med','Date_R\n_contro','Date_R\n_horspoum', 'Reponse']
    all_columns = feature_columns + outputs_columnns

    # We select the data of interest
    data = data[all_columns]

    # We remove all lines after line 181 (because they contain data regarding the average etc)
    data = data[:181]

    # We replace all '-' by nan in dataset
    data = data.apply(lambda x: x.replace('-', np.nan) if x.dtype == "object" else x)

    # For each column we verify the data taken
    ## For sex we check that it can only take 0 or 1
    ## Convert the values from str to int
    data['sexe'] = data['sexe'].apply(pd.to_numeric)
    ## For age we check that they are all int values
    data['age'] = data['age'].apply(pd.to_numeric)
    ## For BMI we check that it's a float value
    data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == "object" else x)
    data['BMI'] = data['BMI'].apply(pd.to_numeric)
    ## For score\n_charlson we check that they all take numeric values
    data["score\n_charlson"] =  data["score\n_charlson"].apply(pd.to_numeric)
    ## For OMS we check that they all take numeric values
    data['OMS'] = data['OMS'].apply(pd.to_numeric)
    ## For tabac, tabac\n_PA, tabac\n_sevre we check that they all take numeric values
    data['tabac'] = data['tabac'].apply(pd.to_numeric)
    data['tabac\n_PA'] = data['tabac\n_PA'].apply(pd.to_numeric)
    # data['tabac\n_sevre'] = data['tabac\n_sevre'].replace('7', '1')
    data['tabac\n_sevre'] = data['tabac\n_sevre'].apply(pd.to_numeric)
    ## For histo we check that they all take numeric values
    data['histo'] = data['histo'].apply(pd.to_numeric)
    ## For T we check that they all take numeric values
    data['T'] = data['T'].apply(pd.to_numeric)
    ## For centrale we check that they all take numeric values
    data['centrale'] = data['centrale'].apply(pd.to_numeric)
    ## For dose\n_tot we check that they all take numeric values
    data['dose\n_tot'] = data['dose\n_tot'].apply(pd.to_numeric)
    ## For etalement we check that they all take numeric values
    data['etalement'] = data['etalement'].apply(pd.to_numeric)
    ## For vol\n_GTV, vol\n_PTV, vol\n_ITV, couv\n_PTV, BED\n_10 we check that they all take numeric values
    data['vol\n_GTV'] = data['vol\n_GTV'].apply(pd.to_numeric)
    data['vol\n_PTV'] = data['vol\n_PTV'].apply(pd.to_numeric)
    data['vol\n_ITV'] = data['vol\n_ITV'].apply(pd.to_numeric)
    data['couv\n_PTV'] = data['couv\n_PTV'].apply(pd.to_numeric)
    data['BED\n_10'] = data['BED\n_10'].apply(pd.to_numeric)

    ## For DC we check that they all take numeric values
    data['DC'] = data['DC'].apply(pd.to_numeric)
    ## For DDD we check that they are all dates values
    data['DDD'] = data['DDD'].apply(pd.to_datetime, dayfirst=True)
    ## For cause_DC we check that they all take numeric values
    data['cause_DC'] = data['cause_DC'].apply(pd.to_numeric)
    ## For Date_R\n_PTV, Date_R\n_homo, Date_R\n_med, Date_R\n_contro, Date_R\n_horspoum we check they are all dates
    ## For each of these dates we build a column if takes 1 if there is a date and 0 otherwise
    data['rechute_PTV'] = data['Date_R\n_PTV'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R\n_PTV'] = data['Date_R\n_PTV'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_homo'] = data['Date_R\n_homo'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R\n_homo'] = data['Date_R\n_homo'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_med'] = data['Date_R\n_med'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R\n_med'] = data['Date_R\n_med'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_contro'] = data['Date_R\n_contro'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R\n_contro'] = data['Date_R\n_contro'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_horspoum'] = data['Date_R\n_horspoum'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R\n_horspoum'] = data['Date_R\n_horspoum'].replace('1', np.nan)
    data['Date_R\n_horspoum'] = data['Date_R\n_horspoum'].apply(pd.to_datetime, dayfirst=True)
    ## For Reponse we check thay all take numeric values
    data['Reponse'] = data['Reponse'].apply(pd.to_numeric)

    ## We save the pre-processed dataset in the folder
    output_file = os.path.join(output_folder, "preprocessed_csv.csv")
    data.to_csv(output_file, index=False)

    return None


if __name__ == "__main__":
    main()