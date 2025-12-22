"""
This file is created to pre-process the data from the lung cancer response dataset put in place by Camille Invernizzi.

Args:
    input: Path to the input csv file
    output-file: Path to the output file

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
    parser.add_argument('--output-file', type=str, help='Path to the output file')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It preprocesses the data from the lung cancer response csv dataset
    """

    # We parse the arguments
    args = parse_args()
    input_path = args.input
    output_file = args.output_file

    # If the output folder does not exist we create it
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # We first load the dataset (in csv)
    data = pd.read_csv(input_path)

    # From our conversations we selected the following features to keep:
    ## sex, age, BMI, Charlson's score, OMS, tabac, tabac_PA, tabac_sevre, histo, T, centrale, dose_tot, etalement, vol_GTV, vol_PTV, vol_ITV, couv-PTV, BED_10
    # The outputs are the following:
    ## deces, date de mort (DDD), cause_deces, toutes les dates de rechute, reponse (le nodule a repondu)
    feature_columns = ['num\n_patient', 'sexe', 'age', 'BMI', 'score\n_charlson', 'dyspnee\n_NYHA', 'OMS', 'tabac', 'tabac\n_PA', 'tabac\n_sevre', 
                'histo', 'T', 'centrale', 'dose\n_tot', 'etalement', 'vol\n_GTV', 'vol\n_PTV', 'vol\n_ITV', 'couv\n_PTV', 'BED\n_10', 'Last\n_news', 'date\n_TDM']
    outputs_columnns = ['DC', 'DDD', 'cause_DC', 'Date_R\n_PTV', 'Date_R\n_homo','Date_R\n_med','Date_R\n_contro','Date_R\n_horspoum', 'Reponse', 'date\n_fin']
    # In the end, we decide to add all features
    newly_added_columns = ['poids', 'taille', 'age_50\n_1', 'IDM\n_1', 'ICC\n_1', 'vasc\n_periph\n_1', 'AIT\n_AVC\n_1', 'demence\n_1', 'BPCO\n_1', 'systeme\n_1',
                            'ulcere\n_1', 'hepatique\n_mod\n_1', 'diabete\n_1', 'diabete\n_comp\n_2', 'hemiplegie\n_2', 'IRC\n_2', 'kc\n_local\n_2', 'leucemie\n_2',
                            'lymphome\n_2', 'hepatique\n_HTP\n_3', 'kc\n_meta\n_6', 'sida\n_6']
    last_added_columns = ['SAS', 'HTA', 'primitif', 'N','M', 'nbre_cibles', 'loc', 'ATCD\n_neo_NP','ATCD\n_neo\n_pulm',
                           '\nATCD\n_loc_1', '\r\nATCD\r\n_loc_2', '\r\nATCD\r\n_loc_3', 'ATCD\n_histo', 'dose\n_fraction',
                           'min\n_PTV', 'mean\n_PTV', 'max\n_PTV', ]
    all_columns = feature_columns + outputs_columnns + newly_added_columns + last_added_columns

    # We select the data of interest
    data = data[all_columns]

    # We remove all lines after line 181 (because they contain data regarding the average etc)
    data = data[:181]

    # We replace all '-' by nan in dataset
    data = data.apply(lambda x: x.replace('-', np.nan) if x.dtype == "object" else x)

    # Renaming of the columns (removing \n)
    change_columns = {
        "num\n_patient": "num_patient",
        "score\n_charlson": "score_charlson", 
        "dyspnee\n_NYHA": "dyspnee_NYHA",
        "tabac\n_PA": "tabac_PA",
        "tabac\n_sevre": "tabac_sevre",
        "dose\n_tot": "dose_tot",
        "vol\n_GTV": "vol_GTV",
        "vol\n_PTV": "vol_PTV",
        "vol\n_ITV": "vol_ITV",
        "couv\n_PTV": "couv_PTV",
        "BED\n_10": "BED_10",
        "Date_R\n_PTV": "Date_R_PTV",
        "Date_R\n_homo": "Date_R_homo",
        "Date_R\n_med": "Date_R_med",
        "Date_R\n_contro": "Date_R_contro",
        "Date_R\n_horspoum": "Date_R_horspoum",
        "date\n_fin": "date_fin",
        "Last\n_news": "last_news",
        "date\n_TDM": "date_TDM",
        "age_50\n_1": "score_age_50",
        "IDM\n_1": "score_IDM",
        "ICC\n_1": "score_ICC",
        "vasc\n_periph\n_1": "score_vasc_periph",
        "AIT\n_AVC\n_1": "score_AIT_AVC",
        "demence\n_1": "score_demence",
        "BPCO\n_1": "score_BPCO",
        "systeme\n_1": "score_systeme",
        "ulcere\n_1": "score_ulcere",
        "hepatique\n_mod\n_1": "score_hepatique_mod",
        "diabete\n_1": "score_diabete",
        "diabete\n_comp\n_2": "score_diabete_comp",
        "hemiplegie\n_2": "score_hemiplegie",
        "IRC\n_2": "score_IRC",
        "kc\n_local\n_2": "score_kc_local",
        "leucemie\n_2": "score_leucemie",
        "lymphome\n_2": "score_lymphome",
        "hepatique\n_HTP\n_3": "score_hepatique_HTP",
        "kc\n_meta\n_6": "score_kc_meta",
        "sida\n_6": "score_sida",
        "ATCD\n_neo_NP": "ATCD_neo_NP",
        "ATCD\n_neo\n_pulm": "ATCD_neo_pulm",
        "\nATCD\n_loc_1": "ATCD_loc_1",
        '\r\nATCD\r\n_loc_2': 'ATCD_loc_2',
        '\r\nATCD\r\n_loc_3': 'ATCD_loc_3',
        'ATCD\n_histo': 'ATCD_histo',
        'dose\n_fraction': 'dose_fraction',
        'min\n_PTV': 'min_PTV',
        'mean\n_PTV': 'mean_PTV',
        'max\n_PTV': 'max_PTV',
    }
    data = data.rename(columns=change_columns)

    # For each column we verify the data taken
    ## For sex we check that it can only take 0 or 1
    ## Convert the values from str to int
    data['sexe'] = data['sexe'].apply(pd.to_numeric)
    ## For age we check that they are all int values
    data['age'] = data['age'].apply(pd.to_numeric)
    ## For BMI we check that it's a float value
    data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == "object" else x)
    data['BMI'] = data['BMI'].apply(pd.to_numeric)
    ## For score_charlson we check that they all take numeric values
    data["score_charlson"] =  data["score_charlson"].apply(pd.to_numeric)
    ## For dyspnee_NYHA we check that they all take numeric values
    data['dyspnee_NYHA'] = data['dyspnee_NYHA'].apply(pd.to_numeric)
    ## For OMS we check that they all take numeric values
    data['OMS'] = data['OMS'].apply(pd.to_numeric)
    ## For tabac, tabac_PA, tabac_sevre we check that they all take numeric values
    data['tabac'] = data['tabac'].apply(pd.to_numeric)
    data['tabac_PA'] = data['tabac_PA'].apply(pd.to_numeric)
    # data['tabac_sevre'] = data['tabac_sevre'].replace('7', '1')
    data['tabac_sevre'] = data['tabac_sevre'].apply(pd.to_numeric)
    ## For histo we check that they all take numeric values
    data['histo'] = data['histo'].apply(pd.to_numeric)
    ## For T we check that they all take numeric values
    data['T'] = data['T'].apply(pd.to_numeric)
    ## For centrale we check that they all take numeric values
    data['centrale'] = data['centrale'].apply(pd.to_numeric)
    ## For dose_tot we check that they all take numeric values
    data['dose_tot'] = data['dose_tot'].apply(pd.to_numeric)
    ## For etalement we check that they all take numeric values
    data['etalement'] = data['etalement'].apply(pd.to_numeric)
    ## For vol_GTV, vol_PTV, vol_ITV, couv_PTV, BED_10 we check that they all take numeric values
    data['vol_GTV'] = data['vol_GTV'].apply(pd.to_numeric)
    data['vol_PTV'] = data['vol_PTV'].apply(pd.to_numeric)
    data['vol_ITV'] = data['vol_ITV'].apply(pd.to_numeric)
    data['couv_PTV'] = data['couv_PTV'].apply(pd.to_numeric)
    data['BED_10'] = data['BED_10'].apply(pd.to_numeric)

    ## For DC we check that they all take numeric values
    data['DC'] = data['DC'].apply(pd.to_numeric)
    ## For DDD we check that they are all dates values
    data['DDD'] = data['DDD'].apply(pd.to_datetime, dayfirst=True)
    ## For cause_DC we check that they all take numeric values
    data['cause_DC'] = data['cause_DC'].apply(pd.to_numeric)
    ## For Date_R_PTV, Date_R_homo, Date_R_med, Date_R_contro, Date_R_horspoum we check they are all dates
    ## For each of these dates we build a column if takes 1 if there is a date and 0 otherwise
    data['rechute_PTV'] = data['Date_R_PTV'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R_PTV'] = data['Date_R_PTV'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_homo'] = data['Date_R_homo'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R_homo'] = data['Date_R_homo'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_med'] = data['Date_R_med'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R_med'] = data['Date_R_med'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_contro'] = data['Date_R_contro'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R_contro'] = data['Date_R_contro'].apply(pd.to_datetime, dayfirst=True)
    data['rechute_horspoum'] = data['Date_R_horspoum'].apply(lambda x: 1 if pd.notnull(x) else 0)
    data['Date_R_horspoum'] = data['Date_R_horspoum'].replace('1', np.nan)
    data['Date_R_horspoum'] = data['Date_R_horspoum'].apply(pd.to_datetime, dayfirst=True)
    ## For Reponse we check thay all take numeric values
    data['Reponse'] = data['Reponse'].apply(pd.to_numeric)
    ## For 'date_fin' we check that it is a date
    data['date_fin'] = data['date_fin'].apply(pd.to_datetime, dayfirst=True)

    # We create a column which is the time between end of treatment and death
    data['delai_fin_DC'] = data['DDD'] - data['date_fin']
    # Convert to days
    data['delai_fin_DC'] = data['delai_fin_DC'].dt.days
    # When delai_fin_DC is negative, we replace by 0
    data['delai_fin_DC'] = data['delai_fin_DC'].apply(lambda x: 0 if x < 0 else x)
    # We create a column which is the time between end of treatment and rechutes
    data['delai_fin_rechutePTV'] = data['Date_R_PTV'] - data['date_fin']
    data['delai_fin_rechutePTV'] = data['delai_fin_rechutePTV'].dt.days
    data['delai_fin_rechutePTV'] = data['delai_fin_rechutePTV'].apply(lambda x: 0 if x < 0 else x)
    data['delai_fin_rechuteHomo'] = data['Date_R_homo'] - data['date_fin']
    data['delai_fin_rechuteHomo'] = data['delai_fin_rechuteHomo'].dt.days
    data['delai_fin_rechuteHomo'] = data['delai_fin_rechuteHomo'].apply(lambda x: 0 if x < 0 else x)
    data['delai_fin_rechuteMed'] = data['Date_R_med'] - data['date_fin']
    data['delai_fin_rechuteMed'] = data['delai_fin_rechuteMed'].dt.days
    data['delai_fin_rechuteMed'] = data['delai_fin_rechuteMed'].apply(lambda x: 0 if x < 0 else x)
    data['delai_fin_rechuteContro'] = data['Date_R_contro'] - data['date_fin']
    data['delai_fin_rechuteContro'] = data['delai_fin_rechuteContro'].dt.days
    data['delai_fin_rechuteContro'] = data['delai_fin_rechuteContro'].apply(lambda x: 0 if x < 0 else x)
    data['delai_fin_rechuteHorspoum'] = data['Date_R_horspoum'] - data['date_fin']
    data['delai_fin_rechuteHorspoum'] = data['delai_fin_rechuteHorspoum'].dt.days
    data['delai_fin_rechuteHorspoum'] = data['delai_fin_rechuteHorspoum'].apply(lambda x: 0 if x < 0 else x)

    # We pre-preprocess the last_news column and the date_debut column
    data['last_news'] = data['last_news'].apply(pd.to_datetime, dayfirst=True)
    data['date_TDM'] = data['date_TDM'].apply(pd.to_datetime, dayfirst=True)

    # Create a column which is follow-up time
    data['follow_up'] = data['last_news'] - data['date_TDM']
    data['follow_up'] = data['follow_up'].dt.days

    # We also preprocess the newly added columns (all should be numeric)
    data['poids'] = data['poids'].apply(pd.to_numeric)
    data['taille'] = data['taille'].apply(pd.to_numeric)
    # For score_age_50 the value is binary (0 or 1) - nan should be replaced by 0
    data['score_age_50'] = data['score_age_50'].apply(pd.to_numeric)
    data['score_age_50'] = data['score_age_50'].fillna(0)
    data['score_IDM'] = data['score_IDM'].apply(pd.to_numeric)
    data['score_IDM'] = data['score_IDM'].fillna(0)
    data['score_ICC'] = data['score_ICC'].apply(pd.to_numeric)
    data['score_ICC'] = data['score_ICC'].fillna(0)
    data['score_vasc_periph'] = data['score_vasc_periph'].apply(pd.to_numeric)
    data['score_vasc_periph'] = data['score_vasc_periph'].fillna(0)
    data['score_AIT_AVC'] = data['score_AIT_AVC'].apply(pd.to_numeric)
    data['score_AIT_AVC'] = data['score_AIT_AVC'].fillna(0)
    data['score_demence'] = data['score_demence'].apply(pd.to_numeric)
    data['score_demence'] = data['score_demence'].fillna(0)
    data['score_BPCO'] = data['score_BPCO'].apply(pd.to_numeric)
    data['score_BPCO'] = data['score_BPCO'].fillna(0)
    data['score_systeme'] = data['score_systeme'].apply(pd.to_numeric)
    data['score_systeme'] = data['score_systeme'].fillna(0)
    data['score_ulcere'] = data['score_ulcere'].apply(pd.to_numeric)
    data['score_ulcere'] = data['score_ulcere'].fillna(0)
    data['score_hepatique_mod'] = data['score_hepatique_mod'].apply(pd.to_numeric)
    data['score_hepatique_mod'] = data['score_hepatique_mod'].fillna(0)
    data['score_diabete'] = data['score_diabete'].apply(pd.to_numeric)
    data['score_diabete'] = data['score_diabete'].fillna(0)
    data['score_diabete_comp'] = data['score_diabete_comp'].apply(pd.to_numeric)
    data['score_diabete_comp'] = data['score_diabete_comp'].fillna(0)
    data['score_hemiplegie'] = data['score_hemiplegie'].apply(pd.to_numeric)
    data['score_hemiplegie'] = data['score_hemiplegie'].fillna(0)
    data['score_IRC'] = data['score_IRC'].apply(pd.to_numeric)
    data['score_IRC'] = data['score_IRC'].fillna(0)
    data['score_kc_local'] = data['score_kc_local'].apply(pd.to_numeric)
    data['score_kc_local'] = data['score_kc_local'].fillna(0)
    data['score_leucemie'] = data['score_leucemie'].apply(pd.to_numeric)
    data['score_leucemie'] = data['score_leucemie'].fillna(0)
    data['score_lymphome'] = data['score_lymphome'].apply(pd.to_numeric)
    data['score_lymphome'] = data['score_lymphome'].fillna(0)
    data['score_hepatique_HTP'] = data['score_hepatique_HTP'].apply(pd.to_numeric)
    data['score_hepatique_HTP'] = data['score_hepatique_HTP'].fillna(0)
    data['score_kc_meta'] = data['score_kc_meta'].apply(pd.to_numeric)
    data['score_kc_meta'] = data['score_kc_meta'].fillna(0)
    data['score_sida'] = data['score_sida'].apply(pd.to_numeric)
    data['score_sida'] = data['score_sida'].fillna(0)

    # we preprocess the last added columns
    data['SAS'] = data['SAS'].apply(pd.to_numeric)
    data['HTA'] = data['HTA'].apply(pd.to_numeric)
    data['primitif'] = data['primitif'].apply(pd.to_numeric)
    data['N'] = data['N'].apply(pd.to_numeric)
    data['M'] = data['M'].apply(pd.to_numeric)
    data['nbre_cibles'] = data['nbre_cibles'].apply(pd.to_numeric)
    data['loc'] = data['loc'].apply(pd.to_numeric)
    data['ATCD_neo_NP'] = data['ATCD_neo_NP'].apply(pd.to_numeric)
    data['ATCD_neo_pulm'] = data['ATCD_neo_pulm'].apply(pd.to_numeric)
    data['ATCD_loc_1'] = data['ATCD_loc_1'].apply(lambda x: str(x).split('_')[0] if pd.notnull(x) else x)
    data['ATCD_loc_1'] = data['ATCD_loc_1'].apply(pd.to_numeric)
    data['ATCD_loc_2'] = data['ATCD_loc_2'].apply(lambda x: str(x).split('_')[0] if pd.notnull(x) else x)
    data['ATCD_loc_2'] = data['ATCD_loc_2'].apply(pd.to_numeric)
    data['ATCD_loc_3'] = data['ATCD_loc_3'].apply(lambda x: str(x).split('_')[0] if pd.notnull(x) else x)
    data['ATCD_loc_3'] = data['ATCD_loc_3'].apply(pd.to_numeric)
    # On enlève la colonne ATCD_histo car elle est trop complexe à traiter
    data = data.drop(columns=['ATCD_histo'])
    data['dose_fraction'] = data['dose_fraction'].apply(pd.to_numeric)
    data['min_PTV'] = data['min_PTV'].apply(pd.to_numeric)
    data['mean_PTV'] = data['mean_PTV'].apply(pd.to_numeric)
    data['max_PTV'] = data['max_PTV'].apply(pd.to_numeric)

    # We remove the columns that are not useful anymore
    data = data.drop(columns=['date_TDM', 'last_news'])
    
    # We remove date_fin
    data = data.drop(columns=['date_fin'])

    ## We save the pre-processed dataset in the folder
    data.to_csv(output_file, index=False)

    return None


if __name__ == "__main__":
    main()