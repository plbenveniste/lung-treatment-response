"""
This script is used to merge the preprocessed dataset built using the clinical data and the preprocessed dataset built using the radiomics data.

Args:
    --clini: path to the csv file containing the preprocessed clinical data
    --radio: path to the csv file containing the preprocessed clinical data
    --output: path to the folder where the merged dataset will be saved

Returns:
    None

Example run:
    python mergin_preprocessed_datasets.py --clinical-data /path/to/clinical.csv --radiomics-data /path/to/radiomics.csv

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    parser = argparse.ArgumentParser(description='Preprocess the data from the lung cancer response dataset')
    parser.add_argument('--clini', type=str, help='Path to the csv file containing the preprocessed clinical data')
    parser.add_argument('--radio', type=str, help='Path to the csv file containing the preprocessed clinical data')
    parser.add_argument('--output', type=str, help='Path to the folder where the merged dataset will be saved')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It does the merge between the two datasets.
    """

    # We parse the arguments
    args = parse_args()
    clini_data = args.clini
    radio_data = args.radio

    # Load the datasets
    clini_data = pd.read_csv(clini_data)
    radio_data = pd.read_csv(radio_data)

    # we initialize the merged dataset
    merged_data = pd.DataFrame()

    # iterate over the lines in the clinical data
    for i, row in clini_data.iterrows():
        # Find the corresponding line in the radiomics data using .loc
        found = False
        patient_id = row['num\n_patient']
        matching_radio_row = None
        if 'H042' in patient_id:
            patient_id = patient_id.replace('H042','H042_LSD')
        for i, radio_row in radio_data.iterrows():
            if radio_row['subject_nodule'].replace('_GTV','') == patient_id:
                found = True
                matching_radio_row = radio_row
        # we add the line to the merged dataset
        if found:
            # we create a line containing the data from the clinical dataset and the radiomics dataset
            merged_line = pd.concat([row, matching_radio_row])
            # we add the line to the merged dataset
            merged_data = merged_data._append(merged_line, ignore_index=True)

        if not found:
            print("Problem with patient {}".format(patient_id))

    # We save the merged dataset
    output_folder = args.output
    merged_data.to_csv(os.path.join(output_folder, 'merged_data.csv'), index=False)
    
    return None


if __name__ == '__main__':
    main()