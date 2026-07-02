"""
This script performs inference on the 3-year survival prediction model trained in this repository.
It takes as input 14 features (radiomics, clinical and dosimetric) and outputs the predicted 3-year survival probability.

Inputs:
    --model_path: path to the trained model
    --sex: sex of the patient (0 for male, 1 for female)
    --BMI: the body mass index of the patient
    --score_charlson: the Charlson comorbidity index of the patient
    --smoke_cessation: the number of years since the patient stopped smoking
    --dose_tot: the total dose of radiation received by the patient
    --BED_10: the biologically effective dose of radiation received by the patient
    --MeanIntensity: the mean intensity of the tumor
    --IntensitySkewness: the skewness of the intensity of the tumor
    --IntensityKurtosis: the kurtosis of the intensity of the tumor
    --AreaUnderCurveCIVH: the area under the curve of the cumulative intensity-volume histogram
    --RootMeanSquareIntensity: the root mean square intensity of the tumor
    --IntensityHistogramMean: the mean of the intensity histogram of the tumor
    --IntensityHistogramVariance: the variance of the intensity histogram of the tumor
    --NGTDM_Strength: the strength of the neighborhood gray-tone difference matrix of the tumor

Outputs:
    --predicted_survival: the predicted 3-year survival probability of the patient

Author: Pierre-Louis Benveniste
"""

import argparse
import pickle
import numpy as np


def parse_args():
    """
    Parse the arguments of the script. The arguments are the path to the trained model and the 14 features.

    Returns:
        args: the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Predict 3-year survival')
    parser.add_argument('--model-path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--sex', type=int, help='Sex of the patient (0 for male, 1 for female)', required=True)
    parser.add_argument('--BMI', type=float, help='Body mass index of the patient', required=True)
    parser.add_argument('--score_charlson', type=float, help='Charlson comorbidity index of the patient', required=True)
    parser.add_argument('--smoke_cessation', type=float, help='Number of years since the patient stopped smoking', required=True)
    parser.add_argument('--dose_tot', type=float, help='Total dose of radiation received by the patient', required=True)
    parser.add_argument('--BED_10', type=float, help='Biologically effective dose of radiation received by the patient', required=True)
    parser.add_argument('--MeanIntensity', type=float, help='Mean intensity of the tumor', required=True)
    parser.add_argument('--IntensitySkewness', type=float, help='Skewness of the intensity of the tumor', required=True)
    parser.add_argument('--IntensityKurtosis', type=float, help='Kurtosis of the intensity of the tumor', required=True)
    parser.add_argument('--AreaUnderCurveCIVH', type=float, help='Area under the curve of the cumulative intensity-volume histogram', required=True)
    parser.add_argument('--RootMeanSquareIntensity', type=float, help='Root mean square intensity of the tumor', required=True)
    parser.add_argument('--IntensityHistogramMean', type=float, help='Mean of the intensity histogram of the tumor', required=True)
    parser.add_argument('--IntensityHistogramVariance', type=float, help='Variance of the intensity histogram of the tumor', required=True)
    parser.add_argument('--NGTDM_Strength', type=float, help='Strength of the neighborhood gray-tone difference matrix of the tumor', required=True)
    return parser.parse_args()


def predict_3year_survival(model, input_data):
    """
    This function performs inference on the 3-year survival prediction model.

    Args:
        model: the trained model
        input_data: the 14 features
    
    Returns:
        predicted_survival: the predicted 3-year survival probability
    """
    # Load the model
    with open(model, 'rb') as f:
        model = pickle.load(f)

    # Perform inference
    prediction = model.predict_proba(input_data)
    
    # Extract the predicted survival probability
    predicted_survival = prediction[:, 0]

    return predicted_survival 


def main():
    """
    This is the main function of the script.
    """
    # Parse the arguments
    args = parse_args()
    
    # Create the input data
    input_data = [args.sex, args.BMI, args.score_charlson, args.smoke_cessation, args.dose_tot, args.BED_10,
                  args.MeanIntensity, args.IntensitySkewness, args.IntensityKurtosis, args.AreaUnderCurveCIVH,
                  args.RootMeanSquareIntensity, args.IntensityHistogramMean, args.IntensityHistogramVariance,
                  args.NGTDM_Strength]

    # predict the 3-year survival
    predicted_survival = predict_3year_survival(args.model_path, np.array(input_data).reshape(1, -1))


    print(f'The predicted 3-year survival probability is: {1 - predicted_survival[0]}')
    print(f'The predicted 3-year death probability is: {predicted_survival[0]}')

    return None


if __name__ == '__main__':
    main()