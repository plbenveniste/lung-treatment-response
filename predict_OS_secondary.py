"""
This script performs OS prediction for patients with secondary lung tumors.
It takes as input 20 features (radiomics, clinical and dosimetric) and outputs the predicted survival probability.

Inputs:
    --model_path: path to the trained model
    --GLRLM_GreyLevelNonUniformity
    --tabac
    --INTENSITY-BASED_MeanIntensity
    --INTENSITY-BASED_IntensitySkewness
    --max_PTV
    --mean_PTV
    --INTENSITY-BASED_IntensityRange
    --etalement
    --age
    --tabac_PA
    --poids
    --MORPHOLOGICAL_Maximum3DDiameter
    --vol_PTV
    --vol_GTV
    --ATCD_loc_1
    --INTENSITY-BASED_25thIntensityPercentile
    --GLCM_JointMaximum
    --INTENSITY-BASED_IntensityBasedEnergy
    --GLRLM_RunLengthNonUniformity
    --GLRLM_ShortRunsEmphasis
Author: Pierre-Louis Benveniste
"""

import argparse
import pickle
import numpy as np


def parse_args():
    """
    Parse the arguments of the script. The arguments are the path to the trained model and the 20 features.

    Returns:
        args: the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Predict survival for primary lung tumors patients')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--GLRLM_GreyLevelNonUniformity', type=float, required=True)
    parser.add_argument('--tabac', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_MeanIntensity', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_IntensitySkewness', type=float, required=True)
    parser.add_argument('--max_PTV', type=float, required=True)
    parser.add_argument('--mean_PTV', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_IntensityRange', type=float, required=True)
    parser.add_argument('--etalement', type=float, required=True)
    parser.add_argument('--age', type=float, required=True)
    parser.add_argument('--tabac_PA', type=float, required=True)
    parser.add_argument('--poids', type=float, required=True)
    parser.add_argument('--MORPHOLOGICAL_Maximum3DDiameter', type=float, required=True)
    parser.add_argument('--vol_PTV', type=float, required=True)
    parser.add_argument('--vol_GTV', type=float, required=True)
    parser.add_argument('--ATCD_loc_1', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_25thIntensityPercentile', type=float, required=True)
    parser.add_argument('--GLCM_JointMaximum', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_IntensityBasedEnergy', type=float, required=True)
    parser.add_argument('--GLRLM_RunLengthNonUniformity', type=float, required=True)
    parser.add_argument('--GLRLM_ShortRunsEmphasis', type=float, required=True)
    return parser.parse_args()


def predict_OS_secondary(model, input_data):
    """
    This function performs inference on the survival prediction model.

    Args:
        model: the trained model
        input_data: the 20 features
    
    Returns:
        predicted_survival: the predicted survival probability
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
    input_data = [args.GLRLM_GreyLevelNonUniformity, args.tabac, args.INTENSITY_BASED_MeanIntensity,
                  args.INTENSITY_BASED_IntensitySkewness, args.max_PTV, args.mean_PTV,
                  args.INTENSITY_BASED_IntensityRange, args.etalement, args.age,
                  args.tabac_PA, args.poids, args.MORPHOLOGICAL_Maximum3DDiameter,
                  args.vol_PTV, args.vol_GTV, args.ATCD_loc_1,
                  args.INTENSITY_BASED_25thIntensityPercentile, args.GLCM_JointMaximum,
                  args.INTENSITY_BASED_IntensityBasedEnergy, args.GLRLM_RunLengthNonUniformity,
                  args.GLRLM_ShortRunsEmphasis]

    # predict the 3-year survival
    predicted_survival = predict_OS_secondary(args.model_path, np.array(input_data).reshape(1, -1))


    print(f'The predicted survival probability is: {1 - predicted_survival[0]}')
    print(f'The predicted death probability is: {predicted_survival[0]}')

    return None


if __name__ == '__main__':
    main()