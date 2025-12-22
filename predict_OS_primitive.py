"""
This script performs OS prediction for patients with primary lung tumors.
It takes as input 20 features (radiomics, clinical and dosimetric) and outputs the predicted survival probability.

Inputs:
    --model_path: path to the trained model
    --max_PTV
    --INTENSITY-HISTOGRAM_MinimumHistogramGradientGreyLevel
    --NGTDM_Complexity
    --LOCAL_INTENSITY_BASED_GlobalIntensityPeak
    --MORPHOLOGICAL_Maximum3DDiameter
    --score_charlson
    --min_PTV
    --vol_ITV
    --mean_PTV
    --BMI
    --INTENSITY-BASED_AreaUnderCurveCIVH
    --GLSZM_SmallZoneHighGreyLevelEmphasis
    --MORPHOLOGICAL_Volume
    --INTENSITY-HISTOGRAM_RootMeanSquare
    --INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog10
    --NGTDM_Coarseness
    --INTENSITY-HISTOGRAM_MaximumHistogramGradient
    --INTENSITY-BASED_TotalLesionGlycolysis
    --GLCM_JointEntropyLog2
    --INTENSITY-BASED_IntensityRange

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
    parser.add_argument('--max_PTV', type=float, required=True)
    parser.add_argument('--INTENSITY-HISTOGRAM_MinimumHistogramGradientGreyLevel', type=float, required=True)
    parser.add_argument('--NGTDM_Complexity', type=float, required=True)
    parser.add_argument('--LOCAL_INTENSITY_BASED_GlobalIntensityPeak', type=float, required=True)
    parser.add_argument('--MORPHOLOGICAL_Maximum3DDiameter', type=float, required=True)
    parser.add_argument('--score_charlson', type=float, required=True)
    parser.add_argument('--min_PTV', type=float, required=True)
    parser.add_argument('--vol_ITV', type=float, required=True)
    parser.add_argument('--mean_PTV', type=float, required=True)
    parser.add_argument('--BMI', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_AreaUnderCurveCIVH', type=float, required=True)
    parser.add_argument('--GLSZM_SmallZoneHighGreyLevelEmphasis', type=float, required=True)
    parser.add_argument('--MORPHOLOGICAL_Volume', type=float, required=True)
    parser.add_argument('--INTENSITY-HISTOGRAM_RootMeanSquare', type=float, required=True)
    parser.add_argument('--INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog10', type=float, required=True)
    parser.add_argument('--NGTDM_Coarseness', type=float, required=True)
    parser.add_argument('--INTENSITY-HISTOGRAM_MaximumHistogramGradient', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_TotalLesionGlycolysis', type=float, required=True)
    parser.add_argument('--GLCM_JointEntropyLog2', type=float, required=True)
    parser.add_argument('--INTENSITY-BASED_IntensityRange', type=float, required=True)
    return parser.parse_args()


def predict_OS_primitive(model, input_data):
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
    input_data = [args.max_PTV, args.INTENSITY_HISTOGRAM_MinimumHistogramGradientGreyLevel,
                  args.NGTDM_Complexity, args.LOCAL_INTENSITY_BASED_GlobalIntensityPeak,
                  args.MORPHOLOGICAL_Maximum3DDiameter, args.score_charlson,
                  args.min_PTV, args.vol_ITV, args.mean_PTV, args.BMI,
                  args.INTENSITY_BASED_AreaUnderCurveCIVH, args.GLSZM_SmallZoneHighGreyLevelEmphasis,
                  args.MORPHOLOGICAL_Volume, args.INTENSITY_HISTOGRAM_RootMeanSquare,
                  args.INTENSITY_HISTOGRAM_IntensityHistogramEntropyLog10, args.NGTDM_Coarseness,
                  args.INTENSITY_HISTOGRAM_MaximumHistogramGradient, args.INTENSITY_BASED_TotalLesionGlycolysis,
                  args.GLCM_JointEntropyLog2, args.INTENSITY_BASED_IntensityRange]

    # predict the 3-year survival
    predicted_survival = predict_OS_primitive(args.model_path, np.array(input_data).reshape(1, -1))


    print(f'The predicted survival probability is: {1 - predicted_survival[0]}')
    print(f'The predicted death probability is: {predicted_survival[0]}')

    return None


if __name__ == '__main__':
    main()