"""
This code loads the model and the dataset and plots the feature importance.
It also uses the log file to get which features were used in the model.

Input:
    -model-path: path to the trained model
    -data-path: path to the dataset
    -log-path: path to the log file
    -output: path to output folder

Output:
    None

Example usage:
    python plot_feature_importance_primitive.py --model_path model.pkl --data_path data.csv --log_path training.log --output output_folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import shap
from loguru import logger
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Plot feature importance from a trained model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--log-path", type=str, required=True, help="Path to the log file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path
    log_path = args.log_path
    output_folder = args.output
    
    nb_years = 100  # Number of years to consider for survival analysis

    # If output folder does not exist, create it
    os.makedirs(output_folder, exist_ok=True)

    # Clear the log file
    log_file = os.path.join(output_folder, f'log_primitive_shap.txt')
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    ##########################################
    ###### Preprocessing the data ############
    ##########################################

    # Load the dataset
    data = pd.read_csv(data_path)

    # We remove data which is note useful to make the averaging easier
    data = data.drop(columns=['DDD', 'cause_DC', 'Date_R_PTV', 'Date_R_homo', 'Date_R_med', 'Date_R_contro', 'Date_R_horspoum', 'Reponse', 'rechute_PTV', 'rechute_homo',
                           'rechute_med', 'rechute_contro', 'rechute_horspoum', 'delai_fin_rechutePTV', 'delai_fin_rechuteHomo','delai_fin_rechuteMed',
                           'delai_fin_rechuteContro', 'delai_fin_rechuteHorspoum','subject_nodule', 'nodule', 'follow_up' ])

    # For dosimetric data, we sum the features together by subject (so that if there is two nodules, the dosimetric data reflects the sum of the two doses)
    data_dosi = data[['subject_id', 'dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']] #, 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV']]
    # We group the data by subject and sum the dosi features
    data_dosi = data_dosi.groupby('subject_id').sum().reset_index()

    # For the rest of the data, we average
    data_rest = data.drop(columns=['dose_tot', 'etalement', 'vol_GTV', 'vol_PTV', 'vol_ITV', 'couv_PTV', 'BED_10']) #, 'dose_fraction', 'min_PTV', 'mean_PTV', 'max_PTV'])
    data_rest = data_rest.groupby('subject_id').mean().reset_index()

    # We concatenate the dosimetric and rest of the data
    data_grouped = pd.merge(data_dosi, data_rest, on='subject_id', how='outer')

    # We keep primitive patients
    data_primitive = data_grouped[data_grouped['primitif'] == 1]

    # Split into features and target
    y = data_primitive[['DC']]
    x = data_primitive.drop(columns=['DC', 'delai_fin_DC', 'subject_id'])
    
    # In this case, because we are only interested in the prediction of survival, we extract only the 'DC'
    y = y[['DC']]
    # We replace all nan values by 0 in 'DC'
    y.fillna(0, inplace=True)

    ###################################################
    ### Load the selected features from the log #######
    ###################################################

    # Now we load the log file to get the features used in the model
    with open(log_path, 'r') as f:
        log_lines = f.readlines()
    # For element in log_lines, we remove the line if it does not contain "Features selected for the final model"
    log_lines = [line for line in log_lines if "Features selected for the final model" in line]
    features_selected = log_lines[-1].split("[")[1].split("]")[0]
    # It looks like this: 'BMI', 'couv_PTV', 'GLRLM_GreyLevelNonUniformity', 'INTENSITY-HISTOGRAM_MaximumHistogramGradient', 'INTENSITY-BASED_TotalLesionGlycolysis', 'GLSZM_SmallZoneHighGreyLevelEmphasis', 'vol_ITV', 'score_charlson', 'INTENSITY-BASED_25thIntensityPercentile', 'GLSZM_SmallZoneLowGreyLevelEmphasis', 'dose_tot', 'tabac_sevre', 'INTENSITY-BASED_IntensitySkewness', 'tabac_PA', 'poids', 'INTENSITY-BASED_MaximumIntensity', 'MORPHOLOGICAL_SurfaceToVolumeRatio', 'histo', 'LOCAL_INTENSITY_BASED_GlobalIntensityPeak', 'GLCM_JointEntropyLog2'
    ## Convert it to a list of strings
    features_selected = [feat.strip().strip("'") for feat in features_selected.split(",")]
    logger.info(f"Features selected for the final model: {features_selected}")

    # Now we select only these features in X
    X = x[features_selected]

    # For each feature, we look at the unique values:
    for feature in X.columns:
        unique_values = np.unique(X[feature])
        # If feature has nan values, we print the numbe of nan values
        num_nan = X[feature].isna().sum()
        if num_nan > 0:
            logger.info(f"Feature {feature} has {num_nan} nan values.")
    
    #############################################################
    ###### Load the model and plot feature importance ###########
    #############################################################

    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Fit the model to the data 
    model.fit(X, y)
    logger.info("Model fitted to the data.")

    # Calculate SHAP values
    shap.initjs()
    explainer = shap.Explainer(model, seed=42)
    importances = np.abs(explainer.shap_values(X)).mean(axis=0)
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    for i, feature in enumerate(feature_importances_df['Feature']):
        # Print feature and importance
        logger.info(f"{i+1}: {feature_importances_df['Importance'].iloc[i]:.4f}: {feature}")

    # We plot the Shap plot of the final model
    shap_values = explainer(X)
    feature_importances_df = feature_importances_df.reset_index(drop=True)
    feat_renaming_dict = dict(zip(feature_importances_df['Feature'], [f'Feat {i+1}' for i in range(X.shape[1])]))
    X_renamed = X.rename(columns=feat_renaming_dict)
    logger.info(f"Feature renaming dictionary: {feat_renaming_dict}")
    shap_values = explainer(X_renamed)
    plt.figure()
    shap.summary_plot(shap_values, X_renamed, show=False)
    plt.title('SHAP summary plot for the final model (features renamed)')
    plt.savefig(os.path.join(output_folder, 'shap_summary_final_model_renamed.png'))
    plt.close()

    # Finally we plot the calibration curves of the final model
    y_prob = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration curve')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'calibration_curve_final_model.png'))
    plt.close()

    
    return None




if __name__ == "__main__":
    main()