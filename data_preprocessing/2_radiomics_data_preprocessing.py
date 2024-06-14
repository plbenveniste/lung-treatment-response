"""
This file deals with the pre-preprocessing of the individual csv files

Args:
    input: Path to the folder containing the csv files
    output: Path to the output folder

Returns:
    None

Example:
    python 

Author: Pierre-Louis Benveniste
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import pingouin as pg



def parse_args():
    """
    This function is used to parse the arguments given to the script
    """
    parser = argparse.ArgumentParser(description='Preprocess the data from the lung cancer response dataset')
    parser.add_argument('--input', type=str, help='Path to the folder containing the csv files')
    parser.add_argument('--output', type=str, help='Path to the output folder')
    return parser.parse_args()


def main():
    """
    This is the main function of the script. 
    It preprocesses the data from the lung cancer response of the multiple csv files
    """

    # We parse the arguments
    args = parse_args()
    input_path = args.input
    output_folder = args.output

    # Initialisation of the merged dataset
    merged_data = pd.DataFrame()

    # We list all the csv files contained in the input folder using rglob (recursive glob)
    csv_files = list(Path(input_path).rglob('*.csv'))
    
    # We iterate over all the csv files
    for csv_file in csv_files:
        # We load the dataset
        data = pd.read_csv(csv_file)

        # We select the following columns
        columns_to_keep = ['INFO_NameOfRoi', 'MORPHOLOGICAL_Volume(IBSI:RNU0)[mm3]','MORPHOLOGICAL_ApproximateVolume(IBSI:YEKZ)[mm3]','MORPHOLOGICAL_voxelsCounting(IBSI:No)[#vx]',
                           'MORPHOLOGICAL_SurfaceArea(IBSI:C0JK)[mm2]','MORPHOLOGICAL_SurfaceToVolumeRatio(IBSI:2PR5)[mm]',	'MORPHOLOGICAL_Compacity(IBSI:No)[]',
                           'MORPHOLOGICAL_Compactness1(IBSI:SKGS)[]', 'MORPHOLOGICAL_Compactness2(IBSI:BQWJ)[]', 'MORPHOLOGICAL_SphericalDisproportion(IBSI:KRCK)[]',
                           'MORPHOLOGICAL_Sphericity(IBSI:QCFX)[]',	'MORPHOLOGICAL_Asphericity(IBSI:25C7)[]', 'MORPHOLOGICAL_MaxIntensityCoor(IBSI:No)[vx]', 
                           'MORPHOLOGICAL_Centroid(IBSI:No)[vx]', 'MORPHOLOGICAL_WeightedCenterOfMass(IBSI:No)[vx]', 'MORPHOLOGICAL_CentreOfMassShift(IBSI:KLMA)[mm]',
                           'MORPHOLOGICAL_Maximum3DDiameter(IBSI:L0JK)[mm]', 'MORPHOLOGICAL_SphereDiameter(IBSI:No)[mm]', 'MORPHOLOGICAL_IntegratedIntensity(IBSI:99N0)[Intensity]',
                           'INTENSITY-BASED_MeanIntensity(IBSI:Q4LE)[HU]', 'INTENSITY-BASED_IntensityVariance(IBSI:ECT3)[HU]', 'INTENSITY-BASED_IntensitySkewness(IBSI:KE2A)[HU]',
                           'INTENSITY-BASED_IntensityKurtosis(IBSI:IPH6)[HU]', 'INTENSITY-BASED_MedianIntensity(IBSI:Y12H)[HU]', 'INTENSITY-BASED_MinimumIntensity(IBSI:1GSF)[HU]',
                           'INTENSITY-BASED_10thIntensityPercentile(IBSI:QG58)[HU]', 'INTENSITY-BASED_25thIntensityPercentile(IBSI:No)[HU]', 'INTENSITY-BASED_50thIntensityPercentile(IBSI:Y12H)[HU]',
                           'INTENSITY-BASED_75thIntensityPercentile(IBSI:No)[HU]', 'INTENSITY-BASED_90thIntensityPercentile(IBSI:8DWT)[HU]', 'INTENSITY-BASED_StandardDeviation(IBSI:No)[HU]', 
                           'INTENSITY-BASED_MaximumIntensity(IBSI:84IY)[HU]', 'INTENSITY-BASED_IntensityInterquartileRange(IBSI:SALO)[HU]', 'INTENSITY-BASED_IntensityRange(IBSI:2OJQ)[HU]', 
                           'INTENSITY-BASED_IntensityBasedMeanAbsoluteDeviation(IBSI:4FUA)[HU]', 'INTENSITY-BASED_IntensityBasedRobustMeanAbsoluteDeviation(IBSI:1128)[HU]', 
                           'INTENSITY-BASED_IntensityBasedMedianAbsoluteDeviation(IBSI:N72L)[HU]', 'INTENSITY-BASED_IntensityBasedCoefficientOfVariation(IBSI:7TET)[HU]', 
                           'INTENSITY-BASED_IntensityBasedQuartileCoefficientOfDispersion(IBSI:9S40)[HU]', 'INTENSITY-BASED_AreaUnderCurveCIVH(IBSI:No)[HU]', 'INTENSITY-BASED_IntensityBasedEnergy(IBSI:N8CA)[HU]',
                           'INTENSITY-BASED_RootMeanSquareIntensity(IBSI:5ZWQ)[HU]', 'INTENSITY-BASED_TotalLesionGlycolysis(IBSI:No)[HU]', 'INTENSITY-BASED_TotalCalciumScore[OnlyOnCT](IBSI:No)[]', 
                           'LOCAL_INTENSITY_BASED_GlobalIntensityPeak(IBSI:0F91)[Intensity]', 'LOCAL_INTENSITY_BASED_LocalIntensityPeak(IBSI:VJGA)[Intensity]', 
                           'INTENSITY-BASED-RIM_RIM-IntensityMin(IBSI:No)[Intensity]', 'INTENSITY-BASED-RIM_RIM-IntensityMean(IBSI:No)[Intensity]', 'INTENSITY-BASED-RIM_RIM-IntensityStd(IBSI:No)[Intensity]',
                           'INTENSITY-BASED-RIM_RIM-IntensityMax(IBSI:No)[Intensity]', 'INTENSITY-BASED-RIM_RIM-CountingVoxels(IBSI:No)[vx]', 'INTENSITY-BASED-RIM_RIM-ApproximateVolume(IBSI:No)[mL]', 
                           'INTENSITY-BASED-RIM_RIM-IntensitySum(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMean(IBSI:X6K6)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramVariance(IBSI:CH89)[Intensity]',
                           'INTENSITY-HISTOGRAM_IntensityHistogramSkewness(IBSI:88K1)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramKurtosis(IBSI:C3I7)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMedian(IBSI:WIFQ)[Intensity]',
                           'INTENSITY-HISTOGRAM_IntensityHistogramMinimumGreyLevel(IBSI:1PR8)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogram10thPercentile(IBSI:GPMT)[]', 'INTENSITY-HISTOGRAM_IntensityHistogram25thPercentile(IBSI:No)[]',
                           'INTENSITY-HISTOGRAM_IntensityHistogram50thPercentile(IBSI:No)[]', 'INTENSITY-HISTOGRAM_IntensityHistogram75thPercentile(IBSI:No)[]', 'INTENSITY-HISTOGRAM_IntensityHistogram90thPercentile(IBSI:OZ0C)[]', 
                           'INTENSITY-HISTOGRAM_IntensityHistogramStd(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMaximumGreyLevel(IBSI:3NCY)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMode(IBSI:AMMC)[Intensity]', 
                           'INTENSITY-HISTOGRAM_IntensityHistogramInterquartileRange(IBSI:WR0O)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramRange(IBSI:5Z3W)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMeanAbsoluteDeviation(IBSI:D2ZX)[Intensity]', 
                           'INTENSITY-HISTOGRAM_IntensityHistogramRobustMeanAbsoluteDeviation(IBSI:WRZB)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramMedianAbsoluteDeviation(IBSI:4RNL)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramCoefficientOfVariation(IBSI:CWYJ)[Intensity]',
                           'INTENSITY-HISTOGRAM_IntensityHistogramQuartileCoefficientOfDispersion(IBSI:SLWD)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog10(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM_IntensityHistogramEntropyLog2(IBSI:TLU2)[Intensity]',
                           'INTENSITY-HISTOGRAM_AreaUnderCurveCIVH(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM_Uniformity(IBSI:BJ5W)[Intensity]', 'INTENSITY-HISTOGRAM_RootMeanSquare(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM_MaximumHistogramGradient(IBSI:12CE)[Intensity]',
                           'INTENSITY-HISTOGRAM_MaximumHistogramGradientGreyLevel(IBSI:8E6O)[Intensity]', 'INTENSITY-HISTOGRAM_MinimumHistogramGradient(IBSI:VQB3)[Intensity]', 'INTENSITY-HISTOGRAM_MinimumHistogramGradientGreyLevel(IBSI:RHQZ)[Intensity]',
                           'LOCAL_INTENSITY_HISTOGRAM_GlobalIntensityPeak(IBSI:No)[Intensity]', 'LOCAL_INTENSITY_HISTOGRAM_LocalIntensityPeak(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMin(IBSI:No)[Intensity]',
                           'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMean(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensityStd(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMax(IBSI:No)[Intensity]',
                           'INTENSITY-HISTOGRAM-RIM_RIM-CountingVoxels(IBSI:No)[vx]', 'INTENSITY-HISTOGRAM-RIM_RIM-ApproximateVolume(IBSI:No)[mL]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensitySum(IBSI:No)[Intensity]', 'GLCM_JointMaximum(IBSI:GYBY)', 
                           'GLCM_JointAverage(IBSI:60VM)', 'GLCM_JointVariance(IBSI:UR99)', 'GLCM_JointEntropyLog2(IBSI:TU9B)', 'GLCM_JointEntropyLog10(IBSI:No)', 'GLCM_DifferenceAverage(IBSI:TF7R)', 'GLCM_DifferenceVariance(IBSI:D3YU)', 'GLCM_DifferenceEntropy(IBSI:NTRS)',
                           'GLCM_SumAverage(IBSI:ZGXS)', 'GLCM_SumVariance(IBSI:OEEB)', 'GLCM_SumEntropy(IBSI:P6QZ)', 'GLCM_AngularSecondMoment(IBSI:8ZQL)', 'GLCM_Contrast(IBSI:ACUI)', 'GLCM_Dissimilarity(IBSI:8S9J)', 'GLCM_InverseDifference(IBSI:IB1Z)',
                           'GLCM_NormalisedInverseDifference(IBSI:NDRX)', 'GLCM_InverseDifferenceMoment(IBSI:WF0Z)', 'GLCM_NormalisedInverseDifferenceMoment(IBSI:1QCO)', 'GLCM_InverseVariance(IBSI:E8JP)', 'GLCM_Correlation(IBSI:NI2N)','GLCM_Autocorrelation(IBSI:QWB0)', 'GLCM_ClusterTendency(IBSI:DG8W)',
                           'GLCM_ClusterShade(IBSI:7NFM)', 'GLCM_ClusterProminence(IBSI:AE86)', 'GLRLM_ShortRunsEmphasis(IBSI:22OV)', 'GLRLM_LongRunsEmphasis(IBSI:W4KF)', 'GLRLM_LowGreyLevelRunEmphasis(IBSI:V3SW)', 'GLRLM_HighGreyLevelRunEmphasis(IBSI:G3QZ)',
                           'GLRLM_ShortRunLowGreyLevelEmphasis(IBSI:HTZT)', 'GLRLM_ShortRunHighGreyLevelEmphasis(IBSI:GD3A)', 'GLRLM_LongRunLowGreyLevelEmphasis(IBSI:IVPO)', 'GLRLM_LongRunHighGreyLevelEmphasis(IBSI:3KUM)', 'GLRLM_GreyLevelNonUniformity(IBSI:R5YN)',
                           'GLRLM_RunLengthNonUniformity(IBSI:W92Y)', 'GLRLM_RunPercentage(IBSI:9ZK5)', 'NGTDM_Coarseness(IBSI:QCDE)', 'NGTDM_Contrast(IBSI:65HE)', 'NGTDM_Busyness(IBSI:NQ30)', 'NGTDM_Complexity(IBSI:HDEZ)', 'NGTDM_Strength(IBSI:1X9X)',
                           'GLSZM_SmallZoneEmphasis(IBSI:5QRC)', 'GLSZM_LargeZoneEmphasis(IBSI:48P8)', 'GLSZM_LowGrayLevelZoneEmphasis(IBSI:XMSY)', 'GLSZM_HighGrayLevelZoneEmphasis(IBSI:5GN9)', 'GLSZM_SmallZoneLowGreyLevelEmphasis(IBSI:5RAI)', 'GLSZM_SmallZoneHighGreyLevelEmphasis(IBSI:HW1V)',
                           'GLSZM_LargeZoneLowGreyLevelEmphasis(IBSI:YH51)', 'GLSZM_LargeZoneHighGreyLevelEmphasis(IBSI:J17V)', 'GLSZM_GreyLevelNonUniformity(IBSI:JNSA)', 'GLSZM_NormalisedGreyLevelNonUniformity(IBSI:Y1RO)', 'GLSZM_ZoneSizeNonUniformity(IBSI:4JP3)', 
                           'GLSZM_NormalisedZoneSizeNonUniformity(IBSI:VB3A)', 'GLSZM_ZonePercentage(IBSI:P30P)', 'GLSZM_GreyLevelVariance(IBSI:BYLV)', 'GLSZM_ZoneSizeVariance(IBSI:3NSA)', 'GLSZM_ZoneSizeEntropy(IBSI:GU8N)'
        ]
        data = data[columns_to_keep]

        # Columns removed because they contain non numeric values : MORPHOLOGICAL_MaxIntensityCoor(IBSI:No)[vx], MORPHOLOGICAL_Centroid(IBSI:No)[vx], MORPHOLOGICAL_WeightedCenterOfMass(IBSI:No)[vx]
        columns_to_remove = ['MORPHOLOGICAL_MaxIntensityCoor(IBSI:No)[vx]', 'MORPHOLOGICAL_Centroid(IBSI:No)[vx]', 'MORPHOLOGICAL_WeightedCenterOfMass(IBSI:No)[vx]', 'INTENSITY-BASED-RIM_RIM-IntensityMin(IBSI:No)[Intensity]', 
                             'INTENSITY-BASED-RIM_RIM-IntensityMean(IBSI:No)[Intensity]', 'INTENSITY-BASED-RIM_RIM-IntensityStd(IBSI:No)[Intensity]', 'INTENSITY-BASED-RIM_RIM-IntensityMax(IBSI:No)[Intensity]', 
                             'INTENSITY-BASED-RIM_RIM-CountingVoxels(IBSI:No)[vx]', 'INTENSITY-BASED-RIM_RIM-ApproximateVolume(IBSI:No)[mL]','INTENSITY-BASED-RIM_RIM-IntensitySum(IBSI:No)[Intensity]', 
                             'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMin(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMean(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-IntensityStd(IBSI:No)[Intensity]', 
                             'INTENSITY-HISTOGRAM-RIM_RIM-IntensityMax(IBSI:No)[Intensity]', 'INTENSITY-HISTOGRAM-RIM_RIM-CountingVoxels(IBSI:No)[vx]', 'INTENSITY-HISTOGRAM-RIM_RIM-ApproximateVolume(IBSI:No)[mL]',
                             'INTENSITY-HISTOGRAM-RIM_RIM-IntensitySum(IBSI:No)[Intensity]']

        data = data.drop(columns=columns_to_remove)
        
        # For each column we check that it only contains numeric values
        for column in data.columns:
            unique_values = data[column].unique()
            # if unique_values contain str values print column name
            if any(isinstance(value, str) for value in unique_values) and column!='INFO_NameOfRoi':
                print(column)

        # We add the subject name : 
        data['file_name'] = csv_file.name
        data['subject_id'] = csv_file.name.split(' ')[1].split('.')[0]

        # We create an empty colum which is the subject and the nodule but not yet assigned
        data['nodule'] = None
        data['subject_nodule'] = None

        # we reset the index and then print it
        data = data.reset_index(drop=True)

        # We build the name of the nodule
        # If there is only one line, we take directly the line and put it in the output_data
        if len(data['INFO_NameOfRoi'].unique())==1:
            # Easy in this case because only one line
            # If the name is 'Test_Camille' we consider that the nodule is GTV
            if data['INFO_NameOfRoi'].unique()[0]=="'Test_Camille":
                data['nodule'] = 'GTV'
                data['subject_nodule'] = data['subject_id'] + '_GTV'
            # Then we can directly add the line in the output dataset
       
        # If there are only two lines, we consider that they concern the same patient and the same nodule
        elif len(data['INFO_NameOfRoi'].unique())==2:

            # for each line we add the subject_nodule
            for index, row in data.iterrows():
                nodule = ''
                if "Test_Camille" in row['INFO_NameOfRoi']:
                    nodule = 'GTV'
                elif "GTV" in row['INFO_NameOfRoi']:
                    nodule = 'GTV'
                else: 
                    print("Problem")
                data.at[index, 'subject_nodule'] = row['subject_id'] + '_' + nodule
                data.at[index, 'nodule'] = nodule
        
        # If there are more than two lines, we consider that they concern the same patient but different nodules
        else:
            
            # We iterate over the lines and add the subject_nodule
            for index, row in data.iterrows():
                nodule = ''
                if "LSD_1" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSD_1'
                elif "LSD_2" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSD_2'
                elif "LSD_a" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSD_a'
                elif "LSD_p" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSD_p'
                elif "LID_m" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LID_m'
                elif "LID_p" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LID_p'
                elif "LSG" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSG'
                elif "LSD" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LSD'
                elif "LS" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LS'
                elif "apexG" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_apexG'
                elif "LIG" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LIG'
                elif "LID" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LID'
                
                elif "LM" in row['INFO_NameOfRoi']:
                    nodule = 'GTV_LM'
                else: 
                    print("Problem")
                # We change only the corresponding line
                data.at[index, 'nodule'] = nodule
                data.at[index, 'subject_nodule'] = row['subject_id'] + '_' + nodule
        # we then merge the data with the merged_data        
        merged_data = pd.concat([merged_data, data])
    
    # We check if some columns only contain nan values
    for column in merged_data.columns:
        if merged_data[column].isnull().all():
            # We remove the column
            merged_data = merged_data.drop(columns=column)
            print("Column {} has been removed because it contained only nan".format(column))
        # if more than 80% of the values are nan we remove the column
        elif merged_data[column].isnull().sum() > 0.5*len(merged_data):
            merged_data = merged_data.drop(columns=column)
            print("Column {} has been removed because it contained more than 50% of nan".format(column))
    
    # We compute the ICC for each feature and keep only the features which have an ICC>0.8
    # We first define the rater column : is rater 1 if "Camille" in INFO_NameOfRoi and 0 otherwise
    merged_data['rater'] = 0
    merged_data.loc[merged_data['INFO_NameOfRoi'].str.contains('Camille'), 'rater'] = 1
    # We iterate over the features except a few ones
    columns_to_iterate = merged_data.columns.drop('INFO_NameOfRoi').drop('file_name').drop('subject_id').drop('nodule').drop('subject_nodule').drop('rater')
    columns_to_remove_after_icc = []
    associated_column_icc_score = []
    for feature in columns_to_iterate:
        icc = pg.intraclass_corr(data=merged_data, targets='subject_nodule', raters='rater', ratings=feature, nan_policy='omit')
        # Because each subject had been rated by all the raters, we will use ICC3
        icc = icc.set_index('Type')
        if icc.loc['ICC3', 'ICC']<0.8:
            columns_to_remove_after_icc.append(feature)
            associated_column_icc_score.append(icc.loc['ICC3', 'ICC'])

    # We remove the columns which have an ICC<0.8
    merged_data = merged_data.drop(columns=columns_to_remove_after_icc)
    # We also remov the rater column
    merged_data = merged_data.drop(columns='rater')
    
    # We now create the output csv file which basically takes the average of the features for each nodule
    output_data = pd.DataFrame()
    for subject_nodule in merged_data['subject_nodule'].unique():
        # We select the data of the subject_nodule
        data = merged_data[merged_data['subject_nodule']==subject_nodule]
        # We create a df which is the mean of the data (but only the numeric columns)
        columns_to_average = data.select_dtypes(include=[np.number]).columns
        mean_data = data[columns_to_average].mean()
        mean_data['subject_nodule'] = subject_nodule
        mean_data['subject_id'] = data['subject_id'].iloc[0]
        mean_data['nodule'] = data['nodule'].iloc[0]
        mean_data['file_name'] = data['file_name'].iloc[0]

        # We add the mean_data to the output_data
        output_data = pd.concat([output_data, mean_data], axis=1)

    # We save the output_data
    output_data = output_data.T
    output_data.to_csv(os.path.join(output_folder, 'radiomics_data_preprocessed.csv'))            

    return None


if __name__ == '__main__':
    main()