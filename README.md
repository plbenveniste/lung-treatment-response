# Lung Treatment response

Machine learning model to investigate lung cancer response after SBRT (radiotherapy treatment). 

We investigated the clinical and radiomics data regarding lung cancer response after SBRT for the following predictions: 
- survival
- local relapse
- remote relapse

We also investigated feature removal, data-preprocessing and prediction timeframe. 

Authors: Camille Invernizzi, Pierre-Louis Benveniste

## Instructions to install everything

Create a new environment

```console
conda create -n venv_lung_response python=3.9
```

Activate it
```console
conda activate venv_lung_response
```

Then install all required libraries
```console
pip install -r requirements.txt
``` 

## Code in this repository

The code is divided in two folders: 
- data_preprocessing: here we investigate the data for data preprocessing, dataset merging, dataset statistics and feature elimination.
- model training: here we investigate the training of model prediction for survival, local relapse and final relapse. 

NB: the investigations are detailed in the issues. 