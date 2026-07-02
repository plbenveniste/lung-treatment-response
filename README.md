# Lung Treatment response

(TODO: ADD icon of publication)

(TODO: ADD publication title)

Machine learning models to investigate lung cancer response after SBRT (radiotherapy treatment). 

We use clinical, dosimetrc and radiomics data to predict overal survival of patients with either primary or secondary lung tumors.

(TODO: ADD citation here)

## Installation

Create a new environment, activate it and install the requirements

```console
conda create -n venv_lung_response python=3.10
conda activate venv_lung_response
pip install -r requirements.txt
``` 

## Code

The code is divided in two folders: 
- `data`: here we perform data preprocessing, dataset merging and dataset statistics.
- `model_training`: here we perform model training and evaluation for OS on different data splits. 

## How to use it?

After doing the steps in installation section (section 1) and downloading the model from the latest [release](https://github.com/plbenveniste/lung-treatment-response/releases), you can run an inference using the file [predict_OS_primitive.py](./predict_OS_primitive.py) or [predict_OS_secondary.py](./predict_OS_secondary.py).