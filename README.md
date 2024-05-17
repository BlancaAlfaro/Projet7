# Projet7
The aim of this project is to deploy a model predicting loan non-reinbursment risk.

## Data
Data comes from [Home Credit Default Risk competition on Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data). To run the code it must be downloaded and extracted into a `data/` folder on the working directory.

## Notebooks and scripts

### Notebooks
- [01-modelisation.ipynb](notebooks/01-modelisation.ipynb) contains model and feature selection. Model runs are logged with MLFlow.
- [02-check_data_drift.ipynb](notebooks/02-check_data_drift.ipynb) is used to generate evidently html reports to check for data drift.

### Utilitary scripts
- [app_utils.py](src/app_utils.py) contains the functions needed by the api and testing scripts
- [lightgbm_with_simple_features.py](src/lightgbm_with_simple_features.py) is a data cleaning and feature engineering kernel from Kaggle modified to fit the project.
- [model_pred.py](src/model_prep.py) contains the functions needed for the modelisation and model selection
- [model.py](src/model.py) trains the selected model and saves the fitted model and scaler to an AWS bucket to be used by the API.

### Scripts
- [app.py](app.py) is the API script, which dowloads the fitted model and scaler from AWS to make predictions based either on a client id (by downloading client data from AWS) or from inputed data.
- [streamlit_dashboard.py](streamlit_dashboard.py) is the script used to run an API interface. It allows to input either a client id or client data to get a prediction on loan reinbursment probability, and the factors on which the prediction is made.

### Tests
- [tests.py](tests/tests.py) is the scrip used to run unit tests on the API with unittest.

## Setup
To run the code, install packages with poetry by running :
```bash
poetry install
```

To generate requirements.txt :
```bash
poetry export --without-hashes -f requirements.txt --output requirements.txt
```