import io
import json
import os

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

#Run `uvicorn app:app --reload` in a terminal to test

app = FastAPI()

os.makedirs("output",exist_ok=True)
#Download model and scaler from AWS
s3_client = boto3.client('s3')
s3_client.download_file("modelandscaler",'model.pkl',"output/model.pkl")
s3_client.download_file("modelandscaler","scaler.pkl", "output/scaler.pkl")

#Load downloaded local files
pipeline=joblib.load("output/model.pkl")
scaler=joblib.load("output/scaler.pkl")

#Download client data (here X_test is used) from AWS
s3_client.download_file("clientsdataxtest",'X_test.csv',"output/X_test.csv")
X_test=pd.read_csv('output/X_test.csv',index_col=0)

class ClientData(BaseModel):
    data : dict

@app.post('/model/predict_from_data')
def predict_class(X :ClientData):
    X=pd.DataFrame(X.data,index=['temp'])
    output=make_prediction_from_data(X,threshold=0.55)
    return output

@app.get('/data/get_client_data')
def load_data_for_client(SK_ID_CURR):
    X=get_data_for_client(SK_ID_CURR)
    output={'client_data':X.to_dict()}
    return output

@app.post('/model/predict_from_SK_ID_CURR')
def predict_class_from_id(SK_ID_CURR):
    X=get_data_for_client(SK_ID_CURR)
    X=X.set_index('SK_ID_CURR')
    output=make_prediction_from_data(X,threshold=0.65)
    return output

def get_data_for_client(SK_ID_CURR):
    """Loads clients data from csv file and returns data for a given client.

    Parameters
    ----------
    SK_ID_CURR : str or int
        id of the client

    Returns
    -------
    pd.DataFrame
        Data for the given client to be used to make predictions.
    """
    if isinstance(SK_ID_CURR,str):
        SK_ID_CURR=int(SK_ID_CURR)
    data=pd.read_csv('output/X_test.csv',index_col=0)
    print('Data loaded')
    X=data[data['SK_ID_CURR']==SK_ID_CURR]
    return X


def explain_prediction(X_scaled,pipeline,X=None):
    """Generate shap plot to explain prediction for given value and put image of the plot to AWS.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Data for which the prediction is made
    pipeline : sklearn pipeline or equivalent
        fitted pipeline to make predictions with.
    X : pd.DataFrame
        unscaled data for lr explainer
    """
    if isinstance(pipeline['classification'],LogisticRegression):
        explainer=shap.LinearExplainer(pipeline['classification'],X)
    else :
        explainer = shap.Explainer(pipeline['classification'])
    shap.waterfall_plot(explainer(X_scaled)[0],show=False)
    fig=plt.gcf()
    #Transform to uploadable image
    img = io.BytesIO()
    fig.savefig(img, format="png",bbox_inches='tight')
    plt.clf()
    img.seek(0)
    image = img.read()
    img.flush()
    #Put to S3 bucket
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('predictionfigures')
    bucket.put_object(Body=image, ContentType='image/png',Key='explain_prediction_for_'+str(X_scaled.index.values[0]))

def make_prediction_from_data(X,threshold):
    """Compute loan reinbursment prediction from client data

    Parameters
    ----------
    X : pd.DataFrame
        client data
    threshold : float
        threshold to apply to reinbursment probability to determine if the load should be given or not.

    Returns
    -------
    dict
        Dictionnay containing the predicted probability and class.
    """
    features_to_keep=['NAME_INCOME_TYPE_Working',
    'HOUSETYPE_MODE_block of flats',
    'NAME_EDUCATION_TYPE_Higher education',
    'FLAG_OWN_CAR',
    'CNT_CHILDREN',
    'WALLSMATERIAL_MODE_Stone, brick',
    'REGION_RATING_CLIENT_W_CITY',
    'DAYS_REGISTRATION',
    'FLAG_PHONE',
    'REGION_RATING_CLIENT',
    'REGION_POPULATION_RELATIVE',
    'NAME_EDUCATION_TYPE_Secondary / secondary special',
    'NAME_INCOME_TYPE_Commercial associate',
    'NAME_INCOME_TYPE_Pensioner',
    'NAME_TYPE_SUITE_Unaccompanied',
    'WEEKDAY_APPR_PROCESS_START_TUESDAY',
    'REG_CITY_NOT_WORK_CITY']
    X=X[features_to_keep]
    X_scaled=pd.DataFrame(scaler.transform(X))
    X_scaled.columns=X.columns
    X_scaled.index=X.index
    proba=pipeline.predict_proba(X_scaled)
    print('proba',proba)
    prediction=proba[0][0]>threshold #class 0 : reinbursing / class 1 : not reimbursing
    print(prediction)
    str_prediction_dict={True: 'Good chance of reimbursing', False : 'Low chance of reimbursing'}
    #Put image of shap waterfall to AWS
    #explain_prediction(X_scaled,pipeline,X)
    #Return output
    output={'prediction':str_prediction_dict[prediction],'probability_of_reinbursing':str(np.round(proba[0][0],2))}
    return output