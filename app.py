import io
import os

#import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI

#Run `uvicorn app:app --reload` in a terminal to test

app = FastAPI()

os.makedirs("output",exist_ok=True)
#Download model and scaler from AWS
s3_client = boto3.client('s3')
s3_client.download_file("modelandscaler",'model.pkl',"output/model.pkl")
s3_client.download_file("modelandscaler","scaler.pkl", "output/scaler.pkl")

#load downloaded local files
pipeline=joblib.load("output/model.pkl")
scaler=joblib.load("output/scaler.pkl")

#Download client data (here X_test is used) from AWS
s3_client.download_file("clientsdataxtest",'X_test.csv',"output/X_test.csv")

#@app.post('/model/predict_from_data')
#def predict_class(X):
#    threshold=0.55
#    X_scaled=scaler.transform(X)
#    proba=pipeline.predict_proba(X_scaled)
#    prediction=proba<threshold
#    output={'prediction':prediction}
#    return output

@app.get('/data/get_client_data')
def load_data_for_client(SK_ID_CURR):
    X=get_data_for_client(SK_ID_CURR)
    output={'client_data':X.to_dict()}
    return output

@app.post('/model/predict_from_SK_ID_CURR')
def predict_class_from_id(SK_ID_CURR):
    X=get_data_for_client(SK_ID_CURR)
    X=X.set_index('SK_ID_CURR')
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
    threshold=0.55
    X_scaled=pd.DataFrame(scaler.transform(X))
    X_scaled.columns=X.columns
    X_scaled.index=X.index
    proba=pipeline.predict_proba(X_scaled)
    print('proba',proba)
    prediction=proba[0][0]<threshold
    print(prediction)
    str_prediction_dict={True: 'Good chance of reimbursing', False : 'Low chance of reimbursing'}
    #Put image of shap waterfall to AWS
    explain_prediction(X_scaled,pipeline)
    #Return output
    output={'prediction':str_prediction_dict[prediction],'probability_of_not_reinbursing':str(np.round(proba[0][0],2))}
    print(output)
    return output

def get_data_for_client(SK_ID_CURR):
    if isinstance(SK_ID_CURR,str):
        SK_ID_CURR=int(SK_ID_CURR)
    data=pd.read_csv('output/X_test.csv',index_col=0)
    print('Data loaded')
    X=data[data['SK_ID_CURR']==SK_ID_CURR]
    return X


def explain_prediction(X_scaled,pipeline):
    explainer = shap.Explainer(pipeline['classification'])
    shap.waterfall_plot(explainer(X_scaled)[0],show=False)#TODO Fix the fact that the image fx doesnt correspond to proba[0][0]
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
