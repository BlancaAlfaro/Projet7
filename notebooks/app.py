import glob
import os
import sys

import joblib
from fastapi import FastAPI

app = FastAPI()
pipeline=joblib.load("output/model.pkl")
scaler=joblib.load("output/scaler.pkl")

@app.post('/model/predict')
def predict_class(X):
    threshold=0.55
    X_scaled=scaler.transform(X)
    proba=pipeline.predict_proba(X_scaled)
    prediction=proba<threshold
    output={'prediction':prediction}
    return output