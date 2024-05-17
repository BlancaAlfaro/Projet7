import glob
import os
import sys

import boto3
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join('..')))

from src.lightgbm_with_simple_features import main
from src.model_prep import get_initial_splits

os.makedirs("output",exist_ok=True)
files = glob.glob('output/*')
for f in files:
    os.remove(f)

#Load data and split into train/test
df=main(data_folder='data/')
kaggle_df,X_train,X_test,y_train,y_test=get_initial_splits(df)
#Filter features to keep
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
X_train=X_train[features_to_keep]
#Scale data
scaler=StandardScaler()
X_train_scaled=pd.DataFrame(scaler.fit_transform(X_train))
X_train_scaled.columns=X_train.columns
#Select model
model=XGBClassifier(reg_alpha=0.001,
                    colsample_bytree=0.3,
                    gamma=0.5,
                    reg_lambda= 0.001,
                    learning_rate=0.01,
                    max_depth= 4,
                    n_estimators=40,
                    subsample=0.7,
                    random_state=33)
#Fit model
pipeline = Pipeline([
        ('sampling', SMOTE()),
        ('classification', model)
    ])

pipeline.fit(X_train_scaled, y_train)

#Save local copy of the fitted model and scaler
joblib.dump(pipeline,"output/model.pkl")
joblib.dump(scaler,"output/scaler.pkl")

#Save a copy in AWS bucket
s3_client = boto3.client('s3')

output_file = 'model.pkl'
s3_client.upload_file("output/model.pkl", "modelandscaler",'model.pkl')

output_file = 'scaler.pkl'
s3_client.upload_file("output/scaler.pkl", "modelandscaler",'scaler.pkl')