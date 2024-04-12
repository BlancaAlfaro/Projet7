import glob
import os
import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler

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
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(max_depth=5, max_features='log2', min_samples_leaf=10,n_estimators=10, random_state=33)

#Fit model
pipeline = Pipeline([
        ('sampling', SMOTE()),
        ('classification', model)
    ])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline,"output/model.pkl")
joblib.dump(scaler,"output/scaler.pkl")