import io

import boto3
import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.title('Loan reimbursment prediction')
selection=st.selectbox('Do you want to predict reimbursment probability from id or raw data?',('Client Id',"Client raw data"))
if selection=='Client raw data':
    st.write('This option is not available for now')
elif selection=='Client Id':
    st.write('**Please input client id:**')
    input=st.text_input('SK_ID_CURR',help="Client id as found under SK_ID_CURR in client's data. Should be 6 digits long.")
    prediction=requests.post('http://127.0.0.1:8000/model/predict_from_SK_ID_CURR?SK_ID_CURR='+str(input)).json()
    st.write('Predicted class: '+prediction['prediction'])
    #TODO add image from AWS
    filename="explain_prediction_for_"+str(input)
    s3_client = boto3.client('s3')
    s3_client.download_file("predictionfigures",filename,"output/"+filename+".png")
    img=Image.open("output/"+filename)
    st.write('Prediction was based of the following factors:')
    st.image(img)