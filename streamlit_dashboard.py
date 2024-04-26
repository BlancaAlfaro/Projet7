import io
from json import JSONDecodeError

import boto3
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.title('Loan reimbursment prediction')
selection=st.selectbox('Do you want to predict reimbursment probability from id or raw data?',('Client Id',"Client raw data"))
if selection=='Client raw data':
    st.write('This option is not available for now')
    #Add fields to input needed features
    st.subheader('**Please provide the following information:**')

    st.write('**Application information**')
    flag_phone=st.radio('Did client provide home phone?',["Yes","No"],horizontal=True)
    days_registration=st.text_input('How many days before the application did the client change his registration?')
    weekday_appr_process_start_tuesday=st.radio('Did the client apply for previous application on a Tuesday?',['Yes','No'],horizontal=True)

    st.write('**Income and assets**')
    name_income=st.selectbox("Clients income type",('Working','Commercial associate','Pensioner','Other'))
    flag_onw_car=st.radio('Does the client own a car?',['Yes','No'],horizontal=True)
    housetype_mode=st.radio('Does the client live in a block of flats?',['Yes','No'],horizontal=True)
    walls_material_type=st.radio('Are the walls of the building the client lives in made of either stone or brick',
                                 ['Yes','No'],horizontal=True)
    reg_city_not_work_city=st.radio('Does permanent address not match work address?',['Yes','No'],horizontal=True)

    st.write('**Personnal information**')
    cnt_children=st.text_input('Number of children')
    name_type_suite=st.radio('Was the client accompanied when he was applying for the loan?',['Yes','No'],horizontal=True)

    st.write('**Education**')
    name_education_type=st.selectbox('Level of highest education the client achieved',
                                     ('Higher education','Secondary / secondary special',
                                      'Other'))

    st.write('**Regional information**')
    region_rating_client=st.text_input('Our rating of the region where client lives (1,2,3)')
    region_rating_client_w_city=st.text_input("Our rating of the region where client lives with taking city into account (1,2,3)")
    region_population=st.text_input("Normalized population of region where client lives (higher number means the client lives in more populated region)")
    #TODO Trasform inputs into usable features

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
    X=pd.DataFrame(columns=features_to_keep)
elif selection=='Client Id':
    st.write('**Please input client id:**')
    input=st.text_input('SK_ID_CURR',help="Client id as found under SK_ID_CURR in client's data. Should be 6 digits long.")
    #Make a prediction request to the API
    try:
        prediction=requests.post('http://127.0.0.1:8000/model/predict_from_SK_ID_CURR?SK_ID_CURR='+str(input)).json()
        st.write('Predicted class: '+prediction['prediction'])
        st.write('Probability of not reimbursing :'+prediction['probability_of_not_reinbursing'])
        # Add image from AWS
        filename="explain_prediction_for_"+str(input)
        s3_client = boto3.client('s3')
        s3_client.download_file("predictionfigures",filename,"output/"+filename+".png")
        img=Image.open("output/"+filename+".png")
        st.write('Prediction was based of the following factors:')
        st.image(img)
    except JSONDecodeError:
        st.write('Please provide a valid id')