import io
import json
from json import JSONDecodeError

import boto3
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

#api_url="http://127.0.0.1:8000/" #Local for testing
api_url="https://ocp7webapp.azurewebsites.net/"

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


    if is_float(region_population):
        try:
            # Transform answers into usable features
            features={}
            str_to_bool_dict={'Yes' : True, 'No': False}

            features['FLAG_PHONE']= str_to_bool_dict[flag_phone]
            features['DAYS_REGISTRATION']=int(days_registration)
            features['WEEKDAY_APPR_PROCESS_START_TUESDAY']=str_to_bool_dict[weekday_appr_process_start_tuesday]
            features['NAME_INCOME_TYPE_Working']=(name_income=='Working')
            features['NAME_INCOME_TYPE_Commercial associate']=(name_income=='Commercial associate')
            features['NAME_INCOME_TYPE_Pensioner']=(name_income=='Pensioner')
            features['FLAG_OWN_CAR']=str_to_bool_dict[flag_onw_car]
            features['HOUSETYPE_MODE_block of flats']=str_to_bool_dict[housetype_mode]
            features['WALLSMATERIAL_MODE_Stone, brick']=str_to_bool_dict[walls_material_type]
            features['REG_CITY_NOT_WORK_CITY']=str_to_bool_dict[reg_city_not_work_city]
            features['CNT_CHILDREN']=int(cnt_children)
            features['NAME_TYPE_SUITE_Unaccompanied']=str_to_bool_dict[name_type_suite]
            features['NAME_EDUCATION_TYPE_Higher education']=(name_education_type=='Higher education')
            features['NAME_EDUCATION_TYPE_Secondary / secondary special']=(name_education_type=='Secondary / secondary special')
            features['REGION_RATING_CLIENT']=int(region_rating_client)
            features['REGION_RATING_CLIENT_W_CITY']=int(region_rating_client_w_city)
            features['REGION_POPULATION_RELATIVE']=float(region_population)

            X=pd.DataFrame(features,index=['temp'])
        except ValueError as err:
            print(err)

        try :
            st.write(X)
        except (ValueError,NameError):
            st.write('Please provide the information needed for prediction')

        try:
            prediction=requests.post(api_url+'model/predict_from_data?',json={'data':features})
            print(prediction)
            prediction=prediction.json()
            print(prediction)
            st.write('Predicted class: '+prediction['prediction'])
            st.write('Probability of not reimbursing :'+prediction['probability_of_not_reinbursing'])
            # Add image from AWS
            filename="explain_prediction_for_"+"temp"
            s3_client = boto3.client('s3')
            s3_client.download_file("predictionfigures",filename,"output/"+filename+".png")
            img=Image.open("output/"+filename+".png")
            st.write('Prediction was based of the following factors:')
            st.image(img)
        except JSONDecodeError:
            st.write('Please provide valid information for the prediction')

elif selection=='Client Id':
    st.write('**Please input client id:**')
    input=st.text_input('SK_ID_CURR',help="Client id as found under SK_ID_CURR in client's data. Should be 6 digits long.")
    #Make a prediction request to the API
    try:
        prediction=requests.post(api_url+'model/predict_from_SK_ID_CURR?SK_ID_CURR='+str(input)).json()
        print(prediction)
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