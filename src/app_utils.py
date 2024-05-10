
import json


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def build_test_features():
    features={}
    features['FLAG_PHONE']= True
    features['DAYS_REGISTRATION']=10
    features['WEEKDAY_APPR_PROCESS_START_TUESDAY']=True
    features['NAME_INCOME_TYPE_Working']=True
    features['NAME_INCOME_TYPE_Commercial associate']=False
    features['NAME_INCOME_TYPE_Pensioner']=False
    features['FLAG_OWN_CAR']=False
    features['HOUSETYPE_MODE_block of flats']=True
    features['WALLSMATERIAL_MODE_Stone, brick']=True
    features['REG_CITY_NOT_WORK_CITY']=False
    features['CNT_CHILDREN']=2
    features['NAME_TYPE_SUITE_Unaccompanied']=True
    features['NAME_EDUCATION_TYPE_Higher education']=True
    features['NAME_EDUCATION_TYPE_Secondary / secondary special']=False
    features['REGION_RATING_CLIENT']=2
    features['REGION_RATING_CLIENT_W_CITY']=1
    features['REGION_POPULATION_RELATIVE']=5000
    return features