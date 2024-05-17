
import json


def is_float(string):
    """Check if string is convertible to float

    Parameters
    ----------
    string : str
        string to test

    Returns
    -------
    bool
        If the strings is convertible to float.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False

def build_test_features():
    """
    Make a dictionnary of features as returned by streamlit dashboard to test the API's function that predicts from raw data.

    Returns
    -------
    dict
        dictionnary of features
    """
    features={}
    features['FLAG_PHONE']= True
    features['DAYS_REGISTRATION']=-10
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
    features['REGION_POPULATION_RELATIVE']=0.1
    return features