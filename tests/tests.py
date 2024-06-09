import json
import unittest

from fastapi.testclient import TestClient

from app import app
from src.app_utils import build_test_features, is_float

#To run locally : run `python -m unittest tests/tests.py` in the terminal

class TestAPI(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.skid=404989
        self.client=TestClient(app)

    def test_predict_from_SK_ID_CURR(self):
        #Retrieve prediction
        prediction=self.client.post('model/predict_from_SK_ID_CURR?SK_ID_CURR='+str(self.skid))
        self.assertEqual(str(prediction),'<Response [200 OK]>')
        #Convert to json
        prediction=prediction.json()
        #Check contents
        self.assertIsInstance(prediction,dict)
        self.assertEqual(list(prediction.keys()),['prediction', 'probability_of_reinbursing'])
        self.assertIsInstance(prediction['prediction'],str)
        self.assertTrue(is_float(prediction['probability_of_reinbursing']))
        self.assertGreaterEqual(float(prediction['probability_of_reinbursing']),0)
        self.assertLessEqual(float(prediction['probability_of_reinbursing']),1)

    def test_get_client_data(self):
        data=self.client.get('data/get_client_data?SK_ID_CURR='+str(self.skid)).json()
        self.assertIsInstance(data,dict)
        self.assertEqual(list(data.keys()),['client_data'])
        self.assertIsInstance(data['client_data'],dict)

    def test_predict_from_data(self):
        features=build_test_features()
        #Retrieve prediction
        prediction=self.client.post('model/predict_from_data?',json={'data':features}).json()
        #Check contents
        self.assertIsInstance(prediction,dict)
        self.assertEqual(list(prediction.keys()),['prediction', 'probability_of_reinbursing'])
        self.assertIsInstance(prediction['prediction'],str)
        self.assertTrue(is_float(prediction['probability_of_reinbursing']))
        self.assertGreaterEqual(float(prediction['probability_of_reinbursing']),0)
        self.assertLessEqual(float(prediction['probability_of_reinbursing']),1)

if __name__ == '__main__':
    unittest.main()
