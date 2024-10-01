import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features): 
        try:
            print("Before Loading Model and Preprocessor")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            print("Model and Preprocessor Loaded Successfully")
            
            # Log the input features shape
            print(f"Input Features Shape: {features.shape}")
            
            # Scale the input features
            data_scaled = preprocessor.transform(features)
            print(f"Scaled Data Shape: {data_scaled.shape}")
            
            # Predict
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 age: int, 
                 sex: str, 
                 bmi: float, 
                 children: int, 
                 smoker: str, 
                 region: str):
                 
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
