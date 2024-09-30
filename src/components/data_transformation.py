from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
import numpy as np
import pandas as pd
import os
import sys

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation_obj(self):
        """This function is responsible for Data Transformation"""
        try:
            Num_columns = ['age', 'bmi', 'children']
            Categorical_columns = ['sex', 'smoker', 'region']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Numerical_pipeline", num_pipeline, Num_columns),
                    ("Categorical_pipeline", categorical_pipeline, Categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  # type: ignore
            test_df = pd.read_csv(test_path)    # type: ignore
            logging.info("Reading train and test data completed")

            preprocessing_obj = self.data_transformation_obj()
            target_column_name = "charges"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully.")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_filepath)

        except Exception as e:
            logging.error("An error occurred during data transformation", exc_info=True)
            raise CustomException(e, sys)
