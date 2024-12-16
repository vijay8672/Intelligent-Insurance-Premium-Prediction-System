import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    input_data_path: str = 'EDA/Medical_insurance.csv'  # Parameterized input path
    test_size: float = 0.3  # Configurable test size

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into the data ingestion method")
        try:
            df = pd.read_csv(self.ingestion_config.input_data_path)
            logging.info("Read the dataset as dataframe")

            # Remove duplicates
            df = df.drop_duplicates()
            logging.info(f"Removed duplicates, new dataframe shape: {df.shape}")

            # Create directories for saving files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=self.ingestion_config.test_size, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")
            return {
                "train_data_path": self.ingestion_config.train_data_path,
                "test_data_path": self.ingestion_config.test_data_path,
                "train_shape": train_set.shape,
                "test_shape": test_set.shape,
            }

        except Exception as e:
            logging.error("An error occurred in Data Ingestion", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    ingestion_results = obj.initiate_data_ingestion()
    train_data, test_data = ingestion_results['train_data_path'], ingestion_results['test_data_path']

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTraining()
    logging.info("Initiating model training")
    score = model_trainer.initiate_model_training(train_arr, test_arr)
    logging.info(f"Model training completed with R2 score: {score}")
