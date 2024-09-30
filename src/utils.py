import os
import sys
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves an object to a specified file path using dill for serialization.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning.
    
    Parameters:
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training labels.
    - X_test (ndarray): Testing features.
    - y_test (ndarray): Testing labels.
    - models (dict): Dictionary of models to evaluate.
    - param (dict): Dictionary of hyperparameters for each model.

    Returns:
    - report (dict): A dictionary containing test R² scores for each model.
    """
    report = {}

    for model_name, model in models.items():
        try:
            para = param[model_name]

            logging.info(f"Evaluating model: {model_name}")

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Training the model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"Model: {model_name}, Train R² Score: {train_model_score}, Test R² Score: {test_model_score}")

        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {str(e)}")
            raise CustomException(e, sys)

    return report
