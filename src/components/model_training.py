import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConifg:
  trainer_model_file_path=os.path.join("artifacts","model.pkl")

class Model_training:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConifg()

  def initiate_model_training(self, train_array, test_array):
    try:
      logging.info("Splitting dataset into train and test data")

      X_train, y_train, X_test, y_test =(train_array[:,:-1],
                                        train_array[:,-1],
                                        test_array[:,:-1],
                                        test_array[:,-1])
      
      models={
        "Random_forest":RandomForestRegressor(),
        "Decision_tree":DecisionTreeRegressor(),
        "Gradient_Boosting":GradientBoostingRegressor(),
        "Linear_Regression": LinearRegression(),
        "K-Neighbours_Regressor": KNeighborsRegressor(),
        "XGB_classifier": XGBRegressor(),
        "CatBoosting classifier": CatBoostRegressor(),
        "AdaBoost classifier":AdaBoostRegressor()
      }


      model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models)

      ## To get the best model score from dict
      best_model_score= max(sorted(model_report.values()))

      best_model_name= list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model=models[best_model_name]

      if best_model_score <=0.60:
        raise CustomException("No best model found")
      
      logging.info(f"Best Model found on both Training and Testing dataset")

      save_object(file_path=self.model_trainer_config.trainer_model_file_path,
                  obj=best_model
                  )
      
      predicted=best_model.predict(X_test)

      r2_scoring =r2_score(y_test, predicted)

      return r2_scoring




    except Exception as e:
      raise CustomException(e, sys)
  