# Model training module

import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

from src.utils import evaluate_model, save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join("artifact","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_path_config = ModelTrainingConfig()

    def initiate_model_training(self, train_data, test_data):
        try:
            logging.info("Data Segregation(Feature Engineering), Train Test Splitting")
            X_train_data = train_data[:,: -1]                        #Data Obtained is ARRAY!!!!!
            X_test_data = test_data[:, : -1]                         #So can't use drop   
            y_train_data = train_data[:, -1]
            y_test_data = test_data[:, -1]

            logging.info("Model Training Begins")
            report, best_model, model_name = evaluate_model(X_train_data, X_test_data, y_train_data, y_test_data)    #Better to get report
            logging.info("Model Training Successful")


            
            logging.info("Best Model Prediction Output Checking")
            y_pred = best_model.predict(X_test_data)
            
            if r2_score(y_test_data, y_pred) < 0.6:
                raise CustomException("No Model performed desirably")
            print("Model Name: ", model_name, "R2_Score: ", r2_score(y_test_data, y_pred))

            save_object(self.model_path_config.model_path, best_model)
            logging.info("Model Saved as pickle")

            return (best_model, report)

        except Exception as e:
            raise CustomException(e, sys)    
