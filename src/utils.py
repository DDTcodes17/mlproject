# Common used Functionalities

import os
import sys
import dill
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)            # Store at file_object not file_string(path)

def evaluate_model(X_train_data, X_test_data, y_train_data, y_test_data):
    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "SVM": SVR(),
        "Tree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "Adaboost": AdaBoostRegressor(),
        "Xgboost": XGBRegressor()
    }
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train_data, y_train_data)

            #Testing
            y_pred = model.predict(X_test_data)

            r2 = r2_score(y_test_data, y_pred)
            report[list(models.keys())[i]] = r2

            best_score = max(sorted(list(report.values())))
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]
            best_model = models[best_model_name]
        return report, best_model, best_model_name

    except Exception as e:
        raise CustomException(e, sys)    
