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
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)            # Store at file_object not file_string(path)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file) 
    
    except Exception as e:
        raise CustomException(e, sys)

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
    params = {
        "Linear": {"tol" : [1e-5, 1e-6, 1e-7, 1e-8]},
        
        "Lasso": {"alpha": [0.8, 0.9, 1, 1.5, 2],
                  "max_iter":[500, 1000, 1500],
                   "tol": [0.0001, 0.0002, 0.0003]},
        
        "Ridge": {"alpha": [0.8, 0.9, 1, 1.5, 2]},
        
        "SVR": {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                "degree": [2, 3, 4],
                "gamma": ['scale', 'auto'],
                "coef0": [0, 1]},
        
        "Tree": {"splitter": ['best', 'random'],
                "max_depth": [2, 3, 4],
                "max_features": [1.0, 'sqrt', 'log2']},

        "RandomForest": {"n_estimators": [50, 100, 150],
                        "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        "max_depth": [2, 3, 4]},

        "Adaboost": { "n_estimators": [50, 100, 150],
                    "learning_rate": [0.8, 0.9, 1.0, 1.5, 2.0],
                    "loss" : ['linear', 'square', 'exponential']},                   

        "Xgboost": {
                    'n_estimators': [100, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.7, 0.8, 0.9]
}
    }
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]                  #Since same order as of models
            randomized_search = RandomizedSearchCV(estimator=model, n_iter = 50, param_distributions=param, cv=3, n_jobs=-1)

            randomized_search.fit(X_train_data, y_train_data)
            best_params = randomized_search.best_params_

            model_pro = model.set_params(**best_params)
            model_pro.fit(X_train_data, y_train_data)
            #Testing
            y_pred = model.predict(X_test_data)

            r2 = r2_score(y_test_data, y_pred)
            report[list(models.keys())[i]] = (r2, best_params)

            best_score = max(sorted(list(report.values())))
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]
            best_model = models[best_model_name]
        return report, best_model, best_model_name

    except Exception as e:
        raise CustomException(e, sys)


