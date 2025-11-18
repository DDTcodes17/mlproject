#Data Preprocessing/ Encoding

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformer_path = DataTransformationConfig()
        logging.info("DataTransformation Object created with config")

    def get_transformer(self):
        '''Responsible for creating Preprocessor Object for all Data Transformation/Encoding of
        numerical and categorical columns'''
        logging.info("Data Preprocessor object creation started")

        try:
            numeric_columns = ["math score", "reading score", "writing score"]
            cat_columns = ["gender","race/ethnicity","parental level of education","lunch",
                           "test preparation course"]
            
            num_pipeline = Pipeline(
                steps = [('imputer', SimpleImputer(strategy="median")),
                         ('scaler', StandardScaler())]
            )
            logging.info("Numerical Pipeline Created")

            cat_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                       ('oneh_encoder', OneHotEncoder())
                       ]
            )
            logging.info("Categorical Pipeline Created")

            #Combining the Pipelines
            preprocessor = ColumnTransformer(
                transformers=[("num_transformer", num_pipeline, numeric_columns ),
                              ("cat_transformer", cat_pipeline, cat_columns)],
                remainder='passthrough'              
            )
            logging.info("Combined Pipeline created using ColumnTransformer")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_transformation(self, train_path, test_path):
        '''Implementing the preprocessor.
        Reading train/test data, obtaining preprocessor and implementing.'''
        logging.info("Data Transformation initiated")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train/test data")

            preprocessor = self.get_transformer()
            logging.info("Obtained Preprocessor object")

            target = "total score"
            X_train_df = train_df.drop([target], axis=1)
            y_train_df = train_df[target]

            X_test_df = test_df.drop([target], axis=1)
            y_test_df = test_df[target]
            logging.info("Input, Output Feature Segregated")

            X_train_arr = preprocessor.fit_transform(X_train_df)
            X_test_arr = preprocessor.transform(X_test_df)
            logging.info("Applied Transformations")

            #Combining these inputs tranformed(now array) and Dataframe output to a final array( for further use)
            train_arr = np.c_[X_train_arr, np.array(y_train_df)]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]
            logging.info("A single array entity of output+input created")

            #Saving file
            # os.makedirs(os.path.dirname(self.transformer_path.preprocessor_path), exist_ok=True)

            # with open(self.transformer_path.preprocessor_path, "wb") as file_path:
            #  pickle.dump(preprocessor,file_path)

            save_object(self.transformer_path.preprocessor_path, preprocessor) # Using utilities
            logging.info("Preprocessing completed and Transformer object stored as pickle file")

            return(
                train_arr,
                test_arr,
                self.transformer_path.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)



                    
