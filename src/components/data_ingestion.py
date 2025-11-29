# Data Reading/Extraction package
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainingConfig, ModelTraining

from sklearn.model_selection import train_test_split as tts


@dataclass                           #Decorator
class DataIngestionConfig:                                    #Input Variables to DataIngestion Class
    train_path: str = os.path.join("artifact", "train.csv")   # Changeable parameters
    test_path: str = os.path.join("artifact", "test.csv")
    raw_path: str = os.path.join("artifact", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_inputs = DataIngestionConfig()

    def initiate_ingestion(self):                            # Internal Logic
        # Reading Data from Database
        logging.info("Data Ingestion Initiated")

        try:
            df = pd.read_csv("notebook/data/StudentsPerformance.csv")
            logging.info("Read data as DataFrame")
            df["total score"] = df["math score"] + df["reading score"] + df["writing score"]
            os.makedirs(os.path.dirname(self.ingestion_inputs.train_path), exist_ok=True)
        
            logging.info("Feeding Raw DataFrame in raw_path")
            df.to_csv(self.ingestion_inputs.raw_path, index=False, header=True)
        
            logging.info("Train_Test_Split Beginning")
            train_data, test_data = tts(df, test_size = 0.2, random_state=42)

            logging.info("Exporting Train, Test data to respective paths")
            train_data.to_csv(self.ingestion_inputs.train_path, index=False, header=True)
            test_data.to_csv(self.ingestion_inputs.test_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_inputs.train_path,
                self.ingestion_inputs.test_path,          # Usable in Data Transformation
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_ingestion()

    # Using Data Transformation
    preprocessor = DataTransformation()
    train_arr, test_arr,_ = preprocessor.initiate_transformation(train_path, test_path)  # Returns train_arr, test_arr and preprocessor obj path where saved

    model_trainer = ModelTraining()
    model_trainer.initiate_model_training(train_data=train_arr, test_data=test_arr)


