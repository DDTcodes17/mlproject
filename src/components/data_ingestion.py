# Data Reading/Extraction package
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split as tts


@dataclass                           #Decorator
class DataIngestionConfig:                                    #Input Variables to DataIngestion Class
    train_path: str = os.path.join("artifact", "train.csv")
    test_path: str = os.path.join("artifact", "test.csv")
    raw_path: str = os.path.join("artifact", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_inputs = DataIngestionConfig()

    def initiate_ingestion(self):
        # Reading Data from Database
        logging.info("Data Ingestion Initiated")

        try:
            df = pd.read_csv("notebook/data/StudentsPerformance.csv")
            logging.info("Read data as DataFrame")

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
    obj.initiate_ingestion()