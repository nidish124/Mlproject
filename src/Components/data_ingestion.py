import pandas as pd
from src.Exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os 
import sys
from src.Components.data_transformation import DataTransformation, DataTransformationConfig
from src.utils import save_object


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Initiating data ingestion...')
        try:
            df = pd.read_csv('data\stud.csv')
            logging.info('csv read is completed')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("train test split has been started")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=43)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info('Data ingestion completed successfully.')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.info(f'Error occurred during data ingestion: {e}')
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    dataingestion = DataIngestion()
    train_data, test_data = dataingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)