import pandas as pd # type: ignore
import os 
import sys
import numpy as np # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from src.Exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder # type: ignore
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join("artifacts", "transformed_data.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info('DataTransformationConfig initialized.')

    def get_Data_Transformer(self):
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline( 
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
             )
            
            cat_pipeline = Pipeline( 
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))  # Fix here
                ]
            )

            logging.info('Numerical and categorical pipelines created successfully.')

            preprocessor = ColumnTransformer(
                [("numerical_feature", num_pipeline, num_features),
                 ("categorical_feature", cat_pipeline, cat_features)]
            )

            logging.info('Data transformation pipelines created successfully.')

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info('Starting data transformation process.')
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Data loaded successfully from CSV files.')
            preprocessor_obj = self.get_Data_Transformer()

            target_column   = 'math_score'
            input_train_df  = train_df.drop(columns=[target_column], axis=1)
            target_train_df = train_df[target_column]

            input_test_df  = test_df.drop(columns=[target_column], axis=1)
            target_test_df = test_df[target_column]
            
            input_feature_train = preprocessor_obj.fit_transform(input_train_df)
            input_feature_test = preprocessor_obj.transform(input_test_df)
            logging.info('Data transformation completed successfully.')

            train_array = np.c_[input_feature_train, np.array(target_train_df)]
            test_array = np.c_[input_feature_test, np.array(target_test_df)]

            save_object(self.config.transformed_data_path, preprocessor_obj)
            
            return train_array, test_array, self.config.transformed_data_path

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e, sys)
