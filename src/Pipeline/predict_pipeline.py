from src.Exception import CustomException
from src.logger import logging
import pandas as pd  # type: ignore
import os
import sys
from src.utils import load_file

class predict_pipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocesser_path = os.path.join('artifacts', 'transformed_data.pkl')
            model_path = os.path.join('artifacts', 'trained_model.pkl')
            preprocesser = load_file(preprocesser_path)
            model = load_file(model_path)
            scaled_data = preprocesser.transform(features)
            logging.info("Data preprocessing completed successfully.")
            prediction = model.predict(scaled_data)
            logging.info("Prediction completed successfully.")
            return prediction

        except Exception as e:
            logging.error(f"Error occurred during prediction: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e:
            logging.error(f"Error occurred while converting custom data to DataFrame: {e}")
            raise CustomException(e, sys)
        