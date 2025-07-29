import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import r2_score # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from xgboost import XGBRegressor
from src.Exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import Predict_model

@dataclass
class ModelTrainerConfig:
    model_train_path : str = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def Initiate_model_training(self, Train_data, Test_data):
        try:
            logging.info("Initiating model training...")
            X_train, X_test, y_train, y_test = Train_data[:,:-1], Test_data[:,:-1],  Train_data[:,-1],Test_data[:,-1]

            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "XGBoost": XGBRegressor(),
                "Catboost": CatBoostRegressor()
            }

            logging.info(f"Model training started")

            model_predict: dict = Predict_model(models,X_train, X_test, y_train, y_test)
            
            best_model_name = max(model_predict, key=model_predict.get)
            best_model_value = model_predict[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {model_predict[best_model_name]}")
            if best_model_value > 0.6:
                save_object(self.model_trainer_config.model_train_path, best_model)
                logging.info(f"Model saved at {self.model_trainer_config.model_train_path}")
            else:
                raise CustomException("No model has an R2 score greater than 0.6", sys)

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            

            return r2_score_value

        except Exception as e:
            logging.error(f"Error occurred while training model: {e}")
            raise CustomException(e, sys) from e