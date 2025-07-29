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
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{
                    # 'fit_intercept':[True,False],
                    # 'normalize':[True,False],
                    # 'copy_X':[True,False]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighbors":{
                    'n_neighbors': [3,5,7,9,11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info(f"Model training started")

            model_predict: dict = Predict_model(models,X_train, X_test, y_train, y_test,parms=params)
            
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