import pandas as pd # type: ignore
import os
import sys
import numpy as np # type: ignore
import dill # type: ignore
from src.Exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score # type: ignore


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)
    
def Predict_model(models, X_train, X_test, y_train, y_test):
    model_predict = {}
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_score_value = r2_score(y_test, y_pred)
            model_predict[model_name] = r2_score_value
            logging.info(f"{model_name} R2 score: {r2_score_value}")
        return model_predict
    
    except Exception as e:
        logging.error(f"Error occurred while predicting with models: {e}")
        raise CustomException(e, sys)