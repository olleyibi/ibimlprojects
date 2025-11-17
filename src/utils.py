import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill serialization.

    Parameters
    ----------
    file_path : str
        Full path including filename where the object should be saved.
    obj : object
        The Python object to save (model, transformer, etc.).

    Raises
    ------
    CustomException
        If an error occurs during the object saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)

        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models without hyperparameter tuning.

    Parameters
    ----------
    X_train : DataFrame or array
        Training feature set.
    y_train : Series or array
        Training labels.
    X_test : DataFrame or array
        Test feature set.
    y_test : Series or array
        Test labels.
    models : dict
        Dictionary of model_name : model_instance.

    Returns
    -------
    dict
        Dictionary of model_name : test_R2_score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R2 evaluation
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Save score
            report[list(models.keys())[i]] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models_param(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models with hyperparameter tuning.

    Parameters
    ----------
    X_train, y_train, X_test, y_test:
        Training and test datasets.
    models : dict
        Dictionary of model_name : model_instance.
    param : dict
        Dictionary of model_name : parameter_grid for GridSearchCV.

    Returns
    -------
    dict
        Dictionary of model_name : best_test_R2_score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            param_grid = param[model_name]

            # Grid search
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Update model with best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R2 scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store score
            report[model_name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
            
        except Exception as e:
            raise CustomException(e, sys)
