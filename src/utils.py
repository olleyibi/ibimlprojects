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
        Full file path (including filename) where the object should be saved.
    obj : object
        Any Python object to save — model, transformer, tokenizer, etc.

    Raises
    ------
    CustomException
        Raised if an error occurs during directory creation or file writing.
    """
    try:
        # Extract directory from given file path
        dir_path = os.path.dirname(file_path)

        # Create directory if missing
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object to disk
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple machine learning models (no hyperparameter tuning).

    This function fits each model, makes predictions on both train and test sets,
    then calculates R² scores to compare model performance.

    Parameters
    ----------
    X_train : array-like
        Training input features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Test input features.
    y_test : array-like
        Test labels.
    models : dict
        Dictionary mapping model name -> model instance.

    Returns
    -------
    dict
        Mapping of model_name -> test_R2_score.

    Notes
    -----
    - A higher R² indicates better predictive accuracy.
    - Only test R² is returned, since it reflects generalization.
    """
    try:
        report = {}

        for model_name, model in models.items():

            # Train model
            model.fit(X_train, y_train)

            # Predict on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute R² scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[model_name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models_param(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models with hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Training and testing datasets.
    models : dict
        Dictionary mapping model_name -> model instance.
    param : dict
        Dictionary mapping model_name -> hyperparameter grid for GridSearchCV.

    Returns
    -------
    dict
        Mapping of model_name -> best_test_R2_score obtained after tuning.

    Notes
    -----
    - Updates each model with its best-performing parameters.
    - Performs 3-fold cross-validation inside GridSearchCV.
    """
    try:
        report = {}

        for model_name, model in models.items():

            # Retrieve parameter grid corresponding to current model
            param_grid = param[model_name]

            # Grid search for best parameters
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Update model with optimal parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Compute R² scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store best test score
            report[model_name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    """
    Load a serialized Python object from disk using dill.

    Parameters
    ----------
    file_path : str
        Path to the serialized object file.

    Returns
    -------
    object
        The loaded Python object.

    Raises
    ------
    CustomException
        Raised if loading fails due to file corruption or path issues.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
