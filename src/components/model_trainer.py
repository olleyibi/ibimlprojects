# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor                     # Gradient boosting with categorical support
from sklearn.ensemble import (                              
    AdaBoostRegressor,                                      # Adaptive boosting
    GradientBoostingRegressor,                              
    RandomForestRegressor                                   
)
from sklearn.linear_model import LinearRegression           # Linear regression
from sklearn.metrics import r2_score                         # Model evaluation metric
from sklearn.neighbors import KNeighborsRegressor           # KNN regressor
from sklearn.tree import DecisionTreeRegressor              # Decision tree regressor
from xgboost import XGBRegressor                             # XGBoost regressor
from src.exception import CustomException                    # Custom exception handling
from src.logger import logging                               # Custom logging utility
from src.utils import save_object, evaluate_models          # Helper functions for saving and evaluating models

# -------------------------------------------------------------
# Model Trainer Configuration
# -------------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    """
    Configuration dataclass for the Model Trainer component.
    Stores the file path where the best trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# -------------------------------------------------------------
# Model Trainer Class
# -------------------------------------------------------------
class ModelTrainer:
    """
    Handles training of multiple regression models, evaluates them,
    selects the best model based on R^2 score, and saves the best model.
    """
    def __init__(self):
        # Initialize configuration for the trainer
        self.model_trainer_config = ModelTrainerConfig()

    # ---------------------------------------------------------
    # Method: initiate_model_trainer
    # ---------------------------------------------------------
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models and returns the R^2 score of the best model.

        Args:
            train_array (np.array): Training data with features and target concatenated
            test_array (np.array): Test data with features and target concatenated

        Returns:
            float: R^2 score of the best performing model on test data

        Raises:
            CustomException: If no model performs adequately or any error occurs during training
        """
        try:
            logging.info("Splitting training and test input data")
            # Split features and target for both training and test sets
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except last
                train_array[:, -1],   # Last column is target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of models to train and evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor() 
            }

            # Evaluate all models using helper function
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            
            # Identify the best model based on highest R^2 score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Ensure the best model has an acceptable performance
            if best_model_score < 0.6:
                raise CustomException("No model achieved sufficient performance")

            logging.info(f"Best model found: {best_model_name} with R^2 score: {best_model_score}")

            # Save the best model to disk for future use
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test data using best model
            predicted = best_model.predict(X_test)

            # Evaluate predictions with R^2 score
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a detailed exception if anything fails
            raise CustomException(e, sys)
