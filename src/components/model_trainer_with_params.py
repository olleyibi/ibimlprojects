# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os
import sys
from dataclasses import dataclass

# ML models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models_param


# -------------------------------------------------------------
# Model Trainer Configuration
# -------------------------------------------------------------
@dataclass
class ModelTrainerConfig:
    """
    Stores configuration settings for the model trainer.

    Attributes
    ----------
    trained_model_file_path : str
        Path where the best trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# -------------------------------------------------------------
# Model Trainer Class
# -------------------------------------------------------------
class ModelTrainer:
    """
    Trains multiple regression models, performs hyperparameter tuning,
    selects the best model based on R² score, and saves it for later use.
    """

    def __init__(self):
        # Initialize configuration
        self.model_trainer_config = ModelTrainerConfig()

    # ---------------------------------------------------------
    # Method: initiate_model_trainer
    # ---------------------------------------------------------
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple ML regression models, evaluates them using R² score,
        runs hyperparameter tuning, and saves the best-performing model.

        Parameters
        ----------
        train_array : np.array
            Numpy array containing training features + target.
        test_array : np.array
            Numpy array containing test features + target.

        Returns
        -------
        float
            R² score of the best model on the test dataset.

        Raises
        ------
        CustomException
            If a training error occurs or no model performs above threshold.
        """
        try:
            logging.info("Splitting input data into training and testing sets.")

            # Split into features and target (last column is y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # -------------------------------------------------
            # Define candidate models
            # -------------------------------------------------
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

            # -------------------------------------------------
            # Hyperparameter search space for each model
            # -------------------------------------------------
            param = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regressor": {},
                "K-Neigbour Regressor": {
                    'n_neighbors': [5, 7, 9, 11]
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # -------------------------------------------------
            # Evaluate models (with tuning)
            # -------------------------------------------------
            model_report = evaluate_models_param(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param
            )

            # -------------------------------------------------
            # Select best model based on highest R² score
            # -------------------------------------------------
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(
                f"Best model identified: {best_model_name} "
                f"with R² score: {best_model_score}"
            )

            # -------------------------------------------------
            # Ensure sufficient performance
            # -------------------------------------------------
            if best_model_score < 0.6:
                raise CustomException("No model achieved the minimum acceptable performance threshold (R² < 0.6).")

            # -------------------------------------------------
            # Save best model
            # -------------------------------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully.")

            # -------------------------------------------------
            # Final evaluation on test data
            # -------------------------------------------------
            predictions = best_model.predict(X_test)
            r2_final = r2_score(y_test, predictions)

            return r2_final

        except Exception as e:
            raise CustomException(e, sys)
