# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os
import sys
from src.exception import CustomException   # Handles exceptions across the project
from src.logger import logging              # Custom logging for monitoring pipeline steps
import pandas as pd                         # For data manipulation and CSV reading
from sklearn.model_selection import train_test_split  # To split data into train/test
from dataclasses import dataclass           # For clean configuration of file paths
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# -------------------------------------------------------------
# Data Configuration Class
# -------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    """
    Stores file paths for saving ingested data.

    Attributes
    ----------
    train_data_path : str
        File path to save the training dataset.
    test_data_path : str
        File path to save the testing dataset.
    raw_data_path : str
        File path to save the original raw dataset.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# -------------------------------------------------------------
# Data Ingestion Class
# -------------------------------------------------------------
class DataIngestion:
    """
    Handles data ingestion: reads raw dataset, performs train-test split,
    and saves datasets to the artifact directory.
    """
    def __init__(self):
        # Initialize configuration paths for data saving
        self.ingestion_config = DataIngestionConfig()

    # ---------------------------------------------------------
    # Method: initiate_data_ingestion
    # ---------------------------------------------------------
    def initiate_data_ingestion(self):
        """
        Reads raw data, saves a backup, performs train-test split,
        and stores resulting datasets.

        Returns
        -------
        tuple
            Paths to train and test datasets.

        Raises
        ------
        CustomException
            Wraps any file I/O or processing error.
        """
        logging.info("Entered the data ingestion component")

        try:
            # Step 1: Read raw CSV dataset
            df = pd.read_csv(r'notebook\data\stud.csv')  # Update path if needed
            logging.info('Dataset successfully read into DataFrame')

            # Step 2: Ensure artifacts directory exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Step 3: Save a raw backup copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Step 4: Perform train-test split (80% train, 20% test)
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 5: Save train/test datasets to artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test datasets saved successfully")

            # Step 6: Log completion and return paths
            logging.info("Data ingestion process completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Use custom exception for detailed error trace
            raise CustomException(e, sys)

# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------
if __name__ == '__main__':
    """
    Executes the data ingestion pipeline when run directly.
    After ingestion, it triggers data transformation and model training.
    """
    # 1. Perform data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Transform data for model training
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3. Train models and print the R^2 score of the best model
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

# -------------------------------------------------------------
# How to Run
# -------------------------------------------------------------
# In terminal:
# python -m src.components.data_ingestion
