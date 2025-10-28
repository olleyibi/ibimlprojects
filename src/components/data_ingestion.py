# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os
import sys
from src.exception import CustomException   # Custom exception handler
from src.logger import logging              # Custom logging utility
import pandas as pd                         # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass           # For clean configuration management


# -------------------------------------------------------------
# Data Configuration Class
# -------------------------------------------------------------
# Purpose:
#   This dataclass holds file path configurations for storing
#   training, testing, and raw data. It centralizes path handling
#   and ensures consistency across the pipeline.
# -------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# -------------------------------------------------------------
# Data Ingestion Class
# -------------------------------------------------------------
# Purpose:
#   This class handles reading the dataset, performing the
#   train-test split, and saving the results to designated paths.
# -------------------------------------------------------------
class DataIngestion:
    def __init__(self):
        # Initialize the configuration object with default paths
        self.ingestion_config = DataIngestionConfig()

    # ---------------------------------------------------------
    # Method: initiate_data_ingestion
    # ---------------------------------------------------------
    # Purpose:
    #   Reads raw data, saves a copy, performs train-test split,
    #   and stores split data in artifact directories.
    #
    # Returns:
    #   Tuple (train_data_path, test_data_path)
    #
    # Exceptions:
    #   Raises CustomException for any file I/O or data issues.
    # ---------------------------------------------------------
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # 1. Read dataset from source CSV
            df = pd.read_csv(r'notebook\data\stud.csv') # this line can be changed to read from any source
            logging.info('Dataset successfully read into DataFrame')

            # 2. Ensure the artifact directory exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # 3. Save raw data as a backup in artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # 4. Perform train-test split (80-20)
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save split datasets to artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved successfully")

            # 6. Log completion and return file paths
            logging.info("Data ingestion process completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Wrap and raise any exception using the custom handler
            raise CustomException(e, sys)


# -------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------
# Purpose:
#   Executes data ingestion as a standalone module.
# -------------------------------------------------------------
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

# Run on terminal using 'python -m src.components.data_ingestion'