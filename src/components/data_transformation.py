import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Configuration class for file paths
@dataclass
class DataTransformationConfig:
    # Location to save the serialized preprocessing object
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Class for performing all data transformation steps
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Function responsible for building the preprocessing pipeline.
        It handles numerical and categorical features differently.
        '''
        try:
            # Define numerical and categorical feature names
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", 
                "race_ethnicity", 
                "parental_level_of_education", 
                "lunch", 
                "test_preparation_course"
            ]

            # Pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),   # Fill missing numeric values with median
                    ("scaler", StandardScaler())                     # Standardize numerical data
                ]
            )

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with most frequent
                    ("one_hot_encoder", OneHotEncoder()),                  # Encode categorical variables
                    ("scale", StandardScaler(with_mean=False))             # Scale encoded values (with_mean=False for sparse matrices)
                ]
            )

            # Logging progress
            logging.info('Numerical columns standard scaling completed')
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info('Category columns encoding completed')
            logging.info(f'Category columns: {categorical_columns}')

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Return the full preprocessing object
            return preprocessor

        except Exception as e:
            # Wrap any errors with custom exception handling
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Function to apply preprocessing on train and test datasets.
        It reads CSVs, applies transformation, and saves the preprocessing object.
        '''
        try:
            # Read train and test data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and numeric features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Split features and target for train set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Split features and target for test set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test dataframes")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features and target into single arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object to artifacts folder")

            # Save the preprocessing object (serializer)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj  # ⚠ Make sure save_object() accepts this keyword argument
            )

            # Return transformed arrays and the preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Catch and raise custom exception for debugging
            raise CustomException(e, sys)
