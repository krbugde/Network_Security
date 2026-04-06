# Import sys for system-level error handling
import sys

# Import os for file and directory operations
import os

# Import numpy for numerical array operations
import numpy as np

# Import custom logger to record events in log file
from networksecurity.logging.logger import logging

# Import custom exception for detailed error messages
from networksecurity.exception.exception import CustomException

# Import pandas to read and manipulate CSV files
import pandas as pd

# Import KNNImputer to fill missing values using K-Nearest Neighbours algorithm
from sklearn.impute import KNNImputer

# ✅ Fix 1 — Wrong import! Pipeline should be imported from sklearn.pipeline not KNNImputer
# from sklearn.pipeline import KNNImputer  ← ❌ Remove this wrong line
from sklearn.pipeline import Pipeline

# Import target column name constant (column we want to predict)
from networksecurity.constant.training_pipeline import TARGET_COLUMN

# Import KNN imputer parameters (n_neighbors, weights etc.) from constants
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

# Import artifact entities for input and output of this step
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,   # Output artifact of this step
    DataValidationArtifact         # Input artifact from previous step
)

# Import config entity for data transformation settings
from networksecurity.entity.config_entity import DataTransformationConfig

# Import utility functions to save numpy arrays and Python objects
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:

    # Constructor: initializes class with validation artifact and transformation config
    # data_validation_artifact  → contains valid train/test file paths from previous step
    # data_transformation_config → contains paths where transformed data will be saved
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            # Store validation artifact to access valid train/test file paths
            self.data_validation_artifact = data_validation_artifact

            # Store transformation config to access output file paths
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise CustomException(e, sys)

    # Static method to read CSV file and return as pandas DataFrame
    # @staticmethod means no 'self' needed, can be called without object
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            # ✅ Fix 2 — 'ps' should be 'pd' (pandas)
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    # Method to create and return a data transformation pipeline
    # Uses KNNImputer to fill missing values in the data
    # Returns a sklearn Pipeline object
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initializes a KNNImputer with parameters from training_pipeline.py
        and returns it wrapped in a sklearn Pipeline object
        """
        logging.info("Entered get_data_transformer_object of Transformation class")

        try:
            # Create KNNImputer object with parameters from constants file
            # KNNImputer fills missing values using K nearest neighbours
            # ** unpacks dictionary into keyword arguments
            # Example: {"n_neighbors": 3} → KNNImputer(n_neighbors=3)
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Wrap imputer inside a Pipeline
            # Pipeline allows chaining multiple steps together
            # ("imputer", imputer) → step name and the actual object
            processor: Pipeline = Pipeline([("imputer", imputer)])

            # Return the pipeline object
            return processor

        except Exception as e:
            raise CustomException(e, sys)

    # Main method to start the complete data transformation process
    # Returns DataTransformationArtifact with paths of transformed data
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:
            logging.info("Starting data transformation")

            # Read valid train CSV file from data validation artifact
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)

            # Read valid test CSV file from data validation artifact
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # ── TRAINING DATAFRAME ──────────────────────────────────

            # Remove target column from train data to get input features
            # axis=1 → drop column (not row)
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)

            # Get only the target column from train data (what we want to predict)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # Replace -1 values with 0 in target column
            # Because many ML models expect binary labels (0 and 1) not (-1 and 1)
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # ── TESTING DATAFRAME ───────────────────────────────────

            # Remove target column from test data to get input features
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            # ✅ Fix 3 — was train_df[TARGET_COLUMN], should be test_df[TARGET_COLUMN]
            # Get only the target column from TEST data (not train!)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Replace -1 values with 0 in test target column
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # ── TRANSFORMATION ──────────────────────────────────────

            # Get the KNNImputer pipeline object
            preprocessor = self.get_data_transformer_object()

            # Fit the imputer on TRAIN data only
            # fit() → learns the patterns from train data (e.g. neighbour values)
            # We NEVER fit on test data to avoid data leakage
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            # Transform train input features using fitted imputer
            # Fills missing values in train data
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)

            # Transform test input features using SAME fitted imputer
            # Uses patterns learned from train data to fill test missing values
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # ── COMBINE FEATURES AND TARGET ─────────────────────────

            # ✅ Fix 4 — np.c_ not np.c[] (underscore not square bracket)
            # Combine transformed input features and target column horizontally
            # np.c_ → concatenates arrays column wise
            # np.array(target) → converts pandas series to numpy array
            # Result: [feature1, feature2, ..., target]
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]

            # Combine transformed test features and test target column
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]


            #===========SAVING TRASNFORMED FILE PATH OF TRAIN AND TEST==================
            # Save transformed train array as .npy binary file
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,
                                  array=train_arr)

            # Save transformed test array as .npy binary file
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,
                                  array=test_arr)

            # Save the fitted preprocessor object as .pkl file
            # So we can reuse it later during prediction without refitting
            save_object(self.data_transformation_config.transformed_object_file_path,
                        preprocessor_object)

            # Create DataTransformationArtifact with paths of saved files
            # This artifact is passed to the next step (Model Training)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data Transformation completed successfully")

            # Return artifact to next step in ML pipeline
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)