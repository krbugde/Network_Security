# Import artifact entity to return validation result at the end
from networksecurity.entity.artifact_entity import DataValidationArtifact

# Import artifact entity to receive train/test file paths from data ingestion
from networksecurity.entity.artifact_entity import DataIngestionArtifact

# Import config entity to get validation configuration settings
from networksecurity.entity.config_entity import DataValidationConfig

# Import custom exception for detailed error messages
from networksecurity.exception.exception import CustomException

# Import logger to record events in log file
from networksecurity.logging.logger import logging

# Import path of the schema file which defines expected columns and data types
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH

# Import ks_2samp for data drift detection
# Checks if train and test data follow same statistical distribution
from scipy.stats import ks_2samp

# Import pandas to read and work with CSV files
import pandas as pd

# Import os and sys for file operations and error handling
import os, sys

# Import utility function to read and write YAML configuration files
from networksecurity.utils.main_utils.utils import read_yaml_file
from networksecurity.utils.main_utils.utils import write_yaml_file


class DataValidation:

    # Constructor: initializes class with ingestion artifact and validation config
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            # Store train/test file paths received from data ingestion step
            self.data_ingestion_artifact = data_ingestion_artifact

            # Store validation configuration settings
            self.data_validation_config = data_validation_config

            # Read schema YAML file — defines expected columns and data types
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    # Static method — reads CSV file and returns pandas DataFrame
    # @staticmethod means no 'self' needed, can be called without object
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    # Method to check if total number of columns matches schema
    # Returns True if columns match, False if they don't
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Get required number of columns from schema file
            number_of_columns = len(self._schema_config)

            # Log both values for comparison
            logging.info(f"Required no of columns: {number_of_columns}")
            logging.info(f"DataFrame has total columns: {len(dataframe.columns)}")

            # Compare and return True or False
            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                return False

        except Exception as e:
            raise CustomException(e, sys)

    # Method to check if all numerical columns from schema exist in dataframe
    # Returns True if all numerical columns present, False if any are missing
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Get list of expected numerical columns from schema file
            schema_numerical_columns = self._schema_config["numerical_columns"]

            # Get actual columns present in the dataframe
            dataframe_columns = dataframe.columns

            # List to store any missing numerical columns
            missing_numerical_columns = []

            # Loop through each expected numerical column
            # Check if it exists in the dataframe
            for column in schema_numerical_columns:
                if column not in dataframe_columns:
                    # If column is missing, add to missing list
                    missing_numerical_columns.append(column)

            # If any numerical columns are missing, log and return False
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
                return False

            # All numerical columns are present
            return True

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            # Start drift detection
            logging.info("Checking for data drift...")

            # Flag to track if drift was found
            status = True

            # Dictionary to store drift report for each column
            report = {}

            # Loop through each column in the dataframe
            for column in base_df.columns:

                # Get the column data from train dataframe (base)
                d1 = base_df[column]

                # Get the column data from test dataframe (current)
                d2 = current_df[column]

                # Apply KS Test to compare both distributions
                # ks_2samp returns:
                # is_same_dist → test statistic (how different they are)
                # p_value → probability that both come from same distribution
                is_same_dist = ks_2samp(d1, d2)

                # If p_value < threshold (0.05) → distributions are different → Drift!
                if threshold <= is_same_dist.pvalue:
                    is_found = False  # No drift found for this column
                else:
                    is_found = True   # Drift found for this column ⚠️
                    status = False    # Overall status becomes False

                # Store the result for this column in report
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),  # probability value
                        "drift_status": is_found                 # True = drift, False = no drift
                    }
                })

            # Path where drift report will be saved
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create the directory if it doesn't exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the drift report as a YAML file
            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

            # Return overall drift status
            # True = No drift, False = Drift detected
            return status

        except Exception as e:
            raise CustomException(e, sys)
        



    # Main method to start the complete data validation process
    # Returns DataValidationArtifact with validation status
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation started")

            # Get train and test file paths from data ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read train and test CSV files into DataFrames
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)
            logging.info("Train and Test data loaded successfully")

            # Define error_message as empty string BEFORE using it
            error_message = ""

            # Validate number of columns in train dataframe
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if status == False:
                # Append error to message if columns don't match
                error_message = f"{error_message} Train dataframe does not contain all columns. \n"

            # Validate number of columns in test dataframe
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if status == False:
                # Append error to message if columns don't match
                error_message = f"{error_message} Test dataframe does not contain all columns. \n"

            # Validate numerical columns in train dataframe
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if status == False:
                error_message = f"{error_message} Train dataframe is missing numerical columns. \n"

            # Validate numerical columns in test dataframe
            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if status == False:
                error_message = f"{error_message} Test dataframe is missing numerical columns. \n"

            # Log final validation status
            logging.info(f"Validation error message: {error_message}")


            ## lets check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_path)
            os.makedirs(dir_path,exist_ok=True)

            
            train_dataframe.to_csv(self.data_validation_config.valid_train_path,index=False,header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_path,index=False,header=True)
            
            logging.info("Validated train and test data saved successfully")
            
              # ✅ Fix 3 — Create DataValidationArtifact to pass to next pipeline step
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_path,
                valid_test_file_path=self.data_validation_config.valid_test_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            # Return artifact to next step in ML pipeline
            return data_validation_artifact
        

        except Exception as e:
            raise CustomException(e, sys)