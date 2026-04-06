# Import DataIngestion component to load and split data
from networksecurity.components.data_ingestion import DataIngestion

# Import custom exception for detailed error messages
from networksecurity.exception.exception import CustomException

# Import logger to record events in log file
from networksecurity.logging.logger import logging

# Import config entities for pipeline, ingestion and validation settings
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

# Import DataValidation component to validate the ingested data
from networksecurity.components.data_validation import DataValidation

# Import sys for error handling in CustomException
import sys

# This block runs only when this file is executed directly
if __name__ == '__main__':
    try:
        # Create the overall training pipeline configuration
        # This holds common settings like artifact directory name
        trainingpipelineconfig = TrainingPipelineConfig()

        # Create data ingestion configuration using pipeline config
        # This holds settings like MongoDB URL, train-test split ratio
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)

        # Create DataIngestion object using the ingestion config
        data_ingestion = DataIngestion(dataingestionconfig)

        logging.info("Data Ingestion started")

        # Run data ingestion — fetches data from MongoDB, splits into train/test
        # Returns DataIngestionArtifact containing train & test file paths
        dataingestionartifact = data_ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion completed")

        # Print train and test file paths to verify
        print(dataingestionartifact.trained_file_path)
        print(dataingestionartifact.test_file_path)

        # Create data validation configuration using pipeline config
        # This holds settings like valid/invalid data paths, drift report path
        data_validation_config = DataValidationConfig(trainingpipelineconfig)

        logging.info("Data Validation started")

        # ✅ Fix — pass dataingestionartifact (not dataingestionconfig)
        # DataValidation needs the ARTIFACT (file paths)
        # not the CONFIG (settings)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)

        # Run data validation — validates columns, checks drift, saves valid data
        # Returns DataValidationArtifact with validation status and file paths
        data_validation_artifact = data_validation.initiate_data_validation()

        logging.info("Data Validation completed")

        # Print validation artifact to verify
        print("valid train file path: ",data_validation_artifact.valid_train_file_path)
        print ("valid test file path:",data_validation_artifact.valid_test_file_path)
        print("invalid train file path: ",data_validation_artifact.invalid_train_file_path)
        print("invalid test file path: ",data_validation_artifact.invalid_test_file_path)
    except Exception as e:
        raise CustomException(e, sys)