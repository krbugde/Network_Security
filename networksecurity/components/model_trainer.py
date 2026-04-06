# Import os for file and directory operations
import os

# Import sys for system level error handling
import sys

# Import custom exception for detailed error messages
from networksecurity.exception.exception import CustomException

# Import custom logger to record events in log file
from networksecurity.logging.logger import logging

# Import artifact entities for input and output of this step
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,  # Input — contains transformed train/test file paths
    ModelTrainerArtifact          # Output — contains trained model file path and scores
)

# Import config entity for model trainer settings
from networksecurity.entity.config_entity import ModelTrainerConfig

# Import utility functions:
# save_object      → save any Python object as .pkl file
# load_object      → load any Python object from .pkl file
# load_numpy_array → load numpy array from .npy file
# evaluate_models  → train and evaluate multiple models using GridSearchCV
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array,
    evaluate_models
)

# Import function to calculate classification metrics
# (f1 score, precision, recall) for model evaluation
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

# Import NetworkModel — custom class that wraps preprocessor + model together
# So both preprocessing and prediction happen in one step
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Import all ML models we want to train and compare
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,         # Boosting — combines weak learners
    GradientBoostingClassifier, # Boosting — builds trees sequentially
    RandomForestClassifier      # Bagging — combines many decision trees
)


class ModelTrainer:

    # Constructor: initializes class with model config and transformation artifact
    # model_trainer_config          → settings like model file path, expected score
    # data_transformation_artifact  → contains transformed train/test numpy file paths
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            # Store model trainer configuration settings
            self.model_trainer_config = model_trainer_config

            # Store data transformation artifact to access transformed data paths
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    # Method to train multiple models and return the best one
    # x_train → input features for training
    # y_train → target labels for training
    # x_test  → input features for testing
    # y_test  → target labels for testing
    def train_model(self, x_train, y_train, x_test, y_test):

        # Dictionary of all ML models we want to train and compare
        # verbose=1 → prints progress during training
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier()
        }

        # Dictionary of hyperparameters for each model
        # GridSearchCV will try all combinations to find best params
        params = {
            "Decision Tree": {
                # criterion → how to measure quality of split
                'criterion': ['gini', 'entropy', 'log_loss'],
                # splitter → strategy to split at each node
                'splitter': ['best', 'random'],
                # max_features → number of features to consider for best split
                'max_features': ['sqrt', 'log2'],
            },
            "Random Forest": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                # n_estimators → number of trees in the forest
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                # loss → loss function to optimize
                'loss': ['log_loss', 'exponential'],
                # learning_rate → how much each tree contributes
                'learning_rate': [.1, .01, .05, .001],
                # subsample → fraction of samples used per tree
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'criterion': ['squared_error', 'friedman_mse'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            # Logistic Regression has no hyperparameters to tune here
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        # Evaluate all models using GridSearchCV and get report
        # Returns dictionary with model name → test score
        model_report: dict = evaluate_models(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params
        )

     # Select best model based on highest test R2 score
        best_model_name=max(model_report,key=lambda x:model_report[x]['score'])
        best_model_score=model_report[best_model_name]['score']
        best_model=model_report[best_model_name]['model']  # this is hyperparameter-tuned

        logging.info(f"Best model: {best_model_name} | R2 Score: {best_model_score}")

            # Check threshold for usability
        if best_model_score<0.6:
                 raise CustomException("No suitable model found")
            

        # Log the best model details
        logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

        # Get classification metrics for train data
        # (f1 score, precision, recall)
        y_train_pred = best_model.predict(x_train)
        classification_train_metric = get_classification_score(
            y_true=y_train,
            y_pred=y_train_pred
        )

        # Get classification metrics for test data
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test,
            y_pred=y_test_pred
        )

        # Load the preprocessor object saved during data transformation
        # So we can combine it with the model for end-to-end prediction
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # Create NetworkModel object combining preprocessor + best model
        # This allows raw data → preprocess → predict in one step
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        # Save the complete NetworkModel (preprocessor + model) as .pkl file
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=network_model
        )

        # Save best model separately as well
        save_object("final_model/model.pkl", obj=best_model)

        # Create and return ModelTrainerArtifact with results
        # This artifact is passed to the next step in pipeline
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    # Main method to start the complete model training process
    # Returns ModelTrainerArtifact with trained model path and scores
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # Get transformed train file path from transformation artifact
            train_file_path = self.data_transformation_artifact.transformed_train_file_path

            # Get transformed test file path from transformation artifact
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load transformed train numpy array from .npy file
            train_arr = load_numpy_array(train_file_path)

            # Load transformed test numpy array from .npy file
            test_arr = load_numpy_array(test_file_path)

            # Split arrays into input features and target labels
            # train_arr[:,:-1] → all columns except last = input features
            # train_arr[:,-1]  → only last column = target labels
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],  # Train input features
                train_arr[:, -1],   # Train target labels
                test_arr[:, :-1],   # Test input features
                test_arr[:, -1]     # Test target labels
            )

        
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)

            # Return the model trainer artifact to next pipeline step
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)