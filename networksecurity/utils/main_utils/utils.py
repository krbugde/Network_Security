from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import yaml
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    


# Function to write content into a YAML file
# file_path → where to save the YAML file
# content   → data to write (dictionary, list, etc.)
# replace   → if True, delete old file and write fresh
#           → if False, keep old file and just update (default)
# → None means this function returns nothing
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        # Check if user wants to replace the existing file
        if replace == True:

            # Check if the file already exists at given path
            if os.path.exists(file_path):

                # Delete the existing file so we can write fresh
                os.remove(file_path)

        # Create the folder/directory if it doesn't exist
        # os.path.dirname() → extracts only the folder path from file_path
        # exist_ok=True → don't throw error if folder already exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file in write mode ("w")
        # "w" → creates new file if not exists, overwrites if exists
        # 'file' is the file object we write into
        with open(file_path, "w") as file:

            # Convert Python dictionary/object into YAML format
            # and write it into the file
            yaml.dump(content, file)

            '''file.write() only accepts plain text/string
                It has no idea how to convert a dictionary into YAML format
                ✅ If you use yaml.dump
                python# yaml.dump automatically:
                # 1. Converts dictionary → proper YAML format
                # 2. Writes it into the file
                yaml.dump(content, file)  # ✅ Works perfectly!'''

    except Exception as e:
        # If any error occurs, raise detailed CustomException
        raise CustomException(e, sys)
    




# Function to save a numpy array to a file
# file_path → location where the numpy array will be saved
# array     → the numpy array we want to save
# np.array  → type hint indicating input should be a numpy array
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        # Extract only the folder path from the full file path
        # Example: "artifacts/data/train.npy" → "artifacts/data"
        dir_path = os.path.dirname(file_path)

        # Create the folder if it doesn't already exist
        # exist_ok=True → no error if folder already exists
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write binary mode ("wb")
        # "wb" → write binary — needed because numpy saves in binary format
        # not plain text, so we use "wb" not "w"
        with open(file_path, "wb") as file_obj:

            # Save the numpy array into the file in .npy binary format
            # np.save() converts array → binary and writes to file
            np.save(file_obj, array)

    except Exception as e:
        # If any error occurs, raise detailed CustomException
        # showing file name, line number and error message
        raise CustomException(e, sys)
    



# Function to save any Python object to a file using pickle
# file_path → location where the object will be saved
# obj       → any Python object to save
#             (model, transformer, encoder, scaler, etc.)
# -> None   → this function returns nothing
def save_object(file_path: str, obj: object) -> None:
    try:
        # Log that we have entered this function
        logging.info("Entered the save object method of MainUtils class")

        # Extract folder path from full file path and create it if not exists
        # Example: "artifacts/model/model.pkl" → "artifacts/model"
        # exist_ok=True → no error if folder already exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file in write binary mode ("wb")
        # "wb" → write binary — needed because pickle saves in binary format
        with open(file_path, "wb") as file_obj:

            # Serialize the Python object and save it to the file
            # pickle.dump() converts any Python object → binary format
            # and writes it into the file
            # Example: saves ML model, scaler, encoder as .pkl file
            pickle.dump(obj, file_obj)

        # Log that we have successfully exited this function
        logging.info("Exited save object method of MainUtils class")

    except Exception as e:
        # If any error occurs, raise detailed CustomException
        # showing file name, line number and error message
        raise CustomException(e, sys)
    

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exists")
        with open(file_path,"rb")as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_numpy_array(file_path:str)->np.array:
    try:
        with open(file_path,"rb")as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains multiple models, performs hyperparameter tuning using GridSearchCV,
    evaluates them on test data, and returns a dictionary mapping:
    
    model_name -> {"score": test_R2_score, "model": trained_model_object}
    
    Parameters:
        X_train, y_train : np.array : Training data
        X_test, y_test   : np.array : Test data
        models           : dict : {model_name: model_object}
        param            : dict : {model_name: hyperparameter_grid}
    
    Returns:
        dict : {model_name: {"score": test_R2_score, "model": trained_model_object}}
    """

    try:
        # empty dictionary to store each model's name → score + trained model object
        report = {}

        # loop through each model name and model object directly using .items()
        # cleaner than the previous index-based loop (no need for list()[i])
        for model_name, model in models.items():

            # log which model is currently being trained and tuned
            logging.info(f"Training and tuning model: {model_name}")

            # create GridSearchCV to try all hyperparameter combinations
            # estimator  → the model to tune
            # param_grid → hyperparameter options for this specific model
            # cv=3       → 3-fold cross validation to evaluate each combination
            # n_jobs=-1  → use all CPU cores to speed up the search
            # verbose=0  → don't print anything during search (silent mode)
            grid = GridSearchCV(
                estimator=model,
                param_grid=param[model_name],
                cv=3,
                n_jobs=-1,
                
            )

            # run the grid search on training data to find best hyperparameters
            grid.fit(X_train, y_train)

            # get the best model directly from GridSearchCV
            # best_estimator_ → already trained model with best hyperparameters found
            best_model = grid.best_estimator_

            # predict on test data using the best trained model
            y_test_pred = best_model.predict(X_test)

            # calculate R² score on test data
            # R² close to 1.0 → model predicts very well
            # R² close to 0.0 → model is no better than guessing the mean
            test_score = r2_score(y_test, y_test_pred)

            # save BOTH the score AND the trained model object in the report
            # this is BETTER than the previous version which only saved the score
            # now we can directly use the model object later without retraining
            # e.g., report["Random Forest"] = {"score": 0.91, "model": <trained model>}
            report[model_name] = {"score": test_score, "model": best_model}

            # log the model name and its test score for tracking
            logging.info(f"{model_name} | Test R2 Score: {test_score}")

        # return the complete report with all model names, scores and model objects
        return report

    except Exception as e:
        # if anything goes wrong, raise a clear custom error with traceback info
        raise CustomException(e, sys)