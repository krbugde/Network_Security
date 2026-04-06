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