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