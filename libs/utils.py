import yaml
import logging
import os
import pandas as pd
import multiprocessing
import shutil
import time  # For simulating time-consuming functions
import sys
import random
import string
from datetime import datetime, timedelta

def generate_dirs() :
    """
        Generate relevant dirs
    """
    os.makedirs("input/", exist_ok = True)
    os.makedirs("output/", exist_ok = True)

def load_config_file(config_path : str) :
    """
        Helper function to load configuration yaml file 

        -----
        Args :
            config_path (str) : the path of our config file

        ------
        Returns :
            (dict) : the dictionary of our config key-value mappings
    """

    with open(config_path, "r") as file_obj :
        return yaml.load(file_obj, Loader=yaml.SafeLoader)

# Function to configure a logger
def configure_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger