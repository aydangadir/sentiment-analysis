import json
import os

def import_json():
    """
    Reads and parses 'config.json' into a dictionary.

    Returns:
        dict: Parsed JSON data from 'config.json'.
    """
    with open("config.json") as f:
        dictionary = json.load(f)
        
    return dictionary

def if_exists(path):
    return os.path.exists(path)

def create_folder(path):
    if if_exists(path):
        return
    
    os.mkdir(path)
