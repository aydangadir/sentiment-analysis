import json

def import_json():
    """
    Reads and parses 'config.json' into a dictionary.

    Returns:
        dict: Parsed JSON data from 'config.json'.
    """
    with open("config.json") as f:
        dictionary = json.load(f)
        
    return dictionary