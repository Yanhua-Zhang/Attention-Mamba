import json


# -----------------------------------------------------------------------------

def read_json(file_path):

    # Opening the JSON file
    with open(file_path, 'r') as file:
        # Parsing the file content into a Python object
        data = json.load(file)

    return data


def write_json(file_path, input_data):

    # Writing to the JSON file
    with open(file_path, 'w') as file:
        json.dump(input_data, file, indent=4)