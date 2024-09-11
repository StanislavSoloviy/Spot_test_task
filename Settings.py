import json
from copy import deepcopy


def save_dataset_path(path, settings_path):
    data = {}
    with open(settings_path, "w") as file:
        data["dataset_path"] = path
        json.dump(data, file)


def load_dataset_path(settings_path):
    try:
        with open(settings_path, "r") as file:
            path = deepcopy(json.load(file)["dataset_path"])
    except:
        path = "./test_data"
    return path


