import yaml
import json
import glob

def load_yaml(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")

def load_documents(raw_data_path):
    docs = []
    json_files = glob.glob(raw_data_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found at path: {raw_data_path}")
    for json_path in json_files:
        data = load_json(json_path)
        if 'paragraph' not in data:
            raise KeyError(f"'paragraph' key not found in JSON file: {json_path}")
        docs.append(data['paragraph'])
    return docs