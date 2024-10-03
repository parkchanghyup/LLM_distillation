import yaml
import json
import glob
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Tuple


def load_file(file_path: str, loader_func: callable, encoding: str = 'utf-8') -> Dict:
    """
    Generic function to load a file using the specified loader function.

    Args:
        file_path (str): Path to the file.
        loader_func (callable): Function to use for loading the file (yaml.safe_load or json.load).
        encoding (str, optional): File encoding. Defaults to 'utf-8'.

    Returns:
        Dict: Loaded data.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If there's an error parsing the file.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return loader_func(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")


def load_yaml(yaml_path: str) -> Dict:
    """
    Load a YAML file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        Dict: Loaded YAML data.
    """
    return load_file(yaml_path, yaml.safe_load)


def load_json(json_path: str) -> Dict:
    """
    Load a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Dict: Loaded JSON data.
    """
    return load_file(json_path, json.load)


def load_documents(raw_data_path: str) -> List[str]:
    """
    Load documents from JSON files in the specified path.

    Args:
        raw_data_path (str): Path pattern to JSON files.

    Returns:
        List[str]: List of paragraphs from the JSON files.

    Raises:
        FileNotFoundError: If no JSON files are found.
        KeyError: If 'paragraph' key is not found in a JSON file.
    """
    json_files = glob.glob(raw_data_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found at path: {raw_data_path}")

    docs = []
    for json_path in json_files:
        data = load_json(json_path)
        if 'paragraph' not in data:
            raise KeyError(f"'paragraph' key not found in JSON file: {json_path}")
        docs.append(data['paragraph'])
    return docs


def load_train_dataset(summarize_data_path: str, test_size: float, train_prompt: str) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess the training dataset from a CSV file.

    Args:
        summarize_data_path (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        train_prompt (str): Template for formatting the training data.

    Returns:
        Tuple[Dataset, Dataset]: Train and test datasets.
    """
    df = pd.read_csv(summarize_data_path, encoding='cp949')

    def format_chat_template(row):
        row['text'] = train_prompt.format(row['document'], row['summary'])
        return row

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_chat_template, num_proc=4)

    print(dataset['text'][3])  # Consider removing or making this optional
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)

    return split_dataset['train'], split_dataset['test']