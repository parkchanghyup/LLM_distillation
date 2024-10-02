import yaml
import json
import glob
import pandas as pd
from datasets import Dataset

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



def load_train_dataset(summarize_data_path, test_size, train_prompt):
    # CSV 파일 로드
    df = pd.read_csv(summarize_data_path, encoding='cp949')


    def format_chat_template(row):
        prompt = train_prompt.format(row['document'], row['summary'])
        row['text'] = prompt
        return row

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )

    print(dataset['text'][3])
    # Dataset의 train_test_split 메서드를 사용하여 분할
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)

    return split_dataset['train'], split_dataset['test']

