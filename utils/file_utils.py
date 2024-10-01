import yaml
import json
import glob
import pandas as pd
from datasets import Dataset
from langchain.prompts import PromptTemplate

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
    df = pd.read_csv(summarize_data_path)
    prompt = PromptTemplate.from_template(train_prompt)
    texts = []
    for i in range(len(df)):
        document = df['document'][i]
        summary = df['sumaization'][i]  # 'summarization'의 오타로 보입니다. 필요시 수정해주세요.

        text = prompt.format(document=document, summary=summary)
        texts.append(text)

    # texts 리스트를 Dataset 객체로 변환
    dataset = Dataset.from_dict({"text": texts})

    # Dataset의 train_test_split 메서드를 사용하여 분할
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)

    return split_dataset['train'], split_dataset['test']

