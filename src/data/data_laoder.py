import os
import pandas as pd
from datasets import Dataset, DatasetDict

# 파일 경로 설정
def get_file_paths(data_root="./data", save_model_path="./models/trained_model"):
    """Get the appropriate file paths"""
    train_file = "train/train.csv"
    val_file = "test/gemini_test_result.csv"
    save_dir = save_model_path
    
    return {
        'train_path': os.path.join(data_root, train_file),
        'val_path': os.path.join(data_root, val_file),
        'save_path': save_dir
    }

# 데이터 로딩 함수
def load_data(paths):
    """Load training and validation datasets"""
    try:
        print(f"Loading training data from: {paths['train_path']}")
        print(f"Loading validation data from: {paths['val_path']}")
    
        train_data = pd.read_csv(paths['train_path'])
        val_data = pd.read_csv(paths['val_path'])
        
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# 학습용 프롬프트 생성 함수
def generate_prompts(examples, tokenizer):
    """Generate prompts for training dataset"""
    instructions = examples["text"]
    results = examples["results"]
    texts = []
    for t, r in zip(instructions, results):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": f"""Please summarize the documentation provided in 3 lines.
            Also, please extract the top five key phrases. See template for the answer format.
            The summary must be written in the same language as the body.
            <template>
            summary
            - summarize 1
            - summarize 2
            - summarize 3

            key phrases
            [key phrase1, key phrase2, key phrase3, key phrase4, key phrase5]
            </template>

            docs:
            {t}"""},
            {"role": "assistant", "content": f"{r}"}
        ]
        chat_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(chat_message)
    
    return {"text": texts}

# Inference용 프롬프트 생성 함수
def generate_inference_prompts(docs, tokenizer):
    """Generate prompts for inference"""
    system_prompt = """Please summarize the documentation provided in 3 lines.
    Also, please extract the top five key phrases. See template for the answer format.
    The summary must be written in Korean.
    <template>
    summary
    - summarize 1
    - summarize 2
    - summarize 3

    key phrases
    [key phrase1, key phrase2, key phrase3, key phrase4, key phrase5]
    </template>

    docs:
    {docs}"""
    
    texts = []
    for doc in docs:
        messages = [
            {'role': 'system', 'content': 'you are a helpful assistant'},
            {'role': 'user', 'content': system_prompt.format(docs=doc)}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    return texts