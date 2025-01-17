import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from utils.prompt_utils import create_prompt_templates

# 로깅 설정 (train_sft.py와 통일)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 파일 경로 설정
def get_file_paths(config=None, data_root="./data", save_model_path="./models/trained_model"):
    """적절한 파일 경로를 반환합니다. config가 제공되면 yaml 설정을 우선 사용."""
    if config and "data" in config and "train_path" in config["data"] and "val_path" in config["data"]:
        return {
            'train_path': config["data"]["train_path"],
            'val_path': config["data"]["val_path"],
            'save_path': save_model_path
        }
    # 기본 경로 (config가 없거나 필드가 누락된 경우)
    train_file = "train/train.csv"
    val_file = "test/gemini_test_result.csv"
    return {
        'train_path': os.path.join(data_root, train_file),
        'val_path': os.path.join(data_root, val_file),
        'save_path': save_model_path
    }

# 데이터 로딩 함수
def load_data(paths):
    """훈련 및 검증 데이터셋을 로드합니다."""
    try:
        logger.info(f"Loading training data from: {paths['train_path']}")
        logger.info(f"Loading validation data from: {paths['val_path']}")
        train_data = pd.read_csv(paths['train_path'])
        val_data = pd.read_csv(paths['val_path'])
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# 학습용 프롬프트 생성 함수
def generate_prompts(examples, tokenizer):
    """훈련 데이터셋용 프롬프트를 생성합니다."""
    training_template, _, _ = create_prompt_templates()
    instructions = examples["text"]
    results = examples["results"]
    texts = []
    for t, r in zip(instructions, results):
        formatted_prompt = training_template.format(docs=t)
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": formatted_prompt.messages[1].content},
            {"role": "assistant", "content": f"{r}"}
        ]
        chat_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(chat_message)
    return {"text": texts}

# Inference용 프롬프트 생성 함수
def generate_inference_prompts(docs, tokenizer):
    """추론을 위한 프롬프트를 생성합니다."""
    _, inference_template, _ = create_prompt_templates()
    texts = []
    for doc in docs:
        formatted_prompt = inference_template.format(docs=doc)
        messages = [
            {'role': 'system', 'content': 'you are a helpful assistant'},
            {'role': 'user', 'content': formatted_prompt.messages[1].content}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    return texts

# DPO용 프롬프트 생성 함수
def generate_prompt_dpo(row, tokenizer):
    """DPO를 위한 프롬프트를 생성합니다."""
    _, _, dpo_template = create_prompt_templates()
    text = row['text']
    text_chosen = row['text_chosen']
    text_reject = row['text_reject']
    formatted_prompt = dpo_template.format(docs=text)
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": formatted_prompt.messages[1].content}
    ]
    message_chosen = [{"role": "assistant", "content": f"{text_chosen}"}]
    message_reject = [{"role": "assistant", "content": f"{text_reject}"}]
    row['prompt'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    row['chosen'] = tokenizer.apply_chat_template(message_chosen, tokenize=False, add_generation_prompt=False)
    row['chosen'] = row['chosen'].replace(
        '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n', ''
    )
    row['rejected'] = tokenizer.apply_chat_template(message_reject, tokenize=False, add_generation_prompt=False)
    row['rejected'] = row['rejected'].replace(
        '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n', ''
    )
    return row

# Gemini 데이터 생성 및 요약 추가
BASE_PATH = '../data'

def load_and_sample_data(data_num, random_state=323):
    """데이터를 로드하고 샘플링한 뒤 train/test로 분할"""
    df = pd.read_csv(os.path.join(BASE_PATH, 'raw/train.csv'))
    df_sample = df.sample(n=data_num, random_state=random_state).reset_index(drop=True)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def generate_summaries(train_df, test_df, summary_func):
    """train/test 데이터프레임에 대해 요약 결과 생성"""
    train_results = []
    for i in tqdm(range(len(train_df)), desc="Summarizing train texts"):
        result = summary_func(train_df['text'][i])
        train_results.append(result)
    train_df['results'] = train_results

    test_results = []
    for i in tqdm(range(len(test_df)), desc="Summarizing test texts"):
        result = summary_func(test_df['text'][i])
        test_results.append(result)
    test_df['results'] = test_results

    return train_df, test_df

def save_results(train_df, test_df):
    """train과 test 결과를 CSV로 저장"""
    train_output_path = os.path.join(BASE_PATH, 'gemini_summary/train_result.csv')
    test_output_path = os.path.join(BASE_PATH, 'gemini_summary/test_result.csv')
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Train results saved to {train_output_path}")
    logger.info(f"Test results saved to {test_output_path}")

# 실행 예시
if __name__ == "__main__":
    from gemini_api import get_summary
    train_df, test_df = load_and_sample_data(data_num=100)  # 예시로 100개 샘플
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)
    save_results(train_df, test_df)