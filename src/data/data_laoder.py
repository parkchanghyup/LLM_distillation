import os
import pandas as pd
from datasets import Dataset, DatasetDict
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage


# 파일 경로 설정
def get_file_paths(data_root="./data", save_model_path="./models/trained_model"):
    """
    적절한 파일 경로를 반환합니다.
    """
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
    """
    훈련 및 검증 데이터셋을 로드합니다.
    """
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


def load_prompt_template(prompt_path="./prompts/prompt.txt"):
    """
    지정된 파일에서 프롬프트 템플릿을 로드합니다.
    프롬프트 파일을 로드하지 못할 경우 프로세스를 종료합니다.
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        return prompt_template
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        print(f"Critical error: Unable to load prompt file from {prompt_path}")
        raise


# 프롬프트 템플릿 생성
def create_prompt_templates():
    """
    프롬프트 템플릿을 생성합니다.
    """
    # 프롬프트 템플릿 로드
    prompt_text = load_prompt_template()

    # 훈련용 프롬프트 템플릿
    training_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])

    inference_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])

    # DPO용 프롬프트 템플릿
    dpo_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])

    return training_prompt_template, inference_prompt_template, dpo_prompt_template
합

# 학습용 프롬프트 생성 함수
def generate_prompts(examples, tokenizer):
    """
    훈련 데이터셋용 프롬프트를 생성합니다.
    """
    training_template, _, _ = create_prompt_templates()

    instructions = examples["text"]
    results = examples["results"]
    texts = []

    for t, r in zip(instructions, results):
        # LangChain 템플릿 포맷에 맞게 변환
        formatted_prompt = training_template.format(docs=t)

        # 메시지 형식 구성
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
    """
    추론을 위한 프롬프트를 생성합니다.
    """
    _, inference_template, _ = create_prompt_templates()

    texts = []
    for doc in docs:
        # LangChain 템플릿 포맷에 맞게 변환
        formatted_prompt = inference_template.format(docs=doc)

        messages = [
            {'role': 'system', 'content': 'you are a helpful assistant'},
            {'role': 'user', 'content': formatted_prompt.messages[1].content}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    return texts


def generate_prompt_dpo(row, tokenizer):
    """
    DPO를 위한 프롬프트를 생성합니다.
    """
    _, _, dpo_template = create_prompt_templates()

    text = row['text']
    text_chosen = row['text_chosen']
    text_reject = row['text_reject']

    # LangChain 템플릿 포맷에 맞게 변환
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