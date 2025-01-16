import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate

def load_prompt_template(prompt_path="./prompts/general_prompt.txt"):
    """지정된 파일에서 프롬프트 템플릿을 로드합니다."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        return prompt_template
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        raise

def create_prompt_templates():
    """프롬프트 템플릿을 생성합니다."""
    prompt_text = load_prompt_template()
    training_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])
    inference_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])
    dpo_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="you are a helpful assistant"),
        HumanMessagePromptTemplate.from_template(prompt_text)
    ])
    return training_prompt_template, inference_prompt_template, dpo_prompt_template