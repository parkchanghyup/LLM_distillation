import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()


def load_prompt_template(prompt_type="general"):
    """지정된 파일에서 프롬프트 템플릿을 로드합니다."""
    prompt_paths = {
        "general": ROOT_DIR / "prompts/general_prompt.txt",
        "evaluation": ROOT_DIR / "prompts/summary_evaluation_prompt.txt",
        "dpo": ROOT_DIR / "prompts/general_prompt.txt"
    }

    prompt_path = prompt_paths.get(prompt_type)
    if not prompt_path:
        logger.warning(f"Unknown prompt type: {prompt_type}")
        return "Please summarize the following text:\n\n{docs}"

    try:
        if not prompt_path.exists():
            # 파일이 없으면 기본 프롬프트 반환
            logger.warning(f"Warning: Prompt file not found at {prompt_path}. Using default prompt.")
            if prompt_type == "dpo":
                return "Please provide a high-quality summary of the following text:\n\n{docs}"
            return "Please summarize the following text:\n\n{docs}"

        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        return prompt_template
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        # 오류 발생 시 기본 프롬프트 반환
        return "Please summarize the following text:\n\n{docs}"


def create_prompt_templates():
    """프롬프트 템플릿을 생성합니다."""
    training_prompt_text = load_prompt_template("general")
    inference_prompt_text = load_prompt_template("general")
    dpo_prompt_text = load_prompt_template("dpo")

    training_prompt_template = ChatPromptTemplate.from_template(training_prompt_text)
    inference_prompt_template = ChatPromptTemplate.from_template(inference_prompt_text)
    dpo_prompt_template = ChatPromptTemplate.from_template(dpo_prompt_text)

    # training_prompt_template = ChatPromptTemplate.from_messages([
    #     HumanMessagePromptTemplate.from_template(training_prompt_text)
    # ])
    # inference_prompt_template = ChatPromptTemplate.from_messages([
    #     HumanMessagePromptTemplate.from_template(inference_prompt_text)
    # ])
    # dpo_prompt_template = ChatPromptTemplate.from_messages([
    #     SystemMessage(content="you are a helpful assistant"),
    #     HumanMessagePromptTemplate.from_template(dpo_prompt_text)
    # ])
    return training_prompt_template, inference_prompt_template, dpo_prompt_template