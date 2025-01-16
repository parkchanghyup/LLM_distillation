import os
from google import genai
from dotenv import load_dotenv
from utils.prompt_utils import create_prompt_templates

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 로드
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not MODEL_NAME:
    raise ValueError("GEMINI_MODEL_NAME 환경 변수가 설정되지 않았습니다.")

client = genai.Client(api_key=API_KEY)

def get_summary(text):
    """주어진 텍스트를 요약하고 주요 문구를 추출하는 함수"""
    training_template, _, _ = create_prompt_templates()
    formatted_prompt = training_template.format(docs=text)
    response = client.models.generate_content(model=MODEL_NAME, contents=formatted_prompt)
    return response.text