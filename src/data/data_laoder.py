from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")
client = genai.Client(api_key=API_KEY)


def get_summary(text):
    """주어진 텍스트를 요약하고 주요 문구를 추출하는 함수"""
    prompt = f"""Please summarize the documentation provided in 3 lines.
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
    {text}
    """
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text