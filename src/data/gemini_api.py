import os
import pandas as pd
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from data.data_loader import load_prompt_template, create_prompt_templates

# .env 파일 로드 (프로젝트 루트 기준)
load_dotenv()

# 환경 변수에서 API 키 로드
API_KEY = os.environ.get("GEMINI_API_KEY")
DATA_NUM = int(os.environ.get("GEMINI_DATA_NUM"))
MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")
client = genai.Client(api_key=API_KEY)

# 데이터 경로 및 샘플 크기 설정
BASE_PATH = '../data'

def get_summary(text):
    """주어진 텍스트를 요약하고 주요 문구를 추출하는 함수"""

    training_template, _, _ = create_prompt_templates()  # training_template만 사용

    # LangChain 템플릿 포맷에 맞게 변환
    formatted_prompt = training_template.format(docs=text)


    response = client.models.generate_content(model=MODEL_NAME, contents=formatted_prompt)
    return response.text

def main():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 데이터 로드 및 전처리
    df = pd.read_csv(os.path.join(BASE_PATH, 'raw/combined_filtered_data.csv'))

    # 무작위 샘플링
    df_sample = df.sample(n=DATA_NUM, random_state=323).reset_index(drop=True)
    
    # train과 test 데이터로 분할 (80% train, 20% test)
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=323)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # train 데이터에 대한 요약 결과 생성
    train_results = []
    for i in tqdm(range(len(train_df)), desc="Summarizing train texts"):
        result = get_summary(train_df['text'][i])
        train_results.append(result)
    
    # test 데이터에 대한 요약 결과 생성
    test_results = []
    for i in tqdm(range(len(test_df)), desc="Summarizing test texts"):
        result = get_summary(test_df['text'][i])
        test_results.append(result)
    
    # 결과 추가
    train_df['results'] = train_results
    test_df['results'] = test_results
    
    # train과 test 결과 저장
    train_output_path = os.path.join(BASE_PATH, 'gemini_summary/gemini_train_result.csv')
    test_output_path = os.path.join(BASE_PATH, 'gemini_summary/gemini_test_result.csv')
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"Train results saved to {train_output_path}")
    print(f"Test results saved to {test_output_path}")

if __name__ == "__main__":
    main()