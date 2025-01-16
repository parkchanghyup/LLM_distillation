import os
from dotenv import load_dotenv
from data.data_loader import load_and_sample_data
from scripts.summarizer import generate_summaries, save_results
from data.gemini_api import get_summary

def main():
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

    # 데이터 로드 및 샘플링
    data_num = int(os.environ.get("GEMINI_DATA_NUM", 100))  # 기본값 1000 설정
    train_df, test_df = load_and_sample_data(data_num)

    # 요약 생성
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)  # summary_func로 get_summary 전달

    # 결과 저장
    save_results(train_df, test_df)

if __name__ == "__main__":
    main()