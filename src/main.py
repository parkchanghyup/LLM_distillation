import os
from dotenv import load_dotenv
from data.data_loader import load_and_sample_data, save_results
from data.preprocess import generate_summaries


def main():
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

    train_df, test_df = load_and_sample_data(int(os.environ.get("GEMINI_DATA_NUM")))
    train_df, test_df = generate_summaries(train_df, test_df)  # Gemini API 호출 포함
    save_results(train_df, test_df)


if __name__ == "__main__":
    main()