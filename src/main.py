import os
from dotenv import load_dotenv
from pathlib import Path
from data.data_loader import load_and_sample_data
from scripts.summarizer import generate_summaries
from data.gemini_api import get_summary
from models.trainer import train_model_a, train_model_b
from evaluation.evaluator import evaluate_models

# 모델 경로 설정
MODEL_A_PATH = Path("outputs/model_a/merged")
MODEL_B_PATH = Path("outputs/model_b/merged")

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "../data/raw",
        "../data/train",
        "../data/test",
        "outputs/model_a",
        "outputs/model_b"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

    create_directories()

    # 1. Gemini API를 사용하여 요약문 생성 (2000개 데이터)
    print("Step 1: Generating summaries using Gemini API...")
    train_df, test_df = load_and_sample_data(num_samples=2000)
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)
    
    # 데이터 저장
    train_df.to_csv("../data/train/llm_train.csv", index=False)
    test_df.to_csv("../data/test/test.csv", index=False)

    # 2. Model A 학습 (Q-LoRA)
    print("Step 2: Training Model A using Q-LoRA...")
    model_a = train_model_a("../data/train/llm_train.csv")

    # 3. 나머지 데이터에 대한 요약문 생성
    print("Step 3: Generating summaries for remaining data...")
    remaining_df = load_and_sample_data(remaining=True)
    remaining_df = generate_summaries(remaining_df, None, model_a.generate)
    remaining_df.to_csv("../data/train/slm_train.csv", index=False)

    # 4. Model B 학습
    print("Step 4: Training Model B...")
    model_b = train_model_b("../data/train/slm_train.csv")

    # 5. 평가
    print("Step 5: Evaluating models...")
    evaluation_results = evaluate_models(
        model_path=str(MODEL_B_PATH),
        test_data_path="../data/test/test.csv"
    )
    
    print("Evaluation Results:")
    print(evaluation_results)

if __name__ == "__main__":
    main()