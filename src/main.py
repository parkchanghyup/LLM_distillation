import os
import json
import logging
from dotenv import load_dotenv
from data.data_loader import load_and_sample_data
from scripts.summarizer import generate_summaries
from data.gemini_api import get_summary
from models.trainer import train_teacher_model, train_student_model
from evaluation.evaluator import evaluate_models
from utils.vllm_utils import load_model_vllm
from utils.config_utils import (
    ROOT_DIR, TEACHER_MODEL_MERGED_DIR, STUDENT_MODEL_MERGED_DIR,
    TRAIN_DATA_DIR, TEST_DATA_DIR, ensure_directories
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("distillation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_environment():
    """필요한 모든 환경 변수를 검증합니다."""
    required_vars = ["GEMINI_API_KEY", "GEMINI_MODEL_NAME", "NUM_SAMPLES"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        raise ValueError(f"다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")


def save_evaluation_results(results, output_path):
    """평가 결과를 JSON 파일로 저장합니다."""
    try:
        # 데이터프레임을 JSON으로 변환
        if hasattr(results, 'to_json'):
            results_json = results.to_json(orient='records')
            results_dict = json.loads(results_json)
        else:
            results_dict = results

        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"평가 결과가 저장되었습니다: {output_path}")
    except Exception as e:
        logger.error(f"평가 결과 저장 중 오류 발생: {e}")


def main():
    load_dotenv()
    validate_environment()

    # 필요한 디렉토리 생성
    ensure_directories()

    # 1. Gemini API를 사용하여 요약문 생성
    logger.info("Step 1: Generating summaries using Gemini API...")
    num_samples = int(os.environ.get("NUM_SAMPLES", 1000))  # 환경변수에서 값을 가져오고, 기본값은 1000
    train_df, test_df = load_and_sample_data(num_samples=num_samples)
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)

    # 데이터 저장
    llm_train_path = TRAIN_DATA_DIR / "llm_train.csv"
    test_path = TEST_DATA_DIR / "test.csv"
    train_df.to_csv(llm_train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"학습 데이터 저장 완료: {llm_train_path}")
    logger.info(f"테스트 데이터 저장 완료: {test_path}")

    # 2. Teacher 모델 학습 (Q-LoRA)
    logger.info("Step 2: Training Teacher model using Q-LoRA...")
    teacher_model = train_teacher_model(str(llm_train_path), str(test_path))  # 테스트 데이터를 평가 데이터로 사용
    logger.info(f"Teacher 모델 학습 완료: {TEACHER_MODEL_MERGED_DIR}")

    # 3. vLLM을 사용하여 나머지 데이터에 대한 요약문 생성
    logger.info("Step 3: Loading Teacher model with vLLM and generating summaries for remaining data...")
    vllm_model = load_model_vllm(str(TEACHER_MODEL_MERGED_DIR))
    remaining_df = load_and_sample_data(remaining=True)[:100]
    remaining_df = generate_summaries(remaining_df, model=vllm_model)
    slm_train_path = TRAIN_DATA_DIR / "slm_train.csv"
    remaining_df.to_csv(slm_train_path, index=False)
    logger.info(f"나머지 데이터 요약 및 저장 완료: {slm_train_path}")

    # 4. Student 모델 학습
    logger.info("Step 4: Training Student model...")
    train_student_model(str(slm_train_path), str(test_path))  # 테스트 데이터를 평가 데이터로 사용
    logger.info(f"Student 모델 학습 완료: {STUDENT_MODEL_MERGED_DIR}")

    # 5. 평가
    logger.info("Step 5: Evaluating models...")
    evaluation_results = evaluate_models(
        model_path=str(STUDENT_MODEL_MERGED_DIR),
        test_data_path=str(test_path)
    )

    # 평가 결과 저장
    eval_output_path = ROOT_DIR / "outputs/evaluation/model_evaluation_results.json"
    save_evaluation_results(evaluation_results, eval_output_path)

    # 평가 결과 요약 출력
    logger.info("평가 결과 요약:")
    if hasattr(evaluation_results, 'describe'):
        logger.info(evaluation_results.describe())
    else:
        logger.info(evaluation_results)


if __name__ == "__main__":
    main()