import os
import json
import logging
import argparse
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


def generate_data():
    """Gemini API를 사용하여 초기 학습 데이터 생성"""
    logger.info("데이터 생성 시작: Gemini API로 요약문 생성...")
    num_samples = int(os.environ.get("NUM_SAMPLES", 1000))
    train_df, test_df = load_and_sample_data(num_samples=num_samples)
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)

    # 데이터 저장
    llm_train_path = TRAIN_DATA_DIR / "llm_train.csv"
    test_path = TEST_DATA_DIR / "test.csv"
    train_df.to_csv(llm_train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"학습 데이터 저장 완료: {llm_train_path}")
    logger.info(f"테스트 데이터 저장 완료: {test_path}")
    return str(llm_train_path), str(test_path)


def train_teacher():
    """Teacher 모델 학습 """
    logger.info("Teacher 모델 학습 시작...")
    llm_train_path = TRAIN_DATA_DIR / "llm_train.csv"
    test_path = TEST_DATA_DIR / "test.csv"

    if not llm_train_path.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {llm_train_path}. 'generate_data'를 먼저 실행하세요.")

    teacher_model = train_teacher_model(str(llm_train_path), str(test_path))
    logger.info(f"Teacher 모델 학습 완료: {TEACHER_MODEL_MERGED_DIR}")
    return teacher_model


def teacher_inference():
    """Teacher 모델로 데이터에 대한 요약문 생성"""
    logger.info("Teacher 모델 추론 시작: vLLM으로 나머지 데이터 요약...")

    if not TEACHER_MODEL_MERGED_DIR.exists():
        raise FileNotFoundError(f"Teacher 모델이 없습니다: {TEACHER_MODEL_MERGED_DIR}. 'train_teacher'를 먼저 실행하세요.")

    vllm_model = load_model_vllm(str(TEACHER_MODEL_MERGED_DIR))
    remaining_df = load_and_sample_data(remaining=True)[:100]
    remaining_df = generate_summaries(remaining_df, model=vllm_model)
    slm_train_path = TRAIN_DATA_DIR / "slm_train.csv"
    remaining_df.to_csv(slm_train_path, index=False)
    logger.info(f"Teacher 모델 추론 완료: {slm_train_path}")
    return str(slm_train_path)


def train_student():
    """Student 모델 학습"""
    logger.info("Student 모델 학습 시작...")
    slm_train_path = TRAIN_DATA_DIR / "slm_train.csv"
    test_path = TEST_DATA_DIR / "test.csv"

    if not slm_train_path.exists():
        raise FileNotFoundError(f"SLM 학습 데이터가 없습니다: {slm_train_path}. 'teacher_inference'를 먼저 실행하세요.")

    train_student_model(str(slm_train_path), str(test_path))
    logger.info(f"Student 모델 학습 완료: {STUDENT_MODEL_MERGED_DIR}")


def evaluate():
    """Student 모델 평가"""
    logger.info("모델 평가 시작...")
    test_path = TEST_DATA_DIR / "test.csv"

    if not STUDENT_MODEL_MERGED_DIR.exists():
        raise FileNotFoundError(f"Student 모델이 없습니다: {STUDENT_MODEL_MERGED_DIR}. 'train_student'를 먼저 실행하세요.")

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

    return evaluation_results


def run_all_steps():
    """모든 단계를 순차적으로 실행"""
    logger.info("전체 LLM Distillation 파이프라인 실행 시작...")

    # 데이터 생성
    generate_data()

    # Teacher 모델 학습
    train_teacher()

    # Teacher 모델 추론
    teacher_inference()

    # Student 모델 학습
    train_student()

    # 평가
    evaluate()

    logger.info("전체 LLM Distillation 파이프라인 실행 완료!")


def main():
    parser = argparse.ArgumentParser(description="LLM Distillation Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        choices=["generate_data", "train_teacher", "teacher_inference", "train_student", "evaluate", "all"],
        help="실행할 단계를 선택하세요:\n"
             "generate_data: Gemini API로 초기 학습 데이터 생성\n"
             "train_teacher: Teacher 모델 학습 \n"
             "teacher_inference: Teacher 모델로 데이터 요약 생성\n"
             "train_student: Student 모델 학습\n"
             "evaluate: 모델 성능 평가\n"
             "all: 모든 단계 순차 실행"
    )

    args = parser.parse_args()

    # 환경 변수 검증 및 디렉토리 생성
    load_dotenv()
    validate_environment()
    ensure_directories()

    try:
        if args.step == "generate_data":
            generate_data()
        elif args.step == "train_teacher":
            train_teacher()
        elif args.step == "teacher_inference":
            teacher_inference()
        elif args.step == "train_student":
            train_student()
        elif args.step == "evaluate":
            evaluate()
        elif args.step == "all":
            run_all_steps()
        else:
            parser.print_help()
            logger.info("\n사용 예시:")
            logger.info("python main.py --step generate_data     # 데이터 생성")
            logger.info("python main.py --step train_teacher     # Teacher 모델 학습")
            logger.info("python main.py --step teacher_inference # Teacher 모델 추론")
            logger.info("python main.py --step train_student     # Student 모델 학습")
            logger.info("python main.py --step evaluate          # 모델 평가")
            logger.info("python main.py --step all               # 전체 실행")

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()