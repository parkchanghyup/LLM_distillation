import pandas as pd
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
from data.data_loader import generate_inference_prompts
from data.gemini_api import get_summary
from utils.config_utils import PROMPT_DIR

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_prompt() -> str:
    """요약 평가를 위한 프롬프트 템플릿 로드"""
    prompt_path = PROMPT_DIR / "summary_evaluation_prompt.txt"

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_summary_with_gemini(reference: str, generated: str, evaluation_prompt: str) -> dict:
    """Gemini API를 사용하여 생성된 요약을 평가"""
    formatted_prompt = evaluation_prompt.format(
        reference_summary=reference,
        generated_summary=generated
    )
    evaluation_result = get_summary(formatted_prompt)
    return evaluation_result


def evaluate_models(model_path: str, test_data_path: str) -> None:
    """Model B와 Gemini API의 성능을 비교 평가

    Args:
        model_path: 학습된 Model B의 저장 경로
        test_data_path: 테스트 데이터 CSV 파일 경로
    """
    logger.info("모델 평가 시작")

    # 평가용 프롬프트 로드
    evaluation_prompt = load_evaluation_prompt()

    # 테스트 데이터 로드
    test_df = pd.read_csv(test_data_path)

    # vLLM 모델 로드
    logger.info(f"vLLM 모델 로드 중: {model_path}")
    llm = LLM(model=model_path)
    tokenizer = llm.get_tokenizer()

    # Sampling 설정
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        repetition_penalty=1.05
    )

    # 프롬프트 생성
    logger.info("추론용 프롬프트 생성 중")
    prompts = generate_inference_prompts(test_df['text'].tolist(), tokenizer)

    # 모델 B로 추론 수행
    logger.info("Model B로 추론 수행 중")
    outputs = llm.generate(prompts, sampling_params)
    sft_results = [output.outputs[0].text for output in outputs]
    test_df['sft_result'] = sft_results

    # Gemini API로 평가 수행
    logger.info("Gemini API로 평가 수행 중")
    evaluations = []
    for _, row in test_df.iterrows():
        # Gemini 생성 결과 평가
        gemini_eval = evaluate_summary_with_gemini(
            reference=row['results'],
            generated=row['sft_result'],
            evaluation_prompt=evaluation_prompt
        )
        evaluations.append(gemini_eval)

    # 평가 결과 저장
    test_df['evaluation'] = evaluations
    output_path = test_data_path.replace('.csv', '_evaluated.csv')
    test_df.to_csv(output_path, index=False)
    logger.info(f"평가 결과 저장 완료: {output_path}")

    return test_df 