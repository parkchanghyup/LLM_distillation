from vllm import LLM, SamplingParams
from data.data_loader import generate_inference_prompts


def load_model_vllm(model_path):
    """
    vLLM을 사용하여 모델을 로드합니다.

    Args:
        model_path (str): 모델이 저장된 경로

    Returns:
        vllm.LLM: 로드된 vLLM 모델
    """
    try:
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1  # GPU 수에 따라 조정 가능
        )
        return model
    except Exception as e:
        raise RuntimeError(f"vLLM 모델 로드 중 오류 발생: {str(e)}")


def generate_summary_vllm(model, texts, max_tokens=2048) -> list[str]:
    """
    vLLM을 사용하여 텍스트 요약을 생성합니다.

    Args:
        model (vllm.LLM): vLLM 모델
        text (str): 요약할 텍스트
        max_tokens (int): 생성할 최대 토큰 수

    Returns:
        str: 생성된 요약문
    """
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=max_tokens
    )

    tokenizer = model.get_tokenizer()
    # 프롬프트 템플릿 적용
    prompts = []
    for text in texts:
        prompt = generate_inference_prompts([text], tokenizer)
        prompts.append(prompt[0])

    # 요약 생성
    outputs = model.generate(prompts, sampling_params)

    summaries = []

    for output in outputs:
        summary = output.outputs[0].text.strip()
        summaries.append(summary)

    return summaries