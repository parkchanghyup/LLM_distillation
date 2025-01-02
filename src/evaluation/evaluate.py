import pandas as pd
from vllm import LLM, SamplingParams
from data.data_loader import generate_inference_prompts

def main():
    # 설정
    model_path = "./Qwen2.5-1.5B-merged"  # 병합된 모델 경로
    input_file = "data/train/train.csv"
    output_file = "data/train/train_dpo.csv"

    # 데이터 로드
    df = pd.read_csv(input_file)
    docs = list(df['text'])

    # vLLM 모델 로드
    llm = LLM(model=model_path)
    tokenizer = llm.get_tokenizer()

    # Sampling 설정
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        repetition_penalty=1.05
    )

    # 프롬프트 생성
    prompts = generate_inference_prompts(docs, tokenizer)

    # 예측 수행
    outputs = llm.generate(prompts, sampling_params)
    results = [output.outputs[0].text for output in outputs]

    # 결과 저장
    df['qwen2.5_result'] = results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()