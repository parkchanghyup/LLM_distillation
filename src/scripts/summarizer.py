import os
from tqdm import tqdm
from data.gemini_api import get_summary
from data.data_loader import load_and_sample_data
from utils.vllm_utils import generate_summary_vllm

BASE_PATH = '../data'


def generate_summaries(train_df, test_df=None, summary_func=None, model=None):
    """train/test 데이터프레임에 대해 요약 결과 생성

    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        test_df (pd.DataFrame, optional): 테스트 데이터프레임
        summary_func (callable, optional): 요약 함수 (Gemini API 사용시)
        model (vllm.LLM, optional): vLLM 모델 (vLLM 사용시)
    """
    train_results = []
    if model is not None:
        # vLLM 모델 사용
        train_results = generate_summary_vllm(model, list(train_df['text']))
    else:
        # Gemini API 또는 다른 요약 함수 사용
        for i in tqdm(range(len(train_df)), desc="Summarizing train texts"):
            result = summary_func(train_df['text'].iloc[i])
            train_results.append(result)
    train_df['results'] = train_results

    if test_df is not None:
        test_results = []
        if model is not None:
            test_results = generate_summary_vllm(model, list(test_df['text']))
        else:
            for i in tqdm(range(len(test_df)), desc="Summarizing test texts"):
                result = summary_func(test_df['text'].iloc[i])
                test_results.append(result)
        test_df['results'] = test_results
        return train_df, test_df

    return train_df


def save_results(train_df, test_df):
    """train과 test 결과를 CSV로 저장"""
    train_output_path = os.path.join(BASE_PATH, 'train/llm_train.csv')
    test_output_path = os.path.join(BASE_PATH, 'test/test.csv')
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"Train results saved to {train_output_path}")
    print(f"Test results saved to {test_output_path}")


def main(data_num=100, random_state=323):
    """Gemini API를 사용해 데이터 요약을 생성하고 저장"""
    # 데이터 로드 및 샘플링
    train_df, test_df = load_and_sample_data(data_num, random_state)
    # 요약 생성
    train_df, test_df = generate_summaries(train_df, test_df, get_summary)
    # 결과 저장
    save_results(train_df, test_df)


if __name__ == "__main__":
    main(data_num=100)  # 예시로 100개 샘플