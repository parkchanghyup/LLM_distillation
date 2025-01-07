from tqdm import tqdm
from data.gemini_api import get_summary


def generate_summaries(train_df, test_df):
    """
    train/test 데이터프레임에 대해 요약 결과 생성
    """
    # Train 데이터 요약
    train_results = []
    for i in tqdm(range(len(train_df)), desc="Summarizing train texts"):
        result = get_summary(train_df['text'][i])
        train_results.append(result)
    train_df['results'] = train_results

    # Test 데이터 요약
    test_results = []
    for i in tqdm(range(len(test_df)), desc="Summarizing test texts"):
        result = get_summary(test_df['text'][i])
        test_results.append(result)
    test_df['results'] = test_results

    return train_df, test_df