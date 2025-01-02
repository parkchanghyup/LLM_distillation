import json
import pandas as pd
import glob
import os
import tiktoken

# 기본 디렉토리 경로 (사용자 환경에 맞게 수정)
base_dir = os.path.expanduser('/Users/ariz1623/Desktop/github/LLM_distillation/data/raw')

# JSON 파일 패턴
json_pattern = os.path.join(base_dir, 'TL_그룹*', '*.json')
json_files = glob.glob(json_pattern, recursive=True)

# tiktoken 설정
encoding = tiktoken.get_encoding("cl100k_base")

# 토큰 수 제한
MIN_TOKEN_COUNT = 100
MAX_TOKEN_COUNT = 1000

# 데이터를 저장할 리스트 초기화
data_list = []

# 각 JSON 파일 읽기 및 필드 추출
for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                paragraph = item.get("paragraph", "")
                token_count = len(encoding.encode(paragraph))
                if MIN_TOKEN_COUNT <= token_count <= MAX_TOKEN_COUNT:
                    selected_item = {
                        "id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "field": item.get("field", ""),
                        "paragraph": paragraph,
                        "terminology": item.get("terminology", ""),
                        "source_file": os.path.basename(file_path),
                        "token_count": token_count  # 디버깅용
                    }
                    data_list.append(selected_item)
        else:
            paragraph = data.get("paragraph", "")
            token_count = len(encoding.encode(paragraph))
            if MIN_TOKEN_COUNT <= token_count <= MAX_TOKEN_COUNT:
                selected_item = {
                    "id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "field": data.get("field", ""),
                    "paragraph": paragraph,
                    "terminology": data.get("terminology", ""),
                    "source_file": os.path.basename(file_path),
                    "token_count": token_count  # 디버깅용
                }
                data_list.append(selected_item)

    except Exception as e:
        print(f"오류 발생 ({file_path}): {e}")

# 데이터프레임 생성
df = pd.DataFrame(data_list)

# 결과 출력
pd.set_option('display.max_colwidth', 5)
print(f"총 {len(df)}개의 항목이 로드되었습니다 (토큰 수 {MIN_TOKEN_COUNT} 이상 {MAX_TOKEN_COUNT} 이하).")
print(df)



# 하나의 파일로 통합 저장
output_path = os.path.join(base_dir, 'combined_filtered_data.csv')

df.to_csv(output_path, index=False)
print(f"필터링된 데이터가 {output_path}에 저장되었습니다.")

# 새로운 파일을 각각 생성 (필요 시 주석 해제)
# for file_path in json_files:
#     file_df = df[df['source_file'] == os.path.basename(file_path)]
#     if not file_df.empty:
#         output_file = file_path.replace('.json', '_filtered.json')
#         file_df.drop(columns=['source_file', 'token_count']).to_json(
#             output_file, orient='records', force_ascii=False
#         )
#         print(f"필터링된 데이터가 {output_file}에 저장되었습니다.")