# Knolwedge  distilation을 통한 sLLM 최적화

이 프로젝트는 대형 언어 모델(LLM)을 활용하여 학습 데이터 셋을 생성하고하고 해당 데이터 셋으로 더 작은 모델을 학습시키는 효율적인 접근 방식을 보여줍니다.   
이는 리소스가 제한된 환경에서 고성능 LLM을 사용하는 과제를 해결하기 위해, 더 큰 모델로 고품질 학습 데이터를 생성하고 이를 사용하여 더 작고 배포가 용이한 모델을 최적화하는 방법을 다룹니다.

## 프로젝트 개요

이 프로젝트의 주요 구성 요소는 다음과 같습니다:

1. **문서 요약**: 대형 모델(예: Gemma2-27B)을 사용하여 입력 문서의 요약을 생성합니다.
2. **모델 학습**: 생성된 요약을 바탕으로 더 작은 모델(예: Gemma2-2B)을 학습하여 성능을 향상시킵니다.
3. **학습 모델 테스트**: 학습된 모델을 사용하여 추론 작업(요약)을 수행합니다.


## Requirements

- Python 3.10+

## Installation

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/yourusername/llm-summarization-training.git
   cd llm-summarization-training
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```
   
3. 데이터를 다운받아 `data/raw` 폴더에 파일별로 저장해줍니다.(폴더 단위 x)
- [AI Hub - 법률안 검토 보고서 요약 데이터]("https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71794")

## Usage

메인 스크립트 `main.py`는 세 가지 주요 작업을 지원합니다:

1. **요약**:
   ```
   python main.py --summarize --config config/config.yaml
   ```

2. **학습** (현재 구현 중):
   ```
   python main.py --train --config config/config.yaml
   ```

3. **학습 모델 테스트** (현재 구현 중):
   ```
   python main.py --infer --config config/config.yaml
   ```

## Configuration

이 프로젝트는 두 개의 주요 구성 파일을 사용합니다:

1. `config/config.yaml`: 데이터 경로, 모델 구성 및 학습 매개변수에 대한 설정을 포함합니다.
2. `config/prompts.yaml`: 요약 생성 및 기타 텍스트 생성 작업에 사용되는 프롬프트를 저장합니다.

## Project Structure

- `src/`: 주요 소스 코드 포함
  - `summarization/`: 요약 관련 코드
  - `model/`: 모델 학습 및 추론 코드
- `utils/`: 파일 작업 등을 위한 유틸리티 함수
- `config/`: 구성 파일
- `data/`: 데이터 디렉토리 (저장소에 포함되지 않음)
