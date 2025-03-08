# Knowledge Distillation을 통한 sLLM 최적화

이 프로젝트는 대형 언어 모델(LLM)을 활용하여 학습 데이터 셋을 생성하고 해당 데이터 셋으로 더 작은 모델을 학습시키는 효율적인 접근 방식을 보여줍니다.   
이는 리소스가 제한된 환경에서 고성능 LLM을 사용하는 과제를 해결하기 위해, 더 큰 모델로 고품질 학습 데이터를 생성하고 이를 사용하여 더 작고 배포가 용이한 모델을 최적화하는 방법을 다룹니다.

## 프로젝트 아키텍쳐

## 프로젝트 개요

이 프로젝트의 주요 구성 요소는 다음과 같습니다.

1. **문서 요약**: 대형 모델(예: Gemini)을 사용하여 입력 문서의 요약을 생성합니다.
2. **모델 학습**: 생성된 요약을 바탕으로 더 작은 모델을 학습하여 성능을 향상시킵니다.
3. **학습 모델 테스트**: 학습된 모델을 사용하여 추론 작업(요약)을 수행하고 평가합니다.

## Requirements

- Python 3.10+
- 필요한 환경 변수: GEMINI_API_KEY, GEMINI_MODEL_NAME, NUM_SAMPLES

## Installation

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/yourusername/LLM_distillation.git
   cd LLM_distillation
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```
   
3. 데이터
- 데이터에서 text로 사용하려는 컬럼은 컬럼명을 `text`로 지정하여야 합니다.

4. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 변수를 설정합니다:
   ```
   GEMINI_API_KEY=your_api_key
   GEMINI_MODEL_NAME=gemini-flash-2.0
   NUM_SAMPLES=1000
   ```

## Usage

메인 스크립트 `src/main.py`를 실행하여 전체 파이프라인을 실행할 수 있습니다:

```
python src/main.py
```

이 스크립트는 다음 단계를 수행합니다:
1. Gemini API를 사용하여 요약문 생성
2. Model A 학습 (Q-LoRA)
3. vLLM을 사용하여 나머지 데이터에 대한 요약문 생성
4. Model B 학습
5. 모델 평가

## 프로젝트 구조

```
LLM_distillation/
├── configs/                  # 구성 파일
│   ├── student_model.yaml    # student 모델  학습 구성
│   ├── teacher_model.yaml    # teacher 모델  학습 구성
│   ├── dpo.yaml              # DPO 학습 구성
├── data/                     # 데이터 디렉토리
│   ├── raw/                  # 원본 데이터
│   ├── train/                # 학습 데이터
│   ├── test/                 # 테스트 데이터
│   ├── gemini_summary/       # Gemini로 생성된 요약
│   └── model_outputs/        # 모델 출력 결과
├── models/                   # 저장된 모델 파일
├── prompts/                  # 프롬프트 템플릿
│   ├── general_prompt.txt    # 일반 프롬프트
│   └── summary_evaluation_prompt.txt  # 요약 평가 프롬프트
├── src/                      # 소스 코드
│   ├── data/                 # 데이터 처리 관련 코드
│   ├── demo/                 # 데모 애플리케이션
│   ├── evaluation/           # 평가 관련 코드
│   ├── models/               # 모델 관련 코드
│   ├── scripts/              # 실행 스크립트
│   ├── utils/                # 유틸리티 함수
│   └── main.py               # 메인 실행 파일
├── .env                      # 환경 변수 파일
├── README.md                 # 프로젝트 설명
└── requirements.txt          # 필요한 패키지 목록
```

## 주요 구성 요소

### 1. 데이터 처리 (src/data/)
- 데이터 로딩 및 전처리
- Gemini API를 통한 요약 생성

### 2. 모델 (src/models/)
- 모델 학습 및 미세 조정
- Q-LoRA 기법을 활용한 효율적인 학습

### 3. 평가 (src/evaluation/)
- 학습된 모델의 성능 평가
- 요약 품질 측정

### 4. 유틸리티 (src/utils/)
- vLLM을 활용한 모델 로딩 및 추론
- 구성 파일 관리

### 5. 데모 (src/demo/)
- 학습된 모델을 활용한 데모 애플리케이션

![img1](./images/img_1.png)
![img2](./images/img_2.png)
