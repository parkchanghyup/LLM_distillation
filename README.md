# LLM Knowledge Distillation Pipeline

이 프로젝트는 LLM을 사용하여 효율적으로 학습 데이터를 생성하고, 생성된 데이터로 더 작은 모델을 훈련하는 지식 증류(Knowledge Distillation) 파이프라인을 구현합니다.  
비교적 큰 사이즈의 LLM의 지식을 작은 모델로 전이하여 리소스 제약 환경에서도 효과적인 텍스트 요약 모델을 만들 수 있습니다.

## 📸 Demo 화면

![img1](./images/img_1.png)
![img2](./images/img_2.png)

## 🏗️ 프로젝트 아키텍처
![Architecture](./images/architecture.png)

## ⚙️ 요구사항

- Python 3.10+
- CUDA 지원 GPU
- 필수 환경변수: `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `NUM_SAMPLES`

## 🚀 설치

1. Clone this repository
   ```
   git clone https://github.com/yourusername/LLM_distillation.git
   cd LLM_distillation
   ```

2. Install required packages
   ```
   pip install -r requirements.txt
   ```
   
3. Data preparation
   - 데이터의 텍스트 컬럼명을 `text`로 변경

4. 환경변수 설정:
   `.env` 파일을 생성하고 다음 변수를 설정:
   ```bash
   GEMINI_API_KEY=your_api_key
   GEMINI_MODEL_NAME=gemini-flash-2.0
   NUM_SAMPLES=1000
   ```

## 🔧 사용법

### 단계별 실행 (권장)

각 단계를 독립적으로 실행할 수 있어 메모리 효율성과 안정성이 향상됩니다:

```bash
# 1. 초기 학습 데이터 생성 (Gemini API 사용)
python src/main.py --step generate_data

# 2. Teacher 모델 학습 (Q-LoRA)
python src/main.py --step train_teacher

# 3. Teacher 모델로 나머지 데이터 요약 생성 (vLLM 사용)
python src/main.py --step teacher_inference

# 4. Student 모델 학습
python src/main.py --step train_student

# 5. 모델 성능 평가
python src/main.py --step evaluate
```

### 전체 파이프라인 실행

모든 단계를 한번에 실행하려면 아래 명령어 실행

```bash
python src/main.py --step all
```

### 도움말 확인

```bash
python src/main.py --help
```

## 📊 파이프라인 단계 설명

| 단계 | 명령어 | 설명                       | 출력 |
|------|--------|--------------------------|------|
| **데이터 생성** | `generate_data` | Gemini API로 초기 요약 데이터 생성 | `llm_train.csv`, `test.csv` |
| **Teacher 학습** | `train_teacher` | Teacher 모델 학습            | `teacher_model/merged/` |
| **Teacher 추론** | `teacher_inference` | Teacher모델로  데이터 요약 생성    | `slm_train.csv` |
| **Student 학습** | `train_student` | Student 모델 학습  | `student_model/merged/` |
| **모델 평가** | `evaluate` | Student 모델 성능 평가         | `model_evaluation_results.json` |

## 📁 프로젝트 구조

```
LLM_distillation/
├── configs/                  # 설정 파일
│   ├── dpo.yaml             # DPO 학습 설정
│   ├── sft.yaml             # SFT 학습 설정
│   └── student_model.yaml   # Student 모델 설정
├── data/                     # 데이터 디렉토리
│   ├── raw/                 # 원본 데이터
│   ├── train/               # 학습 데이터
│   └── test/                # 테스트 데이터
├── outputs/                  # 출력 결과
│   ├── teacher_model/       # Teacher 모델
│   ├── student_model/       # Student 모델
│   └── evaluation/          # 평가 결과
├── prompts/                  # 프롬프트 템플릿
├── src/                      # 소스 코드
│   ├── data/                # 데이터 처리
│   ├── demo/                # 데모 애플리케이션
│   ├── evaluation/          # 평가 코드
│   ├── models/              # 모델 학습 코드
│   ├── scripts/             # 스크립트
│   ├── utils/               # 유틸리티 함수
│   └── main.py              # 메인 실행 파일
├── .env                      # 환경변수 파일
├── README.md                # 프로젝트 설명
└── requirements.txt         # 필요 패키지 목록
```

## 🔍 주요 구성요소

### 1. 데이터 처리 (`src/data/`)
- 데이터 로딩 및 전처리
- Gemini API를 통한 요약 생성
- 데이터 샘플링 및 분할

### 2. 모델 학습 (`src/models/`)
- Teacher/Student 모델 학습
- LoRA, Q-LoRA를 사용한 효율적 파인튜닝 지원

### 3. 평가 시스템 (`src/evaluation/`)
- 학습된 모델 성능 평가
- 요약 품질 측정 지표
- 결과 리포트 생성

### 4. 유틸리티 (`src/utils/`)
- vLLM을 사용한 추론
- 설정 파일 관리
- 모델 유틸리티 함수

### 5. 데모 (`src/demo/`)
- 학습된 모델을 사용한 웹 데모
- REST API 인터페이스



## 🎯 Future Work

다음 기능들이 향후 버전에 추가될 예정입니다:

1. **Direct Preference Optimization (DPO)**: 인간 선호도 기반 모델 최적화로 요약 품질 향상

2. **자동 데이터 필터링**: 저품질 학습 데이터를 자동으로 식별하고 제거하는 기능



