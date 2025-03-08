from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from pathlib import Path
from data.data_loader import generate_inference_prompts
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Text Summarization API",
    description="vLLM을 사용한 텍스트 요약 API",
    version="1.0.0"
)

# 모델 경로 설정
MODEL_PATH = Path("outputs/student_model/merged")

# Request/Response 모델
class SummarizationRequest(BaseModel):
    text: str

class SummarizationResponse(BaseModel):
    summary: str

# vLLM 모델 초기화
@app.on_event("startup")
async def startup_event():
    global llm, tokenizer
    logger.info(f"Loading model from {MODEL_PATH}")
    llm = LLM(model=str(MODEL_PATH))
    tokenizer = llm.get_tokenizer()
    logger.info("Model loaded successfully")

# Sampling 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=2048,
    repetition_penalty=1.05
)

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    """텍스트 요약 엔드포인트"""
    try:
        # 프롬프트 생성
        prompt = generate_inference_prompts([request.text], tokenizer)[0]
        
        # 요약 생성
        outputs = llm.generate([prompt], sampling_params)
        summary = outputs[0].outputs[0].text
        
        return SummarizationResponse(summary=summary)
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"} 