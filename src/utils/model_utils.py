import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_quantization_config() -> BitsAndBytesConfig:
    """QLoRA를 위한 양자화 설정을 반환합니다."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(model_name: str, max_seq_length: int, training_type: str, use_qlora: bool = False) -> Tuple:
    """모델과 토크나이저를 로드합니다."""
    kwargs = {}
    if training_type == "qlora" or (training_type == "lora" and use_qlora):
        kwargs["quantization_config"] = get_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_seq_length

    if training_type == "qlora" or (training_type == "lora" and use_qlora):
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_peft_config(model, r: int, lora_alpha: int, training_type: str, random_state: int = 42):
    """PEFT 설정을 모델에 적용합니다. LoRA 또는 QLoRA일 때만 적용."""
    if training_type in ["lora", "qlora"]:
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        return get_peft_model(model, peft_config)
    return model


def merge_peft_model(base_model_name: str, peft_model_path: str, merged_model_path: str, training_type: str):
    """PEFT 모델을 병합합니다. LoRA 또는 QLoRA일 때만 실행."""
    if training_type in ["lora", "qlora"]:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        peft_config = LoraConfig.from_pretrained(peft_model_path)
        peft_model = get_peft_model(base_model, peft_config)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_model_path)


def load_config(config_path: Path) -> Dict:
    """설정 파일을 로드합니다."""
    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML 파싱 에러: {e}")
        raise


def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """가장 최근 체크포인트를 반환합니다."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        logger.warning("체크포인트가 존재하지 않습니다.")
        return None
    return max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))


def setup_training_args(config: Dict, output_dir: Path) -> "TrainingArguments":
    """TrainingArguments를 설정하고 반환합니다."""
    from transformers import TrainingArguments
    import os

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 기본 설정
    training_args = {
        "per_device_train_batch_size": config["training"]["batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "warmup_steps": config["training"]["warmup_steps"],
        # "num_train_epochs": config["training"]["epochs"],
        "max_steps": config["training"]["max_steps"],
        "learning_rate": float(config["training"]["learning_rate"]),
        "logging_steps": config["training"]["logging_steps"],
        "optim": config["training"]["optimizer"],
        "weight_decay": config["training"]["weight_decay"],
        "lr_scheduler_type": config["training"]["lr_scheduler_type"],
        "seed": config["training"]["seed"],
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "steps",  # 명시적으로 steps로 설정
        "save_steps": config["training"]["save_steps"],
        "save_total_limit": config["training"]["save_total_limit"],
        "fp16": True,  # 학습 속도 향상을 위해 fp16 활성화
        "overwrite_output_dir": True,  # 기존 출력 디렉토리 덮어쓰기
        "disable_tqdm": False,  # 진행 상황 표시
        "load_best_model_at_end": False,  # 평가 데이터가 없을 수 있으므로 False로 설정
    }

    # evaluation_strategy 설정 (config에서 직접 가져오거나 기본값 사용)
    evaluation_strategy = config["training"].get("evaluation_strategy", "no")
    training_args["evaluation_strategy"] = evaluation_strategy

    if evaluation_strategy == "steps":
        training_args["eval_steps"] = config["training"].get("eval_steps", 5)

    return TrainingArguments(**training_args)