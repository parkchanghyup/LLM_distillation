# model_utils.py
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from unsloth import is_bfloat16_supported

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 기존 model_utils.py 함수들 (가정)
def load_model(model_name: str, max_seq_length: int) -> Tuple:
    """모델과 토크나이저를 로드합니다."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_seq_length
    return model, tokenizer


def apply_peft_config(model, r: int, lora_alpha: int, random_state: int = 42):
    """PEFT 설정을 모델에 적용합니다."""
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        random_state=random_state,
    )
    return get_peft_model(model, peft_config)


def merge_peft_model(base_model_name: str, peft_model_path: str, merged_model_path: str):
    """PEFT 모델을 병합합니다."""
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = get_peft_model(base_model, LoraConfig.from_pretrained(peft_model_path))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)


# common_utils.py에서 병합된 함수들
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

    return TrainingArguments(
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config["training"]["logging_steps"],
        optim=config["training"]["optimizer"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        seed=config["training"]["seed"],
        output_dir=str(output_dir),
        report_to="none",
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
    )