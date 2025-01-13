# train.py
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

from models.model_utils import load_model, apply_peft_config, merge_peft_model
from data.data_loader import load_data, get_file_paths, generate_prompts

# 상수 정의
CONFIG_PATH = Path("config/training_config.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MERGED_MODEL_PATH = Path("./Qwen2.5-1.5B-SFT-merged")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict,
    output_dir: Path,
) -> SFTTrainer:
    """SFTTrainer를 설정하고 반환합니다."""
    training_args = TrainingArguments(
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

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        dataset_num_proc=config["data"]["num_proc"],
        packing=False,
        args=training_args,
    )

#TODO: 동적구현
def get_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """가장 최근 체크포인트를 반환합니다."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    return checkpoints[-1]



def main() :
    """메인 학습 프로세스를 실행합니다."""
    try:
        # 설정 로드
        config = load_config(CONFIG_PATH)
        output_dir = DEFAULT_OUTPUT_DIR
        merged_model_path = DEFAULT_MERGED_MODEL_PATH

        logger.info("모델 및 데이터 로딩 시작")
        # 모델 로드
        model, tokenizer = load_model(
            config["model"]["name"],리
            max_seq_length=config["model"]["max_seq_length"],
        )
        model = apply_peft_config(
            model,
            r=config["peft"]["r"],
            lora_alpha=config["peft"]["lora_alpha"],
        )

        # 데이터 로드
        paths = get_file_paths()
        dataset_dict = load_data(paths)
        dataset_dict = dataset_dict.map(
            lambda x: generate_prompts(x, tokenizer),
            batched=True,
        )

        # 트레이너 설정
        trainer = setup_trainer(
            model,
            tokenizer,
            dataset_dict["train"],
            dataset_dict["validation"],
            config,
            output_dir,
        )

        # 학습 실행
        logger.info("학습 시작")
        trainer_stats = trainer.train()
        logger.info(f"학습 완료: {trainer_stats}")

        # 모델 병합
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint:
            logger.info(f"모델 병합 시작: {latest_checkpoint}")
            merge_peft_model(
                base_model_name=config["model"]["name"],
                peft_model_path=str(latest_checkpoint),
                merged_model_path=str(merged_model_path),
            )
            logger.info("모델 병합 완료")

    except Exception as e:
        logger.error(f"학습 중 에러 발생: {e}")
        raise


if __name__ == "__main__":
    main()