from pathlib import Path
from typing import Dict

from trl import DPOTrainer, DPOConfig

from models.model_utils import (
    load_model, apply_peft_config, merge_peft_model,
    load_config, get_latest_checkpoint, setup_training_args, logger
)
from data.data_loader import load_data, get_file_paths, generate_prompt_dpo

# 상수 정의
CONFIG_PATH = Path("config/dpo_config.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs_dpo")
DEFAULT_MERGED_MODEL_PATH = Path("./Qwen2.5-1.5B-DPO-merged")

# Custom DPO Trainer to handle log issue
class CustomDPOTrainer(DPOTrainer):
    def log(self, logs, start_time=None):
        super().log(logs)

def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict,
    output_dir: Path,
) -> CustomDPOTrainer:
    """DPOTrainer를 설정하고 반환합니다."""
    training_args = setup_training_args(config, output_dir)

    return CustomDPOTrainer(
        model=model,
        ref_model=None,  # DPO는 참조 모델 없이도 동작 가능
        args=DPOConfig(
            **training_args.to_dict(),
            beta=config["dpo"]["beta"],
            max_length=config["model"]["max_seq_length"],
            max_prompt_length=config["dpo"]["max_prompt_length"],
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

def main():
    """메인 학습 프로세스를 실행합니다."""
    try:
        # 설정 로드
        config = load_config(CONFIG_PATH)
        output_dir = DEFAULT_OUTPUT_DIR
        merged_model_path = DEFAULT_MERGED_MODEL_PATH
        training_type = config["model"].get("training_type", "sft").lower()

        if training_type not in ["sft", "lora", "qlora"]:
            raise ValueError("training_type은 'sft', 'lora', 'qlora' 중 하나여야 합니다.")

        logger.info(f"학습 방식: {training_type} (DPO)")
        logger.info("모델 및 데이터 로딩 시작")

        # 모델 로드
        use_qlora = config["peft"].get("use_qlora", False)
        model, tokenizer = load_model(
            config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            training_type=training_type,
            use_qlora=use_qlora,
        )
        model = apply_peft_config(
            model,
            r=config["peft"]["r"],
            lora_alpha=config["peft"]["lora_alpha"],
            training_type=training_type,
        )

        # 데이터 로드 (config 전달)
        paths = get_file_paths(config=config)
        dataset_dict = load_data(paths)
        # DPO용 프롬프트 생성
        dataset_dict = dataset_dict.map(
            lambda x: generate_prompt_dpo(x, tokenizer),
            batched=False,  # DPO는 행 단위 처리가 필요하므로 batched=False
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

        # 모델 병합 (LoRA 또는 QLoRA일 경우에만)
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint and training_type in ["lora", "qlora"]:
            logger.info(f"모델 병합 시작: {latest_checkpoint}")
            merge_peft_model(
                base_model_name=config["model"]["name"],
                peft_model_path=str(latest_checkpoint),
                merged_model_path=str(merged_model_path),
                training_type=training_type,
            )
            logger.info("모델 병합 완료")

    except Exception as e:
        logger.error(f"학습 중 에러 발생: {e}")
        raise

if __name__ == "__main__":
    main()