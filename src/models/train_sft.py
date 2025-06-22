from pathlib import Path
from typing import Dict
import os
import traceback

from trl import SFTTrainer, SFTConfig

from utils.model_utils import (
    load_model, apply_peft_config, merge_peft_model,
    load_config, get_latest_checkpoint, setup_training_args, logger
)
from data.data_loader import load_data, get_file_paths, generate_prompts

# 상수 정의
CONFIG_PATH = Path("configs/sft.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MERGED_MODEL_PATH = Path("./Qwen2.5-1.5B-SFT-merged")


def setup_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config: Dict,
        output_dir: Path,
) -> SFTTrainer:
    """SFTTrainer를 설정하고 반환합니다."""
    # eval_dataset이 None이면 evaluation_strategy를 "no"로 설정
    if eval_dataset is None and "training" in config:
        config["training"]["eval_strategy"] = "no"

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 훈련 설정을 SFTConfig에 맞게 변환
    training_config = config["training"]

    # SFTConfig 생성 - 키 이름을 올바르게 매핑
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        dataset_num_proc=config["data"]["num_proc"],
        max_length=config["model"]["max_seq_length"],
        packing=False,
        # 훈련 관련 설정들을 올바른 키 이름으로 매핑
        per_device_train_batch_size=training_config.get("batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        warmup_steps=training_config.get("warmup_steps", 1000),
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=training_config.get("learning_rate", 3e-5),
        eval_steps=training_config.get("eval_steps", 1000),
        eval_strategy=training_config.get("eval_strategy", "steps"),
        logging_steps=training_config.get("logging_steps", 500),
        optim=training_config.get("optimizer", "adamw_8bit"),
        weight_decay=training_config.get("weight_decay", 0.01),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
        seed=training_config.get("seed", 323),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
    )

    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
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

        logger.info(f"학습 방식: {training_type}")
        logger.info("모델 및 데이터 로딩 시작")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(merged_model_path, exist_ok=True)

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

        # 데이터 로드
        paths = get_file_paths()
        dataset_dict = load_data(paths)
        dataset_dict = dataset_dict.map(
            lambda x: generate_prompts(x, tokenizer),
            batched=True,
        )
        print("데이터셋 정보:", dataset_dict)

        # 트레이너 설정
        trainer = setup_trainer(
            model,
            tokenizer,
            dataset_dict["train"],
            dataset_dict.get("validation"),  # validation이 없을 수 있으므로 get 사용
            config,
            output_dir,
        )

        # 학습 실행
        logger.info("학습 시작")
        trainer_stats = trainer.train()
        logger.info(f"학습 완료: {trainer_stats}")

        # 모델 저장 (체크포인트와 별개로 최종 모델 저장)
        logger.info("최종 모델 저장 시작")
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        logger.info(f"최종 모델 저장 완료: {output_dir / 'final_model'}")

        # 모델 병합 (LoRA 또는 QLoRA일 경우에만)
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint and training_type in ["lora", "qlora"]:
            logger.info(f"모델 병합 시작: {latest_checkpoint}")
            try:
                merge_peft_model(
                    base_model_name=config["model"]["name"],
                    peft_model_path=str(latest_checkpoint),
                    merged_model_path=str(merged_model_path),
                    training_type=training_type,
                )
                logger.info("모델 병합 완료")
            except Exception as e:
                logger.error(f"모델 병합 중 오류 발생: {e}")
                logger.error(traceback.format_exc())
                # 병합 실패 시 최종 모델을 복사
                logger.info("병합 실패, 최종 모델을 사용합니다.")
                import shutil
                shutil.copytree(str(output_dir / "final_model"), str(merged_model_path), dirs_exist_ok=True)

    except Exception as e:
        logger.error(f"학습 중 에러 발생: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()