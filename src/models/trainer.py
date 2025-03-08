from pathlib import Path
import yaml
from typing import Dict
import pandas as pd
from datasets import Dataset
import os

from .train_sft import setup_trainer

from utils.model_utils import (
    load_model, apply_peft_config,
    load_config, merge_peft_model, logger
)
from utils.config_utils import (
    SFT_CONFIG_PATH, MODEL_A_OUTPUT_DIR, MODEL_B_OUTPUT_DIR,
    MODEL_A_MERGED_DIR, MODEL_B_MERGED_DIR, ensure_directories
)


def load_training_config() -> Dict:
    """기본 학습 설정을 로드하고 반환"""
    with open(SFT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(csv_path: str) -> Dataset:
    """CSV 파일을 Dataset 형식으로 변환"""
    try:
        df = pd.read_csv(csv_path)
        # 필요한 텍스트 필드가 있는지 확인
        if 'text' not in df.columns:
            # 텍스트 필드 생성 (예: 'prompt'와 'completion' 열을 합쳐서)
            if 'prompt' in df.columns and 'completion' in df.columns:
                df['text'] = df['prompt'] + df['completion']
            else:
                # 다른 열이 있으면 첫 번째 열을 텍스트로 사용
                df['text'] = df.iloc[:, 0]
                logger.warning(f"'text' 열이 없어 첫 번째 열을 사용합니다: {df.columns[0]}")

        return Dataset.from_pandas(df)
    except Exception as e:
        logger.error(f"데이터셋 준비 중 오류 발생: {e}")
        raise


def train_model_a(train_data_path: str, eval_data_path: str = None):
    """Model A 학습 (Q-LoRA 사용)

    Args:
        train_data_path: 학습 데이터 CSV 파일 경로
        eval_data_path: 평가 데이터 CSV 파일 경로 (선택사항)
    """
    logger.info("Model A 학습 시작")

    # 필요한 디렉토리 생성
    ensure_directories()

    try:
        config = load_training_config()

        # Q-LoRA 설정
        config["model"]["training_type"] = "qlora"
        config["peft"]["use_qlora"] = True

        # 모델 로드
        model, tokenizer = load_model(
            config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            training_type="qlora",
            use_qlora=True
        )

        # LoRA 설정 적용
        if config['model']['training_type'] == 'qlora' or config['model']['training_type'] == 'lora':
            model = apply_peft_config(
                model,
                r=config["peft"]["r"],
                lora_alpha=config["peft"]["lora_alpha"],
                training_type="qlora"
            )

        # 데이터셋 준비
        train_dataset = prepare_dataset(train_data_path)
        eval_dataset = prepare_dataset(eval_data_path)

        logger.info(f"train_dataset: {len(train_dataset)}")
        logger.info(f"train_dataset: {len(eval_dataset)}")
        # 트레이너 설정 및 학습
        if "training" not in config:
            config["training"] = {}

        # evaluation_strategy 설정
        config["training"]["eval_strategy"] = "steps"

        trainer = setup_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config,
            MODEL_A_OUTPUT_DIR
        )

        # 학습 실행
        trainer.train()

        # 모델 병합
        logger.info("Model A 병합 시작")

        # 체크포인트 디렉토리가 존재하는지 확인
        checkpoints = list(MODEL_A_OUTPUT_DIR.glob("checkpoint-*"))
        if not checkpoints:
            logger.error("체크포인트가 생성되지 않았습니다. 모델 저장에 실패했을 수 있습니다.")
            # 체크포인트가 없으면 현재 모델 상태 저장
            logger.info("현재 모델 상태를 저장합니다.")
            trainer.save_model(str(MODEL_A_OUTPUT_DIR / "final"))
            checkpoints = [MODEL_A_OUTPUT_DIR / "final"]

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]) if "checkpoint-" in x.name else 0)

        # MODEL_A_MERGED_DIR 디렉토리 생성
        MODEL_A_MERGED_DIR.mkdir(parents=True, exist_ok=True)

        if config['model']['training_type'] == 'qlora' or config['model']['training_type'] == 'lora':
            merge_peft_model(
                base_model_name=config["model"]["name"],
                peft_model_path=str(latest_checkpoint),
                merged_model_path=str(MODEL_A_MERGED_DIR),
                training_type="qlora"
            )
            tokenizer.save_pretrained(str(MODEL_A_MERGED_DIR))
            logger.info(f"Model A 병합 완료: {MODEL_A_MERGED_DIR}")
        else:
            # 모델 저장
            logger.info(f"Model A 저장 시작: {MODEL_A_MERGED_DIR}")
            trainer.save_model(str(MODEL_A_MERGED_DIR))
            tokenizer.save_pretrained(str(MODEL_A_MERGED_DIR))
            logger.info(f"Model A 저장 완료: {MODEL_A_MERGED_DIR}")

        return model
    except Exception as e:
        logger.error(f"Model A 학습 중 오류 발생: {e}")
        raise


def train_model_b(train_data_path: str, eval_data_path: str = None):
    """Model B 학습 (기본 SFT 사용)

    Args:
        train_data_path: 학습 데이터 CSV 파일 경로
        eval_data_path: 평가 데이터 CSV 파일 경로 (선택사항)
    """
    logger.info("Model B 학습 시작")

    # 필요한 디렉토리 생성
    ensure_directories()

    try:
        config = load_training_config()

        # 기본 SFT 설정
        config["model"]["training_type"] = "lora"
        config["peft"]["use_qlora"] = False

        # 모델 로드
        model, tokenizer = load_model(
            config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            training_type=config["model"]["training_type"],
            use_qlora=config["peft"]["use_qlora"]
        )

        # LoRA 설정 적용
        if config['model']['training_type'] == 'qlora' or config['model']['training_type'] == 'lora':
            model = apply_peft_config(
                model,
                r=config["peft"]["r"],
                lora_alpha=config["peft"]["lora_alpha"],
                training_type="qlora"
            )

        # 데이터셋 준비
        train_dataset = prepare_dataset(train_data_path)
        eval_dataset = prepare_dataset(eval_data_path)

        # 트레이너 설정 및 학습
        if "training" not in config:
            config["training"] = {}

        # evaluation_strategy 설정
        config["training"]["eval_strategy"] = "steps"

        trainer = setup_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config,
            MODEL_B_OUTPUT_DIR
        )

        # 학습 실행
        trainer.train()

        # 체크포인트 디렉토리가 존재하는지 확인
        checkpoints = list(MODEL_B_OUTPUT_DIR.glob("checkpoint-*"))
        if not checkpoints:
            logger.error("체크포인트가 생성되지 않았습니다. 모델 저장에 실패했을 수 있습니다.")
            # 체크포인트가 없으면 현재 모델 상태 저장
            logger.info("현재 모델 상태를 저장합니다.")
            trainer.save_model(str(MODEL_B_OUTPUT_DIR / "final"))
            checkpoints = [MODEL_B_OUTPUT_DIR / "final"]

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]) if "checkpoint-" in x.name else 0)

        # MODEL_B_MERGED_DIR 디렉토리 생성
        MODEL_B_MERGED_DIR.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        if config['model']['training_type'] == 'qlora' or config['model']['training_type'] == 'lora':
            merge_peft_model(
                base_model_name=config["model"]["name"],
                peft_model_path=str(latest_checkpoint),
                merged_model_path=str(MODEL_B_MERGED_DIR),
                training_type="qlora"
            )
            tokenizer.save_pretrained(str(MODEL_B_MERGED_DIR))
            logger.info(f"Model A 병합 완료: {MODEL_B_MERGED_DIR}")
        else:

            logger.info(f"Model A 저장 시작: {MODEL_B_MERGED_DIR}")
            trainer.save_model(str(MODEL_B_MERGED_DIR))
            tokenizer.save_pretrained(str(MODEL_B_MERGED_DIR))
            logger.info(f"Model A 저장 완료: {MODEL_B_MERGED_DIR}")

        return
    except Exception as e:
        logger.error(f"Model B 학습 중 오류 발생: {e}")
        raise