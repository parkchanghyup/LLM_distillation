from pathlib import Path
import yaml
from typing import Dict
import pandas as pd
from datasets import Dataset

from .train_sft import (
    load_model, apply_peft_config, setup_trainer,
    load_config, merge_peft_model, logger
)

CONFIG_PATH = Path("configs/sft.yaml")
MODEL_A_OUTPUT_DIR = Path("outputs/model_a")
MODEL_B_OUTPUT_DIR = Path("outputs/model_b")

def load_training_config() -> Dict:
    """기본 학습 설정을 로드하고 반환"""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(csv_path: str) -> Dataset:
    """CSV 파일을 Dataset 형식으로 변환"""
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)

def train_model_a(train_data_path: str):
    """Model A 학습 (Q-LoRA 사용)
    
    Args:
        train_data_path: 학습 데이터 CSV 파일 경로
    """
    logger.info("Model A 학습 시작")
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
    model = apply_peft_config(
        model,
        r=config["peft"]["r"],
        lora_alpha=config["peft"]["lora_alpha"],
        training_type="qlora"
    )
    
    # 데이터셋 준비
    train_dataset = prepare_dataset(train_data_path)
    
    # 트레이너 설정 및 학습
    trainer = setup_trainer(
        model,
        tokenizer,
        train_dataset,
        None,  # eval_dataset
        config,
        MODEL_A_OUTPUT_DIR
    )
    
    trainer.train()
    return model

def train_model_b(train_data_path: str):
    """Model B 학습 (기본 SFT 사용)
    
    Args:
        train_data_path: 학습 데이터 CSV 파일 경로
    """
    logger.info("Model B 학습 시작")
    config = load_training_config()
    
    # 기본 SFT 설정
    config["model"]["training_type"] = "sft"
    config["peft"]["use_qlora"] = False
    
    # 모델 로드
    model, tokenizer = load_model(
        config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        training_type="sft",
        use_qlora=False
    )
    
    # 데이터셋 준비
    train_dataset = prepare_dataset(train_data_path)
    
    # 트레이너 설정 및 학습
    trainer = setup_trainer(
        model,
        tokenizer,
        train_dataset,
        None,  # eval_dataset
        config,
        MODEL_B_OUTPUT_DIR
    )
    
    trainer.train()
    return model 