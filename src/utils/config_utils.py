import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

# 로깅 설정
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# 주요 경로 설정
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PROMPT_DIR = ROOT_DIR / "prompts"

# 설정 파일 경로
SFT_CONFIG_PATH = CONFIG_DIR / "sft.yaml"
DPO_CONFIG_PATH = CONFIG_DIR / "dpo.yaml"

# 모델 경로
MODEL_A_OUTPUT_DIR = OUTPUT_DIR / "model_a"
MODEL_B_OUTPUT_DIR = OUTPUT_DIR / "model_b"
MODEL_A_MERGED_DIR = MODEL_A_OUTPUT_DIR / "merged"
MODEL_B_MERGED_DIR = MODEL_B_OUTPUT_DIR / "merged"

# DPO 모델 경로
DPO_OUTPUT_DIR = OUTPUT_DIR / "dpo"
DPO_MERGED_DIR = DPO_OUTPUT_DIR / "merged"

# 데이터 경로
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"


def load_config(config_path: Path = SFT_CONFIG_PATH) -> Dict:
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


def ensure_directories():
    """필요한 디렉토리가 존재하는지 확인하고, 없으면 생성합니다."""
    directories = [
        RAW_DATA_DIR,
        TRAIN_DATA_DIR,
        TEST_DATA_DIR,
        MODEL_A_OUTPUT_DIR,
        MODEL_B_OUTPUT_DIR,
        DPO_OUTPUT_DIR,
        OUTPUT_DIR / "evaluation",
        PROMPT_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories