from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(model_name="unsloth/Qwen2.5-1.5B-Instruct", max_seq_length=2048, dtype=None, load_in_4bit=False):
    """Load the model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    return model, tokenizer

def apply_peft_config(model, r=16, lora_alpha=16, target_modules=None, **kwargs):
    """Apply PEFT (LoRA) configuration to the model"""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        **kwargs
    )
    return model

def merge_peft_model(base_model_name, peft_model_path, merged_model_path="./merged_model", dtype=torch.float16):
    """Merge the PEFT model with the base model and save it"""
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

    # 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

    # PEFT 모델 로드
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    # LoRA 어댑터 병합 및 제거
    merged_model = model.merge_and_unload()

    # 병합된 모델 저장
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved to: {merged_model_path}")
    return merged_model, tokenizer