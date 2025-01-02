from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported
from models.model_utils import load_model, apply_peft_config, merge_peft_model
from data.data_loader import load_data, get_file_paths, generate_prompts

# Custom DPO Trainer to handle log issue
class CustomDPOTrainer(DPOTrainer):
    def log(self, logs, start_time=None):
        super().log(logs)

def main():
    # 설정
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length = 2048
    output_dir = "outputs_dpo"
    merged_model_path = "./Qwen2.5-1.5B-DPO-merged"
    
    # 모델 로드
    model, tokenizer = load_model(model_name, max_seq_length=max_seq_length)
    model = apply_peft_config(model, r=32, lora_alpha=64, random_state=323)

    # 데이터 로드
    paths = get_file_paths()
    dataset_dict = load_data(paths)
    dataset_dict = dataset_dict.map(lambda x: generate_prompts(x, tokenizer), batched=True)

    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']

    # DPO 학습 설정
    dpo_trainer = CustomDPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=5e-6,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1000,
            evaluation_strategy="steps",
            eval_steps=1000,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=323,
            output_dir=output_dir,
            report_to="none",
            save_steps=1000,
        ),
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=512,
    )

    # 학습 시작
    dpo_trainer.train()

    # 학습 후 최종 체크포인트 병합
    latest_checkpoint = f"{output_dir}/checkpoint-3750"  # 실제로는 마지막 체크포인트 경로로 동적 설정 필요
    merge_peft_model(base_model_name=model_name, peft_model_path=latest_checkpoint, merged_model_path=merged_model_path)

if __name__ == "__main__":
    main()