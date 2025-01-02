from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from models.model_utils import load_model, apply_peft_config, merge_peft_model
from data.data_loader import load_data, get_file_paths, generate_prompts

def main():
    # 설정
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length = 2048
    output_dir = "outputs"
    merged_model_path = "./Qwen2.5-1.5B-SFT-merged"
    
    # 모델 로드
    model, tokenizer = load_model(model_name, max_seq_length=max_seq_length)
    model = apply_peft_config(model, r=16, lora_alpha=16)

    # 데이터 로드
    paths = get_file_paths()
    dataset_dict = load_data(paths)
    dataset_dict = dataset_dict.map(lambda x: generate_prompts(x, tokenizer), batched=True)

    # SFT 학습 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=1000,
            num_train_epochs=3,
            learning_rate=3e-5,
            evaluation_strategy="steps",
            eval_steps=1000,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=500,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            save_steps=1000,
            save_total_limit=3,
        ),
    )

    # 학습 시작
    trainer_stats = trainer.train()
    print(trainer_stats)

    # 학습 후 최종 체크포인트 병합
    latest_checkpoint = f"{output_dir}/checkpoint-3750"  # 실제로는 마지막 체크포인트 경로로 동적 설정 필요
    merge_peft_model(base_model_name=model_name, peft_model_path=latest_checkpoint, merged_model_path=merged_model_path)

if __name__ == "__main__":
    main()