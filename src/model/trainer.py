from transformers import TrainingArguments, Trainer
import os


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_loss = float('inf')
        self.best_model_path = None

    def evaluate(self, *args, **kwargs):
        eval_output = super().evaluate(*args, **kwargs)
        eval_loss = eval_output.get("eval_loss")

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_model_path = os.path.join(self.args.output_dir, f"best_model_eval_loss_{eval_loss:.4f}")
            self.save_model(self.best_model_path)
            print(f"New best model saved with eval loss: {eval_loss:.4f}")

        return eval_output


def train_model(model, tokenizer, train_dataset, test_dataset, config):


    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        save_strategy="no",  # 자동 저장을 비활성화
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
    )

    trainer.train()

    print(f"Best model saved at: {trainer.best_model_path}")
    print(f"Best eval loss: {trainer.best_eval_loss:.4f}")

    # 최종 모델 저장
    final_model_path = os.path.join(config["training"]["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    print(f"Final model saved at: {final_model_path}")