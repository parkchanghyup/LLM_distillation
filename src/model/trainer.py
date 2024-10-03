from trl import SFTTrainer, SFTConfig
import os
from typing import Any, Dict

class CustomTrainer(SFTTrainer):
    """Custom trainer class that saves the best model based on evaluation loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_loss = float('inf')
        self.best_model_path = None

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model and save if it's the best one so far.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        eval_output = super().evaluate(*args, **kwargs)
        eval_loss = eval_output.get("eval_loss")

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_model_path = os.path.join(self.args.output_dir, f"best_model_eval_loss_{eval_loss:.4f}")
            self.save_model(self.best_model_path)
            print(f"New best model saved with eval loss: {eval_loss:.4f}")

        return eval_output

def train_model(model: Any, tokenizer: Any, train_dataset: Any, test_dataset: Any, config: Dict[str, Any]) -> None:
    """
    Train the model using the provided datasets and configuration.

    Args:
        model (Any): The model to train.
        tokenizer (Any): The tokenizer for the model.
        train_dataset (Any): The training dataset.
        test_dataset (Any): The test dataset.
        config (Dict[str, Any]): Configuration parameters for training.
    """
    training_args = SFTConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_steps=config["training"]["warmup_steps"],
        save_strategy="no",  # Disable automatic saving
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        dataset_text_field='text',
        max_seq_length=config['model']['sLLM']['max_seq_length'],
        packing=False
    )

    trainer.train()

    # Save the final model
    final_model_path = os.path.join(config["training"]["output_dir"], "final_model")
    trainer.save_model(final_model_path)
    print(f"Final model saved at: {final_model_path}")