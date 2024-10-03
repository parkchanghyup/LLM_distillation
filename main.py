import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.summarization.summarizer import generate_summaries
from src.model.trainer import train_model
from src.model.inference import run_inference
from src.utils.file_utils import load_documents, load_train_dataset, load_json


def save_summaries(summaries_path: str, docs: list, summaries: list) -> None:
    """
    Save generated summaries to a CSV file.

    Args:
        summaries_path (str): Path to save the summaries.
        docs (list): List of original documents.
        summaries (list): List of generated summaries.
    """
    summaries_path = Path(summaries_path)
    summaries_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'document': docs, 'summary': summaries})
    df.to_csv(summaries_path / 'summaries.csv', index=False, encoding='utf-8-sig')

    print("Summaries generated and saved successfully as CSV!")


def setup_model(model_name: str, device: str):
    """
    Set up the model and tokenizer.

    Args:
        model_name (str): Name of the model to load.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        tuple: Loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def summarize(config: dict, prompts: dict) -> None:
    """
    Summarize documents and save results.

    Args:
        config (dict): Configuration dictionary.
        prompts (dict): Prompts dictionary.
    """
    docs = load_documents(config['data']['raw_data_path'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model(config['model']['LLM']['name'], device)

    generate_prompt = prompts['inference']
    summary_results, use_docs = generate_summaries(docs, model, tokenizer, generate_prompt, config)
    save_summaries(config['data']['summaries_path'], use_docs, summary_results)


def train(config: dict, prompt: dict) -> None:
    """
    Train the model using the provided configuration and prompt.

    Args:
        config (dict): Configuration dictionary.
        prompt (dict): Prompt dictionary.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model(config['model']['sLLM']['name'], device)

    train_prompt = prompt['train']
    train_dataset, valid_dataset = load_train_dataset(
        f"{config['data']['summaries_path']}/summaries.csv",
        config['training']['test_size'],
        train_prompt
    )

    train_model(model, tokenizer, train_dataset, valid_dataset, config)
    print("Model trained successfully!")


def infer(config: dict, prompts: dict) -> None:
    """
    Run inference using the trained model.

    Args:
        config (dict): Configuration dictionary.
        prompts (dict): Prompts dictionary.
    """
    pretrained_model = os.path.join(config["training"]["output_dir"], "final_model")
    generate_prompt = prompts['inference']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model(pretrained_model, device)

    test_document = load_json(config['data']['test_data_path'])
    inference_results = run_inference(model, tokenizer, test_document, generate_prompt, config['summarization'])

    print("Inference completed and results saved successfully!")
    print('-' * 30)
    print(inference_results)


def main(args: argparse.Namespace, config: dict, prompts: dict) -> None:
    """
    Main function to run the summarization process.

    Args:
        args (argparse.Namespace): Command-line arguments.
        config (dict): Configuration dictionary.
        prompts (dict): Prompts dictionary.
    """
    if args.summarize:
        summarize(config, prompts)
    elif args.train:
        train(config, prompts)
    elif args.infer:
        infer(config, prompts)
    else:
        print("Please specify an action: --summarize, --train, or --infer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM-based JSON summarization and sLLM training pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--prompts', type=str, default='config/prompts.yaml', help='Path to the prompts file')
    parser.add_argument('--summarize', action='store_true', help='Generate summaries')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Run inference')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)

    main(args, config, prompts)