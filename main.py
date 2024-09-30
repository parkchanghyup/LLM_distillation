import argparse
import csv
import yaml
from pathlib import Path
import torch
from src.summarization.summarizer import generate_summaries
from src.model.trainer import train_model
from src.model.inference import run_inference, load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.file_utils import load_yaml, load_json, load_documents


import pandas as pd
from pathlib import Path

def save_summaries(summaries_path, docs, summaries):
    summaries_path = Path(summaries_path)
    summaries_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'Original Document': docs,
        'Summary': summaries
    })
    
    # CSV 파일로 저장 (UTF-8 인코딩)
    df.to_csv(summaries_path / 'summaries.csv', index=False, encoding='utf-8-sig')
    
    print("Summaries generated and saved successfully as CSV!")



def setup_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def summarize(config, prompts):
    """Summarize documents and save results."""
    # try:
        # Load documents
    docs = load_documents(config['data']['raw_data_path'])

    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = setup_model(config['model']['LLM']['name'], device)

    # Generate summaries
    generate_prompt = prompts['inference']['user']
    summary_results, use_docs = generate_summaries(docs, model, tokenizer, generate_prompt, config)

    # Save summaries
    save_summaries(config['data']['summaries_path'], use_docs, summary_results)

    # except Exception as e:
    #     print(f"An error occurred during summarization: {e}")



def train(config):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load and preprocess the dataset
    dataset = load_dataset(
        'json',
        data_files={
            'train': config['data']['train_data_path'],
            'test': config['data']['test_data_path']
        }
    )

    # Preprocess the dataset if needed
    # This step depends on your specific preprocessing needs

    # Call the train_model function
    train_model(model, tokenizer, dataset, config)
    print("Model trained successfully!")


def infer(config):
    trained_model = load_model(config['model']['path'])
    test_data = load_json_data(config['test_data_path'])
    inference_results = run_inference(trained_model, test_data, config['inference'])

    inference_results_path = Path(config['data']['inference_results_path'])
    inference_results_path.mkdir(parents=True, exist_ok=True)
    with open(inference_results_path / 'inference_results.json', 'w') as f:
        json.dump(inference_results, f)

    print("Inference completed and results saved successfully!")


def main(args, config, prompts):
    """Main function to run the summarization process."""
    if args.summarize:
        summarize(config, prompts)
    # elif args.train:
    #     train(config)
    # elif args.infer:
    #     infer(config)
    else:
        print("Please specify an action: --summarize, --train, or --infer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the LLM-based JSON summarization and sLLM training pipeline"
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--prompts', type=str, default='config/prompts.yaml',
        help='Path to the prompts file'
    )
    parser.add_argument(
        '--summarize', action='store_true',
        help='Generate summaries'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--infer', action='store_true',
        help='Run inference'
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)

    main(args, config, prompts)
