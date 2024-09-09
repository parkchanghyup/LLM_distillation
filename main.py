import argparse
import yaml
import json
from pathlib import Path

from src.data_processing.json_loader import load_json_data
from src.summarization.summarizer import generate_summaries
from src.model.trainer import train_model
from src.model.inference import run_inference, load_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_yaml(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def summarize(config, prompts):
    raw_data = load_json_data(config['data']['raw_data_path'])
    generate_prompt = prompts['inference']['user']
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    summaries = generate_summaries(raw_data, model, tokenizer, generate_prompt, config)

    summaries_path = Path(config['data']['summaries_path'])
    summaries_path.mkdir(parents=True, exist_ok=True)
    with open(summaries_path / 'summaries.json', 'w') as f:
        json.dump(summaries, f)

    print("Summaries generated and saved successfully!")


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
    test_data = load_json_data(config['data']['test_data_path'])
    inference_results = run_inference(trained_model, test_data, config['inference'])

    inference_results_path = Path(config['data']['inference_results_path'])
    inference_results_path.mkdir(parents=True, exist_ok=True)
    with open(inference_results_path / 'inference_results.json', 'w') as f:
        json.dump(inference_results, f)

    print("Inference completed and results saved successfully!")


def main(args, config, prompts):
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
    config = load_yaml(args.config)
    prompts = load_yaml(args.prompts)
    main(args, config, prompts)