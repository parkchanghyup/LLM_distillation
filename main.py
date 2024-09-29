import argparse
import yaml
import json
import csv
from pathlib import Path
import torch
from src.summarization.summarizer import generate_summaries
from src.model.trainer import train_model
from src.model.inference import run_inference, load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def summarize(config, prompts):
    """Summarize documents and save results."""
    docs = []
    raw_data_path = config['data']['raw_data_path']
    llm_model = config['model']['LLM']['name']
    try:
        raw_data_path = glob.glob(raw_data_path)
        for json_path in raw_data_path:
            data = load_json(json_path)
            docs = docs.append(data['paragraph'])

        generate_prompt = prompts['inference']['user']

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            llm_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(llm_model)

        summaries = generate_summaries(docs, model, tokenizer,
                                       generate_prompt, config)

        summaries_path = Path(config['data']['summaries_path'])
        summaries_path.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        with open(summaries_path / 'summaries.csv', 'w', newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Original Document', 'Summary'])
            for doc, summary in zip(docs, summaries):
                writer.writerow([doc, summary])

        print("Summaries generated and saved successfully as CSV!")
    except Exception as e:
        print(f"An error occurred during summarization: {e}")



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

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.prompts, 'r') as f:
        prompts = yaml.safe_load(f)

    main(args, config, prompts)
