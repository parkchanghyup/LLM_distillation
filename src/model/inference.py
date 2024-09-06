import torch
from tqdm import tqdm


def generate_text(model, tokenizer, user_message, config):
    input_ids = tokenizer(user_message, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=config['max_new_tokens'],
        eos_token_id=tokenizer.eos_token_id,
        do_sample=config['do_sample'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def run_inference(model, test_data, config):
    tokenizer = model.tokenizer  # Assuming the model object has a tokenizer attribute
    model.eval()
    results = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="Running inference"):
            user_message = item['input']  # Adjust this according to your data structure
            generated_text = generate_text(model, tokenizer, user_message, config['generation'])
            
            result = {
                'input': user_message,
                'output': generated_text
            }
            results.append(result)

    return results

# Helper function to load the model (to be used in main.py)
def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model

# If you want to run this script standalone for testing
# if __name__ == "__main__":
    # import argparse
    # import yaml
    # from pathlib import Path
    # import json

    # parser = argparse.ArgumentParser(description="Run inference on test data")
    # parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    # args = parser.parse_args()

    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)

    # model = load_model(config['model']['path'])
    # test_data = load_json_data(config['data']['test_data_path'])  # You need to implement load_json_data
    
    # inference_results = run_inference(model, test_data, config['inference'])
    
    # Path(config['data']['inference_results_path']).mkdir(parents=True, exist_ok=True)
    # with open(Path(config['data']['inference_results_path']) / 'inference_results.json', 'w') as f:
    #     json.dump(inference_results, f)
    
    # print("Inference completed and results saved successfully!")