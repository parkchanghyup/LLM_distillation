import torch
from langchain.prompts import PromptTemplate
from typing import Any, Dict

def generate_text(model: Any, tokenizer: Any, user_message: str, config: Dict[str, Any]) -> str:
    """
    Generate text using the provided model and tokenizer.

    Args:
        model (Any): The language model.
        tokenizer (Any): The tokenizer for the model.
        user_message (str): The input message to generate text from.
        config (Dict[str, Any]): Configuration parameters for text generation.

    Returns:
        str: The generated text.
    """
    input_ids = tokenizer(user_message, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids.input_ids,
        max_new_tokens=config['max_length'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        top_k=config['top_k'],
        do_sample=config['do_sample'],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][input_ids.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def run_inference(model: Any, tokenizer: Any, test_document: str, generate_prompt: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Run inference on the provided test document using the model.

    Args:
        model (Any): The language model.
        tokenizer (Any): The tokenizer for the model.
        test_document (str): The document to run inference on.
        generate_prompt (str): The prompt template for generating text.
        config (Dict[str, Any]): Configuration parameters for inference.

    Returns:
        Dict[str, str]: A dictionary containing the input document and generated output.
    """
    model.eval()
    prompt = PromptTemplate.from_template(generate_prompt)
    formatted_prompt = prompt.format(document=test_document)

    with torch.no_grad():
        generated_text = generate_text(model, tokenizer, formatted_prompt, config)

        result = {
            'input': test_document,
            'output': generated_text
        }

    return result