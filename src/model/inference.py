from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline
from typing import Dict, Any


def generate_text(model: Any, tokenizer: Any, messages: str, config: Dict[str, Any]) -> str:
    """
    Generate text using the provided model and tokenizer with transformers pipeline.

    Args:
        model (Any): The language model.
        tokenizer (Any): The tokenizer for the model.
        messages (str): The input message to generate text from.
        config (Dict[str, Any]): Configuration parameters for text generation.

    Returns:
        str: The generated text.
    """
    # pipeline 생성 (미리 로드된 모델과 토크나이저 사용)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cuda")

    # 텍스트 생성
    outputs = generator(
        messages,
        max_length=config['max_length'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        top_k=config['top_k'],
        do_sample=config['do_sample'],
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # 생성된 텍스트에서 입력 텍스트를 제외한 새로 생성된 부분만 추출
    generated_text = outputs[0]['generated_text']
    new_text = generated_text[len(messages):]

    return new_text.strip()



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
    messages = [{"role": "user", "content": formatted_prompt}]
    messages = tokenizer.apply_chat_template(messages, tokenize=False)

    with torch.no_grad():
        generated_text = generate_text(model, tokenizer, messages, config)

        result = {
            'input': test_document,
            'output': generated_text
        }

    return result