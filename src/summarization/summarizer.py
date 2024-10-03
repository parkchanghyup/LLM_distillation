from tqdm import tqdm
from langchain.prompts import PromptTemplate
from typing import List, Tuple, Dict, Any


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
    inputs = tokenizer(user_message, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=config['summarization']['max_length'],
            temperature=config['summarization']['temperature'],
            top_p=config['summarization']['top_p'],
            top_k=config['summarization']['top_k'],
            do_sample=config['summarization']['do_sample'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0][inputs.input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def generate_summaries(docs: List[str], model: Any, tokenizer: Any, generate_prompt: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Generate summaries for the provided documents.

    Args:
        docs (List[str]): List of documents to summarize.
        model (Any): The language model.
        tokenizer (Any): The tokenizer for the model.
        generate_prompt (str): The prompt template for generating summaries.
        config (Dict[str, Any]): Configuration parameters.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of summaries and the list of used documents.
    """
    summary_results = []
    use_docs = []
    prompt = PromptTemplate.from_template(generate_prompt)

    for doc in tqdm(docs, desc="Generating summaries"):
        if len(tokenizer.encode(doc)) > config['data']['max_input_length']:
            print(f"Skipping document with length {len(tokenizer.encode(doc))} tokens")
            continue
        formatted_prompt = prompt.format(document=doc)
        summary = generate_text(model, tokenizer, formatted_prompt, config)
        if summary:
            summary_results.append(summary)
            use_docs.append(doc)

    return summary_results, use_docs