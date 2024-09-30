from tqdm import tqdm
from langchain.prompts import PromptTemplate


def generate_text(model, tokenizer, user_message, config):
    """Generate text using the provided model and tokenizer."""
    inputs = tokenizer(user_message, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config['summarization']['max_length'],
            temperature=config['summarization']['temperature'],
            top_p=config['summarization']['top_p'],
            top_k=config['summarization']['top_k'],
            do_sample=config['summarization']['do_sample'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""


def generate_summaries(docs, model, tokenizer, generate_prompt, config):
    """Generate summaries for the provided documents."""
    summary_results = []
    use_docs = []
    prompt = PromptTemplate.from_template(generate_prompt)

    for doc in tqdm(docs):
        if len(tokenizer.encode(doc)) > config['data']['max_input_length']:
            print(f"Skipping document with length "
                  f"{len(tokenizer.encode(doc))} tokens")
            continue
        formatted_prompt = prompt.format(document=doc)
        summary = generate_text(model, tokenizer, formatted_prompt, config)
        if summary:
            summary_results.append(summary)
            use_docs.append(doc)

    return summary_results, use_docs
