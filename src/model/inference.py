import torch
from langchain.prompts import PromptTemplate


def generate_text(model, tokenizer, user_message, config):
    input_ids = tokenizer(user_message, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=config['max_length'],
        temperature=config['temperature'],
        top_p=config['top_p'],
        top_k=config['top_k'],
        do_sample=config['do_sample'],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def run_inference(model, tokenizer, test_document, generate_prompt, config):

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
