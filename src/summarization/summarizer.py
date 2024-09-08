from tqdm import tqdm
from model.inference import generate_text
from langchain.prompts import PromptTemplate


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

def generate_summaries(docs, model, tokenizer, generate_prompt, config):
    summary_results = []
    use_docs = []
    prompt = PromptTemplate.from_template(generate_prompt)
    

    for i in tqdm(range(len(docs))):
        if len(docs[i]) > 2048:
            continue
        
        prompt = prompt.format(docs[i])
        
        gemma_summary = generate_text(model, tokenizer, prompt, config)
        
        summary_results.append(gemma_summary)
        use_docs.append(docs[i])
    
    return summary_results, use_docs