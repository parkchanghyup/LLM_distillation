import json
import glob

def load_json_files():
    json_files = []
    for i in range(1, 5):
        json_files.extend(glob.glob(f'./data/group_{i}/*.json'))
    
    docs = []
    gen_summaries = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            docs.append(data['paragraph'])
            gen_summaries.append(data['gen_summary'])
    
    return docs, gen_summaries