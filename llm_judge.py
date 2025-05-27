from openai import OpenAI
from tqdm import tqdm
import os
import requests
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from datasets import load_dataset
import re
datasets = ["DisgustingOzil/Academic_dataset_ShortQA","YokyYao/Diversity_Challenge", "sentence-transformers/natural-questions"]

def access_dataset(dataset):
    input_list = []
    ds = load_dataset(dataset)
    if 'Academic' in dataset:
        path = "academic"
        # dataset="DisgustingOzil/Academic_dataset_ShortQA"
        for i in range(100): # 调整测试的数量
            response = ds['train']['response'][i]
            # 使用正则表达式匹配 <question> 标签之间的内容
            question = re.search(r'<question>(.*?)</question>', response)
            question = question.group(1).strip()  # 打印匹配到的 question 内容
            input_list.append(question)
        return input_list, path

    if 'natural' in dataset:
        path = "natural"
        # dataset = "sentence-transformers/natural-questions"
        for i in range(100):
            question = ds['train']['query'][i]
            input_list.append(question)
        return input_list, path
    if 'Diversity' in dataset:
        path = "diversity"
        for i in range(100):
            question = ds['train']['question'][i]
            input_list.append(question)
        return input_list, path

model = "DeepSeek-R1-Distill-Qwen-32B"
for dataset in datasets:
    input_list, path = access_dataset(dataset)
    result_path = f"/root/repeatcurse/{path}.jsonl"
    items = [] 
    with open(result_path, "w", encoding="utf-8") as f:
        for item in tqdm(input_list, desc="processing"):
            retry_time = 0
\
            url = "https://chatapi.sensenova.cn/v1/llm/chat-completions"
            payload = {
                "model": model,
                "stream": False,
                "temperature": 0.1,
                "top_p": 0.3,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "messages": [
                    {
                    "content": '''
                        判断该问题是否是数学/时间/人名相关的问题，如果是则输出1,若不是则输出0，不要轻易输出1(不需要回答问题，只需要输出1或0)
                        ''',                           
                        "role": "system"
                    },
                    {
                        "content": f"Question: {item}",
                        "role": "user"
                    }
                    
                    
                ]
            }
            headers = {
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJFRUNGNDYwMDhBRTI0Q0MwOTNGMDg2NTY3QkY1MDZBNiIsImV4cCI6MTc0MzMyOTMzMywibmJmIjoxNzQzMjQyOTI4LCJpYXQiOjE3NDMyNDI5MzN9.gpOLQv3VMUq-lnVLpqBA_GID9IiWogAjosI_1T96CFQ",
                "Content-Type": "application/json"
            }
            _ = requests.request("POST", url, json=payload, headers=headers)
            llm_judge = _.json()['data']['choices'][0]['message']
    
            eval = {
                "question": item,
                "llm_judge": llm_judge
            }
            print(eval)
