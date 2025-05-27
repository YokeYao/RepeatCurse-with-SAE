
import os
# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = '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'
import sys
# 将父文件夹的路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate_agent import evaluate_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse
import torch
from tqdm import tqdm
import metrics.n_gram
import metrics.Information_Entropy
from datasets import load_dataset
import re
import json

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
# model_path = 'meta-llama/Llama-3.1-8B'
model_path = 'google/gemma-2-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_path)
if 'llama' in model_path: # 降低精度
    model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16).to(torch_device)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id).to(torch_device)
prompt = '''
Answer the question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. choices A=0, B=4, C=2, D=6'''
# prompt = '''You are a good mathematician, please answer the following question. 1+1=?'''
GENERATE_KWARGS = dict(repetition_penalty=2.0)
# 定义对话历史
conversation = [
    {"role": "user", "content": "You are a good mathematician, please answer the following question. Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. choices A=0, B=4, C=2, D=6"},
]

# 使用 apply_chat_template 转换为模型输入
model_inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(torch_device)
print(type(model_inputs))
print(type(GENERATE_KWARGS))
output = model.generate(input_ids=model_inputs, **GENERATE_KWARGS, )
decoded = tokenizer.decode(output[0])
decoded_edited = decoded.split("\n", 1)[-1].strip()
print(decoded_edited)
