# Standard imports
import os
import logging
# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = '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'
from transformer_lens.hook_points import HookPoint
import torch
import pandas as pd
from accelerate import Accelerator
import random
import concurrent.futures
from functools import partial
torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()
from jaxtyping import Float
from torch import Tensor
import queue
import threading
from datasets import load_dataset
import re
import time
from functools import partial
torch.set_grad_enabled(False)

from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
# print(df) # 打印可使用的sae
# from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer
import os


# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 记录 INFO 级别及以上的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler('1.3.log'),  # 输出到文件 app.log
        logging.StreamHandler()  # 输出到控制台
    ]
)

all_feature = [i for i in range(0, 16383)] # gpt2 24576 gemma 16383
# 设置批次大小和并行数
BATCH_SIZE = 2000
all_feature_batches = [all_feature[i:i + BATCH_SIZE] for i in range(0, len(all_feature), BATCH_SIZE)]

'''
Part Ablation Function
'''

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)


def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        steering_coefficient=steering_coefficient,
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output

'''
Part Concurrent
'''
def process_feature_batch(batch, input_list, model, sae, compute_repeat_score, GENERATE_KWARGS):
    """
    处理一个批次的特征进行消融实验（不重新加载模型）
    """
    feature_ablation_results_batch = {}
    for feature_idx in batch:
        ablation_effects = []  # 每个 feature_idx 开始时清空列表
        input = random.choice(input_list)  # 从 input_list 中随机选择一个输入

        part_start_time = time.time()  # 记录当前部分的开始时间
        # 生成未应用 steering 的输出
        no_steering_output = model.generate(input, max_new_tokens=50, **GENERATE_KWARGS)
        original_repeat_score = compute_repeat_score(no_steering_output)
        repeat_scores = []  # 用于存储每个 steered output 的重复分数

        for i in range(3):
            steered_output = generate_with_steering(
                model,
                sae,
                input,
                feature_idx,
                steering_coefficient=500,  # 设个常量的 steering 系数
            ).replace("\n", "↵")

            row_repeat_score = compute_repeat_score(steered_output)
            repeat_scores.append(row_repeat_score)


        # 计算 ablation 效果
        for score in repeat_scores:
            ablation_effect = score - original_repeat_score
            ablation_effects.append(ablation_effect)

        max_ablation_effect = max(ablation_effects)
        mean_ablation_effect = sum(ablation_effects) / len(ablation_effects)

        feature_ablation_results_batch[feature_idx] = {
            'max_ablation_effect': max_ablation_effect,
            'mean_ablation_effect': mean_ablation_effect
        }

        part_end_time = time.time()
        part_cost_time = part_end_time - part_start_time
        logging.info(f"Processed Feature {feature_idx} changed repeat score by {ablation_effect:.2f}, cost time: {part_cost_time:.2f} seconds") # type: ignore
        # 将批次的输出添加到队列中

    return feature_ablation_results_batch

def parallel_feature_processing(all_feature_batches, input_list, model, sae, compute_repeat_score, GENERATE_KWARGS):
    """
    使用并行计算处理所有特征批次，模型只加载一次
    """
    all_results = {}


    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 设置最大工作线程数
        futures = []
        for batch in all_feature_batches:
            futures.append(executor.submit(process_feature_batch, batch, input_list, model, sae, compute_repeat_score, GENERATE_KWARGS))

        # 获取所有并行任务的结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_results.update(result)

    return all_results


# n-gram
import n_gram
def compute_repeat_score(input_sentence):
    return n_gram.calculate_rep_n(input_sentence, 1)

def main():
    # 初始化并加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(device)
    '''
    Part1 Model Preparation
    '''
    # # gpt2
    # llm = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    #
    # # gpt2 sae
    # sae, cfg_dict, sparsity = SAE.from_pretrained(
    #     release="gpt2-small-res-jb",  # <- Release name
    #     sae_id="blocks.11.hook_resid_pre",  # <- SAE id (not always a hook point!)
    #     device=device
    # )

    # # llama
    # llm = HookedSAETransformer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device=device)
    #
    # # llama sae
    # sae, cfg_dict, sparsity = SAE.from_pretrained(
    #     release="llama-3-8b-it-res-jh	",  # <- Release name
    #     sae_id="blocks.25.hook_resid_post",  # <- SAE id (not always a hook point!)
    #     device=device
    # )

    # gemma
    llm = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device=device)

    # gemma sae
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",  # <- Release name
        sae_id="layer_22/width_16k/canonical",  # <- SAE id (not always a hook point!)
        device=device
    )

    '''
    Part Load the dataset
    '''

    # Dataset
    ds = load_dataset("DisgustingOzil/Academic_dataset_ShortQA")

    input_list = []

    for i in range(50):  # 调整测试的数量
        response = ds['train']['response'][i]
        # 使用正则表达式匹配 <question> 标签之间的内容
        question = re.search(r'<question>(.*?)</question>', response)
        question = question.group(1).strip()  # 打印匹配到的 question 内容
        input_list.append(question)
    logging.info(f"input list: {input_list}")


    # 并行处理
    whole_start_time = time.time()
    ablation_results = parallel_feature_processing(
        all_feature_batches, input_list, llm, sae, compute_repeat_score, GENERATE_KWARGS
    )

    # 按照 max_ablation_effect 排序
    sorted_feature_results = sorted(
        ablation_results.items(),
        key=lambda x: x[1]['max_ablation_effect'],
        reverse=True  # 如果要从高到低排序
    )

    # 输出排序后的结果
    for feature_idx, result in sorted_feature_results:
        logging.info(f"Feature {feature_idx} - Max Ablation Effect: {result['max_ablation_effect']:.2f}")
        logging.info(f"Feature {feature_idx} - Mean Ablation Effect: {result['mean_ablation_effect']:.2f}")

    whole_end_time = time.time()
    logging.info(f"Total time: {whole_end_time - whole_start_time:.2f} seconds")

if __name__ == "__main__":
    main()