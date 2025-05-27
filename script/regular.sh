#!/bin/bash

MODELS=(
    "google/gemma-2-2b:gemma-2-2b"
    "meta-llama/Llama-3.1-8B:Llama-3.1-8B"
    "gpt2:gpt2"
)

DATASETS=(
    "DisgustingOzil/Academic_dataset_ShortQA:AQ"
    "YokyYao/Diversity_Challenge:EQ"
    "sentence-transformers/natural-questions:NQ"
)

METHODS=(
    "greedy"
    "beam"
    "topk"
    "topp"
)

# 遍历所有模型
for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL%%:*}"  # 提取冒号前的模型路径
    MNAME="${MODEL##*:}"       # 提取冒号后的显示名称

    # 遍历所有数据集
    for DATASET in "${DATASETS[@]}"; do
        DATASET_PATH="${DATASET%%:*}"  # 提取冒号前的数据集路径
        DNAME="${DATASET##*:}"         # 提取冒号后的显示名称

        SAVE_PATH="/root/repeatcurse/results2/${MNAME}/${DNAME}"
        mkdir -p "$SAVE_PATH"

        # 遍历所有推理方法
        for METHOD in "${METHODS[@]}"; do
            echo "Running model: $MNAME, dataset: $DNAME, method: $METHOD"
            python ../regular.py \
                --model_path "$MODEL_PATH" \
                --dataset "$DATASET_PATH" \
                --method "$METHOD" \
                --save_path "${SAVE_PATH}/${DNAME}_${METHOD}_result.json"
        done
    done
done

echo "All completed!"