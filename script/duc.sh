#!/bin/bash

MODELS=(
    "meta-llama/Llama-3.1-8B:Llama-3.1-8B"
    #"google/gemma-2-2b:gemma-2-2b"
    #"gpt2:gpt2"
)

DATASETS=(
    #"DisgustingOzil/Academic_dataset_ShortQA:AQ"

    "YokyYao/Diversity_Challenge:EQ"

    #"sentence-transformers/natural-questions:NQ"
)

METHODS=("ours")

# 遍历所有模型
for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL%%:*}" 
    MNAME="${MODEL##*:}"       

    # 遍历所有数据集
    for DATASET in "${DATASETS[@]}"; do
        DATASET_PATH="${DATASET%%:*}"  
        DNAME="${DATASET##*:}"         

        SAVE_PATH="/root/repeatcurse/test/${MNAME}/${DNAME}"
        mkdir -p "$SAVE_PATH"

        # 遍历所有推理方法
        for METHOD in "${METHODS[@]}"; do
            echo "Running model: $MNAME, dataset: $DNAME, method: $METHOD"
            python /root/repeatcurse/Feature_Repeatscore_Batch_Activation.py \
                --model_path "$MODEL_PATH" \
                --dataset "$DATASET_PATH" \
                --save_path "${SAVE_PATH}/${DNAME}_${METHOD}_result.json"
        done
    done
done

echo "All completed!"