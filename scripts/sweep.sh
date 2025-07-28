#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=4

# This script runs evaluations by sweeping over a parameter.
# In this example, we are sweeping the 'alpha' parameter.
# You can change PARAM_NAME and PARAM_VALUES to sweep over other parameters.

PARAM_NAME="alpha"
PARAM_VALUES=(0  0.2  0.4  0.6  0.8 0.9 1)

for val in "${PARAM_VALUES[@]}"
do
    # Format value for file name (e.g., 0.9 -> 09, 0.95 -> 095)
    val_str=$(echo $val | sed 's/\.//g')

    # Modify the save_path to include the parameter name and value
    SAVE_PATH="./outputs/amc23_dsqwen7b_rkv_${PARAM_NAME}_${val_str}.jsonl"

    python3 ./run_math.py \
    --dataset_path ./data/amc23.jsonl \
    --save_path ${SAVE_PATH} \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --max_length 16384 \
    --eval_batch_size 320 \
    --method ikv \
    --kv_budget 512 \
    --window_size 16 \
    --divide_length 128 \
    --suppressing_redundancy \
    --mix_lambda 0.3 \
    --enable_score_cache \
    --${PARAM_NAME} ${val} \
    --n_sample 32 \
    --do_sample \
    --top_p 0.95 \
    --temperature 0.6
done
