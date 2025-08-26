#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

# This script runs evaluations by sweeping over a parameter.
# In this example, we are sweeping the 'alpha' parameter.
# You can change PARAM_NAME and PARAM_VALUES to sweep over other parameters.


PARAM_NAME="kv_budget"
PARAM_VALUES=(2048)

for val in "${PARAM_VALUES[@]}"
do
    # Format value for file name (e.g., 0.9 -> 09, 0.95 -> 095)
    val_str=$(echo $val | sed 's/\.//g')

    # Modify the save_path to include the parameter name and value
    SAVE_PATH="./outputs/aime24_dsllama8b_h2o_${PARAM_NAME}_${val_str}.jsonl"

    python3 ./run_math.py \
    --dataset_path ./data/aime24.jsonl \
    --save_path ${SAVE_PATH} \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max_length 32768 \
    --eval_batch_size 90 \
    --method ikv \
    --window_size 16 \
    --divide_length 128 \
    --record_pos_ids \
    --${PARAM_NAME} ${val} \
    --n_sample 32 \
    --do_sample \
    --top_p 0.95 \
    --temperature 0.6
done
