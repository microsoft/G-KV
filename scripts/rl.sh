#!/bin/bash
export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
DATASET_PATH="agentica-org/DeepScaleR-Preview-Dataset"


# Sparse parameters
COMPRESS_STEP=128
WINDOW_SIZE=16
KV_BUDGET=512
ALPHA=0.8
MIX_LAMBDA=0.5

# train parameters
LEARNING_RATE=1e-6
EXP_NAME="qwen7b_gkv"
MAX_TRAIN_STEPS=1000
MAX_OUTPUT_LEN=4096

accelerate launch \
    --num_processes 4 \
    --config_file ./configs/acce_config.yaml \
    -m gkv.rl_main \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_path agentica-org/DeepScaleR-Preview-Dataset \
    --max_new_tokens 4096 \
    --learning_rate 1e-6 \
    --divide_length 128 \
    --window_size 16 \
    --enable_score_cache \
    --suppressing_redundancy \
    --kv_budget 512 \
    --alpha 0.8 \
    --mix_lambda 0.5 \
    --train_micro_batch_size_per_gpu 1 \
    --train_batch_size_per_gpu 2 \
    --generate_batch_size_per_gpu 32 \
    --eval_steps 10 \
    --trunk_length 4096 \

