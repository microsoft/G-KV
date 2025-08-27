#!/bin/bash

# 设置环境变量
export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
DATASET_PATH="data/valid_train_data_27k.json"


# Sparse parameters
COMPRESS_STEP=128
WINDOW_SIZE=16
KV_BUDGET=512
ALPHA=0.8
MIX_LAMBDA=0.5

# train parameters
LEARNING_RATE=1e-6
EXP_NAME="test"
MAX_TRAIN_STEPS=1000
MAX_OUTPUT_LEN=4096

VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --config_file ./configs/acce_config.yaml \
    -m train.tain_main \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --learning_rate $LEARNING_RATE \
    --compress_step $COMPRESS_STEP \
    --window_size $WINDOW_SIZE \
    --kv_budget $KV_BUDGET \
    --alpha $ALPHA \
    --mix_lambda $MIX_LAMBDA \
    --exp_name $EXP_NAME \
    --max_train_steps $MAX_TRAIN_STEPS \
    --max_output_len $MAX_OUTPUT_LEN

