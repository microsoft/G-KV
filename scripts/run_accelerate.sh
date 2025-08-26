#!/bin/bash

# 设置环境变量
export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

# 模型和数据集参数
MODEL_NAME="Qwen/Qwen2-1.5B"  # 修改为您的模型路径
DATASET_PATH="data/train.json"  # 修改为您的数据集路径

# 训练参数
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16
EPOCHS=1
LEARNING_RATE=1e-6

# Sparse模型参数
COMPRESS_STEP=128
WINDOW_SIZE=16
KV_BUDGET=512
ALPHA=0.8
MIX_LAMBDA=0.5

# 使用accelerate launch启动训练
accelerate launch \
    --num_processes 2 \
    --config_file ./configs/config.yaml \
    -m train.tain_main \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --compress_step $COMPRESS_STEP \
    --window_size $WINDOW_SIZE \
    --kv_budget $KV_BUDGET \
    --alpha $ALPHA \
    --mix_lambda $MIX_LAMBDA \
    --seed 42

