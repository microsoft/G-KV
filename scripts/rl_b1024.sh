
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_PATH="agentica-org/DeepScaleR-Preview-Dataset"



# train parameters
LEARNING_RATE=1e-6
EXP_NAME="qwen7b_rl_b1024"
MAX_TRAIN_STEPS=400
TRAIN_BATCH_SIZE_PER_GPU=2
BUDGET=1024
TRAIN_MICRO_BATCH_SIZE_PER_GPU=1

accelerate launch \
    --num_processes 8 \
    --config_file ./configs/acce_config.yaml \
    -m gkv.rl_main \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --max_new_tokens 6144 \
    --learning_rate $LEARNING_RATE \
    --divide_length 128 \
    --window_size 16 \
    --enable_score_cache \
    --suppressing_redundancy \
    --budget $BUDGET \
    --alpha 0.8 \
    --mix_lambda 0.5 \
    --train_micro_batch_size_per_gpu $TRAIN_MICRO_BATCH_SIZE_PER_GPU \
    --train_batch_size_per_gpu $TRAIN_BATCH_SIZE_PER_GPU \
    --generate_batch_size_per_gpu 64 \
    --eval_steps 10 \
    --trunk_length 4096 \
    --exp_name $EXP_NAME \
    --max_train_steps $MAX_TRAIN_STEPS \
    --eval_do_sample \

