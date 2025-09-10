set -x

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_PATH="mqliao/gkv_distill_math_27k"
EVAL_DATASET_PATH="agentica-org/DeepScaleR-Preview-Dataset"


# Sparse parameters
DIVIDE_LENGTH=128
WINDOW_SIZE=16
BUDGET=2048
ALPHA=0.8
MIX_LAMBDA=0.5

# train parameters
LEARNING_RATE=5e-7
EXP_NAME="qwen7b_sft_kl"
MAX_TRAIN_STEPS=250
EVAL_STEPS=20
MAX_OUTPUT_LEN=4096

accelerate launch \
    --num_processes 8 \
    --config_file ./configs/acce_config.yaml \
    -m gkv.sft_main \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --eval_dataset_path $EVAL_DATASET_PATH \
    --eval_steps $EVAL_STEPS \
    --eval_do_sample \
    --learning_rate $LEARNING_RATE \
    --method score \
    --use_kl_loss \
    --ref_model_offload \
    --divide_length $DIVIDE_LENGTH \
    --window_size $WINDOW_SIZE \
    --budget $BUDGET \
    --alpha $ALPHA \
    --mix_lambda $MIX_LAMBDA \
    --exp_name $EXP_NAME \
    --max_train_steps $MAX_TRAIN_STEPS \
    --max_output_len $MAX_OUTPUT_LEN \
    --save_steps 50

