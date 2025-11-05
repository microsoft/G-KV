set -x


# method

# SnapKV
# --method score
# --enable_pooling

# R-KV
# --method score
# --enable_pooling
# --suppressing_redundancy
# --mix_lambda 0.1

# G-KV
# --method score
# --suppressing_redundancy
# --mix_lambda 0.5
# --enable_score_cache
# --alpha 0.8

# StreamingLLM
# --method streamingllm
# --sink_len 4
# --window_size 512

# H2O
# --method score
# --enable_score_cache
# --smooth_method sum
# --window_size 1

# FullKV
# --method fullkv


# for efficiency analysis
# DATASET_PATH="agentica-org/DeepScaleR-Preview-Dataset"
# --n_sample 1
# --split_len 1024

export CUDA_VISIBLE_DEVICES=7
#DATASET_PATH="zwhe99/amc23"
DATASET_PATH="math-ai/aime24"
MODEL="./checkpoints/qwen7b_rl/checkpoint-400"
#MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

python3 -m gkv.inference_main \
--dataset_path $DATASET_PATH \
--model_path $MODEL \
--save_path ./outputs/aime_4k_rl_2048.jsonl \
--budget 2048 \
--max_new_tokens 4096 \
--eval_batch_size 128 \
--method score \
--divide_length 128 \
--window_size 16 \
--suppressing_redundancy \
--mix_lambda 0.5 \
--enable_score_cache \
--alpha 0.8 \
--n_sample 32 \
--do_sample \
--top_p 0.95 \
--temperature 0.6