set -x

export CUDA_VISIBLE_DEVICES=1

DATASET_PATH="zwhe99/amc23"
# DATASET_PATH="math-ai/aime24"

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

# FullKV
# --method fullkv


# for efficiency analysis
# DATASET_PATH="agentica-org/DeepScaleR-Preview-Dataset"
# --n_sample 1
# --split_len 1024

python3 -m gkv.inference_main \
--dataset_path $DATASET_PATH \
--model_path ./checkpoints/qwen7b_sft_kl/checkpoint-250 \
--save_path ./outputs/train_qwen_sft_1024.jsonl \
--max_new_tokens 16384 \
--eval_batch_size 128 \
--method score \
--divide_length 128 \
--window_size 16 \
--budget 1024 \
--suppressing_redundancy \
--mix_lambda 0.5 \
--enable_score_cache \
--alpha 0.8 \
--n_sample 32 \
--do_sample \
--top_p 0.95 \
--temperature 0.6
