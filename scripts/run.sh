set -x

export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

# --enable_pooling
# --suppressing_redundancy
# --causal_salience_score 
# --shift 
# --combine_entropy_layer
# --mix_lambda
# --enable_score_cache
# --alpha

# /data/MaoXiaowei/models/model/LLM-Research/Llama-3___2-3B-Instruct
# /data/MaoXiaowei/models/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B



# python3 ./run_math.py \
# --dataset_path ./data/amc23.jsonl \
# --save_path ./outputs/amc23_dsllma8b_gkv_alpha_0.jsonl \
# --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
# --max_length 16384 \
# --eval_batch_size 160 \
# --method ikv \
# --window_size 16 \
# --divide_length 128 \
# --kv_budget 512 \
# --enable_score_cache \
# --alpha 0 \
# --suppressing_redundancy \
# --mix_lambda 0.9 \
# --n_sample 32 \
# --do_sample \
# --top_p 0.95 \
# --temperature 0.6

# python3 ./run_math.py \
# --dataset_path ./data/aime24.jsonl \
# --save_path ./outputs/aime24_dsllama8b_fullkv.jsonl \
# --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
# --max_length 32768 \
# --eval_batch_size 12 \
# --method fullkv \
# --window_size 16 \
# --divide_length 128 \
# --n_sample 32 \
# --do_sample \
# --top_p 0.95 \
# --temperature 0.6

python3 ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./outputs/aime24_dsqwen7b_fullkv.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
--max_length 32768 \
--eval_batch_size 16 \
--method fullkv \
--window_size 16 \
--divide_length 128 \
--n_sample 32 \
--do_sample \
--top_p 0.95 \
--temperature 0.6

# bash /cosmos/liaomengqi/gkv/scripts/loop.sh