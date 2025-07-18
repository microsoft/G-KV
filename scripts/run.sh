set -x
export CUDA_VISIBLE_DEVICES=3

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

python3 ./run_math.py \
--dataset_path ./data/amc23.jsonl \
--save_path ./outputs/amc23_h2o_mean_a08.jsonl \
--model_path /data/MaoXiaowei/models/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B \
--max_length 16384 \
--eval_batch_size 160 \
--method ikv \
--kv_budget 512 \
--window_size 32 \
--divide_length 128 \
--smooth_method mean \
--enable_score_cache \
--alpha 0.8 \
--n_sample 32 \
--do_sample \
--top_p 0.95 \
--temperature 0.6