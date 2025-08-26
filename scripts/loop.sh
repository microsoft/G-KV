ROOT_DIR=/cosmos/liaomengqi/gkv

/home/aiscuser/.conda/envs/py10/bin/python ${ROOT_DIR}/loop.py \
--dataset_path ${ROOT_DIR}/data/amc23.jsonl \
--save_path ${ROOT_DIR}/outputs/test.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
--max_length 16384 \
--eval_batch_size 160 \
--method ikv \
--kv_budget 512 \
--window_size 16 \
--divide_length 128 \
--enable_score_cache \
--alpha 0 \
--n_sample 32 \
--do_sample \
--top_p 0.95 \
--temperature 0.6