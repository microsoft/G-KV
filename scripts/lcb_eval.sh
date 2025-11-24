export CUDA_VISIBLE_DEVICES=0
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

python3 -m lcb_pred \
--model $MODEL \
--save_path ./outputs/test.json \
--budget 512 \
--max_new_tokens 32768 \
--batch_size 128 \
--method score \
--divide_length 128 \
--window_size 16 \
--suppressing_redundancy \
--mix_lambda 0.1 \
--enable_pooling \
--n 4 \
--top_p 0.95 \
--temperature 0.6