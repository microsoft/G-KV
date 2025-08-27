export HF_HOME=/local_nvme/liaomengqi/hugingface
export TORCH_EXTENSIONS_DIR=/local_nvme/.cache/torch_extensions
export VLLM_CACHE_ROOT=/local_nvme/.cache/vllm
export TRITON_CACHE_DIR=/local_nvme/.cache/triton
export TORCH_COMPILE_DEBUG_DIR=/local_nvme/.cache/torch_compile
export TEMP=/local_nvme/tmp

CUDA_VISIBLE_DEVICES=3 python -m inference.vllm_infer \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 1 \
    --output-file data/inference_results.jsonl \
    --split_index 3 \
    --num_split 4 \
    --sample_num 1