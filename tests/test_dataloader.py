import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from train.dataloader.dataloader import get_dataloader
from transformers import AutoTokenizer

dataset_path = "/local_nvme/liaomengqi/gkv/data/math1k.json"
model_name = "/local_nvme/liaomengqi/hugingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

batch_size = 2
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataloader = get_dataloader(dataset_path, batch_size, tokenizer)

print(len(dataloader))

for batch in dataloader:
    print(batch)
    break