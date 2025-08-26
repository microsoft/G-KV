import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, AutoConfig
from train.model.modeling_sparse_qwen2 import Qwen2SparseModelForCausalLM

model_name = "/local_nvme/liaomengqi/hugingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.compress_step = 128
config.window_size = 16
config.kv_budget = 512
config.alpha = 0.9
config.mix_lambda = 0.1
model = Qwen2SparseModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

model.to("cuda")
model.train()

prompt = ["Hello, how are you?", "I am fine, thank you!"]

inputs = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
position_ids = inputs.attention_mask.cumsum(dim=-1) - 1

with torch.no_grad():
    logits = model(
        inputs.input_ids.to("cuda"),
        attention_mask=inputs.attention_mask.to("cuda"),
        position_ids=position_ids.to("cuda"),
        input_length=inputs.input_ids.shape[-1],
        logits_to_keep=1,
    )
    print(logits.shape)
