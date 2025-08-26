from train.model.modeling_qwen2_mask_builder import Qwen2MaskBuilderAttention
from argparse import Namespace
import torch
config={
    "kv_budget":512,
    "window_size":16,
    "alpha":0.8,
    "mix_lambda":0.1,
    "compress_step":128,
    "head_dim":64,
    "num_attention_heads":12,
    "num_key_value_heads":12,
    "hidden_size":768,
    "attention_dropout":0.0
}


config=Namespace(**config)



atten=Qwen2MaskBuilderAttention(config,0)

bsz=2
seqlen=1024
attention_mask=torch.LongTensor([[1]*1024,[0]*128+[1]*896])

query_states=torch.randn(bsz,config.num_attention_heads,seqlen,config.head_dim)
key_states=torch.randn(bsz,config.num_key_value_heads,seqlen,config.head_dim)
input_len=256

atten.build_sparse_mask(query_states,key_states,attention_mask,input_len)







