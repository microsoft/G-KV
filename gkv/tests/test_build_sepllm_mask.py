import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gkv.model.sparse_mask import build_SepLLM_mask
import torch

sep_len = 4096
input_ids = torch.randint(0, 100, (2, sep_len))
attention_mask = torch.Tensor([[0] * 100 + [1] * (sep_len - 100), [1] * sep_len]).long()
keep_dis = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).long()
sink_len = 4
sep_cache_len = 512
window_size = 16

mask = build_SepLLM_mask(
    input_ids, attention_mask, keep_dis, sink_len, sep_cache_len, window_size
)
print(mask)
print(mask.shape)
