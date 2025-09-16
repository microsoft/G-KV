import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gkv.model.sparse_mask import build_sparse_mask_from_pos_cache
import torch


def test_build_sparse_mask_from_pos_cache():
    pos_cache_history = [
        (torch.tensor([[0, 2, 3], [1, 2, 3]]).unsqueeze(1), 4),
        (torch.tensor([[2, 4, 5], [1, 4, 5]]).unsqueeze(1), 6),
        (torch.tensor([[4, 5, 7], [1, 5, 7]]).unsqueeze(1), 8),
    ]
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1,1,1], [0, 1, 1, 1, 1, 1, 1, 1,1,1]])
    num_kv_heads = 1
    sparse_mask = build_sparse_mask_from_pos_cache(
        pos_cache_history, attention_mask, num_kv_heads
    )
    print(sparse_mask)


if __name__ == "__main__":
    test_build_sparse_mask_from_pos_cache()
