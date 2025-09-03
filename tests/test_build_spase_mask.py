import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gkv.model.sparse_mask import build_sparse_mask
import torch


def test_build_sparse_mask():
    bsz = 2
    kv_head = 2
    seq_len = 10
    head_dim = 64
    query_states = torch.randn(bsz, kv_head, seq_len, head_dim)
    key_states = torch.randn(bsz, kv_head, seq_len, head_dim)
    attention_mask = torch.tensor(
        [
            [0]*1+[1]*(seq_len-1),
            [1]*seq_len,
        ]
    )
    input_length = 2
    divide_length = 2
    window_size = 2
    budget = 4
    alpha = 0.8
    mix_lambda = 0.5
    sparse_mask = build_sparse_mask(
        query_states,
        key_states,
        attention_mask,
        input_length,
        divide_length,
        window_size,
        budget,
        alpha,
        mix_lambda,
    )
    print(sparse_mask)


if __name__ == "__main__":
    test_build_sparse_mask()
