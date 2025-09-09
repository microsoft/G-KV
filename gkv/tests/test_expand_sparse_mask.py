import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gkv.model.sparse_mask import expand_sparse_mask
import torch


def test_expand_sparse_mask():

    sparse_mask = torch.tensor(
        [
            [[True, False, True], [True, True, False], [True, True, True]],
            [[False, True, False], [False, True, False], [False, True, True]],
        ]
    )
    sparse_mask = sparse_mask.unsqueeze(1).expand(-1, 2, -1, -1)

    expand_len = 6
    kept_pos = torch.tensor([[ 0, 3,4, 5], [1,3, 4, 5]])
    kept_pos = kept_pos.unsqueeze(1).expand(-1, 2, -1)
    expanded_sparse_mask = expand_sparse_mask(sparse_mask, expand_len, kept_pos)
    print(expanded_sparse_mask)


test_expand_sparse_mask()
