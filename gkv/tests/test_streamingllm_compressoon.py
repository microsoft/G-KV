import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from typing import Optional


def update_kv(
    window_size=512,
    sink_len=4,
    key_states: Optional[torch.Tensor] = None,
    value_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    key_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
    query_states: (bsz, num_q_heads, window_size, head_dim)
    value_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
    """
    if key_states.shape[-2] < window_size + sink_len:
        return key_states, value_states, None

    if torch.all(attention_mask != 0):
        sink_key_states = key_states[:, :, :sink_len, :]
        window_key_states = key_states[:, :, -window_size:, :]
        key_states = torch.cat([sink_key_states, window_key_states], dim=-2)

        sink_value_states = value_states[:, :, :sink_len, :]
        window_value_states = value_states[:, :, -window_size:, :]
        value_states = torch.cat([sink_value_states, window_value_states], dim=-2)
    else:
        pos = torch.cumsum(attention_mask, dim=1)
        sink_mask = (pos <= sink_len) & (attention_mask != 0)
        window_mask = torch.ones_like(
            attention_mask, dtype=torch.bool, device=attention_mask.device
        )
        window_mask[:, :-window_size] = False
        keep_mask = sink_mask | window_mask
        sorted_mask, sorted_indices = torch.sort(keep_mask.float(), dim=1)
        keep_indices = sorted_indices[:, -(window_size + sink_len) :]
        keep_indices = torch.sort(keep_indices, dim=1).values
        keep_indices = keep_indices[:, None, :, None].expand(
            key_states.shape[0], key_states.shape[1], -1, key_states.shape[-1]
        )
        key_states = key_states.gather(dim=2, index=keep_indices)
        value_states = value_states.gather(dim=2, index=keep_indices)
    return key_states, value_states, None


window_size = 16
sink_len = 2
seq_len = window_size + sink_len + 1
key_states = torch.randn(2, 1, seq_len, 128)
value_states = torch.randn(2, 1, seq_len, 128)
attention_mask = torch.LongTensor([[0] * 3 + [1] * (seq_len - 3), [1] * seq_len])
compressed_key_states, compressed_value_states, _ = update_kv(
    window_size, sink_len, key_states, value_states, attention_mask
)

print((key_states[1,:,0,:]-compressed_key_states[1,:,0,:]).sum())
