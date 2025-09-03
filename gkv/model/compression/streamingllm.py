import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class StreamingLLMKV:
    def __init__(
        self,
        window_size=512,
        sink_len=4,
        **kwargs,
    ):
        self.window_size = window_size
        self.sink_len = sink_len

    def update_kv(
        self,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        key_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
        value_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
        """
        if key_states.shape[-2] < self.window_size + self.sink_len:
            return key_states, value_states, None

        if torch.all(attention_mask != 0):
            sink_key_states = key_states[:, :, : self.sink_len, :]
            window_key_states = key_states[:, :, -self.window_size :, :]
            key_states = torch.cat([sink_key_states, window_key_states], dim=-2)

            sink_value_states = value_states[:, :, : self.sink_len, :]
            window_value_states = value_states[:, :, -self.window_size :, :]
            value_states = torch.cat([sink_value_states, window_value_states], dim=-2)
        else:
            pos = torch.cumsum(attention_mask, dim=1)
            sink_mask = (pos <= self.sink_len) & (attention_mask != 0)
            window_mask = torch.ones_like(
                attention_mask, dtype=torch.bool, device=attention_mask.device
            )
            window_mask[:, : -self.window_size] = False
            keep_mask = sink_mask | window_mask
            sorted_mask, sorted_indices = torch.sort(keep_mask.float(), dim=1)
            keep_indices = sorted_indices[:, -(self.window_size + self.sink_len) :]
            keep_indices = torch.sort(keep_indices, dim=1).values
            keep_indices = keep_indices[:, None, :, None].expand(
                key_states.shape[0], key_states.shape[1], -1, key_states.shape[-1]
            )
            key_states = key_states.gather(dim=2, index=keep_indices)
            value_states = value_states.gather(dim=2, index=keep_indices)
            
        return key_states, value_states, None
