import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SepLLMKV:
    def __init__(
        self,
        budget=512,
        window_size=16,
        sink_len=4,
        **kwargs,
    ):
        self.budget = budget
        self.window_size = window_size
        self.sink_len = sink_len

    def update_kv(
        self,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sep_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        raise NotImplementedError("SepLLMKV is not implemented yet")
