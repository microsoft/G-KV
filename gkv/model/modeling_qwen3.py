from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3Config,
    FlashAttentionKwargs,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
    apply_rotary_pos_emb,
    Unpack,
    Qwen3RMSNorm,
    Cache,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)

from typing import Optional, Tuple, Callable

import torch
from torch import nn


from .compression import ScoreBasedKV, SepLLMKV, StreamingLLMKV


# flex_attention = torch.compile(flex_attention, dynamic=False)
KV_COMPRESSION_MAP = {
    "score": ScoreBasedKV,
    "sepllm": SepLLMKV,
    "streamingllm": StreamingLLMKV,
}


def qwen3_attn_init(self, config: Qwen3Config, layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )
    self.q_norm = Qwen3RMSNorm(
        self.head_dim, eps=config.rms_norm_eps
    )  # unlike olmo, only on the head dim!
    self.k_norm = Qwen3RMSNorm(
        self.head_dim, eps=config.rms_norm_eps
    )  # thus post q_norm does not need reshape
    self.sliding_window = config.sliding_window
    if not (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        self.sliding_window = None

    self.kv_cluster = KV_COMPRESSION_MAP[config.method](**config.method_config)
    self.divide_length = config.method_config.get("divide_length", 128)
    self.window_size = config.method_config.get("window_size", 16)
    self.budget = config.method_config.get("budget", 512)
    self.alpha = config.method_config.get("alpha", 0.8)
    self.mix_lambda = config.method_config.get("mix_lambda", 0.5)
    self.method = config.method_config.get("method", "score")


def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        # =============== query cache ================
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}
        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage, initial query cache
            if self.method == "score":
                past_key_value.query_cache[self.layer_idx] = query_states[
                    :, :, -self.config.method_config["window_size"] :, :
                ]
                if (
                    self.config.method_config["enable_score_cache"]
                    and self.config.method_config["smooth_method"] == "sum"
                    and self.alpha == 1
                ):
                    # h2o initial score cache
                    score_cache = self.kv_cluster.initial_score_cache(
                        key_states=key_states,
                        query_states=query_states,
                        attention_mask=attention_mask,
                    )
                    past_key_value.score_cache[self.layer_idx] = score_cache

            else:
                past_key_value.query_cache[self.layer_idx] = None
        else:
            # decoding stage, add new query to cache
            if self.method == "score":
                past_key_value.query_cache[self.layer_idx] = torch.cat(
                    (past_key_value.query_cache[self.layer_idx], query_states),
                    dim=2,
                )
                # keep only window_size most recent queries
                window_size = self.config.method_config["window_size"]
                if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                    past_key_value.query_cache[self.layer_idx] = (
                        past_key_value.query_cache[self.layer_idx][
                            :, :, -window_size:, :
                        ]
                    )
        # =============== end query cache ================
        if hasattr(past_key_value, "pos_ids_cache"):
            pos_ids_cache = past_key_value.pos_ids_cache[self.layer_idx]
        else:
            pos_ids_cache = None
        # =============== kv cache compression ================
        if kwargs["enable_compress"]:
            query_cache = past_key_value.query_cache[self.layer_idx]
            # notice if the length of kv cache is smaller than budget, then the kv cache will not be compressed
            (
                compressed_key_states,
                compressed_value_states,
                pos_ids_cache,
                score_cache,
            ) = self.kv_cluster.update_kv(
                key_states=key_states,
                query_states=query_cache,
                value_states=value_states,
                pos_ids_cache=pos_ids_cache,
                cur_len=kwargs["cur_len"],
                attention_mask=(
                    past_key_value.attention_mask
                    if hasattr(past_key_value, "attention_mask")
                    else None
                ),
                unfinished_sequences=past_key_value.unfinished_sequences,
                score_cache=past_key_value.score_cache[self.layer_idx],
            )
            past_key_value.key_cache[self.layer_idx] = compressed_key_states
            past_key_value.value_cache[self.layer_idx] = compressed_value_states
            past_key_value.score_cache[self.layer_idx] = score_cache
            if pos_ids_cache is not None:
                past_key_value.pos_ids_cache[self.layer_idx] = pos_ids_cache
        # =============== end kv cache compression ================

    # =============== keep only the most recent attention mask ================
    if attention_mask is not None and attention_mask.shape[-1] > key_states.shape[-2]:
        if len(attention_mask.shape) == 4:
            # mask for eager attention
            attention_mask = attention_mask[
                :, :, -key_states.shape[-2] :, -key_states.shape[-2] :
            ].contiguous()
        elif len(attention_mask.shape) == 2:
            # mask for flash attention
            attention_mask = attention_mask[:, -key_states.shape[-2] :].contiguous()
    # =============== end keep only the most recent attention mask ================

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
