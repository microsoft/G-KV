from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2Model,
)
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    FlashAttentionKwargs,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    Unpack,
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2RotaryEmbedding,
    add_start_docstrings,
    QWEN2_START_DOCSTRING,
    GenerationMixin,
    KwargsForCausalLM,
    CausalLMOutputWithPast,
    logger,
)
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)
from functools import partial
import math
from .sparse_mask import build_sparse_mask
import torch.nn.functional as F

KV_BLOCK_SIZE = 128
Q_BLOCK_SIZE = 128


class Qwen2SparseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.compress_step = (
            config.compress_step if hasattr(config, "compress_step") else 128
        )
        self.window_size = config.window_size if hasattr(config, "window_size") else 16
        self.kv_budget = config.kv_budget if hasattr(config, "kv_budget") else 1024
        self.alpha = config.alpha if hasattr(config, "alpha") else 0.8
        self.mix_lambda = config.mix_lambda if hasattr(config, "mix_lambda") else 0.5

    def build_block_mask(
        self, attention_mask: torch.Tensor, num_query_heads: int
    ) -> torch.Tensor:
        """
        build block mask from attention mask
        attention mask: bool (bsz,kv_head,seq_len,seq_len), True means valid, False means masked
        """

        def repeat_mask(mask: torch.Tensor, n_rep: int) -> torch.Tensor:
            """
            This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The mask go from (batch,kv_head,seq_len,seq_len) to (batch,kv_head,seq_len,seq_len)
            """
            bsz, kv_head, seq_len, seq_len = mask.shape
            return (
                mask.unsqueeze(2)
                .expand(bsz, kv_head, n_rep, seq_len, seq_len)
                .reshape(bsz, kv_head * n_rep, seq_len, seq_len)
            )

        attention_mask = repeat_mask(
            attention_mask, num_query_heads // attention_mask.shape[1]
        )

        def sparse_kernel(b, h, q_idx, kv_idx):
            return attention_mask[b, h, q_idx, kv_idx].view([]).detach().clone()

        B, H, Sq, Sk = attention_mask.shape

        block_mask = create_block_mask(
            sparse_kernel,
            B,
            H,
            Sq,
            Sk,
            BLOCK_SIZE=(KV_BLOCK_SIZE, Q_BLOCK_SIZE),
            device=attention_mask.device,
            _compile=False,
        )
        return block_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        input_length: int,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if len(attention_mask.shape) == 2:
            sparse_mask = build_sparse_mask(
                query_states,
                key_states,
                attention_mask,
                input_length,
                compress_step=self.compress_step,
                window_size=self.window_size,
                kv_budget=self.kv_budget,
                alpha=self.alpha,
                mix_lambda=self.mix_lambda,
            )
        else:
            sparse_mask = attention_mask

        block_mask = self.build_block_mask(sparse_mask, query_states.shape[1])

        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=True,
        ).transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2SparseDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2SparseAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            input_length=input_length,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2SparseModel(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2SparseDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        input_length: int,
    ):

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__),
                    hidden_states,
                    input_length,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    input_length=input_length,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2SparseModelForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2SparseModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_length: int,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        logits_to_keep: int,
        labels: torch.LongTensor,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_length=input_length,
            **kwargs,
        )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
        )

        return loss
    
    def loss_function(self, logits, labels, vocab_size, **kwargs):
        logits=logits[:,:-1,:]
        labels=labels[:,1:]
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
        return loss
