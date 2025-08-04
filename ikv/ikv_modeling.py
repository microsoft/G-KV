import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from .compression import ImformativeKV
import os
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers.generation.streamers import BaseStreamer

KV_COMPRESSION_MAP = {
    "ikv": ImformativeKV,
}

logger = logging.get_logger(__name__)


def Qwen2Attention_init(
    self, config: Qwen2Config, layer_idx: int, compression_config: dict
):
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

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================


def LLamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
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
    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================


def Qwen2Attention_forward(
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

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}
        # =============== query cache ================
        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage, initial query cache
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # decoding stage, add new query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]
            # keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== end query cache ================

        if hasattr(past_key_value, "pos_ids_cache"):
            pos_ids_cache = past_key_value.pos_ids_cache[self.layer_idx]
        else:
            pos_ids_cache = None

        # =============== kv cache compression ================
        if kwargs["enable_compress"]:
            query_cache = past_key_value.query_cache[self.layer_idx]
            # notice if the length of kv cache is smaller than budget, then the kv cache will not be compressed
            compressed_key_states, compressed_value_states, pos_ids_cache = (
                self.kv_cluster.update_kv(
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
                )
            )
            past_key_value.key_cache[self.layer_idx] = compressed_key_states
            past_key_value.value_cache[self.layer_idx] = compressed_value_states
            if pos_ids_cache is not None:
                past_key_value.pos_ids_cache[self.layer_idx] = pos_ids_cache
        # =============== end kv cache compression ================

    # keep only the most recent attention mask
    if attention_mask is not None and attention_mask.shape[-1] > key_states.shape[-2]:
        if len(attention_mask.shape) == 4:
            # mask for eager attention
            attention_mask = attention_mask[
                :, :, -key_states.shape[-2] :, -key_states.shape[-2] :
            ].contiguous()
        elif len(attention_mask.shape) == 2:
            # mask for flash attention
            attention_mask = attention_mask[:, -key_states.shape[-2] :].contiguous()

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

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
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def LLamaAttention_forward(
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

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # decoding stage, add new query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]
            # keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== end query cache ================

        if hasattr(past_key_value, "pos_ids_cache"):
            pos_ids_cache = past_key_value.pos_ids_cache[self.layer_idx]
        else:
            pos_ids_cache = None

        # =============== kv cache compression ================
        if kwargs["enable_compress"]:
            query_cache = past_key_value.query_cache[self.layer_idx]
            # notice if the length of kv cache is smaller than budget, then the kv cache will not be compressed
            compressed_key_states, compressed_value_states, pos_ids_cache = (
                self.kv_cluster.update_kv(
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
                )
            )
            past_key_value.key_cache[self.layer_idx] = compressed_key_states
            past_key_value.value_cache[self.layer_idx] = compressed_value_states
            if pos_ids_cache is not None:
                past_key_value.pos_ids_cache[self.layer_idx] = pos_ids_cache
        # =============== end kv cache compression ================

    if attention_mask is not None and attention_mask.shape[-1] > key_states.shape[-2]:
        if len(attention_mask.shape) == 4:
            # mask for eager attention
            attention_mask = attention_mask[
                :, :, -key_states.shape[-2] :, -key_states.shape[-2] :
            ].contiguous()
        elif len(attention_mask.shape) == 2:
            # mask for flash attention
            attention_mask = attention_mask[:, -key_states.shape[-2] :].contiguous()

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
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__

    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = (
            model_kwargs["past_key_values"].is_compileable
            and self._supports_static_cache
        )
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda"
            or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    #  =============== initialize pos ids (optional for analysis) ===============

    if hasattr(self.config, "record_pos_ids") and self.config.record_pos_ids:
        if model_kwargs.get("attention_mask") is not None:
            initial_pos_ids = torch.cumsum(model_kwargs["attention_mask"], dim=-1) - 1
        else:
            initial_pos_ids = (
                torch.cumsum(
                    torch.ones((batch_size, cur_len), device=input_ids.device), dim=-1
                )
                - 1
            )
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            past_key_values.pos_ids_cache = {}
            for layer_idx in range(len(self.model.layers)):
                # batch_size, kv_heads, seq_len
                past_key_values.pos_ids_cache[layer_idx] = initial_pos_ids.unsqueeze(
                    1
                ).expand(-1, self.config.num_key_value_heads, -1)

    #  =============== end initialize pos ids ===============
    output_length = 0
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        # prepare model inputs

        # =============== logic of whether to compress ================
        # during decoding, conduct compression every divide_length steps
        # if the prefill length is greater than budget, then the prefilled prompt will be compressed immediately

        model_kwargs["cur_len"] = cur_len
        enable_compress = output_length % self.config.divide_length == 0
        model_kwargs["enable_compress"] = enable_compress
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if model_kwargs.get("past_key_values", None) is not None:
            if model_kwargs.get("attention_mask", None) is not None:
                # huggingface may process the attention mask in the prepare_inputs_for_generation function
                # we need to pass the original attention mask to the update function
                model_kwargs["past_key_values"].attention_mask = model_kwargs[
                    "attention_mask"
                ]
            model_kwargs["past_key_values"].unfinished_sequences = unfinished_sequences

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update(
            {"output_attentions": output_attentions} if output_attentions else {}
        )
        model_inputs.update(
            {"output_hidden_states": output_hidden_states}
            if output_hidden_states
            else {}
        )
        outputs = model_forward(**model_inputs, return_dict=True)
        output_length += 1

        # update pos ids cache
        if model_kwargs.get("past_key_values") is not None:
            if hasattr(model_kwargs["past_key_values"], "pos_ids_cache"):
                pos_ids_cache = model_kwargs["past_key_values"].pos_ids_cache
                for layer_idx in pos_ids_cache:
                    new_pos = pos_ids_cache[layer_idx][:, :, -1].unsqueeze(-1) + 1
                    pos_ids_cache[layer_idx] = torch.cat(
                        [pos_ids_cache[layer_idx], new_pos], dim=2
                    )

        # =============== end logic of whether to compress ================

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    #  =============== record pos ids ===============

    if model_kwargs.get("past_key_values") is not None:
        if hasattr(model_kwargs['past_key_values'], "pos_ids_cache"):
            pos_ids_cache = model_kwargs['past_key_values'].pos_ids_cache
            np_cache=[]
            for layer_idx in pos_ids_cache:
                np_cache.append(pos_ids_cache[layer_idx].cpu().numpy())
            self.pos_ids_cache = np.array(np_cache)

    #  =============== end record pos ids ===============
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def clear_score_cache(self):
    all_scores = []
    for layer in self.model.layers:
        if hasattr(layer.self_attn, "kv_cluster"):
            if hasattr(layer.self_attn.kv_cluster, "cached_score"):
                if layer.self_attn.kv_cluster.cached_score is not None:
                    # score shape (B,H,L)
                    score = layer.self_attn.kv_cluster.cached_score.float().cpu().numpy()
                    all_scores.append(score)
                    layer.self_attn.kv_cluster.cached_score = None
    return np.array(all_scores)
