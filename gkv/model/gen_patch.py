import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Tuple, Union, Callable
import os
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers.cache_utils import Cache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerationMixin

from gkv.model.sparse_mask import build_causal_mask
from gkv.model.sparse_mask import expand_sparse_mask
from gkv.model.sparse_mask import build_sparse_mask_from_pos_cache

from time import time


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

    #  =============== initialize ===============

    past_key_values = model_kwargs["past_key_values"]

    if model_kwargs.get("attention_mask") is not None:
        num_pad = (
            (model_kwargs["attention_mask"] == 0)
            .long()
            .sum(dim=-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    else:
        num_pad = 0

    if hasattr(self.config, "record_pos_ids") and self.config.record_pos_ids:
        initial_pos_ids = (
            torch.arange(cur_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        past_key_values.pos_ids_cache = {}
        for layer_idx in range(len(self.model.layers)):
            # batch_size, kv_heads, seq_len
            past_key_values.pos_ids_cache[layer_idx] = initial_pos_ids.unsqueeze(
                1
            ).expand(-1, self.config.num_key_value_heads, -1)

    past_key_values.sparse_mask_cache = {}

    pos_cache_history = {}
    return_sparse_mask = False
    if hasattr(self.config, "return_sparse_mask"):
        return_sparse_mask = self.config.return_sparse_mask

    past_key_values.score_cache = {}
    for layer_idx in range(len(self.model.layers)):
        past_key_values.score_cache[layer_idx] = None

    #  =============== end initialize ===============
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

        if enable_compress and return_sparse_mask:
            # record the pos_ids_cache for sparse mask
            for layer_idx in range(len(self.model.layers)):
                if layer_idx not in pos_cache_history:
                    pos_cache_history[layer_idx] = []
                pos_cache_history[layer_idx].append(
                    (
                        model_kwargs["past_key_values"]
                        .pos_ids_cache[layer_idx]
                        .clone()
                        .cpu(),
                        cur_len,
                    )
                )

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
            if hasattr(past_key_values, "pos_ids_cache"):
                pos_ids_cache = past_key_values.pos_ids_cache
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

    # =============== build sparse mask from pos cache ===============
    if return_sparse_mask:
        bsz = input_ids.shape[0]

        if torch.distributed.get_rank() == 3:
            print("rank")
        attention_mask = model_kwargs["past_key_values"].attention_mask
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    bsz,
                    1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )
        sparse_mask_list = [[] for _ in range(bsz)]
        for layer_idx in range(len(self.model.layers)):
            # (bsz,kv_head,seq_len,seq_len)
            layer_sparse_mask = build_sparse_mask_from_pos_cache(
                pos_cache_history[layer_idx],
                attention_mask,
                self.config.num_key_value_heads,
            ).cpu()
            for i in range(bsz):  # per layer sparse mask for each input
                sparse_mask_list[i].append(layer_sparse_mask[i])
            del layer_sparse_mask
        for i in range(bsz):
            # (layer,kv_head,seq_len,seq_len)
            sparse_mask_list[i] = torch.stack(sparse_mask_list[i], dim=0)
        model_kwargs["past_key_values"].sparse_mask_cache = sparse_mask_list
        model_kwargs["past_key_values"].attention_mask = attention_mask

    # =============== end build sparse mask from pos cache ===============

    #  =============== record pos ids ===============

    if model_kwargs.get("past_key_values") is not None:
        if hasattr(model_kwargs["past_key_values"], "pos_ids_cache"):
            pos_ids_cache = model_kwargs["past_key_values"].pos_ids_cache
            for layer_idx in pos_ids_cache:
                pos_ids_cache[layer_idx] = pos_ids_cache[layer_idx] - num_pad

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


def patch_sample():
    GenerationMixin._sample = _sample
