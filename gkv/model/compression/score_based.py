import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from gkv.utils.compression_score import (
    compute_attention_scores,
    cal_similarity_raw,
    cal_similarity_triton,
)
from gkv.model.sparse_mask import expand_sparse_mask


class ScoreBasedKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=5,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        enable_pooling=False,
        suppressing_redundancy=False,
        smooth_method="max",
        enable_score_cache=False,
        disable_norm=False,
        alpha=0.8,
        compress_mode="budget",
        compress_ratio=0.2,
        triton_similarity=True,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.enable_pooling = enable_pooling
        self.suppressing_redundancy = suppressing_redundancy
        # ikv parameters
        self.smooth_method = smooth_method
        self.enable_score_cache = enable_score_cache
        self.alpha = alpha
        self.disable_norm = disable_norm
        self.compress_mode = compress_mode
        self.compress_ratio = compress_ratio
        self.triton_similarity = triton_similarity

    def initial_score_cache(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        score_cache = compute_attention_scores(
            query_states, key_states, attention_mask=attention_mask, remove_query=False
        )
        # Replace NaN values in score_cache with 0
        score_cache = torch.nan_to_num(score_cache, nan=0.0)
        score_cache = score_cache.sum(dim=2)

        return score_cache

    def update_kv(
        self,
        key_states: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        pos_ids_cache: Optional[torch.Tensor] = None,
        cur_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        unfinished_sequences: Optional[torch.Tensor] = None,
        score_cache: Optional[torch.Tensor] = None,
    ):
        """
        key_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
        query_states: (bsz, num_q_heads, window_size, head_dim)
        value_states: (bsz, num_kv_heads, kv_cache_len, head_dim)
        entropy_cache: (bsz, window_size)
        """

        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if self.compress_mode == "budget":
            budget = self.budget
        elif self.compress_mode == "ratio":
            if (cur_len is not None) and (self.budget / cur_len < self.compress_ratio):
                budget = int(cur_len * self.compress_ratio)
            else:
                budget = self.budget
        else:
            raise ValueError("compress mode must be budget or ratio")

        if kv_cache_len < budget:
            return key_states, value_states, pos_ids_cache, score_cache
        else:
            if attention_mask is not None and attention_mask.shape[-1] > kv_cache_len:
                attention_mask = attention_mask[:, -kv_cache_len:].contiguous()
            # shape: (bsz, num_kv_heads, len, len)
            attn_scores = compute_attention_scores(
                query_states, key_states, attention_mask=attention_mask
            )
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                raise ValueError("attn_scores is nan or inf")

            # shape: (bsz, num_kv_heads, len)
            final_score = attn_scores.mean(dim=2)

            if self.enable_score_cache and not self.disable_norm:
                # normalize final_score
                final_score = final_score.div_(
                    final_score.max(dim=-1, keepdim=True).values
                )

            if self.enable_score_cache and score_cache is not None and self.alpha > 0:
                # cached score shape: (bsz, num_kv_heads, cached_score_len)
                cached_score_len = score_cache.shape[-1]
                if not self.smooth_method == "sum":
                    old_score = torch.cat(
                        [score_cache, final_score[:, :, cached_score_len:]], dim=-1
                    )
                else:
                    zero_pad_len = final_score.shape[-1] - cached_score_len
                    zero_pad = torch.zeros(
                        (score_cache.shape[0], score_cache.shape[1], zero_pad_len),
                        device=score_cache.device,
                        dtype=score_cache.dtype,
                    )
                    old_score = torch.cat([score_cache, zero_pad], dim=-1)

                if self.smooth_method == "mean":
                    final_score = old_score * self.alpha + final_score * (
                        1 - self.alpha
                    )
                elif self.smooth_method == "max":
                    final_score = torch.max(old_score * self.alpha, final_score)
                elif self.smooth_method == "sum":
                    # score function for H2O
                    final_score = old_score * self.alpha + final_score
                else:
                    raise ValueError("smooth method must be mean or max")

            if self.enable_pooling:
                pooled_score = F.max_pool1d(
                    final_score,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )
            else:
                pooled_score = final_score

            if self.suppressing_redundancy:
                if self.triton_similarity:
                    similarity_cos = cal_similarity_triton(
                        key_states,
                        attention_mask=attention_mask,
                    )[:, :, : -self.window_size]
                else:
                    similarity_cos = cal_similarity_raw(
                        key_states,
                        retain_ratio=self.retain_ratio,
                    )[:, : -self.window_size]

                if self.enable_score_cache and not self.disable_norm:
                    # disable normalization for the original similarity score
                    # normalize similarity_cos
                    similarity_cos = similarity_cos.div_(
                        similarity_cos.max(dim=-1, keepdim=True).values
                    )

                pooled_score = pooled_score * self.mix_lambda - similarity_cos * (
                    1 - self.mix_lambda
                )

            if attention_mask is not None:
                # similarity cos may make score to be negative
                # non-padding tokens may have lower score than the padding tokens
                # so we need to set the score of padding tokens to the minimum value
                # this will make sure the padding tokens always be evicted first
                mask = (attention_mask == 0)[:, : -self.window_size].unsqueeze(1)
                min_values = pooled_score.min().item()
                pooled_score.masked_fill_(mask, min_values - 1e-6)

            # shape: (bsz, num_kv_heads, budget - window_size)
            topk_indices = pooled_score.topk(budget - self.window_size, dim=-1).indices
            # sort the indices to keep the padding always at the left
            # this will make sure the attention mask match the KV caches
            indices = torch.sort(topk_indices, dim=-1).values

            if self.enable_score_cache:
                new_cached_score = final_score.gather(dim=2, index=indices)
                if unfinished_sequences is not None and score_cache is not None:
                    # keep score of the last compressed tokens for analyze
                    bsz, num_heads, seq_len = new_cached_score.shape
                    if seq_len <= score_cache.shape[-1]:
                        mask = (
                            (unfinished_sequences == 0)
                            .view(bsz, 1, 1)
                            .expand(bsz, num_heads, seq_len)
                        )
                        new_cached_score = torch.where(
                            mask, score_cache[:, :, :seq_len], new_cached_score
                        )
                score_cache = new_cached_score

            if pos_ids_cache is not None:
                cur_pos_ids_cache = pos_ids_cache[:, :, -self.window_size :]
                compressed_pos_ids_cache = pos_ids_cache[
                    :, :, : -self.window_size
                ].gather(dim=2, index=indices)
                new_pos_ids_cache = torch.cat(
                    [compressed_pos_ids_cache, cur_pos_ids_cache], dim=2
                )
                if unfinished_sequences is not None:
                    # if finished, keep the previous pos_ids
                    bsz, num_heads, seq_len = new_pos_ids_cache.shape
                    mask = (
                        (unfinished_sequences == 0)
                        .view(bsz, 1, 1)
                        .expand(bsz, num_heads, seq_len)
                    )
                    new_pos_ids_cache = torch.where(
                        mask, pos_ids_cache[:, :, :seq_len], new_pos_ids_cache
                    )
                pos_ids_cache = new_pos_ids_cache

            # shape: (bsz, num_kv_heads, budget - window_size, head_dim)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states, pos_ids_cache, score_cache
