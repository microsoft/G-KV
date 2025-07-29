import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from . import cal_similarity
import math


@torch.no_grad()
def compute_attention_scores(query_states, key_states, pooling="max",attention_mask=None):
    """
    query_states: (bsz, q_heads, q_len, head_dim)
    key_states: (bsz, kv_heads, kv_cache_len, head_dim)
    return: (bsz, kv_heads, q_len, kv_cache_len - q_len)
    attention_mask: eager attention mask (bsz, 1, kv_cache_len, kv_cache_len) or
                    flash attention mask (bsz, kv_cache_len)   
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    kv_cache_len = key_states.shape[2]
    query_group_size = q_heads // kv_heads

    # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
    query_states = query_states.view(
        batch_size, kv_heads, query_group_size, q_len, head_dim
    )

    # shape: [batch_size, kv_heads, 1, kv_cache_len, head_dim]
    key_states = key_states.unsqueeze(2)

    # we first normalize the key_states for better numerical stability
    key_states = key_states / math.sqrt(head_dim)
    # shape: [batch_size, kv_heads, query_group_size, q_len, kv_cache_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(3, 4))


    if attention_mask is not None:
        if attention_mask.dim() == 2:
            # build causal mask (bsz,1,kv_cache_len,kv_cache_len) from attention_mask (bsz,kv_cache_len)
            # shape: (kv_cache_len, kv_cache_len)
            causal_mask = torch.triu(
                torch.ones(kv_cache_len, kv_cache_len, device=attn_weights.device, dtype=torch.bool), diagonal=1
            )  
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1,1,kv_cache_len,kv_cache_len)
            # shape: (bsz,1,kv_cache_len,kv_cache_len)
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(1)  # (bsz,1,1,kv_cache_len)
            causal_mask = causal_mask.masked_fill(mask, True)
            # shape: (bsz,kv_heads,query_group_size,q_len,kv_cache_len)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(2)[:,:,:,-q_len:,:], -float("inf"))
        elif attention_mask.dim() == 4:
            # shape: (bsz,1,kv_cache_len,kv_cache_len)
            attn_weights = attn_weights.add_(attention_mask.unsqueeze(1)[:,:,:,-q_len:,:])
            pass
        else:
            raise ValueError("attention_mask must be 2D or 4D")
    else:
        # shape: (q_len, q_len)
        # query can see all key before it
        mask = torch.triu(
            torch.ones(q_len, q_len, device=attn_weights.device), diagonal=1
        ).bool()
        attn_weights[:, :, :, :, -q_len:].masked_fill_(mask, -float("inf"))

    attn_scores = torch.softmax(
        attn_weights - attn_weights.max(dim=-1, keepdim=True).values, dim=-1
    )
    # apply pooling over attention head
    if pooling == "mean":
        attn_scores = attn_scores.mean(dim=2)
    elif pooling == "max":
        attn_scores = attn_scores.max(dim=2).values
    else:
        raise ValueError("Pooling method not supported")
    return attn_scores[:, :, :, :-q_len]


class ImformativeKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=5,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        record_kept_token_indices=False,
        enable_pooling=False,
        suppressing_redundancy=False,
        smooth_method="mean",
        enable_score_cache=False,
        disable_norm=False,
        alpha=0.8,
        compress_mode="budget",
        compress_ratio=0.2,
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

        self.cached_score = None

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def update_kv(
        self,
        key_states: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        cur_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
            return key_states, value_states
        else:
            # shape: (bsz, num_kv_heads, len, len)
            attn_scores = compute_attention_scores(query_states, key_states,attention_mask=attention_mask)
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                raise ValueError("attn_scores is nan or inf")

            # shape: (bsz, num_kv_heads, len)
            final_score = attn_scores.mean(dim=2)

            if self.enable_score_cache and not self.disable_norm:
                # normalize final_score
                final_score = final_score.div_(
                    final_score.max(dim=-1, keepdim=True).values
                )

            if self.enable_score_cache and self.cached_score is not None:
                # cached score shape: (bsz, num_kv_heads, cached_score_len)
                cached_score_len = self.cached_score.shape[-1]
                old_score = torch.cat(
                    [self.cached_score, final_score[:, :, cached_score_len:]], dim=-1
                )
                if self.smooth_method == "mean":
                    final_score = old_score * self.alpha + final_score * (
                        1 - self.alpha
                    )
                elif self.smooth_method == "max":
                    final_score = torch.max(old_score * self.alpha, final_score)
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
                similarity_cos = cal_similarity(
                    key_states,
                    retain_ratio=self.retain_ratio,
                    retain_direction=self.retain_direction,
                )[:, : -self.window_size]

                if self.enable_score_cache and not self.disable_norm:
                    # normalize similarity_cos
                    similarity_cos = similarity_cos.div_(
                        similarity_cos.max(dim=-1, keepdim=True).values
                    )

                pooled_score = pooled_score * self.mix_lambda - similarity_cos * (
                    1 - self.mix_lambda
                )
            
            # shape: (bsz, num_kv_heads, budget - window_size)
            topk_indices = pooled_score.topk(budget - self.window_size, dim=-1).indices
            # sort the indices to keep the padding always at the left
            # this will make sure the score of padding tokens are zeros and always be evicted first
            indices=torch.sort(topk_indices, dim=-1).values

            if self.enable_score_cache:
                self.cached_score = final_score.gather(dim=2, index=indices)
            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, budget - window_size)
            if self.record_kept_token_indices:
                pass

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
            return key_states, value_states
