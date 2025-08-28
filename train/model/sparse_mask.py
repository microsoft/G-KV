import torch
from packaging.version import Version
from typing import Optional
from .utils import cal_similarity, compute_attention_scores

if Version(torch.__version__) >= Version("2.5.0"):
    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,
    )

KV_BLOCK_SIZE = 128
Q_BLOCK_SIZE = 128


def build_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    build causal mask from attention mask
    attention mask: int (bsz,seq_len)
    causal mask: bool (bsz,seq_len,seq_len), True means valid, False means masked
    """
    bsz, seq_len = attention_mask.shape
    mask = ~torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device),
        diagonal=1,
    )
    causal_mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    valid_token_mask = attention_mask.unsqueeze(1).expand(bsz, seq_len, -1).bool()
    final_mask = causal_mask & valid_token_mask
    return final_mask


def build_StreamingLLM_mask(
    attention_mask: torch.Tensor,
    sink_len: int,
    window_size: int,
    return_all_mask: bool = False,
):
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    causal_mask = build_causal_mask(attention_mask)

    # keep the first sink_len non-padding tokens
    valid_token_count = attention_mask.cumsum(dim=1)
    sink_mask = (valid_token_count <= sink_len) & (attention_mask == 1)
    sink_mask = sink_mask.unsqueeze(1) & causal_mask

    pos_indices = torch.arange(seq_len, device=device)
    pos_diff = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(
        0
    )  # pos_diff[i, j] = i - j
    window_mask = (pos_diff <= window_size).unsqueeze(0).expand(bsz, -1, -1)
    window_mask = window_mask & causal_mask

    if return_all_mask:
        return causal_mask, sink_mask, window_mask
    else:
        return sink_mask | window_mask


def build_SepLLM_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_dis: torch.Tensor,
    sink_len: int,
    sep_cache_len: int,
    window_size: int,
):
    """
    Build SepLLM sparse attention mask

    Args:
        input_ids: (bsz, seq_len) Input token ids
        attention_mask: (bsz, seq_len) Input attention mask
        keep_dis: (n,) Token ids to be kept
        sink_len: The first sink_len valid (non-padding) tokens will always be kept
        sep_cache_len: Maximum cache length; non-sep tokens beyond this length will be dropped. If still exceeding kv_budget, the oldest tokens are dropped in order
        window_size: The rightmost window_size tokens will be kept

    Returns:
        Sparse attention mask: (bsz, seq_len, seq_len), True means valid, False means masked
    """
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    causal_mask, sink_mask, window_mask = build_StreamingLLM_mask(
        attention_mask, sink_len, window_size, return_all_mask=True
    )

    sep_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for token_id in keep_dis:
        sep_mask |= input_ids == token_id
    sep_mask = sep_mask.unsqueeze(1) & causal_mask

    # a non-sep token can be reserved only when
    # 1) left sep token < sep_cache_len
    # 2) right token < window_size + sep_cache_len - (left sep token)
    non_sink_sep_mask = sep_mask & ~sink_mask
    non_sep_mask = causal_mask & ~sep_mask & ~sink_mask
    num_sep_left = non_sink_sep_mask.long().cumsum(dim=-1)
    right_token = causal_mask.flip(dims=[2]).long().cumsum(dim=-1).flip(dims=[2])
    kept_non_sep_mask = non_sep_mask & (
        (num_sep_left < sep_cache_len)
        & (right_token < (window_size + sep_cache_len - num_sep_left))
    )

    # a sep token can be reserved only when
    # 1) right sep token < sep_cache_len
    non_window_sep_mask = sep_mask & ~window_mask
    num_sep_right = (
        non_window_sep_mask.flip(dims=[2]).long().cumsum(dim=-1).flip(dims=[2])
    )
    kept_sep_mask = sep_mask & (num_sep_right < sep_cache_len)

    sepllm_mask = sink_mask | window_mask | kept_non_sep_mask | kept_sep_mask

    return sepllm_mask


@torch.no_grad()
def build_sparse_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_length: int,
    compress_step: int,
    window_size: int,
    kv_budget: int,
    alpha: float,
    mix_lambda: float,
) -> torch.Tensor:
    # shape (bsz,num_head,seq_len,head_dim)
    bsz, kv_head, seq_len, head_dim = key_states.shape
    causal_mask = build_causal_mask(attention_mask)
    causal_mask = causal_mask.unsqueeze(1).expand(-1, kv_head, -1, -1)

    pos_cache = None
    score_cache: Optional[torch.Tensor] = None
    pre_window_end = 0
    for pos in range(input_length, seq_len, compress_step):
        if pos > kv_budget:
            window_start = pos + compress_step - window_size
            window_end = pos + compress_step
            if window_end > seq_len:
                break
            if pos_cache is None:
                # initialize pos_cache and key_cache
                pos_cache = (
                    torch.arange(0, window_end, device=key_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                # (bsz,kv_head,len)
                pos_cache = pos_cache.unsqueeze(1).expand(-1, kv_head, -1)
                key_cache = key_states[:, :, :window_end, :]
            else:
                new_pos = (
                    torch.arange(pre_window_end, window_end, device=key_states.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                new_pos = new_pos.unsqueeze(1).expand(-1, kv_head, -1)
                pos_cache = torch.cat(
                    [pos_cache, new_pos],
                    dim=2,
                )
                key_cache = torch.gather(
                    key_states,
                    2,
                    pos_cache.unsqueeze(-1).expand(-1, -1, -1, head_dim),
                )
            query_cache = query_states[:, :, window_start:window_end, :]
            window_attention_mask = attention_mask[
                :, window_end - key_cache.shape[2] : window_end
            ]
            # get attention scores
            attn_scores = compute_attention_scores(
                query_cache, key_cache, attention_mask=window_attention_mask
            )
            final_score = attn_scores.mean(dim=2)
            final_score = final_score.div_(final_score.max(dim=-1, keepdim=True).values)
            if score_cache is not None:
                cached_score_len = score_cache.shape[-1]
                old_score = torch.cat(
                    [score_cache, final_score[:, :, cached_score_len:]], dim=-1
                )
                final_score = torch.max(old_score * alpha, final_score)

            similarity_cos = cal_similarity(key_cache)
            similarity_cos = similarity_cos.div_(
                similarity_cos.max(dim=-1, keepdim=True).values
            )[:, :-window_size]
            combined_score = final_score * mix_lambda - similarity_cos * (
                1 - mix_lambda
            )
            # mask padding score
            mask = (window_attention_mask == 0)[:, :-window_size].unsqueeze(1)
            min_values = combined_score.min().item()
            combined_score.masked_fill_(mask, min_values - 1e-6)
            # get indices
            keep_indices = combined_score.topk(kv_budget - window_size, dim=-1).indices
            keep_indices = torch.sort(keep_indices, dim=-1).values
            num_drop = combined_score.shape[-1] - (kv_budget - window_size)
            drop_indices = combined_score.topk(num_drop, dim=-1, largest=False).indices
            drop_indices = torch.sort(drop_indices, dim=-1).values
            # (bsz,kv_head,num_drop)
            drop_pos = pos_cache.gather(dim=2, index=drop_indices)

            # update cache
            score_cache = final_score.gather(dim=2, index=keep_indices)
            window_pos = pos_cache[:, :, -window_size:]
            pos_cache = pos_cache.gather(dim=2, index=keep_indices)
            pos_cache = torch.cat([pos_cache, window_pos], dim=2)

            # update causal_mask
            temp_mask = torch.ones_like(causal_mask, dtype=torch.bool)
            # expand drop_pos to match temp_mask dimensions for masking columns (key positions)
            # drop_pos: (bsz, kv_head, num_drop) -> expand to (bsz, kv_head, seq_len, num_drop)
            drop_pos_expanded = drop_pos.unsqueeze(2).expand(-1, -1, seq_len, -1)
            temp_mask.scatter_(3, drop_pos_expanded, False)
            temp_mask[:, :, :window_end, :] = True
            causal_mask = temp_mask & causal_mask
            pre_window_end = window_end
        else:
            continue
    return causal_mask


def build_block_mask(
    attention_mask: torch.Tensor, num_query_heads: int
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

    if len(attention_mask.shape) == 4:

        attention_mask = repeat_mask(
            attention_mask, num_query_heads // attention_mask.shape[1]
        )

        def sparse_kernel(b, h, q_idx, kv_idx):
            return attention_mask[b, h, q_idx, kv_idx].view([]).detach().clone()

        B, H, Sq, Sk = attention_mask.shape
    elif len(attention_mask.shape)==3:
        B, Sq, Sk = attention_mask.shape
        H = num_query_heads
        def sparse_kernel(b, h, q_idx, kv_idx):
            return attention_mask[b, q_idx, kv_idx].view([]).detach().clone()
    else:
        raise ValueError(f"Invalid attention mask shape: {attention_mask.shape}")
        
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
