from numpy import True_
import torch
from packaging.version import Version
from typing import Optional, List, Tuple
from .utils import cal_similarity, compute_attention_scores

if Version(torch.__version__) >= Version("2.5.0"):
    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,
    )
from torch.nn.attention.flex_attention import BlockMask

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
    attention_mask: Optional[torch.Tensor],
    input_length: int,
    divide_length: int,
    window_size: int,
    budget: int,
    alpha: float,
    mix_lambda: float,
) -> torch.Tensor:
    # shape (bsz,num_head,seq_len,head_dim)
    bsz, seq_len = key_states.shape[0], key_states.shape[2]
    if attention_mask is None:
        attention_mask = torch.ones(bsz, seq_len, device=key_states.device).long()
    assert len(attention_mask.shape) == 2, "attention_mask must be 2D"
    bsz, kv_head, seq_len, head_dim = key_states.shape
    causal_mask = build_causal_mask(attention_mask).unsqueeze(1)
    causal_mask = causal_mask.expand(-1, kv_head, -1, -1).clone()
    pos_cache = None
    score_cache: Optional[torch.Tensor] = None
    pre_window_end = 0
    for pos in range(input_length, seq_len, divide_length):
        if pos > budget:
            window_start = pos - window_size
            window_end = pos
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
            keep_indices = combined_score.topk(budget - window_size, dim=-1).indices
            keep_indices = torch.sort(keep_indices, dim=-1).values

            # update cache
            score_cache = final_score.gather(dim=2, index=keep_indices)
            window_pos = pos_cache[:, :, -window_size:]
            pos_cache = pos_cache.gather(dim=2, index=keep_indices)
            pos_cache = torch.cat([pos_cache, window_pos], dim=2)

            # update causal_mask
            next_window_end = min(window_end + divide_length, seq_len)
            step_length = next_window_end - window_end
            if not (step_length > 0):
                break
            sparse_mask = torch.zeros(
                bsz,
                kv_head,
                step_length,
                seq_len,
                device=causal_mask.device,
            ).bool()
            sparse_mask[:, :, :, window_end:] = True
            # expand drop_pos to match temp_mask dimensions for masking columns (key positions)
            # drop_pos: (bsz, kv_head, num_drop) -> expand to (bsz, kv_head, seq_len, num_drop)
            pos_cache_expanded = pos_cache.unsqueeze(2).expand(-1, -1, step_length, -1)
            sparse_mask.scatter_(3, pos_cache_expanded, True)
            causal_mask[:, :, window_end:next_window_end, :] &= sparse_mask
            pre_window_end = window_end
        else:
            continue
    return causal_mask


def build_sparse_mask_from_pos_cache(
    pos_cache_history: List[Tuple[torch.Tensor, int]],
    attention_mask: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Args:
        pos_cache: List[Tuple[torch.Tensor, int]], the positions to keep
        attention_mask: (bsz,seq_len), the attention mask
    Return:
        causal_mask: (bsz,kv_head,seq_len,seq_len), True means valid, False means masked
    """
    device = attention_mask.device

    causal_mask = (
        build_causal_mask(attention_mask).unsqueeze(1).expand(-1, num_kv_heads, -1, -1)
    ).clone()
    bsz, seq_len = attention_mask.shape
    end_raw = attention_mask.shape[-1]
    for i in range(len(pos_cache_history) - 1, -1, -1):
        pos_cache, statrt_raw = pos_cache_history[i]
        pos_cache = pos_cache.to(device)
        if not (end_raw - statrt_raw > 0):
            end_raw = statrt_raw
            continue
        sparse_mask = torch.zeros(
            bsz, num_kv_heads, end_raw - statrt_raw, seq_len, device=device
        ).bool()

        col_indices = pos_cache.unsqueeze(2)  # [bsz, kv_head, 1, num_kept]
        col_indices = col_indices.expand(-1, -1, end_raw - statrt_raw, -1)
        sparse_mask.scatter_(3, col_indices, True)
        sparse_mask[:, :, :, statrt_raw:] = True_
        sparse_mask &= causal_mask[:, :, statrt_raw:end_raw, :]
        causal_mask[:, :, statrt_raw:end_raw, :] = sparse_mask
        end_raw = statrt_raw
    return causal_mask


def expand_sparse_mask(
    sparse_mask: torch.Tensor, expand_len: int, kept_pos: torch.Tensor
) -> torch.Tensor:
    """
    args:
        sparse_mask: (bsz,kv_head,seq_len,seq_len), True means valid, False means masked
        expand_len: int, the length to expand
        kept_pos: (bsz,kv_head,num_kept), the positions to keep
    return:
        expanded_sparse_mask: (bsz,kv_head,expand_len,expand_len), True means valid, False means masked
    """
    len_increase = expand_len - sparse_mask.shape[-1]
    if len_increase > 0:
        bsz, kv_head, seq_len, _ = sparse_mask.shape
        expanded_mask = torch.zeros(
            bsz, kv_head, expand_len, expand_len, device=sparse_mask.device
        ).bool()
        expanded_mask[:, :, :seq_len, :seq_len] = sparse_mask

        new_rows_mask = torch.zeros(
            bsz, kv_head, len_increase, expand_len, device=kept_pos.device
        ).bool()

        col_indices = kept_pos.unsqueeze(2)  # [bsz, kv_head, 1, num_kept]
        col_indices = col_indices.expand(-1, -1, len_increase, -1)
        new_rows_mask.scatter_(3, col_indices, True)
        new_rows_mask[:, :, :, seq_len:] = True

        # kept_pos may contain padding tokens
        # so we mask the padding tokens according to the last row of the sparse mask
        last_row_mask = sparse_mask[:, :, -1, :].to(kept_pos.device)
        last_row_mask = last_row_mask.unsqueeze(2).expand(-1, -1, len_increase, -1)
        new_rows_mask[:, :, :, :seq_len] = (
            new_rows_mask[:, :, :, :seq_len] & last_row_mask
        )

        row_indices = torch.arange(seq_len, expand_len, device=kept_pos.device)
        col_indices = torch.arange(expand_len, device=kept_pos.device)
        causal_mask = col_indices.unsqueeze(0) <= row_indices.unsqueeze(1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, kv_head, -1, -1)
        new_rows_mask = new_rows_mask & causal_mask
        expanded_mask[:, :, seq_len:, :] = new_rows_mask.to(expanded_mask.device)
        return expanded_mask
    else:
        return sparse_mask


def build_block_mask(attention_mask: torch.Tensor, num_query_heads: int) -> BlockMask:
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
    elif len(attention_mask.shape) == 3:
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
