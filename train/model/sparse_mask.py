import torch
from packaging.version import Version
from typing import Optional
from .utils import cal_similarity, compute_attention_scores

if Version(torch.__version__) >= Version("2.5.0"):
    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,
    )


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


def build_stream_mask(input_ids, attention_mask):
    pass


def build_sepllm_mask(input_ids, attention_mask):
    pass


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
            combined_score = final_score * mix_lambda - similarity_cos * (1 - mix_lambda)
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


# def get_block_mask(
#     input_ids, attention_mask
# ):

#     def sparse_kernel(b, h, q_idx, kv_idx):
#         aa = attention_mask[b, h, q_idx, kv_idx]
#         return aa.view([]).detach().clone()

#     B, H, Sq, Sk = attention_mask.shape

#     block_mask = create_block_mask(
#         sparse_kernel,
#         B,
#         H,
#         Sq,
#         Sk,
#         BLOCK_SIZE=(KV_BLOCK_SIZE, Q_BLOCK_SIZE),
#         device=input_ids.device,
#         _compile=False,
#     )
#     return block_mask
