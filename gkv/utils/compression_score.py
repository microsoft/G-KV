import torch
import math
import triton
import triton.language as tl


def cal_similarity_raw(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    """
    raw implementation of similarity score
    from https://github.com/Zefan-Cai/R-KV
    """
    k = key_states[0]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    seq_len = similarity_mask.size(-1)
    k = int(seq_len * retain_ratio)

    indices = torch.where(
        similarity_mask,
        torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    # find the last True index in each row
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]

    # find the first True index in each row
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]

    # keep the last_percent% elements
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

    # keep the first_percent% elements
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # create indices for zeroing
    batch_idx = (
        torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
    )
    seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    return similarity_cos.mean(dim=1).softmax(dim=-1)


def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


@triton.jit
def similarity_score_kernel(
    key_states_ptr,
    similarity_cos_ptr,
    key_norm_ptr,
    num_pad_ptr,
    zero_out_ptr,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sl,
    stride_nb,
    stride_nh,
    stride_nl,
    stride_zb,
    stride_zh,
    stride_zl,
    threshold: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    head_idx = pid // batch_size

    num_pad = tl.load(num_pad_ptr + batch_idx)

    for m_offset in range(seq_len - BLOCK_M, num_pad - BLOCK_M, -BLOCK_M):

        km_ptr = tl.make_block_ptr(
            base=key_states_ptr + batch_idx * stride_kb + head_idx * stride_kh,
            shape=(seq_len, head_dim),
            strides=(stride_kl, stride_kd),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, head_dim),
            order=(1, 0),
        )
        # (m, d)
        km = tl.load(km_ptr, boundary_check=(0,))
        km_norm_ptr = tl.make_block_ptr(
            base=key_norm_ptr + batch_idx * stride_nb + head_idx * stride_nh,
            shape=(seq_len, 1),
            strides=(stride_nl, 1),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        # (m,1)
        km_norm = tl.load(km_norm_ptr, boundary_check=(0,))
        km = km / (km_norm + 1e-6)
        similarity_ptr = tl.make_block_ptr(
            base=similarity_cos_ptr + batch_idx * stride_sb + head_idx * stride_sh,
            shape=(seq_len, 1),
            strides=(stride_sl, 1),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        s = tl.load(similarity_ptr, boundary_check=(0,))
        raw_dtype = s.dtype
        s = s.to(tl.float32)
        for n_offset in range(num_pad, seq_len, BLOCK_N):
            kn_ptr = tl.make_block_ptr(
                base=key_states_ptr + batch_idx * stride_kb + head_idx * stride_kh,
                shape=(head_dim, seq_len),
                strides=(stride_kd, stride_kl),
                offsets=(0, n_offset),
                block_shape=(head_dim, BLOCK_N),
                order=(0, 1),
            )
            # (d, n)
            kn = tl.load(kn_ptr, boundary_check=(1,))
            kn_norm_ptr = tl.make_block_ptr(
                base=key_norm_ptr + batch_idx * stride_nb + head_idx * stride_nh,
                shape=(1, seq_len),
                strides=(1, stride_nl),
                offsets=(0, n_offset),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            kn_norm = tl.load(kn_norm_ptr, boundary_check=(1,))
            kn = kn / (kn_norm + 1e-6)
            # (m, n)
            similarity = tl.dot(km, kn)
            # zero out the similarity of the same key
            m_indices = m_offset + tl.arange(0, BLOCK_M)[:, None]
            n_indices = n_offset + tl.arange(0, BLOCK_N)[None, :]
            same_key_mask = m_indices == n_indices
            similarity = tl.where(same_key_mask, 0.0, similarity)
            # zero out the last token with high similarity
            zo_ptr = tl.make_block_ptr(
                base=zero_out_ptr + batch_idx * stride_zb + head_idx * stride_zh,
                shape=(1, seq_len),
                strides=(1, stride_zl),
                offsets=(0, n_offset),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            zo = tl.load(zo_ptr, boundary_check=(1,))
            threshold_mask = ((similarity > threshold) & (~zo)).to(tl.int1)
            col_indices = tl.arange(0, BLOCK_N)[None, :]
            max_col_per_row = tl.max(
                tl.where(
                    threshold_mask,
                    col_indices,
                    -1,
                ),
                axis=1,
                keep_dims=True,
            )
            last_threshold_mask = (col_indices == max_col_per_row) & threshold_mask
            similarity = tl.where(last_threshold_mask, 0.0, similarity)
            last_threshold_mask = tl.max(last_threshold_mask, axis=0, keep_dims=True)
            last_threshold_mask = last_threshold_mask | zo
            last_threshold_mask = last_threshold_mask.to(zo.dtype)
            tl.store(zo_ptr, last_threshold_mask, boundary_check=(1,))
            # reduce cosine similarity
            similarity = tl.sum(similarity, axis=1, keep_dims=True)
            s = similarity + s
        tl.store(similarity_ptr, s.to(raw_dtype), boundary_check=(0,))


def cal_similarity_triton(
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold=0.5,
    temperature=1.0,
):
    """
    calculate cosine similarity score between key states
    the last token with high similarity (similarity > threshold) will be masked out
    the similarity score of the same key will also be masked out
    Args:
        key_states: (batch_size, num_kv_heads, seq_len, head_dim)
        attention_mask: (batch_size,  seq_len)
        threshold: float
    Returns:
        similarity_score: (batch_size, num_heads, seq_len)
    """
    BLOCK_M = 16
    BLOCK_N = 64

    batch_size, num_heads, seq_len, head_dim = key_states.shape

    key_norm = key_states.norm(dim=-1)
    similarity_cos = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=key_states.dtype,
        device=key_states.device,
    )
    # tag for whether the similarity score of the same key has been masked out
    zero_out = torch.full(
        (batch_size, num_heads, seq_len),
        False,
        device=key_states.device,
        dtype=torch.bool,
    )
    num_pad = (attention_mask == 0).sum(dim=-1).to(torch.int32)

    grid = (batch_size * num_heads,)
    similarity_score_kernel[grid](
        key_states,
        similarity_cos,
        key_norm,
        num_pad,
        zero_out,
        **_strides(key_states, "kb", "kh", "kl", "kd"),
        **_strides(similarity_cos, "sb", "sh", "sl"),
        **_strides(key_norm, "nb", "nh", "nl"),
        **_strides(zero_out, "zb", "zh", "zl"),
        threshold=threshold,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    seq_len = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
    similarity_cos.div_(seq_len * temperature)
    similarity_cos = similarity_cos.masked_fill(
        (attention_mask == 0).unsqueeze(1), -float("inf")
    )

    return torch.softmax(similarity_cos, dim=-1)


@torch.no_grad()
def compute_attention_scores(
    query_states, key_states, pooling="max", attention_mask=None, remove_query=True
):
    """
    query_states: (bsz, q_heads, q_len, head_dim)
    key_states: (bsz, kv_heads, kv_cache_len, head_dim)
    return: (bsz, kv_heads, q_len, kv_cache_len - q_len)
    attention_mask: attention mask (bsz, kv_cache_len)
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

    if attention_mask is not None and torch.any(attention_mask == 0):
        if attention_mask.dim() == 2:
            # build causal mask (bsz,1,kv_cache_len,kv_cache_len) from attention_mask (bsz,kv_cache_len)
            # shape: (kv_cache_len, kv_cache_len)
            causal_mask = torch.triu(
                torch.ones(
                    kv_cache_len,
                    kv_cache_len,
                    device=attn_weights.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                1
            )  # (1,1,kv_cache_len,kv_cache_len)
            # shape: (bsz,1,kv_cache_len,kv_cache_len)
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            mask = (
                (attention_mask == 0).unsqueeze(1).unsqueeze(1)
            )  # (bsz,1,1,kv_cache_len)
            causal_mask = causal_mask.masked_fill(mask, True)
            # shape: (bsz,kv_heads,query_group_size,q_len,kv_cache_len)
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(2)[:, :, :, -q_len:, :], -float("inf")
            )
        else:
            raise ValueError("attention_mask must be 2D")
    else:
        # shape: (q_len, q_len)
        # no left padding, query can see all key before it
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
    if remove_query:
        return attn_scores[:, :, :, :-q_len]
    else:
        return attn_scores
