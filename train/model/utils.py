import torch
import math


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
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


@torch.no_grad()
def compute_attention_scores(
    query_states, key_states, pooling="max", attention_mask=None
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
    return attn_scores[:, :, :, :-q_len]