import torch
import torch.nn.functional as F

@torch.compile
def compute_entropy(logits: torch.Tensor):
    """
    Adapted from https://github.com/OpenRLHF/OpenRLHF
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """
    Adapted from https://github.com/OpenRLHF/OpenRLHF
    """
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)

    return logsumexp_values


def log_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Adapted from https://github.com/OpenRLHF/OpenRLHF
    """
    if temperature != 1.0:
        logits.div_(temperature)
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(
                logits.reshape(-1, last_dim), labels.reshape(-1)
            )
            log_probs_labels = -output[0].view(*batch_dim)
        except ImportError:
            logits_labels = torch.gather(
                logits, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
            logsumexp_values = _logsumexp_by_chunk(logits.reshape(-1, last_dim))
            logsumexp_values = logsumexp_values.view(*batch_dim)
            log_probs_labels = (
                logits_labels - logsumexp_values
            )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(
            logits, labels
        ):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels

def masked_sum(values, mask, axis=None):
    """
    Adapted from https://github.com/volcengine/verl
    Compute mean of tensor with a masked values.
    """
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Adapted from https://github.com/volcengine/verl
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)