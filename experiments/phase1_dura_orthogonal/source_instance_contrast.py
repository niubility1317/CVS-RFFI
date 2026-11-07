"""Source-only same-record contrast, not a transmitter-identity objective."""
import torch

TEMPERATURE = .1


def instance_contrast_loss(first, second, record_ids):
    """Treat every occurrence of the same physical source record as positive."""
    if (first.ndim != 2 or first.shape != second.shape or len(first) < 2
            or record_ids.shape != (len(first),) or record_ids.dtype != torch.int64
            or first.device != second.device or first.device != record_ids.device):
        raise ValueError('Matching source features and integer record IDs required')
    similarities = first @ second.T / TEMPERATURE
    positive = record_ids[:, None] == record_ids[None, :]
    positive_logits = similarities.masked_fill(~positive, -torch.inf)
    forward = torch.logsumexp(similarities, 1) - torch.logsumexp(positive_logits, 1)
    backward = torch.logsumexp(similarities, 0) - torch.logsumexp(positive_logits, 0)
    return .5*(forward.mean() + backward.mean())
