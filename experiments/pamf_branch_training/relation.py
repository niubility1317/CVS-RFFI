"""Identity relations within each actual PAMF feature path.

All inputs must come from the current source batch. The caller controls whether
labels are real L labels or accepted pseudo labels; target truth is never read.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


@contextmanager
def capture_branches(core, sequence, reference=None):
    """Capture deployed pre-fusion paths in one existing model forward."""
    modules = {'time': core.t_proj, 'frequency_main': core.f_proj,
               'frequency_stats': core.freq_stats_proj, 'pa_main': core.pa_proj,
               'pa_stats': core.pa_stats_proj, 'path': sequence}
    if reference is not None:
        modules['reference'] = reference
    captured, packet, handles = {}, {}, []
    try:
        for name, module in modules.items():
            if module is None:
                raise ValueError('Expected PAMF branch graph required')

            def record(_module, _inputs, output, name=name):
                if name in captured or not isinstance(output, torch.Tensor) or output.ndim != 2:
                    raise ValueError('Exactly one vector output per branch required')
                captured[name] = output

            handles.append(module.register_forward_hook(record))
        yield packet
        if captured.keys() != modules.keys():
            raise ValueError('An expected branch was not executed')
        packet.update(time=captured['time'],
                      frequency=captured['frequency_main'] + captured['frequency_stats'],
                      pa=captured['pa_main'] + .25 * captured['pa_stats'],
                      path=captured['path'])
        if reference is not None:
            packet['reference'] = captured['reference']
    finally:
        for handle in handles:
            handle.remove()


def branch_relation_loss(features, labels, receivers, record_ids, accepted, *, single_receiver, temperature=0.2):
    """Average supervised contrastive loss across paths, with cross-RX positives.

    A single-RX source contract uses different-record same-RX positives instead.
    Unaccepted examples do not act as anchors, positives, or negatives.
    """
    if not features or temperature <= 0:
        raise ValueError('At least one feature path and positive temperature required')
    n = labels.numel()
    if (labels.ndim != 1 or receivers.shape != labels.shape or record_ids.shape != labels.shape
            or accepted.shape != labels.shape or accepted.dtype != torch.bool):
        raise ValueError('Aligned labels, receiver, record ID and boolean mask required')
    if any(v.ndim != 2 or v.shape[0] != n for v in features.values()):
        raise ValueError('Each branch must contain the same batch of vectors')
    if single_receiver and receivers[accepted].unique().numel() > 1:
        raise ValueError('Single-receiver fallback cannot consume multiple receivers')
    if bool((labels[accepted] < 0).any()):
        raise ValueError('Accepted identities must be nonnegative')

    same_identity = labels[:, None] == labels[None, :]
    distinct_record = record_ids[:, None] != record_ids[None, :]
    eligible = accepted[:, None] & accepted[None, :] & distinct_record
    cross_receiver = receivers[:, None] != receivers[None, :]
    positive = eligible & same_identity & (torch.ones_like(cross_receiver) if single_receiver else cross_receiver)
    negative = eligible & ~same_identity
    valid = positive.any(dim=1) & negative.any(dim=1)
    count = int(valid.sum().item())
    counts = {'cross_receiver_anchors': 0 if single_receiver else count,
              'same_receiver_anchors': count if single_receiver else 0}
    losses = []
    for value in features.values():
        if not count:
            losses.append(value.sum() * 0.)
            continue
        z = F.normalize(value, dim=1, eps=1e-4)
        scores = (z @ z.T / temperature)[valid]
        positives = positive[valid]
        candidates = (positive | negative)[valid]
        log_denominator = torch.logsumexp(scores.masked_fill(~candidates, -torch.inf), dim=1)
        positive_mean = scores.masked_fill(~positives, 0.).sum(dim=1) / positives.sum(dim=1)
        losses.append((log_denominator - positive_mean).mean())
    return torch.stack(losses).mean(), counts
