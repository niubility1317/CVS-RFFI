import torch
from torch.nn import functional as F
from . import design as d


def group_risk(logits, labels, receivers, logq, mode, *, role='L_s'):
    if role != 'L_s' or mode not in ('ERM', 'BALANCED', 'DRO'):
        raise ValueError('Only labelled-source receiver risk is registered')
    if logits.ndim != 2 or logits.shape != (len(labels), 6) or not len(labels) or receivers is None:
        raise ValueError('Aligned default-head labelled-source logits required')
    if labels.shape != receivers.shape or labels.ndim != 1 or not torch.isfinite(logits.detach()).all():
        raise ValueError('Invalid source arrays')
    if set(receivers.detach().cpu().tolist()) - set(d.RX) or (labels < 0).any() or (labels >= 6).any():
        raise ValueError('Source receiver/TX set differs')
    if logq.shape != (5,) or logq.requires_grad or not torch.isfinite(logq).all():
        raise ValueError('Detached five-group log weights required')
    loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=.01)
    masks = [receivers == r for r in d.RX]
    counts = torch.stack([m.sum() for m in masks])
    means = torch.stack([loss[m].sum() / counts[k].clamp_min(1) for k, m in enumerate(masks)])
    present = counts > 0
    updated = logq.detach().double().cpu().clone()
    if mode == 'DRO':
        updated += d.STEP_SIZE * means.detach().double().cpu()
        updated -= torch.logsumexp(updated, dim=0)
    q = updated.softmax(0).to(device=logits.device, dtype=logits.dtype)
    applied = (q * present if mode == 'DRO' else counts.to(logits.dtype)
               if mode == 'ERM' else present.to(logits.dtype))
    applied = applied / applied.sum()
    selected = loss.mean() if mode == 'ERM' else (applied * means).sum()
    if not torch.isfinite(selected.detach()):
        raise FloatingPointError('Nonfinite receiver-risk loss')
    details = dict(group_counts=counts.detach().cpu().tolist(), group_mean_CE=means.detach().cpu().tolist(),
        group_weights=q.detach().double().cpu().tolist(), applied_weights=applied.detach().double().cpu().tolist(),
        original_CE=float(loss.detach().mean()), selected_CE=float(selected.detach()),
        detached_group_state=True, source_only=True)
    return selected, updated, details
