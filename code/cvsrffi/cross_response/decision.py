"""Single-sided protection of inference-head margins on training features."""
import torch


def decision_margin_loss_reference(logits, labels, rx_ids, physical_ids, condition_ids,
                         day_ids=None, delta=0.0, min_records=2, min_receivers=1):
    """Caller must supply the deployment head's logits with labels=None.

    Reliable references are correct, positive-margin source records of the same
    TX/day/condition, excluding this physical record and RX. Duplicate physical
    records never increase reference count. Each RX contributes equally to the
    median (midpoint median for an even number of receivers).
    """
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('finite inference logits with at least two classes required')
    if delta < 0 or min_records < 1 or min_receivers < 1:
        raise ValueError('invalid fixed reference thresholds')
    n,c = logits.shape
    labels = torch.as_tensor(labels,device=logits.device,dtype=torch.long)
    if labels.shape != (n,) or (labels < 0).any() or (labels >= c).any():
        raise ValueError('invalid source labels')
    def metadata(x):
        vals = x.detach().cpu().tolist() if torch.is_tensor(x) else list(x)
        if len(vals) != n or any(v is None for v in vals):
            raise ValueError('missing reference metadata')
        return vals
    rx, pid, cond = metadata(rx_ids), metadata(physical_ids), metadata(condition_ids)
    day = metadata(day_ids) if day_ids is not None else [None]*n
    ys = labels.detach().cpu().tolist()
    margins = logits.gather(1,labels[:,None])-logits
    refs, valid = torch.zeros_like(margins), torch.zeros_like(margins,dtype=torch.bool)
    detached = margins.detach()
    correct = (logits.detach().argmax(1)==labels).cpu().tolist()
    with torch.no_grad():
        for i in range(n):
            # Physical uniqueness is global within the comparable reference pool.
            groups, seen = {}, set()
            for j in range(n):
                if (ys[j]!=ys[i] or rx[j]==rx[i] or pid[j]==pid[i] or cond[j]!=cond[i]
                        or day[j]!=day[i] or not correct[j] or pid[j] in seen):
                    continue
                seen.add(pid[j])
                groups.setdefault(rx[j],[]).append(j)
            for b in range(c):
                if b == ys[i]:
                    continue
                per_rx = []
                for idx in groups.values():
                    good_records = detached[idx,b]
                    good_records = good_records[good_records>0]
                    if good_records.numel() >= min_records:
                        per_rx.append(good_records.mean())
                if len(per_rx) >= min_receivers:
                    good = torch.stack(per_rx)
                    # CUDA autocast does not promote quantile's half input.
                    # Preserve the mathematical median then cast on assignment.
                    quantile_input = good.float() if good.dtype in (torch.float16,torch.bfloat16) else good
                    refs[i,b] = quantile_input.quantile(0.5)
                    valid[i,b] = True
    gap = torch.relu(refs.detach()-delta-margins)
    loss = gap[valid].square().mean() if valid.any() else logits.sum()*0
    return loss, dict(valid_comparisons=int(valid.sum()), possible_comparisons=n*(c-1),
                      valid_fraction=float(valid.sum())/max(n*(c-1),1),
                      reference=refs.detach(), valid_mask=valid.detach(),
                      mean_gap=gap[valid].mean().detach() if valid.any() else logits.detach().sum()*0,
                      reason='ok' if valid.any() else 'no_reliable_other_rx_reference')


def decision_margin_loss_vectorized(logits, labels, rx_ids, physical_ids, condition_ids,
                                   day_ids=None, delta=0.0, min_records=2, min_receivers=1,
                                   pair_weights=None):
    """Exact reference selection with batched receiver means and midpoint median.

    Optional detached [N,C] noise/weight tensors are supplied only by an explicitly
    frozen source calibration. NaN noise means insufficient evidence, not zero.
    """
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('finite inference logits with at least two classes required')
    if min_records < 1 or min_receivers < 1:
        raise ValueError('invalid fixed reference thresholds')
    n, c = logits.shape
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    if labels.shape != (n,) or (labels < 0).any() or (labels >= c).any():
        raise ValueError('invalid source labels')
    def metadata(values):
        result = values.detach().cpu().tolist() if torch.is_tensor(values) else list(values)
        if len(result) != n or any(v is None for v in result):
            raise ValueError('missing reference metadata')
        return result
    rx, pid, cond = metadata(rx_ids), metadata(physical_ids), metadata(condition_ids)
    day = metadata(day_ids) if day_ids is not None else [None] * n
    ys = labels.detach().cpu().tolist()
    correct = (logits.detach().argmax(1) == labels).cpu().tolist()
    receiver_ids = list(dict.fromkeys(rx))
    # Preserve first eligible physical occurrence, even when duplicated records
    # disagree on RX. Deduplicating the whole batch first changes legacy semantics.
    selection = []
    for i in range(n):
        seen, row = set(), []
        for j in range(n):
            eligible = (ys[j] == ys[i] and rx[j] != rx[i] and pid[j] != pid[i]
                        and cond[j] == cond[i] and day[j] == day[i] and correct[j]
                        and pid[j] not in seen)
            row.append(eligible)
            if eligible:
                seen.add(pid[j])
        selection.append(row)
    margins = logits.gather(1, labels[:, None]) - logits
    with torch.no_grad():
        select = torch.tensor(selection, dtype=torch.bool, device=logits.device)
        receiver = torch.tensor([[r == key for key in receiver_ids] for r in rx],
                                dtype=torch.bool, device=logits.device)
        positive = margins.detach() > 0
        membership = select[:, :, None] & receiver[None, :, :]
        # Explicit multiply/sum keeps autocast from silently lowering reference
        # precision through GEMM. No gradient graph is built for source references.
        mask = membership[:, :, :, None] & positive[None, :, None, :]
        counts = mask.sum(1)
        accumulation = margins.detach().float() if margins.dtype in (torch.float16,torch.bfloat16) else margins.detach()
        sums = (mask * accumulation[None, :, None, :]).sum(1)
        means = (sums / counts.clamp_min(1)).to(margins.dtype)
        reliable = counts >= min_records
        rx_count = reliable.sum(1)
        sorted_means = means.masked_fill(~reliable, float('inf')).sort(dim=1).values
        low = ((rx_count - 1).clamp_min(0) // 2)[:, None, :]
        high = (rx_count.clamp_min(1) // 2)[:, None, :]
        quantile_values = sorted_means.float() if margins.dtype in (torch.float16,torch.bfloat16) else sorted_means
        midpoint = ((quantile_values.gather(1, low).squeeze(1) +
                     quantile_values.gather(1, high).squeeze(1)) * 0.5).to(margins.dtype)
        valid = rx_count >= min_receivers
        valid.scatter_(1, labels[:, None], False)
        refs = torch.where(valid, midpoint, torch.zeros_like(midpoint))
        noise = torch.as_tensor(delta, dtype=logits.dtype, device=logits.device).detach()
        if (noise < 0).any() or torch.isinf(noise).any():
            raise ValueError('delta must be nonnegative or NaN for unavailable evidence')
        noise = torch.broadcast_to(noise, margins.shape)
        valid &= torch.isfinite(noise)
        weight = torch.ones_like(margins) if pair_weights is None else torch.broadcast_to(
            torch.as_tensor(pair_weights, dtype=logits.dtype, device=logits.device).detach(), margins.shape)
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('pair weights must be finite and nonnegative')
        valid &= weight > 0
    gap = torch.relu(refs.detach() - torch.nan_to_num(noise) - margins)
    loss = (gap.square() * weight)[valid].mean() if valid.any() else logits.sum() * 0
    selected = gap[valid].detach()
    positive_gap = valid & (gap.detach() > 0)
    possible = n * (c - 1)
    count = int(valid.sum())
    quantiles = torch.quantile(selected.float(), torch.tensor([.5,.9,.95], device=logits.device)) if count else torch.zeros(3, device=logits.device)
    pair_budget = {}
    for y in sorted(set(ys)):
        for b in range(c):
            if b != y:
                choose = valid & (labels[:,None] == y)
                choose[:, :b] = False
                choose[:, b+1:] = False
                # Exact weighted derivative budget with respect to the margin;
                # parameter-gradient budgets still require the training audit.
                pair_budget[(y,b)] = float((2 * gap.detach() * weight * choose).sum() / max(count,1))
    return loss, dict(valid_comparisons=count, possible_comparisons=possible,
        valid_fraction=count/max(possible,1), reference=refs.detach(), valid_mask=valid.detach(),
        mean_gap=selected.mean() if count else logits.detach().sum()*0,
        positive_gap_comparisons=int(positive_gap.sum()),
        positive_gap_fraction_valid=float(positive_gap.sum())/max(count,1),
        positive_gap_fraction_all=float(positive_gap.sum())/max(possible,1),
        reference_record_counts=counts.detach(), reference_receiver_counts=rx_count.detach(),
        gap_p50=quantiles[0], gap_p90=quantiles[1], gap_p95=quantiles[2],
        weighted_margin_gradient_budget_by_pair=pair_budget,
        reason='ok' if count else 'no_reliable_other_rx_reference')


# Keep the existing default and historical numerical path for legacy configs.
decision_margin_loss = decision_margin_loss_reference
