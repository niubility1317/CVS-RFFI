"""Single-sided protection of inference-head margins on training features."""
import torch


def decision_margin_loss(logits, labels, rx_ids, physical_ids, condition_ids,
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
                    refs[i,b] = good.quantile(0.5)
                    valid[i,b] = True
    gap = torch.relu(refs.detach()-delta-margins)
    loss = gap[valid].square().mean() if valid.any() else logits.sum()*0
    return loss, dict(valid_comparisons=int(valid.sum()), possible_comparisons=n*(c-1),
                      valid_fraction=float(valid.sum())/max(n*(c-1),1),
                      reference=refs.detach(), valid_mask=valid.detach(),
                      mean_gap=gap[valid].mean().detach() if valid.any() else logits.detach().sum()*0,
                      reason='ok' if valid.any() else 'no_reliable_other_rx_reference')
