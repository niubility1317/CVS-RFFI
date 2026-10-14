"""No-fit geometry analysis of balanced visible source feature witnesses.

The margin is fixed before inspecting geometry. This is a source result
analysis, not a training objective, model selection, or transfer evaluation.
"""
import torch
import torch.nn.functional as F

MARGIN = .1


def measure(features, labels, receivers, days, physical_ids):
    count = len(features)
    if count not in (90, 6300) or features.shape != (count, 160) or not torch.isfinite(features).all():
        raise ValueError('Actual finite balanced source L90 or full L6300 joint160 required')
    if any(len(v) != count for v in (labels, receivers, days, physical_ids)):
        raise ValueError('Visible L metadata must match each feature')
    keys = [(int(y), int(r), int(d)) for y, r, d in zip(labels, receivers, days)]
    expected = {(y, r, d) for y in range(6) for r in (1, 3, 4, 6, 8)
                for d in (1, 2, 3)}
    if set(keys) != expected:
        raise ValueError('Original source TX/RX/day cells required')
    if len({tuple(int(v) for v in pid) for pid in physical_ids}) != count:
        raise ValueError('Physical packets must be distinct')
    if (features.float().norm(dim=1) <= 1e-4).any():
        raise ValueError('Undefined normalized feature geometry')
    index = {}
    offsets = []
    for i, key in enumerate(keys):
        indices = index.setdefault(key, [])
        offsets.append(len(indices))
        indices.append(i)
    if any(len(indices) != count // 90 for indices in index.values()):
        raise ValueError('Equal original source packet count per cell required')
    triplets = []
    for i, (y, r, day) in enumerate(keys):
        for reference_rx in (1, 3, 4, 6, 8):
            if reference_rx == r:
                continue
            p = index[y, reference_rx, day][offsets[i]]
            for negative_y in range(6):
                if negative_y != y:
                    triplets.append((i, p, index[negative_y, reference_rx, day][offsets[i]]))
    relation_count = count * 20
    if len(triplets) != relation_count:
        raise ValueError('Fixed all-reference source relation count differs')
    # A detached feature copy measures local derivative without modifying any
    # model, saved feature, optimizer, teacher, or training RNG.
    x = features.detach().float().clone().requires_grad_(True)
    z = F.normalize(x, dim=1, eps=1e-4)
    ids = torch.tensor(triplets, dtype=torch.long, device=x.device)
    positive = (z[ids[:, 0]] * z[ids[:, 1]]).sum(dim=1)
    negative = (z[ids[:, 0]] * z[ids[:, 2]]).sum(dim=1)
    gap = positive - negative
    loss = F.relu(MARGIN - gap)
    gradient = torch.autograd.grad(loss.mean(), x)[0]
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite source local feature derivative')

    def summary(mask):
        g, l = gap[mask].detach(), loss[mask].detach()
        return dict(relations=len(g), positive_negative_gap_mean=float(g.mean()),
                    gap_min=float(g.min()), gap_q10=float(torch.quantile(g, .1)),
                    gap_median=float(g.median()), gap_q90=float(torch.quantile(g, .9)),
                    ordering_violations=int((g < 0).sum()),
                    margin_unsatisfied=int((g < MARGIN).sum()),
                    margin_unsatisfied_rate=float((g < MARGIN).float().mean()),
                    hinge_mean=float(l.mean()))

    result = dict(status='SOURCE_GEOMETRY_COMPLETE', source_count=count,
                  feature='actualjoint160', margin=MARGIN,
                  relation_rule='Same day; anchor RX differs from one reference RX; '
                                'positive and all five other-TX negatives share reference RX; '
                                'same fixed within-cell ordinal, no mining or margin scan.',
                  metadata='Observed visible L labels/RX/day and matched physical IDs',
                  overall=summary(torch.ones(relation_count, dtype=torch.bool, device=x.device)),
                  local_feature_gradient_norm=float(gradient.norm()),
                  local_feature_gradient_nonzero_packets=int((gradient.norm(dim=1) > 0).sum()),
                  extra_IQ_forwards=0, optimizer_updates=0, head_fit_updates=0,
                  U_truth_read=False, target_access=False,
                  limitation='Seen-source packets, deterministic within-cell ordinal pairing; '
                             '20 correlated relations per anchor, not independent samples. '
                             'Nonzero hinge is only potential supervision, not efficacy, '
                             'novelty, cumulative training effect or Phase2 evidence.')
    anchors = ids[:, 0].cpu().tolist()
    for name, field, values in (('RX', 1, (1, 3, 4, 6, 8)), ('day', 2, (1, 2, 3))):
        result[name] = {str(value): summary(torch.tensor(
            [keys[i][field] == value for i in anchors], dtype=torch.bool, device=x.device))
            for value in values}
    return result
