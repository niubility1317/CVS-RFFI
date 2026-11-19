"""Source-only Hybrid losses, pairing and branch gradient protection."""

import torch
from torch.nn import functional as F


TEMPERATURE = .2


def masked_mean(values, mask):
    return (values * mask).mean()


def packet_contrast(raw, altered, y, accepted, selected):
    raw = F.normalize(raw.detach(), dim=-1)
    altered = F.normalize(altered, dim=-1)
    negative = (y[:, None] != y[None, :]) & accepted[None, :]
    active = selected & negative.any(dim=-1)
    candidates = negative | torch.eye(len(y), device=y.device, dtype=torch.bool)
    score = altered @ raw.transpose(-1, -2) / TEMPERATURE
    nll = score.masked_fill(~candidates, -torch.inf).logsumexp(-1) - score.diagonal(dim1=-2, dim2=-1)
    return (nll * active).mean(), active.sum()


def day_relation(branches, y, days, record_ids, accepted):
    distinct = record_ids[:, None] != record_ids[None, :]
    eligible = distinct & accepted[:, None] & accepted[None, :]
    same = y[:, None] == y[None, :]
    positive = eligible & same & (days[:, None] != days[None, :])
    negative = eligible & ~same
    valid = positive.any(dim=1) & negative.any(dim=1)
    branches = F.normalize(branches, dim=-1)
    score = branches @ branches.transpose(-1, -2) / TEMPERATURE
    # The identity diagonal supplies a finite dummy denominator for invalid rows.
    mask = positive | negative | (~valid[:, None] & torch.eye(len(y), device=y.device, dtype=torch.bool))
    lse = score.masked_fill(~mask, -torch.inf).logsumexp(-1)
    average_pos = (score * positive).sum(-1) / positive.sum(-1).clamp_min(1)
    return ((lse - average_pos) * valid).sum() / (branches.shape[0] * valid.sum().clamp_min(1)), valid.sum()


def centers(branches, y, days, classes=66):
    other_day = torch.arange(8, device=y.device)[:, None] != days[None, :]
    membership = F.one_hot(y, classes).T.unsqueeze(0) * other_day.unsqueeze(1)
    valid = membership.any(-1).all(-1)
    center = torch.einsum("dcj,bjf->dbcf", membership.float(), F.normalize(branches.detach().float(), dim=-1))
    return F.normalize(center, dim=-1), valid


def prototypes(centroids, query, days):
    selected = centroids[days]
    return torch.einsum("ncfk,nck->ncf", selected, F.normalize(query.transpose(0, 1), dim=-1)) / TEMPERATURE


def prototype_loss(scores, y, selected):
    branches = scores.shape[1]
    loss = F.cross_entropy(scores.transpose(1, 2), y[:, None].expand(-1, branches), reduction="none")
    return masked_mean(loss.mean(1), selected)


def supervised(raw, ordinary, physical, y, days, ids, use_hybrid, has_physics):
    logits, zi, branches = raw
    logits_o = ordinary
    logits_p, z_p, branches_p = physical
    p, q = logits.log_softmax(1), logits_o.log_softmax(1)
    kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
    basic = (.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits_o, y))
             + kl + .68 * F.cross_entropy(logits_p, y))
    metrics = {"base": basic.detach(), "symmetric_kl": kl.detach()}
    if not use_hybrid:
        return basic, metrics
    accepted = torch.ones_like(y, dtype=torch.bool)
    selected = accepted & has_physics
    fused, packets = packet_contrast(zi, z_p, y, accepted, selected)
    paths, _ = packet_contrast(branches, branches_p, y, accepted, selected)
    relation, anchors = day_relation(branches, y, days, ids, accepted)
    centroids, available = centers(branches, y, days)
    proto_mask = selected & available[days]
    proto = prototype_loss(prototypes(centroids, branches_p, days), y, proto_mask)
    metrics.update(packet_fusion=fused.detach(), packet_branches=paths.detach(),
                   day_relation=relation.detach(), prototype=proto.detach(), packet_anchors=packets,
                   day_anchors=anchors, prototype_anchors=proto_mask.sum())
    return basic + .05 * (fused + paths + proto) + .025 * relation, metrics


def partners(y, days, ids, accepted):
    candidates = ((y[:, None] == y[None, :]) & (days[:, None] != days[None, :])
                  & (ids[:, None] != ids[None, :]) & accepted[:, None] & accepted[None, :])
    return candidates.int().argmax(dim=1), candidates.any(dim=1)


def style_extrapolation(z, partner, selected):
    peer = z[partner].detach()
    mean = z.mean(-1, keepdim=True)
    scale = z.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    pmean = peer.mean(-1, keepdim=True)
    pscale = peer.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    style = (z - mean) / scale * (.75 * scale + .25 * pscale) + .75 * mean + .25 * pmean
    return torch.where(selected[:, None], 1.25 * style - .25 * peer, z)


def group(name):
    if "cls_head." in name:
        return "head"
    if ".t2." in name:
        return "t2"
    if name.startswith("sequence_pool."):
        return "path"
    if name.startswith(("response_projection.", "response_gain")):
        return "reference"
    leaf = name.removeprefix("core.id_backbone.").split(".")[0]
    if leaf.startswith(("f1", "f2", "f3", "f_proj", "freq")):
        return "frequency"
    if leaf.startswith("pa"):
        return "pa"
    if leaf == "t_proj":
        return "time"
    return "other"


def protected_backward(model, labeled, unlabeled):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    left = torch.autograd.grad(labeled, params, allow_unused=True)
    right = torch.autograd.grad(unlabeled, params, allow_unused=True)
    groups = {}
    for (name, p), a, b in zip(named, left, right, strict=True):
        if a is not None or b is not None:
            groups.setdefault(group(name), []).append(
                (p, torch.zeros_like(p) if a is None else a, torch.zeros_like(p) if b is None else b))
    stats = {}
    for name, values in groups.items():
        ln2 = torch.stack([a.square().sum() for _, a, _ in values]).sum()
        dot = torch.stack([(a * b).sum() for _, a, b in values]).sum()
        coefficient = torch.minimum(dot, torch.zeros_like(dot)) / ln2.clamp_min(1e-12)
        projected = [b - coefficient * a for _, a, b in values]
        un2 = torch.stack([b.square().sum() for b in projected]).sum()
        factor = (ln2 / un2.clamp_min(1e-12)).sqrt().clamp(max=1)
        for (p, a, _), b in zip(values, projected, strict=True):
            p.grad = a + factor * b
        stats[f"guard_{name}_conflict"] = (dot < 0).int()
        stats[f"guard_{name}_cap"] = (factor < 1).int()
        stats[f"guard_{name}_zero"] = ((ln2 < 1e-12) | (un2 < 1e-12)).int()
    return stats
