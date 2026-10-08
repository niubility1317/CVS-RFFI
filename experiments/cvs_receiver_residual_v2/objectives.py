"""Detached L-only targets. No domain adversary or global orthogonality."""
import torch
import torch.nn.functional as F
from experiments.cvs_receiver_residual_v2.design import RECIPE, SOURCE_RXS


def cell_means(values, y, rx):
    means, counts = [], []
    for c in range(6):
        rows, nums = [], []
        for d in SOURCE_RXS:
            selected = values[(y == c) & (rx == d)]
            rows.append(selected.mean(0) if len(selected) else values.sum(0)*0.)
            nums.append(len(selected))
        means.append(torch.stack(rows)); counts.append(nums)
    return torch.stack(means), torch.tensor(counts, device=values.device)


def leave_tx_targets(centroids):
    # [held TX, receiver d, receiver e, feature]; held TX never enters its target.
    return torch.stack([torch.stack([
        centroids[k, :, None] - centroids[k, None, :] for k in range(6) if k != c
    ]).median(dim=0).values for c in range(6)]).detach()


def displacement_loss(prediction, y, rx, bank):
    means, counts = cell_means(prediction, y, rx)
    present = counts > 0
    pairs = present[:, :, None] & present[:, None, :]
    pairs &= ~torch.eye(5, device=y.device, dtype=torch.bool)[None]
    scale = bank['scale'].to(prediction)
    target = bank['targets'].to(prediction)
    difference = means[:, :, None] - means[:, None, :]
    loss = F.smooth_l1_loss(difference[pairs]/scale, target[pairs]/scale) if pairs.any() else prediction.sum()*0.
    gauges = (means*present[:, :, None]).sum(1)/present.sum(1).clamp_min(1)[:, None]
    gauge = (gauges[present.any(1)]/scale).square().mean()
    return loss, gauge, int(pairs.sum())


def contribution_targets(model, d, y):
    with torch.no_grad():
        z = d['base'].detach()
        full = F.cross_entropy(model.encoder.classify_features(z), y, reduction='none')
        deleted = torch.stack([F.cross_entropy(model.encoder.classify_features(
            z-d['branches'][:, b].detach()), y, reduction='none') for b in range(3)], dim=1)
        return (deleted-full[:, None]).clamp(-RECIPE['contribution_clip'], RECIPE['contribution_clip'])


def auxiliary_losses(model, d, y, rx, bank, strength):
    zero = d['base'].sum()*0.
    displacement = gauge = contribution = zero
    pairs = 0
    target_mean = None
    if strength > 0 and d['correction'] is not None:
        if bank is None:
            raise ValueError('Active compensation requires own detached L-only centroids')
        displacement, gauge, pairs = displacement_loss(d['correction'][:len(y)], y, rx, bank)
    if strength > 0 and d['contribution_prediction'] is not None:
        clean = {k: v[:len(y)] for k, v in d.items() if k in ('base', 'branches')}
        target = contribution_targets(model, clean, y)
        contribution = F.smooth_l1_loss(d['contribution_prediction'][:len(y)], target)
        target_mean = target.mean(0).cpu().tolist()
    loss = strength*(RECIPE['displacement_weight']*displacement + RECIPE['gauge_weight']*gauge
        + RECIPE['contribution_weight']*contribution)
    return loss, dict(displacement_loss=displacement, gauge_loss=gauge,
        contribution_loss=contribution, cross_tx_pair_count=pairs,
        branch_deletion_ce_delta_mean=target_mean)


@torch.no_grad()
def collect_bank(model, loader, device):
    """Current-model evaluation pass over L, never V/U or target. No EMA."""
    model.eval()
    sums = torch.zeros(6, 5, 160, device=device)
    counts = torch.zeros(6, 5, device=device)
    square_sum = 0.; n = 0
    for batch in loader:
        z = model.base_details(batch['iq'].to(device))['base']
        y, rx = batch['label'].to(device), batch['receiver'].to(device)
        means, nums = cell_means(z, y, rx)
        sums += means*nums[:, :, None]; counts += nums
        square_sum += float(z.square().sum()); n += len(z)
    if n != 6300 or not torch.all(counts == 210):
        raise ValueError('L-only centroid cells must each contain 210 physical observations')
    centroids = sums/counts[:, :, None]
    targets = leave_tx_targets(centroids)
    own = centroids[:, :, None] - centroids[:, None, :]
    mask = ~torch.eye(5, device=device, dtype=torch.bool)
    error = (own[:, mask]-targets[:, mask]).square().sum()
    denom = own[:, mask].square().sum().clamp_min(1e-8)
    scale = (square_sum/(n*160))**.5
    diagnostics = dict(role='L_s', count=n, cell_counts=counts.cpu().tolist(),
        cross_tx_relative_prediction_error=float(error/denom),
        cross_tx_direction_cosine=float(F.cosine_similarity(own[:, mask], targets[:, mask], dim=-1).mean()),
        feature_rms_scale=scale, target_rms=float(targets.square().mean().sqrt()),
        target_access=False, V_used=False, ema=False)
    return dict(targets=targets.cpu(), scale=torch.tensor(max(scale, 1e-4))), diagnostics
