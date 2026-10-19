"""Joint default concat CE, native DG objectives, and optional ordinary PL."""
import torch
import torch.nn.functional as F

from experiments.phase1_dg_nativebudget.pipeline import finite
from experiments.phase1_external_dg.models import source_coral
from .models import identity
from .pl import pseudo_targets, masked_pl_loss


def step(net, c, x, y, rx, optimizer, augment, progress, batch_index, *,
         stage=1, fed_optimizer=None, unlabeled=None, native_views=None,
         author=None, fishr=None, unlabeled_receivers=None):
    enabled = bool(c['U_s_use'])
    if enabled != (unlabeled is not None):
        raise ValueError('The PL/supervised input permission differs')
    if c['satellite_train_augmentation'] is not True or c['lambda_sat_cls'] != .68:
        raise ValueError('Every formal row must retain matched concat augmentation')
    full = c['variant'] == 'full'
    sdg_full = c['family'] == 'sdg' and full
    b = len(y)
    if sdg_full:
        if author is None or native_views is None or len(native_views) != 2:
            raise ValueError('SDG requires its two native source views')
        views = list(native_views)
        base = views[-1]
    else:
        views = [x]
        base = x
    satellite, sat_info = augment(base, progress, batch_index, stage)
    sat_offset = sum(len(v) for v in views)
    parts = views + [satellite]
    targets = accepted = confidence = None
    if enabled:
        targets, accepted, confidence = pseudo_targets(net, lambda z: identity(net, c, z), unlabeled, c['pl_threshold'])
        parts.append(unlabeled)
    joined = torch.cat(parts, dim=0)
    net.train()
    if c['family'] == 'resnet18':
        net.context = (y, rx)
    clean_offset = (len(views) - 1) * b
    z = None
    clean_drift = None
    if c['family'] == 'riei':
        features = net.fed(joined)
        logits = net.ec(features[:, :net.emitter_feature_dim])
    elif c['family'] == 'drift':
        from baselines.drift.grl import gradient_reverse
        features = net.encoder(joined)
        z_tx, z_rx = features[:, :net.split_dim], features[:, net.split_dim:]
        logits = net.tx_classifier(z_tx)
        if full:
            clean_drift = dict(z=features[:b], z_tx=z_tx[:b], z_rx=z_rx[:b],
                tx_logits=logits[:b], rx_logits=net.rx_classifier(z_rx[:b]),
                domain_logits=net.domain_discriminator(gradient_reverse(z_tx[:b], 1.)))
    else:
        z, logits = net(joined)
    ce = F.cross_entropy(logits[clean_offset:clean_offset+b], y)
    sat_ce = F.cross_entropy(logits[sat_offset:sat_offset+b], y)
    pl = logits.sum() * 0.
    if enabled:
        pl = masked_pl_loss(logits[sat_offset+b:], targets, accepted)
    objective = ce + .68 * sat_ce + c['pl_weight'] * pl
    native_loss = ce.new_zeros(())
    if c['family'] == 'riei' and full:
        if fed_optimizer is None:
            raise ValueError('RIEI alternating feature optimizer required')
        native_loss = F.cross_entropy(net.rc(features[:b, net.emitter_feature_dim:]), rx)
        objective = objective + native_loss
    elif clean_drift is not None:
        from baselines.drift.losses import compute_drift_loss
        native_total = compute_drift_loss(clean_drift, y, rx, lambda_grl=1.,
            lambda_center=.01, lambda_mse=.02, mse_reduction='sum', mse_cap=0.,
            normalize_features_for_mse=False, center_mode='batch')['loss']
        native_loss = native_total - ce
        objective = objective + native_loss
    elif c['family'] == 'ibrff' and full and stage == 2:
        from utils.hsic import hsic_normalized_cca
        native_loss = .04 * hsic_normalized_cca(z[:b].flatten(1), x.flatten(1))
        objective = objective + native_loss
    elif sdg_full:
        labels = y.repeat(2)
        domain = torch.arange(2, device=y.device).repeat_interleave(b)
        contrastive = author.SupConLoss().to(y.device)
        native_loss = contrastive(z[:2*b].unsqueeze(1), labels, adv=False)
        native_loss = native_loss + 100. * contrastive(z[:2*b].unsqueeze(1), domain, adv=True)
        objective = objective + native_loss
    elif c['family'] == 'resnet18' and c['route'] == 'fishr':
        if fishr is None: raise ValueError('Fishr state is required')
        penalty = fishr(z[:b], logits[:b], y, rx)
        native_loss = (c['fishr_lambda'] if batch_index >= c['fishr_warmup'] else 0.) * penalty
        objective = objective + native_loss
    elif c['family'] == 'resnet18' and c['route'] == 'coral':
        native_loss = c['coral_lambda'] * source_coral(z[:b], rx, c['num_domains'])
        objective = objective + native_loss
    optimizer.zero_grad()
    if fed_optimizer is not None: fed_optimizer.zero_grad()
    objective.backward(); finite(net, objective); optimizer.step()
    if fed_optimizer is not None: fed_optimizer.step()
    extra = 0.
    if c['family'] == 'riei' and full:
        from baselines.riei_fd.losses import mutual_independence_loss, entropy_from_logits
        for module in (net.ec, net.rc):
            for p in module.parameters(): p.requires_grad_(False)
        try:
            output = net(x)
            dis = 1.2 * mutual_independence_loss(output['z_e'], output['z_r'], mode='cosine_abs', reduction='mean')
            dis = dis - 1.2 * (entropy_from_logits(output['cross_emitter_logits']) + entropy_from_logits(output['cross_receiver_logits']))
            fed_optimizer.zero_grad(); dis.backward(); finite(net, dis); fed_optimizer.step()
            extra = float(dis.detach())
        finally:
            for module in (net.ec, net.rc):
                for p in module.parameters(): p.requires_grad_(True)
    if c['family'] == 'resnet18': net.context = None
    receiver_counts = {}
    if enabled and unlabeled_receivers is not None:
        urx = unlabeled_receivers.detach().cpu()
        if len(urx) != len(accepted): raise ValueError('U receiver metadata length differs')
        mask = accepted.detach().cpu()
        for domain in range(c['num_domains']):
            membership = urx == domain
            receiver_counts[str(domain)] = [int(membership.sum()), int((membership & mask).sum())]
    return dict(loss=float(objective.detach()) + extra, identity_ce=float(ce.detach()),
        satellite_ce=float(sat_ce.detach()), native_penalty=float(native_loss.detach()) + extra,
        pl_loss=float(pl.detach()), pl_enabled=enabled,
        pl_seen=len(unlabeled) if enabled else 0,
        pl_accepted=int(accepted.sum()) if enabled else 0,
        pl_confidence_sum=float(confidence.sum()) if enabled else 0.,
        pl_receiver_counts=receiver_counts,
        satellite_samples=b, satellite_applied_samples=b if sat_info['applied'] else 0,
        satellite_scenario=sat_info['scenario'], satellite_probability=sat_info['probability'],
        satellite_weight=.68, concat_sat_ce_only=True,
        forward_examples=len(joined) + (len(unlabeled) if enabled else 0) + (b if c['family']=='riei' and full else 0),
        optimizer_steps=3 if c['family']=='riei' and full else 1)
