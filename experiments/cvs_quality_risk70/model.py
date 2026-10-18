"""Keep the six native L forwards; relocate only complete labeled As packs."""
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F


def weighted_satellite_terms(first, second, y, groups, p_g, method):
    """Finite-pool CE estimator and the original unweighted same-IQ KL.

    The four nominal weights are fixed by the complete generated pool.  They
    are deliberately not replaced by the current batch's group frequencies.
    This helper has no model forward and also accepts small synthetic batches.
    """
    if (method not in ('POOL_RANDOM', 'QUALITY_STRAT') or first.ndim != 2
            or first.shape != second.shape or first.shape != (len(y), 6)
            or len(y) == 0 or y.dtype != torch.long or y.shape != (len(y),)
            or not bool(((y >= 0) & (y < 6)).all())
            or not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all())):
        raise ValueError('Finite paired default-head logits and visible source labels required')
    groups = torch.as_tensor(groups, device=y.device, dtype=torch.long)
    weights = torch.as_tensor(p_g, device=first.device, dtype=first.dtype)
    if (groups.shape != y.shape or not bool(((groups >= 0) & (groups < 4)).all())
            or weights.shape != (4,) or not bool(torch.isfinite(weights).all())
            or not bool((weights >= 0).all())
            or not torch.isclose(weights.sum(), weights.new_tensor(1.), atol=1e-6, rtol=0)):
        raise ValueError('Four valid fixed finite-pool weights and packet quality groups required')
    per_ce = .5 * (F.cross_entropy(first, y, reduction='none', label_smoothing=.01)
                    + F.cross_entropy(second, y, reduction='none', label_smoothing=.01))
    p, q = first.log_softmax(1), second.log_softmax(1)
    per_kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1)
    counts = [int((groups == g).sum()) for g in range(4)]
    plain_ce = per_ce.mean()
    if method == 'POOL_RANDOM':
        ce = plain_ce
    else:
        if any(float(weights[g]) > 0 and counts[g] == 0 for g in range(4)):
            raise ValueError('Every positive nominal group must occur in a stratified step')
        ce = sum(weights[g] * per_ce[groups == g].mean()
                 for g in range(4) if float(weights[g]) > 0)
    kl = per_kl.mean()
    return ce, kl, dict(counts=counts, p_g=[float(v) for v in weights.detach().cpu()],
                       group_CE=[float(per_ce[groups == g].mean().detach()) if counts[g] else None
                                 for g in range(4)],
                       CE_unweighted=float(plain_ce.detach()), CE_weighted=float(ce.detach()),
                       KL_unweighted=float(kl.detach()))


def _labeled_hook(self, *, model, labeled_output, labels, extra, domains,
                  optimizer_step, identity_loss, labeled_count, labeled_inputs):
    from experiments.cvs_crossview_supcon41 import model as base, relations
    from experiments.cvs_fullsource_concat20 import supcon
    from experiments.cvs_noise_shape_factorial43 import noise

    if (model is not self or not self.training or not torch.is_grad_enabled()
            or labeled_count != 128 or labels.dtype != torch.long or labels.shape != (128,)
            or not bool(((labels >= 0) & (labels < 6)).all())
            or labeled_inputs.shape != (128, 2, 256)
            or optimizer_step != self._feature_l_steps or optimizer_step != self._full_calls + 1
            or not self._quality_pool_bound):
        raise ValueError('Only ordered actual L128 default-head steps may consume the pool')
    if not isinstance(extra, (tuple, list)) or len(extra) < 2 or not isinstance(extra[1], Mapping):
        raise ValueError('Original visible L physical metadata required')
    rx = torch.as_tensor(extra[1]['rx_i'], device=labels.device, dtype=torch.long)
    if (domains is None or domains.shape != labels.shape
            or not bool(((domains >= 0) & (domains < 15)).all())
            or not torch.equal(rx, torch.tensor(supcon.RX_BY_DOMAIN, device=domains.device)[domains])):
        raise ValueError('Original five-source-RX/day domains required')
    step = int(optimizer_step)
    epoch = (step - 1) // 222 + 1
    weight = supcon.schedule(epoch)
    pending = self._supcon_pending
    ordinary = self._legacy_actual_L_input
    self._legacy_actual_L_input = None
    if ordinary is None or ordinary.shape != (128, 2, 256):
        raise ValueError('Reuse the already generated ordinary IQ')
    first = labeled_output['tx_logits']
    if not torch.allclose(F.cross_entropy(first, labels, label_smoothing=.01).detach(),
                          identity_loss.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError('Native ordinary CE scale changed')
    if weight and (pending is None or tuple(pending['first']) != supcon.LEVELS):
        raise ValueError('Actual first ordinary maps absent')
    first_maps = pending['first'] if weight else {}
    supcon.clear(self)
    second, second_maps = base.capture_forward(self, ordinary, domains, weight > 0)
    common, ce, kl, _, _ = base.terms(first, second, labels, 1.)

    raw = labeled_inputs.detach()
    smask = (torch.arange(128, device=labels.device) + step).remainder(2) != 0
    satellite, sat_labels, sat_domains, observed, allocation = self._full_channel.consume(
        step, epoch, labels.device)
    if (satellite.shape != (64, 2, 256) or sat_labels.shape != (64,)
            or sat_domains.shape != (64,) or len(observed) != 64
            or not bool(torch.isfinite(satellite).all())
            or not bool(((sat_labels >= 0) & (sat_labels < 6)).all())
            or not bool(((sat_domains >= 0) & (sat_domains < 15)).all())):
        raise ValueError('One complete labeled/domain-aligned As64 pack required')
    view = ordinary.clone()
    view[smask] = satellite
    mixed_y, mixed_d = labels.clone(), domains.clone()
    mixed_y[smask], mixed_d[smask] = sat_labels, sat_domains
    sat_rx = torch.tensor(supcon.RX_BY_DOMAIN, device=domains.device)[sat_domains]
    for y, r, o in zip(sat_labels.tolist(), sat_rx.tolist(), observed):
        if int(o['sample_id'][1]) != int(r):
            raise ValueError('The original As RX/domain must move with its IQ and label')
        self._full_coverage[(int(y), int(r), o['scene'], o['state_end'], o['receiver_lock_status'])] += 1

    # Both stochastic forwards see exactly the same IQ, target and domain.
    z0, sat0 = base.capture_forward(self, view, mixed_d, weight > 0)
    z1, sat1 = base.capture_forward(self, view, mixed_d, weight > 0)
    original_extra, original_extra_ce, extra_kl, _, _ = base.terms(z0, z1, mixed_y, 1.)
    sat_ce, sat_kl, sat_stats = weighted_satellite_terms(
        z0[smask], z1[smask], sat_labels, allocation['groups'], allocation['p_g'], self._quality_method)
    _, ordinary64_ce, ordinary64_kl, _, _ = base.terms(
        z0[~smask], z1[~smask], labels[~smask], 1.)
    # Preserve the old mixed loss exactly for RANDOM, including unweighted KL.
    # Half of mixed128 are As, and the mixed pair has outer coefficient .5.
    if self._quality_method == 'POOL_RANDOM':
        correction_to_mixed = sat_ce * 0
    else:
        # Keep gradients of the subtracted ordinary estimator; detached
        # telemetry must never enter this objective algebra.
        plain_as_ce = .5 * (F.cross_entropy(z0[smask], sat_labels, label_smoothing=.01)
                             + F.cross_entropy(z1[smask], sat_labels, label_smoothing=.01))
        correction_to_mixed = .5 * (sat_ce - plain_as_ce)
    extra_loss = original_extra + correction_to_mixed
    extra_ce = original_extra_ce + correction_to_mixed

    details = []
    if weight:
        values = []
        for phase, a, b in [('first', first_maps, sat0), ('second', second_maps, sat1)]:
            value, records = relations.penalty(a, b, labels, rx, smask, self._crossview_mode)
            values.append(value)
            details.extend(dict(phase=phase, **r) for r in records)
        aux = torch.stack(values).mean()
        self._supcon_applied += 1
    else:
        aux = common * 0
    total_before_raw = .5 * common + .5 * extra_loss + weight * aux
    original_before_raw = .5 * common + .5 * original_extra + weight * aux
    self._full_calls += 1
    self._full_L_packets += 512
    self._supcon_calls = step
    self._legacy_calls += 1
    self._legacy_extra_packets += 128
    self._legacy_positive_KL_calls += int(float(kl.detach()) > 0)
    record = dict(step=step, epoch=epoch, total_L=float(total_before_raw.detach()),
        common_RD=float(common.detach()), common_CE=float(ce.detach()), common_KL=float(kl.detach()),
        extra_RD=float(extra_loss.detach()), extra_CE=float(extra_ce.detach()), extra_KL=float(extra_kl.detach()),
        SupCon=float(aux.detach()), SupCon_weight=weight, SupCon_details=details,
        L_forwards=4, Ag_packets=0, As_packets=64,
        ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),
        second_raw_MSE=float((view-raw).square().mean()),
        same_IQ_each_pair=True, ordinary_redraws=0,
        scene_counts=dict(Counter(o['scene'] for o in observed)),
        state_counts=dict(Counter(o['state_end'] for o in observed)),
        all_satellite_L_supervised=True, crossview_mode=self._crossview_mode,
        new_losses=0, extra_U_NN=0)
    self._full_records.append(record)

    # The existing final raw pair remains byte-for-byte the P43 route.
    cumulative_step = step + self._shape_offset
    raw_view, noise_meta = noise.apply(raw, cumulative_step, self._shape_mode)
    r0 = self.forward_identity_only(raw_view, domain_labels=domains)['tx_logits']
    r1 = self.forward_identity_only(raw_view, domain_labels=domains)['tx_logits']
    raw_rd, raw_ce, raw_kl, _, _ = base.terms(r0, r1, labels, 1.)
    self._shape_NN += 256
    self._shape_packets += noise_meta['applied']
    self._shape_records.append(dict(step=step, cumulative_step=cumulative_step,
        shape_RD=float(raw_rd.detach()), shape_CE=float(raw_ce.detach()),
        shape_KL=float(raw_kl.detach()), weight=.25, NN=256, same_IQ_each_pair=True, **noise_meta))
    total = total_before_raw + .25 * raw_rd
    original_total = original_before_raw + .25 * raw_rd
    record['base_L_before_shape'] = record['total_L']
    record.update(total_L=float(total.detach()), L_forwards=6, extra_control_packets=128,
                  Ag_mode=self._shape_mode, same_IQ_each_pair=True)
    quality_record = dict(step=step, cumulative_step=cumulative_step, epoch=epoch,
        cumulative_epoch=20+epoch, method=self._quality_method,
        counts=list(allocation['counts']), groups=list(allocation['groups']),
        used_pool_indices=list(allocation['used_pool_indices']), p_g=list(allocation['p_g']),
        origin_steps=list(allocation['origin_steps']), origin_slots=list(allocation['origin_slots']),
        As_CE_unweighted=sat_stats['CE_unweighted'], As_CE_weighted=float(sat_ce.detach()),
        group_CE=list(sat_stats['group_CE']),
        As_KL_unweighted=float(sat_kl.detach()), ordinary64_CE=float(ordinary64_ce.detach()),
        ordinary64_KL=float(ordinary64_kl.detach()), original_total_L=float(original_total.detach()),
        used_total_L=float(total.detach()), CE_correction_to_total=float((.5*correction_to_mixed).detach()),
        extra_NN=0, all_labeled_CE_retained=True, same_IQ_each_RD_pair=True,
        As_KL_quality_weighted=False, U_truth_read=False,
        ordinary_raw_U_unchanged=True, labels_domains_move_with_As=True,
        original_total_is_same_logits_counterfactual=True)
    self._quality_records.append(quality_record)
    metrics = {'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
    metrics.update({'shape/RD':float(raw_rd.detach()),
                    'quality/As_CE':float(sat_ce.detach()), 'quality/CE_correction':quality_record['CE_correction_to_total']})
    return total - identity_loss, metrics


_labeled_hook._accepts_labeled_inputs = True


def classifier(*args, **kwargs):
    from experiments.cvs_noise_shape_factorial43 import model as original
    return original.classifier(*args, **kwargs)


def source_domain_count(*args, **kwargs):
    from experiments.cvs_noise_shape_factorial43 import model as original
    return original.source_domain_count(*args, **kwargs)


def _attach(m, c):
    from . import design as d
    d.validate(c)
    if c['Ag_mode'] != 'NONE' or not c['As'] or m._crossview_mode != 'ORD128':
        raise ValueError('Only fixed P43 As/NONE/ORD128 six-forward training')
    m._quality_method = c['method']
    m._quality_records = []
    m._quality_pool_bound = False
    m._source_labeled_feature_hook = MethodType(_labeled_hook, m)
    m._feature_learning_report['quality_risk70'] = dict(c['quality_pool'], method=c['method'])
    return m


def install(m, c):
    from . import design as d
    from experiments.cvs_noise_shape_factorial43 import model as original
    d.validate(c)
    return _attach(original.install(m, d.old_config()), c)


def build(c, device='cpu'):
    from . import design as d
    from experiments.cvs_noise_shape_factorial43 import model as original
    d.validate(c)
    return _attach(original.build(d.old_config(), device), c)


def bind_pool(m, c, directory):
    from . import design as d, pool
    d.validate(c)
    if m._quality_pool_bound or m._full_calls or m._quality_records:
        raise ValueError('Bind the fixed shared pool once before any update')
    m._full_channel = pool.load(directory, c['method'], seed=c['quality_pool']['allocation_seed'])
    m._quality_pool_bound = True
    return m


def own_model(c, device='cpu'):
    from . import design as d
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(c)
    parent = parent_payload(c)
    out = Path(c['output_root'])
    final_metadata(c, d.read(out/'completion.json'), d.read(out/'source_frozen.json'),
                   d.read(out/'initialization.json'), d.read(out/'source_contract.json'),
                   d.read(d.SOURCE)['role_ids'])
    cp = torch.load(out/'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(c, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Final default-head student/ownEMA state keys differ')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
