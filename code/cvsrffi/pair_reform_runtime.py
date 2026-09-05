"""Execution primitives for the opt-in, source-only paired reform."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import numbers
import torch
import torch.nn.functional as F
from . import pair_reform as objectives
from .deployment_orbit import physical_reliability, apply_physical_probe, sample_pair_reform_probe


def sample_seed(run_seed, physical_id, epoch, family):
    payload = f'{int(run_seed)}|{physical_id}|{int(epoch)}|{family}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') % (2**63-1)


@contextmanager
def deterministic_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def deterministic_pair(model, base, perturbed, *, domain_labels):
    with deterministic_mode(model):
        kwargs = dict(y_tx=None, grl_lambda=1., return_aux=True, domain_labels=domain_labels)
        return model(base, **kwargs), model(perturbed, **kwargs)


class StepStateTransaction:
    """Candidate auxiliary writes are discarded on skipped optimizer steps."""
    def __init__(self):
        self.pending = []
    def stage(self, callback):
        self.pending.append(callback)
    def finish(self, *, applied):
        pending, self.pending = self.pending, []
        if applied:
            for callback in pending:
                callback()


def direction_objectives(model, clean, channel, *, domain_labels,
        tangent_weight, route_weight, ratio, seed, delta, reference_scale,
        budget, nuisance_transform, probe_transform):
    result = {'weighted_components': {}, 'diagnostics': {}, 'forward_samples': 0}
    if tangent_weight <= 0 and route_weight <= 0:
        return result
    if not 0 <= ratio <= 1 or delta <= 0 or reference_scale <= 0:
        raise ValueError('invalid paired direction budget')
    count = int(math.floor(len(clean) * ratio))
    if count == 0:
        return result
    generator = torch.Generator(device=clean.device).manual_seed(int(seed))
    indices = torch.randperm(len(clean), device=clean.device, generator=generator)[:count]
    c, x = clean[indices], channel[indices]
    domains = domain_labels[indices] if domain_labels is not None else None
    base_n, changed_n = deterministic_pair(model, x, nuisance_transform(x), domain_labels=domains)
    result['forward_samples'] += 2 * count
    def distance(a, b):
        return (F.normalize(a.float(), dim=-1)-F.normalize(b.float(), dim=-1)).square().sum(-1) * .5
    di = distance(base_n['z_id'], changed_n['z_id'])
    dd = distance(base_n['z_dom'], changed_n['z_dom'])
    if tangent_weight > 0:
        energy = 2 * di / ((delta/reference_scale)**2 + 1e-12)
        result['weighted_components']['tangent'] = float(tangent_weight) * (energy-float(budget)).relu().mean()
    if route_weight > 0:
        base_f, changed_f = deterministic_pair(model, c, probe_transform(c), domain_labels=domains)
        result['forward_samples'] += 2 * count
        fi = distance(base_f['z_id'], changed_f['z_id'])
        fd = distance(base_f['z_dom'], changed_f['z_dom'])
        result['weighted_components']['route'] = float(route_weight) * ((.05+di-dd).relu()+(.05+fd-fi).relu()).mean()
        result['diagnostics'].update(delta_nui_id=di.detach(),delta_nui_dom=dd.detach(),delta_fp_id=fi.detach(),delta_fp_dom=fd.detach())
    result['diagnostics'].update(formula='normalized_chordal_linear_v3', sampled_count=count)
    return result


def _teacher_outputs(teacher, clean, channel, domains):
    batch = torch.cat([clean, channel])
    d = torch.cat([domains, domains]) if domains is not None else None
    with deterministic_mode(teacher), torch.no_grad():
        fn = getattr(teacher, 'forward_identity_only', None)
        out = fn(batch, domain_labels=d) if callable(fn) else teacher(
            batch, y_tx=None, grl_lambda=0., return_aux=True, domain_labels=d)
    n = len(clean)
    return ({k:v[:n] for k,v in out.items() if torch.is_tensor(v) and v.ndim and len(v)==2*n},
            {k:v[n:] for k,v in out.items() if torch.is_tensor(v) and v.ndim and len(v)==2*n})


def _fixed_head_weights(teacher, teacher_clean):
    backbone = getattr(teacher, 'id_backbone', None)
    head = getattr(getattr(backbone, 'cls_head', None), 'head', None)
    if head is None or type(head).__name__ != 'CosFaceHead':
        raise ValueError('safe region requires the verified bias-free CosFaceHead')
    w = head.weight.detach().float()
    expected = F.normalize(teacher_clean['z_id'].float(),dim=-1) @ F.normalize(w,dim=-1).t() * float(head.s)
    if not torch.allclose(expected,teacher_clean['tx_logits'].float(),atol=.03,rtol=.002):
        raise ValueError('safe region z_id and actual teacher classifier feature spaces differ')
    return w


@torch.no_grad()
def fixed_head_pair_diagnostics(student_clean, student_leo, teacher_clean, weights, labels, *, alpha):
    """L-only cosine margins under one fixed teacher W (without its logit scale)."""
    w = F.normalize(weights.detach().float(), dim=-1)
    clean = F.normalize(student_clean.detach().float(), dim=-1)
    leo = F.normalize(student_leo.detach().float(), dim=-1)
    anchor = F.normalize(teacher_clean.detach().float(), dim=-1)
    labels = labels.detach().long()
    clean_scores, leo_scores = clean @ w.T, leo @ w.T
    def margin(scores):
        target = scores.gather(1, labels[:, None]).squeeze(1)
        other = scores.clone().scatter_(1, labels[:, None], float('-inf')).amax(-1)
        return target - other
    radius, valid = objectives.cosine_safety_radius(teacher_clean, weights, labels, alpha)
    distance = (leo-anchor).norm(dim=-1)
    return {
        'fixed_head_clean_margin': margin(clean_scores),
        'fixed_head_leo_margin': margin(leo_scores),
        'safe_anchor_valid': valid.float(),
        # Conditional on valid anchors; never count zero-radius invalid anchors as safe.
        'safe_radius_inside': (distance[valid] <= radius[valid]).float(),
        'safe_radius': radius[valid],
        'fixed_head_classification_flip': (clean_scores.argmax(-1) != leo_scores.argmax(-1)).float(),
    }


@torch.no_grad()
def grouped_pair_weight_diagnostics(physical_ids, r_phys, feature_weight, pseudo_weight):
    """RX x r_phys quartile-bin totals; bins represent weight, not measured severity.

    Bins are [0,.25), [.25,.5), [.5,.75), [.75,1]. Opaque identities have
    unknown RX and are counted separately. No TX field or class label is read.
    """
    if len(physical_ids) != len(r_phys):
        raise ValueError('physical ID batch does not match pair diagnostics')
    receivers = [int(key[0]) if isinstance(key, (tuple, list)) and len(key) == 5
                 and isinstance(key[0], numbers.Integral) and int(key[0]) >= 0 else -1
                 for key in physical_ids]
    rx = torch.tensor(receivers, device=r_phys.device, dtype=torch.long)
    bands = (r_phys.detach().float().clamp(0, 1) * 4).long().clamp_max(3)
    groups = {}
    for receiver in sorted(set(receivers) - {-1}):
        for band in range(4):
            mask = (rx == receiver) & (bands == band)
            groups[f'rx_{receiver}/r_phys_bin_{band}'] = {
                'samples_count': mask.sum(),
                'feature_weight_sum': feature_weight.detach()[mask].sum(),
                'pseudo_weight_sum': pseudo_weight.detach()[mask].sum(),
            }
    return {'rx_unknown_count': (rx < 0).sum(), 'weight_groups': groups}


def compute_pair_batch(*,model,teacher,clean,channel,student_clean,student_channel,
        domains,labels,metadata,args,epoch,physical_ids,memory=None,transaction=None):
    """Reuse one student pair. Cached features never enter teacher classification votes."""
    zero = student_channel['z_id'].new_zeros(())
    weighted, diag = {}, {'formula':'pair_reform_v3','student_extra_views':0,'teacher_views':0}
    active = epoch >= int(args.pair_start_epoch) and float(args.pair_weight)>0
    cls_active = labels is None and epoch>=int(args.pair_pseudo_start_epoch) and float(args.pair_pseudo_weight)>0
    if not active and not cls_active:
        return {'loss':zero,'components':{},'weighted_components':{},'diagnostics':diag}
    tc, tl = _teacher_outputs(teacher, clean, channel, domains)
    r, known = physical_reliability(metadata or {},batch_size=len(clean),device=clean.device)
    r, known = objectives.physical_reliability(r,known,unknown_policy=args.pair_unknown_quality)
    p0=F.softmax(tc['tx_logits'].float(),dim=-1)
    p1=F.softmax(tl['tx_logits'].float(),dim=-1)
    q=objectives.classification_confidence(p0,p1)
    target=objectives.asymmetric_teacher_target(tc['z_id'],tl['z_id'],leo_mix=float(args.pair_teacher_mix)*r)
    if memory is not None and active:
        from .deployment_orbit import stable_orbit_key_tensor
        keys=stable_orbit_key_tensor(physical_ids,device=clean.device)
        history,found,meta=memory.lookup(keys,step=int(epoch))
        # Historical target availability is independent of current physical quality.
        history_agreement = (.5 + .5 * (F.normalize(target.detach().float(), dim=-1)
            * F.normalize(history.detach().to(clean.device).float(), dim=-1)).sum(-1)).clamp(0., 1.)
        cache_quality=meta['reliability'].to(clean.device) * found.to(clean.device) * history_agreement
        if str(args.pair_reform) != 'safe':
            target=objectives.asymmetric_teacher_target(target,history.to(clean.device),leo_mix=.25*cache_quality)
        if transaction is None:
            raise ValueError('memory writes require optimizer transaction')
        payload=dict(keys=keys.detach().clone(),features=tl['z_id'].detach().clone(),
            reliability=r.detach().clone(),scenario_bin=torch.zeros_like(keys),
            receiver_bin=domains.detach().clone() if domains is not None else torch.full_like(keys,-1),step=int(epoch))
        transaction.stage(lambda: memory.update(**payload))
        diag['cache_weight_sum']=cache_quality.sum().detach()
        diag['q_cache']=cache_quality.detach()
    if active:
        if str(args.pair_reform)=='safe' and labels is not None:
            w=_fixed_head_weights(teacher,tc)
            pair=objectives.safe_pair_loss(student_channel['z_id'],tc['z_id'],w,labels,r,alpha=args.pair_alpha)
            diag.update(fixed_head_pair_diagnostics(student_clean['z_id'], student_channel['z_id'],
                tc['z_id'], w, labels, alpha=args.pair_alpha))
        elif str(args.pair_reform)=='asymmetric':
            pair=objectives.point_pair_loss(student_channel['z_id'],target,r)
        elif labels is None and str(args.pair_reform)=='safe':
            pair=objectives.point_pair_loss(student_channel['z_id'],tc['z_id'],r,tolerance=args.pair_u_tolerance)
        else:
            pair=.5*(objectives.point_pair_loss(student_clean['z_id'],target,r)+
                     objectives.point_pair_loss(student_channel['z_id'],target,r))
        weighted['orbit_z']=float(args.pair_weight)*pair
    if cls_active:
        # Two fresh observations, never a cached feature/duplicated logits vote.
        target_prob=(p0+r[:,None]*p1)/(1+r[:,None])
        weighted['orbit_logit']=float(args.pair_pseudo_weight)*objectives.unified_soft_ce(
            student_channel['tx_logits'],target_prob,r,q)
    tw=float(getattr(args,'pair_tangent_weight',0)) if active and labels is not None else 0.
    rw=float(getattr(args,'pair_route_weight',0)) if active and labels is not None else 0.
    if tw>0 or rw>0:
        seed=sample_seed(getattr(args,'seed',0),'|'.join(map(str,physical_ids)),epoch,'directions')
        delta=float(args.pair_direction_delta)
        directions=direction_objectives(model,clean,channel,domain_labels=domains,
            tangent_weight=tw,route_weight=rw,ratio=args.pair_direction_ratio,seed=seed,
            delta=delta,reference_scale=args.pair_direction_scale,budget=args.pair_direction_budget,
            nuisance_transform=lambda x: apply_physical_probe(x,name='sto',amount=delta,sample_rate_hz=args.sat_fs_hz),
            probe_transform=lambda x: sample_pair_reform_probe(x,seed=seed+1,strength=delta,sample_rate_hz=args.sat_fs_hz)[0])
        weighted.update(directions['weighted_components'])
        diag.update(directions['diagnostics'])
        diag['student_extra_views']=directions['forward_samples']
    feature_weight = r.detach() if active else torch.zeros_like(r)
    if 'safe_anchor_valid' in diag:
        feature_weight = feature_weight * diag['safe_anchor_valid']
    pseudo_weight = (r*q).detach() if cls_active else torch.zeros_like(r)
    diag.update(grouped_pair_weight_diagnostics(physical_ids, r, feature_weight, pseudo_weight))
    diag.update(teacher_views=2*len(clean),physical_quality_unknown_count=(~known).sum().detach(),
        feature_weight_sum=feature_weight.sum(),pseudo_weight_sum=pseudo_weight.sum(),
        pair_shift=(F.normalize(student_channel['z_id'].detach().float(),dim=-1)-F.normalize(tc['z_id'].float(),dim=-1)).square().sum(-1).mean(),
        q_cls=q.detach(),r_phys=r.detach(),pseudo=p0.argmax(-1))
    return {'loss':sum(weighted.values(),zero),'components':weighted,'weighted_components':weighted,'diagnostics':diag}
