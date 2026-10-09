"""Source-L-only receiver-condition action audit, never a pure RX hardware claim.

Group observations are distributional, not same-transmission counterfactuals.
Fit and audit physical packets are disjoint and the reference E/G stays frozen.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_multi_disentangle.model import (
    FixedViews, BoundedConditionalOperator, intermediate,
    identity_from_intermediate, classify_intermediate,
)


def packet_time_statistics(x):
    """12 global-phase-invariant numbers per packet; aggregate *after* this."""
    z = torch.complex(x[:, 0], x[:, 1])
    values = []
    for lag in (1, 20):
        a, b = z[:, lag:], z[:, :-lag]
        products = a * b.conj()
        cross = products.mean(1)
        scale = (a.abs().square().mean(1) * b.abs().square().mean(1)).sqrt().clamp_min(1e-8)
        unit = products / products.abs().clamp_min(1e-8)
        circular = unit.mean(1)
        values.extend((cross.real / scale, cross.imag / scale, cross.abs() / scale,
                       circular.real, circular.imag, circular.abs()))
    return torch.stack(values, 1)


class ContributionStatistics:
    """Timestamped sufficient-statistic buckets, without refresh-based rejuvenation.

    Every contribution retains its own step and reference version. Aggregation
    requires a specific version; features from different E/G coordinates never
    mix. Only sums/counts are kept, not IQ or individual packet features.
    """
    def __init__(self, max_age_steps=888, min_count=2):
        self.max_age_steps = int(max_age_steps)
        self.min_count = int(min_count)
        self.buckets = defaultdict(list)
        self.expired_samples = 0
        self.usage = Counter()
        self.seen_tx = set()
        self.seen_tx_pairs = set()
        self.seen_rx_pairs = set()
        self.seen_conditions = set()
        self.seen_relations = set()

    @staticmethod
    def keys(y, rx, views):
        quality = torch.bucketize(views[:, 24].contiguous(), views.new_tensor([.25, .75]))
        response = (views[:, :24].square().mean(1).sqrt() >= 1.).long()
        cfo = (views[:, 27].abs() >= .1).long()
        return [tuple(map(int, row)) for row in torch.stack((y, rx, quality, response, cfo), 1).cpu().tolist()]

    def expire(self, step):
        for key in list(self.buckets):
            kept = []
            for bucket in self.buckets[key]:
                if step < bucket['step']:
                    raise ValueError('Statistics clock must not move backwards')
                if step - bucket['step'] <= self.max_age_steps:
                    kept.append(bucket)
                else:
                    self.expired_samples += bucket['count']
            if kept:
                self.buckets[key] = kept
            else:
                del self.buckets[key]

    def update(self, y, rx, views, h, step, reference_version, z=None, margin=None):
        if bool((y < 0).any()) or bool((rx < 0).any()):
            raise ValueError('Only visible source-L TX/RX labels are permitted')
        self.expire(step)
        grouped = defaultdict(list)
        for index, key in enumerate(self.keys(y, rx, views)):
            grouped[key].append(index)
        for key, indices in grouped.items():
            bucket=dict(step=int(step), version=str(reference_version),
                count=len(indices), view_sum=views[indices].detach().sum(0).cpu(),
                h_sum=h[indices].detach().sum(0).cpu())
            if z is not None:
                bucket['z_sum']=z[indices].detach().sum(0).cpu()
            if margin is not None:
                bucket['margin_sum']=margin[indices].detach().sum().cpu()
            self.buckets[key].append(bucket)

    def groups(self, step, reference_version):
        self.expire(step)
        groups = {}
        for key, buckets in self.buckets.items():
            active = [b for b in buckets if b['version'] == str(reference_version)]
            count = sum(b['count'] for b in active)
            if count >= self.min_count:
                groups[key] = dict(count=count,
                    view=sum(b['view_sum'] for b in active) / count,
                    h=sum(b['h_sum'] for b in active) / count)
                if all('z_sum' in b for b in active):
                    groups[key]['z']=sum(b['z_sum'] for b in active)/count
                if all('margin_sum' in b for b in active):
                    groups[key]['margin']=sum(b['margin_sum'] for b in active)/count
        return groups

    def age_report(self, step):
        self.expire(step)
        ages, versions = Counter(), Counter()
        for buckets in self.buckets.values():
            for b in buckets:
                age = step - b['step']
                ages[str(age)] += b['count']
                versions[b['version']] += b['count']
        return dict(contribution_age_steps=dict(ages), reference_versions=dict(versions),
                    expired_samples=self.expired_samples, refresh_extends_old_age=False,
                    max_allowed_age_steps=self.max_age_steps)

    @staticmethod
    def all_pairs(groups):
        indexed = defaultdict(list)
        for key in groups:
            indexed[(key[0], *key[2:])].append(key)
        pairs = []
        for keys in indexed.values():
            for source in keys:
                for destination in keys:
                    if source[1] != destination[1]:
                        pair=dict(source=source, destination=destination,
                            v0=groups[source]['view'], v1=groups[destination]['view'],
                            h0=groups[source]['h'], h1=groups[destination]['h'])
                        for field in ('z','margin'):
                            if field in groups[source] and field in groups[destination]:
                                pair[field+'0']=groups[source][field]
                                pair[field+'1']=groups[destination][field]
                        pairs.append(pair)
        return pairs

    def sample_pairs(self, groups, generator, limit=32):
        candidates = self.all_pairs(groups)
        # Randomize ties before balancing lifetime relation use: numeric TX
        # order can never systematically select the first eight classes.
        order = torch.randperm(len(candidates), generator=generator).tolist()
        shuffled = [candidates[i] for i in order]
        shuffled.sort(key=lambda p: self.usage[(p['source'], p['destination'])])
        selected = shuffled[:limit]
        relation_tx = defaultdict(set)
        for pair in selected:
            a, b = pair['source'], pair['destination']
            self.usage[(a, b)] += 1
            self.seen_tx.add(a[0])
            self.seen_rx_pairs.add((a[1], b[1]))
            self.seen_conditions.add(a[2:])
            self.seen_relations.add((a, b))
            relation_tx[(a[1], b[1], *a[2:])].add(a[0])
        for txs in relation_tx.values():
            self.seen_tx_pairs.update(tuple(sorted(p)) for p in combinations(txs, 2))
        return selected

    def coverage(self):
        return dict(unique_tx=sorted(self.seen_tx), unique_tx_pairs=sorted(self.seen_tx_pairs),
                    unique_rx_pairs=sorted(self.seen_rx_pairs),
                    unique_condition_strata=sorted(self.seen_conditions),
                    unique_group_relations=len(self.seen_relations),
                    uses_by_group_relation={str(k): v for k, v in self.usage.items()})


class ReceiverAction(nn.Module):
    def __init__(self, width=49):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 8))
        self.operator = BoundedConditionalOperator(349, rank=8, state_dim=8)

    def forward(self, v0, v1, h):
        return self.operator(h, self.encoder(v1) - self.encoder(v0))


def anchor_loss(pred_z, target_z, anchors):
    """Frozen destination-source-RX class directions, Eq. V.3 in the report."""
    a = F.normalize(anchors.detach(), dim=1)
    target = F.normalize(target_z.detach(), dim=1) @ a.T
    return (F.normalize(pred_z, dim=1) @ a.T - target).square().mean()


def relation_loss(source_z, moved_z):
    a, b = F.normalize(source_z.detach(), dim=1), F.normalize(moved_z, dim=1)
    mask = ~torch.eye(len(a), device=a.device, dtype=torch.bool)
    return ((a @ a.T - b @ b.T)[mask]).square().mean()


def _margin(logits, y):
    other = logits.clone().scatter(1, y[:, None], float('-inf'))
    return logits.gather(1, y[:, None])[:, 0] - other.max(1).values


@contextmanager
def _fixed(identity):
    training = [(m, m.training) for m in identity.modules()]
    grads = [(p, p.requires_grad) for p in identity.parameters()]
    identity.eval()
    for p, _ in grads:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in grads:
            p.requires_grad_(enabled)
        for m, enabled in training:
            m.training = enabled


def _stack(pairs, device):
    return [torch.stack([p[k] for p in pairs]).to(device) for k in ('v0', 'v1', 'h0', 'h1')]


def _error_report(identity, h0, h1, predicted, y, pairs):
    with torch.no_grad():
        target = h1-h0
        energy = float(target.square().sum())
        error = float((predicted-target).square().sum())
        z_actual = identity_from_intermediate(identity, h1)
        z_pred = identity_from_intermediate(identity, h0+predicted)
        z0 = identity_from_intermediate(identity, h0)
        margin0 = _margin(classify_intermediate(identity, h0), y)
        margin1 = _margin(classify_intermediate(identity, h1), y)
        ma = margin1-margin0
        mp = _margin(classify_intermediate(identity, h0+predicted), y)-margin0
        mean_z0,mean_z1,mean_m0,mean_m1=[torch.stack([p[k] for p in pairs]).to(h0.device)
            for k in ('z0','z1','margin0','margin1')]
        group_delta_z=mean_z1-mean_z0
        group_delta_margin=mean_m1-mean_m0
        return dict(sse=error, target_energy=energy, target_rms=float(target.square().mean().sqrt()),
            skill_over_zero=None if energy <= 1e-12 else 1-error/energy,
            low_target_energy=energy <= 1e-12,
            centroid_G_mse=float((z_pred-z_actual).square().mean()),
            centroid_G_actual_change_mse=float((z_actual-z0).square().mean()),
            centroid_margin_delta_mae=float((ma-mp).abs().mean()),
            centroid_margin_actual_delta_abs_mean=float(ma.abs().mean()),
            group_mean_G_actual_change_mse=float(group_delta_z.square().mean()),
            group_mean_packet_margin_actual_delta_abs_mean=float(group_delta_margin.abs().mean()),
            predicted_centroid_vs_destination_mean_G_mse=float((z_pred-mean_z1).square().mean()),
            predicted_centroid_delta_vs_group_mean_G_delta_mse=float((z_pred-z0-group_delta_z).square().mean()),
            predicted_centroid_margin_delta_vs_mean_packet_margin_delta_mae=float((mp-group_delta_margin).abs().mean()),
            source_centroid_vs_mean_G_gap_mse=float((z0-mean_z0).square().mean()),
            destination_centroid_vs_mean_G_gap_mse=float((z_actual-mean_z1).square().mean()),
            centroid_delta_vs_group_mean_G_delta_gap_mse=float((z_actual-z0-group_delta_z).square().mean()),
            source_centroid_vs_mean_packet_margin_gap_mae=float((margin0-mean_m0).abs().mean()),
            destination_centroid_vs_mean_packet_margin_gap_mae=float((margin1-mean_m1).abs().mean()),
            centroid_margin_delta_vs_mean_packet_margin_delta_gap_mae=float((ma-group_delta_margin).abs().mean()),
            group_comparison_scope='Predictions act on mean h; group errors include action error and nonlinear centroid approximation; no packet counterfactual',
            prediction_norm_mean=float(predicted.norm(dim=1).mean()),
            target_norm_mean=float(target.norm(dim=1).mean()))


def _direction_diagnostics(identity, pairs, groups, device, deltas):
    by_relation = defaultdict(list)
    for index,p in enumerate(pairs):
        a,b = p['source'],p['destination']
        by_relation[(a[1], b[1], *a[2:])].append((index,p))
    records = []
    for key, selected in by_relation.items():
        if len(selected) < 2:
            continue
        indices,selected=zip(*selected)
        _, _, h0, h1 = _stack(selected, device)
        # Genuine mean final-feature class anchors, not G(mean h): G contains
        # nonlinear LayerNorm, so the two operations must not be conflated.
        target_z=torch.stack([groups[p['destination']]['z'] for p in selected]).to(device).detach()
        anchor_keys=[k for k in groups if k[1]==key[1] and k[2:]==key[2:]]
        anchors=torch.stack([groups[k]['z'] for k in anchor_keys]).to(device).detach()
        with torch.no_grad():
            source_z = identity_from_intermediate(identity, h0).detach()
        # Both objectives see the same predicted action. Only gradients with
        # respect to its input are inspected; identity is not optimized.
        live = (h0+deltas[list(indices)].detach()).requires_grad_(True)
        moved = identity_from_intermediate(identity, live)
        old = relation_loss(source_z, moved)
        new = anchor_loss(moved, target_z, anchors)
        go = torch.autograd.grad(old, live, retain_graph=True)[0]
        gn = torch.autograd.grad(new, live)[0]
        records.append(dict(source_rx=key[0], destination_source_rx=key[1], condition=list(key[2:]),
            classes=[p['source'][0] for p in selected], original_relation_loss=float(old.detach()),
            destination_anchor_classes=[k[0] for k in anchor_keys],
            anchor_loss=float(new.detach()), original_input_gradient_norm=float(go.norm()),
            anchor_input_gradient_norm=float(gn.norm()), anchors_frozen=not target_z.requires_grad,
            input_gradient_cosine=float(F.cosine_similarity(go.flatten()[None],gn.flatten()[None]))))
    return records


def _repeatability(views, keys):
    grouped = defaultdict(list)
    for i, key in enumerate(keys):
        grouped[key].append(i)
    values = []
    for key, indices in grouped.items():
        if len(indices) < 4:
            continue
        a, b = views[indices[::2], -12:].mean(0), views[indices[1::2], -12:].mean(0)
        values.append(dict(key=list(key), packets=len(indices), half_group_mse=float((a-b).square().mean())))
    return dict(groups=len(values), rows=values,
                mean_half_group_mse=sum(v['half_group_mse'] for v in values)/len(values) if values else None,
                reason_if_missing=None if values else 'Fewer than four packets per matched group')


def receiver_audit(identity, fit_x, fit_y, fit_rx, audit_x, audit_y, audit_rx,
                   generator, steps=200, *, fit_ids, audit_ids, lr=2e-4):
    """Fit source-only action; audit disjoint physical packets once after freeze.

    No epoch/model selection or training feedback uses audit values. Returned
    state_dict belongs to this auxiliary action only; identity never changes.
    """
    fit_ids, audit_ids = list(fit_ids), list(audit_ids)
    if len(fit_ids)!=len(fit_x) or len(audit_ids)!=len(audit_x):
        raise ValueError('Physical ID count does not match IQ')
    if len(set(fit_ids))!=len(fit_ids) or len(set(audit_ids))!=len(audit_ids):
        raise ValueError('Physical IDs must be unique within each source-L partition')
    if set(fit_ids) & set(audit_ids):
        raise ValueError('Auxiliary fit/audit physical packet overlap')
    device = fit_x.device
    reference_version = 'frozen_source_reference_v1'
    pools = [ContributionStatistics(), ContributionStatistics()]
    views_module = FixedViews().to(device).eval()
    raw = []
    logs = []
    with _fixed(identity):
        with torch.no_grad():
            for pool, x,y,rx in zip(pools,(fit_x,audit_x),(fit_y,audit_y),(fit_rx,audit_rx)):
                hs, vs = [], []
                for start in range(0,len(x),64):
                    batch=x[start:start+64]
                    hs.append(intermediate(identity,batch))
                    vs.append(torch.cat((views_module.receiver(batch),packet_time_statistics(batch)),1))
                h, v = torch.cat(hs),torch.cat(vs)
                z=identity_from_intermediate(identity,h)
                packet_margin=_margin(classify_intermediate(identity,h),y)
                pool.update(y,rx,v,h,step=0,reference_version=reference_version,z=z,margin=packet_margin)
                raw.append((v, pool.keys(y,rx,v)))
        fit_groups,audit_groups=[p.groups(0,reference_version) for p in pools]
        fit_pairs,audit_pairs=[p.all_pairs(g) for p,g in zip(pools,(fit_groups,audit_groups))]
        report=dict(scope='source_L_only', target_access=False, identity_updated=False,
            receiver_semantics='condition-matched total source RX differences, not pure hardware',
            subtract_synthetic_LT=False, fit_physical_packets=len(fit_ids),
            audit_physical_packets=len(audit_ids), fit_audit_physical_disjoint=True,
            fit_groups=len(fit_groups), audit_groups=len(audit_groups),
            fit_group_relations=len(fit_pairs), audit_group_relations=len(audit_pairs),
            reference_version=reference_version, audit_selection_or_feedback=False,
            temporal_repeatability_fit=_repeatability(*raw[0]),
            temporal_repeatability_audit=_repeatability(*raw[1]),
            contribution_statistics=[p.age_report(0) for p in pools])
        if not fit_pairs or not audit_pairs:
            report.update(status='INSUFFICIENT_MATCHED_GROUPS', predictive_metrics=None,
                reason='At least two packets per TX/RX/condition group and a matched other RX required')
            return dict(report=report,state_dict={},step_logs=[])
        devices = [device.index] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generator.initial_seed()+8301)
            branch=ReceiverAction().to(device)
        optimizer=torch.optim.Adam(branch.parameters(),lr=lr)
        _,_,fh0,fh1=_stack(fit_pairs,device)
        fit_target=fh1-fh0
        block_scales=[b.square().mean().clamp_min(1e-8).detach() for b in fit_target.split((160,160,29),1)]
        # Input normalization derives only from fit groups.
        fv=torch.stack([g['view'] for g in fit_groups.values()]).to(device)
        center=fv.mean(0);scale=fv.std(0,unbiased=False).clamp_min(.01)
        for step in range(int(steps)):
            pairs=pools[0].sample_pairs(fit_groups,generator,32)
            v0,v1,h0,h1=_stack(pairs,device)
            optimizer.zero_grad(set_to_none=True)
            pred=branch((v0-center)/scale,(v1-center)/scale,h0)
            errors=(pred-(h1-h0)).split((160,160,29),1)
            block=torch.stack([e.square().mean()/s for e,s in zip(errors,block_scales)]).mean()
            zp=identity_from_intermediate(identity,h0+pred)
            with torch.no_grad():zt=identity_from_intermediate(identity,h1)
            g_loss=(zp-zt).square().mean()
            loss=block+.1*g_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite R action fit')
            loss.backward()
            grad=torch.stack([p.grad.norm() for p in branch.parameters() if p.grad is not None]).norm()
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite R action gradient')
            optimizer.step()
            logs.append(dict(step=step+1,loss=float(loss.detach()),block_loss=float(block.detach()),
                             G_loss=float(g_loss.detach()),gradient_norm=float(grad),lr=lr))
        branch.eval()
        v0,v1,h0,h1=_stack(audit_pairs,device)
        labels=torch.tensor([p['source'][0] for p in audit_pairs],device=device)
        with torch.no_grad():pred=branch((v0-center)/scale,(v1-center)/scale,h0)
        mean=fit_target.mean(0).expand_as(pred)
        report.update(status='VERIFIED',steps=int(steps),
            predictive_metrics={name:_error_report(identity,h0,h1,value,labels,audit_pairs)
                for name,value in [('learned',pred),('zero',torch.zeros_like(pred)),('fit_mean',mean)]},
            relation_vs_anchor=_direction_diagnostics(identity,audit_pairs,audit_groups,device,pred),
            sampling_coverage=pools[0].coverage(),
            auxiliary_parameters=sum(p.numel() for p in branch.parameters()),
            inference_auxiliary_parameters=0,
            limitations=['Audit groups estimate distributions, not paired transmissions',
                         'New anchor loss is diagnosed; no identity optimization or performance claim'])
        state=dict(branch={k:v.detach().cpu() for k,v in branch.state_dict().items()},
                   view_center=center.cpu(),view_scale=scale.cpu(),reference_version=reference_version)
    return dict(report=report,state_dict=state,step_logs=logs)
