"""Independent source TX readout and fit-only cross-RX reference centers."""
from dataclasses import dataclass
import time

import torch
from torch import nn
from torch.nn import functional as F

from .source_audit import (AuditConfig, _coverage_reason, classification_metrics,
                           fit_probe, isolated_rng, validate_groups)


@dataclass(frozen=True)
class CapabilityConfig:
    steps: int = 40
    lr: float = 0.02
    weight_decay: float = 0.0
    min_samples_per_class: int = 2
    minimum_margin_coverage: float = 1.0
    margin_quantile: float = 0.1
    collapse_variance: float = 1e-8
    seed: int = 2718

    def __post_init__(self):
        if not 0 < self.minimum_margin_coverage <= 1 or not 0 <= self.margin_quantile <= 1:
            raise ValueError('invalid capability coverage/quantile')
        if self.collapse_variance < 0:
            raise ValueError('invalid collapse threshold')
        AuditConfig(steps=self.steps, lr=self.lr, weight_decay=self.weight_decay,
                    min_samples_per_class=self.min_samples_per_class)


def cross_rx_margin(fit_z, fit_tx, fit_rx, monitor_z, monitor_tx, monitor_rx, *, quantile=0.1):
    """Centers use fit samples only; each comparison has a different RX.

    Missing same-class or competitor centers invalidate that pair, not silently
    improve its score. Report eligible-pair coverage and the weakest RX pair.
    """
    fit, monitor = F.normalize(fit_z.detach(), dim=-1), F.normalize(monitor_z.detach(), dim=-1)
    references = {}
    for rx in fit_rx.unique().tolist():
        rows = fit_rx == rx
        references[rx] = {int(tx): F.normalize(fit[rows & (fit_tx == tx)].mean(0), dim=0)
                          for tx in fit_tx[rows].unique().tolist()}
    pairs, potential, covered, sample_covered = {}, 0, 0, set()
    for index in range(len(monitor)):
        for other_rx, centers in references.items():
            if other_rx == int(monitor_rx[index]):
                continue
            potential += 1
            label = int(monitor_tx[index])
            if label not in centers or len(centers) < 2:
                continue
            positive = torch.dot(monitor[index], centers[label])
            negative = torch.stack([torch.dot(monitor[index], center) for tx, center in centers.items() if tx != label]).max()
            key = f'{int(monitor_rx[index])}->{other_rx}'
            pairs.setdefault(key, []).append(float(positive - negative))
            covered += 1
            sample_covered.add(index)
    values = [v for vs in pairs.values() for v in vs]
    pair_means = {key: sum(vs) / len(vs) for key, vs in pairs.items()}
    return dict(valid=bool(values), mean=sum(values) / len(values) if values else None,
                low_quantile=float(torch.tensor(values).quantile(quantile)) if values else None,
                weakest_pair_margin=min(pair_means.values()) if pair_means else None,
                pair_means=pair_means, pair_counts={key: len(vs) for key, vs in pairs.items()},
                coverage=covered / potential if potential else 0.0, eligible_pairs=potential,
                evaluated_pairs=covered, sample_coverage=len(sample_covered) / max(len(monitor), 1))


def evaluate_capability(fit_z, fit_tx, fit_rx, monitor_z, monitor_tx, monitor_rx,
                        *, fit_groups, monitor_groups, config=CapabilityConfig(), monitor_aug=None,
                        step=0, encoder_version=0, independence_verified=False, source_role='train'):
    """Fit a separate linear TX head; features remain detached from the encoder."""
    if source_role != 'train':
        raise ValueError('capability fitting requires labeled source training data')
    validate_groups(fit_groups, monitor_groups, len(fit_z), len(monitor_z))
    if any(len(a) != len(fit_z) for a in (fit_tx, fit_rx)) or any(len(a) != len(monitor_z) for a in (monitor_tx, monitor_rx)):
        raise ValueError('capability label/sample length mismatch')
    if not len(fit_z) or not len(monitor_z):
        raise ValueError('nonempty capability sets required')
    if (fit_tx < 0).any() or (monitor_tx < 0).any():
        raise ValueError('hidden or unknown TX labels forbidden')
    started = time.perf_counter()
    reason = _coverage_reason(fit_tx, monitor_tx, config.min_samples_per_class)
    if not independence_verified:
        reason = reason or 'acquisition_independence_unverified'
    probe_config = AuditConfig(steps=config.steps, lr=config.lr, weight_decay=config.weight_decay,
                               min_samples_per_class=config.min_samples_per_class, seed=config.seed)
    with isolated_rng(config.seed):
        classes = int(max(fit_tx.max(), monitor_tx.max())) + 1
        head = nn.Linear(fit_z.shape[1], classes).to(device=fit_z.device, dtype=fit_z.dtype)
        performed = fit_probe(head, fit_z, fit_tx, probe_config) if not reason or reason == 'acquisition_independence_unverified' else 0
        with torch.no_grad():
            readout = classification_metrics(head(monitor_z.detach()), monitor_tx)
            margin = cross_rx_margin(fit_z, fit_tx, fit_rx, monitor_z, monitor_tx, monitor_rx,
                                     quantile=config.margin_quantile)
            variance = float(monitor_z.detach().double().var(dim=0, unbiased=False).mean())
            collapsed = not torch.isfinite(torch.tensor(variance)) or variance <= config.collapse_variance
            consistency = None
            if monitor_aug is not None:
                if monitor_aug.shape != monitor_z.shape:
                    raise ValueError('clean/aug sample correspondence required')
                consistency = float(F.cosine_similarity(monitor_z.detach(), monitor_aug.detach(), dim=-1).mean())
    if collapsed:
        reason = reason or 'representation_collapsed'
    if not margin['valid'] or margin['coverage'] < config.minimum_margin_coverage:
        reason = reason or 'insufficient_cross_rx_margin_coverage'
    if performed != config.steps:
        reason = reason or 'readout_budget_not_completed'
    return dict(valid=not reason, reason=reason, step=int(step), encoder_version=int(encoder_version),
                identity_valid=not reason, identity=readout['balanced_accuracy'], margin=margin['low_quantile'],
                readout=readout, cross_rx=margin, feature_variance=variance, collapsed=bool(collapsed),
                consistency=consistency, probe_steps=performed, elapsed_seconds=time.perf_counter() - started,
                independence_verified=bool(independence_verified))


class BranchCapabilityRegistry:
    """Conditional interface only; CORE90's z_dom is not an identity branch."""
    def __init__(self):
        self.branches = {}

    def register(self, name, capability, *, identity_threshold, margin_threshold):
        self.branches[name] = dict(capability=dict(capability), identity_threshold=identity_threshold,
                                   margin_threshold=margin_threshold)

    def fusion_gate(self, name=None):
        if name is None:
            return dict(applicable=False, allowed=False, reason='NOT_APPLICABLE_NO_RESPONSE_FUSION')
        item = self.branches[name]
        evidence = item['capability']
        allowed = bool(evidence.get('valid') and not evidence.get('collapsed', True) and
                       evidence.get('identity', -1) >= item['identity_threshold'] and
                       evidence.get('margin') is not None and evidence['margin'] >= item['margin_threshold'])
        return dict(applicable=True, allowed=allowed, reason='independent_branch_capable' if allowed else 'branch_capability_insufficient')
