"""Bounded source-training probes; monitor features never enter optimization.

Groups denote independently acquired records, not arbitrarily sliced windows.
Group disjointness alone does not establish physical independence: callers must
explicitly certify that metadata. Native domain labels and RX labels are distinct.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def isolated_rng(seed=0):
    """Keep Torch CPU/CUDA, Python and NumPy caller streams exactly unchanged."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def validate_groups(fit_groups, monitor_groups, n_fit, n_monitor):
    if fit_groups is None or monitor_groups is None:
        raise ValueError('acquisition group metadata required')
    if len(fit_groups) != n_fit or len(monitor_groups) != n_monitor:
        raise ValueError('group/sample length mismatch')
    if any(g is None or str(g) == '' for g in list(fit_groups) + list(monitor_groups)):
        raise ValueError('missing acquisition group')
    if set(map(str, fit_groups)) & set(map(str, monitor_groups)):
        raise ValueError('fit/monitor acquisition group overlap')


def partition_groups(groups, *, monitor_fraction=0.3, seed=0):
    """Deterministic group-level split, without touching caller random streams."""
    if not 0 < monitor_fraction < 1:
        raise ValueError('monitor_fraction must be inside (0, 1)')
    unique = set(map(str, groups))
    if len(unique) < 2:
        raise ValueError('at least two acquisition groups required')
    ordered = sorted(unique, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).digest())
    count = min(len(unique) - 1, max(1, round(len(unique) * monitor_fraction)))
    monitor = set(ordered[:count])
    return ([i for i, g in enumerate(groups) if str(g) not in monitor],
            [i for i, g in enumerate(groups) if str(g) in monitor])


@dataclass(frozen=True)
class AuditConfig:
    steps: int = 40
    lr: float = 0.02
    weight_decay: float = 0.0
    min_samples_per_class: int = 2
    independent_interval: int = 10
    mlp_width: int = 32
    seed: int = 1729

    def __post_init__(self):
        if self.steps < 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError('invalid probe optimizer budget')
        if self.min_samples_per_class < 1 or self.independent_interval < 1 or self.mlp_width < 1:
            raise ValueError('invalid audit coverage or cadence')


@dataclass
class AuditResult:
    metrics: dict
    recovered_head: nn.Module
    independent_heads: dict


def classification_metrics(logits, labels):
    """Empirical monitor prior, including imbalance-aware reference scores."""
    labels = labels.detach().long()
    logits = logits.detach()
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits) or not len(labels):
        raise ValueError('expected nonempty [N,C] logits and [N] labels')
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError('label outside classifier output space')
    classes, counts = labels.unique(return_counts=True)
    prior = counts.to(torch.float64) / len(labels)
    prediction = logits.argmax(-1)
    accuracy = (prediction == labels).float().mean()
    balanced = torch.stack([(prediction[labels == c] == c).float().mean() for c in classes]).mean()
    return dict(ce=float(F.cross_entropy(logits, labels)), accuracy=float(accuracy),
                balanced_accuracy=float(balanced), prior_entropy=float(-(prior * prior.log()).sum()),
                majority_accuracy=float(prior.max()), prior_sampling_accuracy=float(prior.square().sum()),
                effective_classes=len(classes), samples=len(labels),
                prior={int(c): float(p) for c, p in zip(classes, prior)})


def fit_probe(head, features, labels, config):
    """Fit an owned head on detached tensors with a new standard AdamW state."""
    head.train()
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = 0
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features.detach()), labels.detach().long())
        if not torch.isfinite(loss):
            break
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):
            break
        optimizer.step()
        steps += 1
    head.eval()
    return steps


def _coverage_reason(fit_labels, monitor_labels, min_samples):
    fc, fn = fit_labels.unique(return_counts=True)
    mc, mn = monitor_labels.unique(return_counts=True)
    if len(fc) < 2 or len(mc) < 2:
        return 'single_class'
    if set(fc.tolist()) != set(mc.tolist()):
        return 'fit_monitor_label_coverage_mismatch'
    if int(fn.min()) < min_samples or int(mn.min()) < min_samples:
        return 'insufficient_samples_per_class'
    return ''


class SourceAuditor:
    def __init__(self, config=AuditConfig()):
        self.config = config
        self.calls = 0

    def state_dict(self):
        return dict(config=asdict(self.config), calls=self.calls)

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('audit configuration differs from checkpoint')
        self.calls = int(state['calls'])

    def run(self, online_head, fit_features, fit_labels, monitor_features, monitor_labels,
            *, fit_groups, monitor_groups, step, encoder_version, label_kind='rx',
            independence_verified=False, source_role='train', rx_fit_labels=None, rx_monitor_labels=None,
            grouping_kind=None):
        """Return a recovered COPY plus scalar evidence; never change online head.

        `label_kind='domain'` recovers the native head, emits S_domain, and runs
        separate independently initialized RX probes when RX labels are supplied.
        Step and encoder_version are integers captured at feature extraction.
        """
        if source_role != 'train':
            raise ValueError('probe fitting requires legal source training data; V/target forbidden')
        if label_kind not in ('rx', 'domain'):
            raise ValueError('label_kind must distinguish rx from native domain')
        validate_groups(fit_groups, monitor_groups, len(fit_features), len(monitor_features))
        if len(fit_features) != len(fit_labels) or len(monitor_features) != len(monitor_labels):
            raise ValueError('feature/label length mismatch')
        if fit_features.ndim != 2 or monitor_features.ndim != 2 or fit_features.shape[1] != monitor_features.shape[1]:
            raise ValueError('features must be matching [N,D] matrices')
        if not torch.isfinite(fit_features).all() or not torch.isfinite(monitor_features).all():
            raise ValueError('nonfinite source features')
        started = time.perf_counter()
        reason = _coverage_reason(fit_labels, monitor_labels, self.config.min_samples_per_class)
        if not independence_verified:
            reason = reason or 'acquisition_independence_unverified'
        heads, probe_metrics, total_steps = {}, {}, 0
        with isolated_rng(self.config.seed + self.calls):
            recovered = deepcopy(online_head)
            original = deepcopy(online_head).eval()
            with torch.no_grad():
                online = classification_metrics(original(monitor_features.detach()), monitor_labels)
            if not reason or reason == 'acquisition_independence_unverified':
                total_steps = fit_probe(recovered, fit_features, fit_labels, self.config)
            recovered.eval()
            with torch.no_grad():
                recovered_metrics = classification_metrics(recovered(monitor_features.detach()), monitor_labels)
            if self.calls % self.config.independent_interval == 0:
                labels_by_task = {label_kind: (fit_labels, monitor_labels)}
                if label_kind == 'domain' and rx_fit_labels is not None and rx_monitor_labels is not None:
                    labels_by_task['rx'] = (rx_fit_labels, rx_monitor_labels)
                for task, (fit_y, monitor_y) in labels_by_task.items():
                    coverage = _coverage_reason(fit_y, monitor_y, self.config.min_samples_per_class)
                    if coverage:
                        probe_metrics[task] = dict(valid=False, reason=coverage)
                        continue
                    classes = int(max(fit_y.max(), monitor_y.max())) + 1
                    probe_metrics[task] = {}
                    for kind in ('linear', 'mlp'):
                        d = fit_features.shape[1]
                        head = (nn.Linear(d, classes) if kind == 'linear' else
                                nn.Sequential(nn.Linear(d, self.config.mlp_width), nn.ReLU(),
                                              nn.Linear(self.config.mlp_width, classes)))
                        head = head.to(device=fit_features.device, dtype=fit_features.dtype)
                        count = fit_probe(head, fit_features, fit_y, self.config)
                        total_steps += count
                        with torch.no_grad():
                            evidence = classification_metrics(head(monitor_features.detach()), monitor_y)
                        evidence.update(valid=bool(independence_verified and count == self.config.steps), steps=count)
                        probe_metrics[task][kind] = evidence
                        heads[f'{task}_{kind}'] = head
        entropy = online['prior_entropy']
        raw_gap = online['ce'] - recovered_metrics['ce']
        finite = all(np.isfinite(x) for x in (raw_gap, entropy, recovered_metrics['ce']))
        if not finite:
            reason = reason or 'nonfinite_probe_output'
        recovery_steps = total_steps - sum(p.get('steps', 0) for task in probe_metrics.values()
                                           for p in task.values() if isinstance(p, dict))
        if recovery_steps != self.config.steps:
            reason = reason or 'recovery_budget_not_completed'
        metrics = dict(valid=not reason, reason=reason, step=int(step), encoder_version=int(encoder_version),
                       label_kind=label_kind, online=online, recovered=recovered_metrics,
                       ce_gap_raw=raw_gap, G_lag=max(raw_gap, 0.) / (entropy + 1e-12),
                       independent=probe_metrics, independence_verified=bool(independence_verified),
                       probe_steps=total_steps, recovery_steps=recovery_steps,
                       elapsed_seconds=time.perf_counter() - started,
                       fit_groups=len(set(map(str, fit_groups))), monitor_groups=len(set(map(str, monitor_groups))),
                       seed=self.config.seed + self.calls, optimizer='AdamW_reset', grouping_kind=grouping_kind)
        metrics['S_' + label_kind] = max(entropy - recovered_metrics['ce'], 0.) / (entropy + 1e-12)
        self.calls += 1
        return AuditResult(metrics, recovered, heads)
