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
    """Legacy v1 fitting API (also used by v1 capability); no v2 quality claim."""
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
        metrics = dict(schema='game_audit_v1', valid=not reason, reason=reason, step=int(step), encoder_version=int(encoder_version),
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


@dataclass(frozen=True)
class ProbeConfigV2:
    """Source-development constants, not target-calibrated or proven optimal.

    Low gap requires a completed minimum budget and stationary/plateau evidence.
    High gap only certifies the measured improvement, not a global best response.
    """
    steps: int = 40
    lr: float = 2e-3
    weight_decay: float = 0.
    objective_scale: float = 1.
    min_steps: int = 40
    high_gap_threshold: float = .1
    gradient_tolerance: float = 1e-4
    plateau_tolerance: float = 1e-5
    plateau_window: int = 5
    instability_absolute: float = .05
    instability_relative: float = .25
    transfer_tolerance: float = .01
    monitor_interval: int = 10
    mlp_width: int = 32
    seed: int = 1729
    threshold_source: str = 'fixed_source_development_unvalidated'

    def __post_init__(self):
        numeric = (self.lr, self.weight_decay, self.objective_scale, self.high_gap_threshold,
                   self.gradient_tolerance, self.plateau_tolerance,
                   self.instability_absolute, self.instability_relative, self.transfer_tolerance)
        if not all(np.isfinite(v) and v >= 0 for v in numeric) or self.lr == 0:
            raise ValueError('invalid v2 numerical threshold')
        if min(self.steps, self.min_steps) < 0 or min(self.plateau_window,self.monitor_interval,self.mlp_width) < 1:
            raise ValueError('invalid v2 budget')


def _weighted_inputs(features, labels, weights):
    x, y = features.detach(), labels.detach().long()
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError('expected nonempty matching features and labels')
    w = (torch.ones(len(y), device=x.device, dtype=x.dtype) if weights is None else
         weights.detach().to(device=x.device, dtype=x.dtype))
    if w.shape != y.shape or not torch.isfinite(w).all() or (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError('sample weights must be finite, nonnegative, and have positive sum')
    return x, y, w / w.sum()


def _scale(values):
    values = values.detach().float().reshape(-1)
    if not values.numel() or not torch.isfinite(values).all():
        return None
    return dict(zip(('min','p25','median','p75','max'),
                    torch.quantile(values,values.new_tensor([0.,.25,.5,.75,1.])).cpu().tolist()))


def _fit_v2(online_head, features, labels, sample_weights, config, objective_scope,
            fixed_head_state=None, monitor=None):
    """Owned-copy fixed-objective optimizer; monitor never selects or backprops.

    Every forward resets the same head buffers and RNG. Mixed module training
    flags are copied from online_head (or explicitly supplied module_training).
    This supports a fixed train-mode Dropout realization, not an expectation over
    future random masks. Encoder features must already represent the claimed view.
    """
    x,y,w = _weighted_inputs(features,labels,sample_weights)
    head = deepcopy(online_head)
    for p in head.parameters():
        p.requires_grad_(True)
        p.grad = None
    modes = dict((name,module.training) for name,module in head.named_modules())
    if objective_scope == 'eval_clean_surrogate':
        head.eval()
        modes = dict((name,False) for name in modes)
    elif fixed_head_state is not None and 'module_training' in fixed_head_state:
        modes = fixed_head_state['module_training']
        if set(modes) != set(dict(head.named_modules())):
            raise ValueError('fixed module_training must cover every head module')
        for name,module in head.named_modules():
            module.training = bool(modes[name])
    buffers = {name:b.detach().clone() for name,b in head.named_buffers()}
    state = fixed_head_state or {}
    cpu_rng = state.get('cpu_rng',torch.get_rng_state()).clone()
    cuda_rng = state.get('cuda_rng')
    if cuda_rng is None and x.is_cuda:
        cuda_rng = torch.cuda.get_rng_state_all()

    def forward(z):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.no_grad():
            for name,b in head.named_buffers():
                b.copy_(buffers[name])
        return head(z)

    def measure(z, target, weights):
        logits = forward(z)
        loss = (F.cross_entropy(logits,target,reduction='none')*weights).sum()
        return loss,logits

    training_random_fixed = (not any(modes.values()) or
                            (fixed_head_state is not None and 'cpu_rng' in state and
                             (not x.is_cuda or 'cuda_rng' in state)))
    counts = torch.zeros(int(y.max())+1,device=x.device,dtype=x.dtype)
    if int(y.min()) < 0:
        raise ValueError('negative class label')
    counts.scatter_add_(0,y,w)
    nonzero = counts[counts>0]
    entropy = float(-(nonzero*nonzero.log()).sum())
    result = dict(estimand='bounded_empirical_head_optimization_gap',objective_scope=objective_scope,
                  status='UNAVAILABLE',quality_pass=False,control_ready=False,
                  fit_before=None,fit_after=None,monitor_before=None,monitor_after=None,
                  gap_raw=None,gap_normalized=None,readability=None,prior_entropy=entropy,
                  trajectory=[],reason_codes=[],selection='fixed_endpoint',
                  optimizer='AdamW_cold_reset', optimizer_state_mode='cold_reset',
                  objective_scale=config.objective_scale,
                  budget=dict(requested_steps=config.steps,minimum_steps=config.min_steps,actual_steps=0),
                  thresholds=asdict(config),threshold_source=config.threshold_source,
                  module_training=modes,fixed_stochastic_state=training_random_fixed,
                  feature_norm_quantiles=_scale(x.norm(dim=1)),
                  monitor_feature_norm_quantiles=(_scale(monitor[0].norm(dim=1)) if monitor is not None else None),
                  sample_weights=dict(sum=float(w.sum()),min=float(w.min()),max=float(w.max())),
                  extrapolation='bounded_training_subset_only_no_independent_monitor')
    if config.objective_scale == 0:
        result['reason_codes'] = ['inactive_head_objective']
        result['stop_reason'] = 'inactive_head_objective'
        return AuditResult(result,head,{})
    if entropy <= 0:
        result['reason_codes'] = ['single_effective_class']
        return AuditResult(result,head,{})
    optimizer = torch.optim.AdamW(head.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    failure = not bool(torch.isfinite(x).all())
    stop = 'nonfinite_features' if failure else 'fixed_budget_completed'
    if not failure:
        for step in range(config.steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss,logits = measure(x,y,w)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                failure,stop = True,'nonfinite_fit'
                break
            (loss*config.objective_scale).backward()
            grads = [p.grad.detach().norm().square() for p in head.parameters() if p.grad is not None]
            grad_norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.
            ce = float(loss.detach())
            row = dict(step=step,fit_ce=ce,fit_objective=ce*config.objective_scale,gradient_norm=grad_norm,
                       head_parameter_norm=float(torch.stack([p.detach().norm().square() for p in head.parameters()]).sum().sqrt()),
                       logit_scale=_scale(logits),monitor_ce=None)
            # Monitor executes after backward, never entering the fit graph.
            if monitor is not None and (step == 0 or step == config.steps or step % config.monitor_interval == 0):
                with torch.no_grad():
                    ml,_ = measure(*monitor)
                row['monitor_ce'] = float(ml) if bool(torch.isfinite(ml)) else None
            result['trajectory'].append(row)
            if not np.isfinite(grad_norm):
                failure,stop = True,'nonfinite_gradient'
                break
            start_ce = result['trajectory'][0]['fit_ce']
            if ce > start_ce + config.instability_absolute + config.instability_relative*abs(start_ce):
                failure,stop = True,'fit_instability'
                break
            if step == config.steps:
                break
            optimizer.step()
            result['budget']['actual_steps'] += 1
    trace = result['trajectory']
    if trace:
        result.update(fit_before=trace[0]['fit_ce'],fit_after=trace[-1]['fit_ce'],
                      monitor_before=trace[0]['monitor_ce'],monitor_after=trace[-1]['monitor_ce'])
    if failure:
        result.update(status='OPTIMIZATION_FAILURE',reason_codes=[stop])
    elif config.steps < config.min_steps or not config.steps:
        result.update(status='BUDGET_INCONCLUSIVE',reason_codes=['insufficient_fixed_budget'])
    else:
        raw = result['fit_before']-result['fit_after']
        normalized = raw/entropy
        tail = [r['fit_ce'] for r in trace[-(config.plateau_window+1):]]
        stationary = trace[-1]['gradient_norm'] <= config.gradient_tolerance
        plateau = (len(tail) == config.plateau_window+1 and max(tail)-min(tail) <= config.plateau_tolerance)
        result['convergence'] = dict(stationary=stationary,plateau=plateau)
        if raw < -config.plateau_tolerance:
            result.update(status='OPTIMIZATION_FAILURE',reason_codes=['endpoint_fit_worsened'])
        elif normalized >= config.high_gap_threshold or stationary or plateau:
            result.update(status=('RELIABLE_HIGH_GAP' if normalized >= config.high_gap_threshold else 'RELIABLE_LOW_GAP'),
                          quality_pass=True,gap_raw=raw,gap_normalized=normalized,
                          readability=(entropy-result['fit_after'])/entropy)
        else:
            result.update(status='BUDGET_INCONCLUSIVE',reason_codes=['low_gap_without_convergence_support'])
    result['stop_reason'] = stop
    result['control_ready'] = bool(result['quality_pass'] and training_random_fixed and
                                   objective_scope == 'current_training_head_objective')
    if not training_random_fixed:
        result['reason_codes'].append('training_stochastic_state_unverified')
    if objective_scope != 'current_training_head_objective':
        result['reason_codes'].append('surrogate_not_control_ready')
    with torch.no_grad():
        for name,b in head.named_buffers():
            b.copy_(buffers[name])
    for p in head.parameters():
        p.grad = None
    return AuditResult(result,head,{})


def fit_empirical_lag(online_head, features, labels, *, sample_weights, objective_scope,
                      config=ProbeConfigV2(), fixed_head_state=None, source_role='train'):
    """V2 lag on one fixed weighted target; no fabricated independent monitor.

    Caller certifies legal source labels, representative coverage, current main
    view and head scale. Passing a training scope does not certify those claims.
    Returned head is owned; online parameters/buffers/gradients/RNG are untouched.
    """
    if source_role != 'train':
        raise ValueError('probe fitting requires source train data; V/target forbidden')
    if objective_scope not in ('current_training_head_objective','eval_clean_surrogate'):
        raise ValueError('unsupported head objective scope')
    with isolated_rng(config.seed):
        return _fit_v2(online_head,features,labels,sample_weights,config,objective_scope,fixed_head_state)


def cross_tx_readout(features, domain_labels, rx_labels, tx, groups, *,
                     config=ProbeConfigV2(), seed=0, standardize=False, source_role='train'):
    """Two swapped TX-disjoint folds, fresh linear/MLP domain and RX heads.

    Standardization is fit-only. No transformed-feature comparison to online
    weights, no monitor-driven endpoint selection, and no lag control evidence.
    """
    if source_role != 'train':
        raise ValueError('cross TX probes require source train data')
    if len(groups) != len(features) or len(tx) != len(features):
        raise ValueError('cross TX metadata length mismatch')
    unique = sorted(set(tx.detach().cpu().tolist()),key=lambda t:hashlib.sha256(f'{seed}:{t}'.encode()).digest())
    if len(unique) < 2:
        return dict(status='UNAVAILABLE',quality_pass=False,control_ready=False,folds=[],reason_codes=['insufficient_tx'])
    partitions = (unique[:len(unique)//2],unique[len(unique)//2:])
    folds = []
    with isolated_rng(config.seed+seed):
        for fold_id in range(2):
            fit_tx,monitor_tx = partitions[fold_id],partitions[1-fold_id]
            fi = torch.tensor([i for i,t in enumerate(tx.tolist()) if t in fit_tx],device=features.device)
            mi = torch.tensor([i for i,t in enumerate(tx.tolist()) if t in monitor_tx],device=features.device)
            fg,mg = [groups[i] for i in fi.tolist()],[groups[i] for i in mi.tolist()]
            validate_groups(fg,mg,len(fi),len(mi))
            xf,xm = features[fi].detach(),features[mi].detach()
            transform = None
            if standardize:
                mean,std = xf.mean(0),xf.std(0,unbiased=False).clamp_min(1e-6)
                xf,xm = (xf-mean)/std,(xm-mean)/std
                transform = dict(mean=mean.cpu().tolist(),std=std.cpu().tolist(),scope='fit_only')
            fold = dict(fit_tx=sorted(fit_tx),monitor_tx=sorted(monitor_tx),
                        fit_groups=sorted(set(map(str,fg))),monitor_groups=sorted(set(map(str,mg))),
                        standardization=transform,readouts={})
            for task,all_y in (('domain',domain_labels),('rx',rx_labels)):
                yf,ym = all_y[fi],all_y[mi]
                reason = _coverage_reason(yf,ym,1)
                fold['readouts'][task] = {}
                for kind in ('linear','mlp'):
                    if reason:
                        fold['readouts'][task][kind] = dict(status='UNAVAILABLE',quality_pass=False,reason_codes=[reason])
                        continue
                    classes = int(all_y.max())+1
                    head = (nn.Linear(xf.shape[1],classes) if kind == 'linear' else
                            nn.Sequential(nn.Linear(xf.shape[1],config.mlp_width),nn.ReLU(),nn.Linear(config.mlp_width,classes)))
                    head = head.to(device=xf.device,dtype=xf.dtype).eval()
                    monitor = _weighted_inputs(xm,ym,None)
                    fitted = _fit_v2(head,xf,yf,None,config,'cross_tx_readout',monitor=monitor)
                    metric = fitted.metrics
                    metric['estimand'] = 'cross_tx_domain_readout' if task == 'domain' else 'cross_tx_rx_readout'
                    metric['extrapolation'] = 'held_out_tx_groups_only'
                    metric['control_ready'] = False
                    # These are fit optimization diagnostics, never a cross-TX lag.
                    metric['fit_improvement_raw'] = metric.pop('gap_raw')
                    metric['fit_improvement_normalized'] = metric.pop('gap_normalized')
                    metric['fit_optimization_status'] = metric['status']
                    if metric['status'] != 'OPTIMIZATION_FAILURE':
                        if metric['monitor_after'] is None:
                            metric.update(status='TRANSFER_FAILURE',quality_pass=False,reason_codes=['nonfinite_monitor'])
                        elif metric['monitor_after'] > metric['monitor_before'] + config.transfer_tolerance:
                            metric.update(status='TRANSFER_FAILURE',quality_pass=False,reason_codes=['monitor_ce_worsened'])
                    # CE metrics at the fixed endpoint, for readout interpretation.
                    with torch.no_grad():
                        monitor_logits = fitted.recovered_head(xm)
                        metric['monitor'] = (classification_metrics(monitor_logits,ym)
                                             if bool(torch.isfinite(monitor_logits).all()) else None)
                    fold['readouts'][task][kind] = metric
            folds.append(fold)
    return dict(estimand='cross_tx_readout',folds=folds,control_ready=False,
                quality_pass=all(p['quality_pass'] for f in folds for task in f['readouts'].values() for p in task.values()),
                seed=seed,grouping_kind='whole_tx_swapped_two_fold')
