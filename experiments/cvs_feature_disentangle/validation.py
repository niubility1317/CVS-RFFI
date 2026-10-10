"""Read-only full source-V diagnostics and immutable source-only selection."""
import copy
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import design as d
from experiments.cvs_multi_action_risk.physics import (
    factorial_batch, apply_action, sample_parameters, phase_process, quality_proxies)


def margins(logits, labels):
    other = logits.clone()
    other.scatter_(1, labels[:, None], float('-inf'))
    return logits.gather(1, labels[:, None]).squeeze(1) - other.max(1).values


def _metric(logits, y):
    pred = logits.argmax(1)
    margin = margins(logits, y)
    k = max(1, math.ceil(len(y) * d.STRESS['tail_alpha']))
    cm = torch.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
    denom = cm.sum(0) + cm.sum(1)
    f1 = torch.where(denom > 0, 2 * cm.diag() / denom, torch.zeros_like(denom)).mean()
    tail_indices = torch.topk(-margin, k).indices
    return dict(count=len(y), accuracy=float((pred == y).float().mean()),
                macro_f1=float(f1), margin_mean=float(margin.mean()),
                margin_q10=float(torch.quantile(margin, .1)),
                tail_risk=float(F.softplus(-margin[tail_indices]).mean()),
                error_tail_rate=float((pred[tail_indices] != y[tail_indices]).float().mean()),
                mean_ce=float(F.cross_entropy(logits, y)))


def metrics(logits, y, rx, day):
    logits = logits.float()
    if len(y) == 0 or logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
        raise ValueError('Invalid source logits')
    groups = {name: {str(int(v)): _metric(logits[values == v], y[values == v])
                     for v in values.unique()}
              for name, values in [('receiver', rx), ('day', day), ('transmitter', y)]}
    acc = {name: {k: value['accuracy'] for k, value in records.items()} for name, records in groups.items()}
    return dict(_metric(logits, y), group_accuracy=acc, group_metrics=groups,
                worst_rx=min(acc['receiver'].values()), worst_day=min(acc['day'].values()),
                worst_tx=min(acc['transmitter'].values()))


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right.to(left))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


@contextmanager
def time_contribution_off(model):
    """Zero exactly the final time embedding; leave all other branch inputs intact."""
    # The native dual wrapper retains a domain backbone even when all domain
    # losses are disabled. Its t_proj is not part of the identity dependency
    # probe. Restrict resolution to identity before checking uniqueness; a
    # standalone identity adapter (used by diagnostics) is already that scope.
    identity = getattr(model, 'id_backbone', model)
    prefix = 'id_backbone.' if identity is not model else ''
    points = [(prefix + name, module) for name, module in identity.named_modules()
              if name.split('.')[-1] == 't_proj']
    if len(points) != 1:
        raise ValueError('Expected exactly one identity downstream time t_proj for dependency probe; '
                         f'found {[name for name, _ in points]}')
    name, module = points[0]
    calls = {'count': 0, 'hook': name}
    def zero_time(_module, _inputs, output):
        if output.ndim != 2 or output.shape[1] != 160:
            raise ValueError('Time contribution probe requires final [B,160] vector')
        calls['count'] += 1
        return torch.zeros_like(output)
    handle = module.register_forward_hook(zero_time)
    try:
        yield calls
    finally:
        handle.remove()


@contextmanager
def source_readonly(model, style=None, fishr=None):
    modes = [(m, m.training) for m in model.modules()]
    before = copy.deepcopy(model.state_dict())
    fishr_before = copy.deepcopy(fishr.state_dict()) if fishr is not None else None
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    attached = style is not None and style.handle is not None
    style_before = copy.deepcopy(style.state_dict()) if style is not None else None
    context = copy.deepcopy(style._context) if style is not None else None
    if attached:
        style.handle.remove()
        style.handle = None
    model.eval()
    try:
        yield
        if not _equal(before, model.state_dict()):
            raise ValueError('Source V changed model state')
        if fishr is not None and not _equal(fishr_before, fishr.state_dict()):
            raise ValueError('Source V changed Fishr buckets')
        if style is not None and not _equal(style_before, style.state_dict()):
            raise ValueError('Source V changed style state')
    finally:
        for module, mode in modes:
            module.training = mode
        if attached:
            style.attach(model)
            style._context = context
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _quality_layers(logits, y, rx, day, quality):
    out = {}
    for column, name in [(0, 'lag20_coherence'), (1, 'fitted_repeat_residual')]:
        values = quality[:, column].contiguous()
        cuts = torch.quantile(values, torch.tensor([.25, .5, .75]))
        bins = torch.bucketize(values, cuts, right=True)
        out[name] = dict(cutpoints=cuts.tolist(), definition='source V clean quartiles; same packet bins across views; proxy is not SNR',
                         bins={str(i): metrics(logits[bins == i], y[bins == i], rx[bins == i], day[bins == i])
                               if (bins == i).any() else {'count': 0, 'metrics': None}
                               for i in range(4)})
    return out


def evaluate_source_stress(model, loader, device, c, output, *, style=None, fishr=None):
    with source_readonly(model, style, fishr), torch.no_grad():
        result = _evaluate_source_stress(model, loader, device, c, output)
    d.write(Path(output) / 'source_stress.json', result)
    return result


def _evaluate_source_stress(model, loader, device, c, output):
    generator = torch.Generator().manual_seed(d.STRESS['seed'])
    collected, applied = defaultdict(list), defaultdict(int)
    labels, receivers, days, quality, off_logits = [], [], [], [], []
    total, probe_hook = 0, None
    for step, (x, y, domain, meta) in enumerate(loader):
        x = x.to(device)
        b = factorial_batch(x, generator, step=0)
        tl = apply_action(b['x01'], b['linear_parameters'], 'linear')
        heldout = apply_action(b['x11'], .5 * b['linear_parameters'], 'linear')
        p = sample_parameters('temporal', len(x), generator, x, pressure=True)
        pressure = apply_action(x, p, 'temporal', phase_process(len(x), generator, x))
        noise = torch.randn(b['x11'].shape, generator=generator).to(x)
        noise = noise / noise.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-10)
        noisy = b['x11'] + noise * b['x11'].square().mean((1, 2), keepdim=True).sqrt() * 10 ** (-25/20)
        views = dict(clean=x, linear=b['x10'], temporal=b['x01'], LT=b['x11'], TL=tl,
                     noise_fixed_LT=noisy, LTL_heldout=heldout, curvature_pressure=pressure)
        if set(views) != set(d.STRESS['views']):
            raise ValueError('Source pressure views differ from preregistration')
        for name, view in views.items():
            collected[name].append(model(view).detach().cpu())
            applied[name] += int((view != x).flatten(1).any(1).sum())
        with time_contribution_off(model) as probe:
            off_logits.append(model(x).detach().cpu())
        if probe['count'] != 1:
            raise ValueError('Time branch probe did not execute once')
        probe_hook = probe['hook']
        quality.append(quality_proxies(x).cpu())
        labels.append(y.cpu()); receivers.append(meta['rx_i'].cpu()); days.append(meta['day_i'].cpu())
        total += len(y)
    if total != 27000:
        raise ValueError('Incomplete source V pressure validation')
    y, rx, day, q = map(torch.cat, (labels, receivers, days, quality))
    if set(rx.tolist()) != {1, 3, 4, 6, 8}:
        raise ValueError('Non-source receiver in V')
    full = torch.cat(collected['clean']); off = torch.cat(off_logits)
    difference = margins(full, y) - margins(off, y)
    result = dict(status='SOURCE_STRESS_COMPLETE', row_id=c['row_id'], model_seed=c['model_seed'], arm=c['arm'],
                  source_role='V only; not auxiliary holdout', target_access=False,
                  model_state_unchanged=True, fishr_state_unchanged=True, style_disabled=True,
                  config=d.STRESS, views={})
    for name, values in collected.items():
        logits = torch.cat(values)
        result['views'][name] = dict(metrics(logits, y, rx, day), applied_packets=applied[name],
                skipped_clean_copies=total-applied[name], heldout_composition=name == 'LTL_heldout',
                artificial_curvature_pressure=name == 'curvature_pressure',
                quality_strata=_quality_layers(logits, y, rx, day, q))
    result['time_dependency_probe'] = dict(scope='full source V clean only', hook=probe_hook,
        operation='zero identity downstream t_proj output; domain, frequency, PA and reference inputs unchanged',
        interpretation='dependency probe; feature intervention may be out of distribution, not causal contribution',
        full=metrics(full, y, rx, day), time_off=metrics(off, y, rx, day),
        paired_margin_difference_mean=float(difference.mean()),
        paired_margin_difference_q10=float(torch.quantile(difference, .1)),
        paired_margin_difference_by_rx={str(int(r)): float(difference[rx == r].mean()) for r in rx.unique()})
    return result


def source_select():
    """Open exactly 12 predeclared source artifacts, never discover target files."""
    rule = d.SELECTION_RULE
    arms = [rule['baseline']] + rule['source_arms']
    groups = defaultdict(list)
    rows = [row for row in d.rows() if row['arm'] in arms]
    for arm in arms:
        if sorted(r['model_seed'] for r in rows if r['arm'] == arm) != sorted(d.SEEDS):
            raise ValueError('Incomplete source selection seed matrix')
    for row in rows:
        path = Path(d.config(row)['output_root']) / 'source_stress.json'
        value = d.read(path)
        if (value['status'] != 'SOURCE_STRESS_COMPLETE' or value['target_access']
                or value['row_id'] != row['row_id'] or value['model_seed'] != row['model_seed']
                or value['arm'] != row['arm'] or value['config'] != d.STRESS
                or set(value['views']) != set(d.STRESS['views'])
                or any(v['count'] != 27000 for v in value['views'].values())):
            raise ValueError('Invalid full source stress evidence')
        if any(not math.isfinite(v[k]) or not 0 <= v[k] <= 1
               for v in value['views'].values() for k in ('accuracy', 'worst_rx')):
            raise ValueError('Invalid source selection metrics')
        groups[row['arm']].append(value)
    mean = lambda values: sum(values) / len(values)
    baseline = mean([v['views']['clean']['accuracy'] for v in groups[rule['baseline']]])
    records = []
    for percent, arm in zip(rule['candidates'], rule['source_arms']):
        values = groups[arm]
        clean = mean([v['views']['clean']['accuracy'] for v in values])
        score = mean([(q['accuracy'] + q['worst_rx']) / 2 for v in values for q in v['views'].values()])
        records.append(dict(arm=arm, ratio=percent/100., seeds=sorted(v['model_seed'] for v in values),
                            clean_accuracy=clean, score=score,
                            feasible=clean >= baseline-rule['clean_noninferiority_pp']/100.))
    feasible = [v for v in records if v['feasible']]
    winner = (max(feasible, key=lambda v: (v['score'], -v['ratio'])) if feasible else
              max(records, key=lambda v: (v['clean_accuracy'], v['score'], -v['ratio'])))
    percent = int(round(winner['ratio']*100))
    result = dict(status='SOURCE_SELECTION_FROZEN', selected_ratio=winner['ratio'], selected_arm=winner['arm'],
                  selected_companion=rule['selected_companion'].format(ratio=percent),
                  baseline_clean_accuracy=baseline, no_clean_noninferior_candidate=not bool(feasible),
                  records=records, rule=copy.deepcopy(rule), target_scores_consumed=False,
                  trained_identity_rows=48, target_test_rows=36,
                  selection_scope='Only current-run fixed baseline and AF3/AF5/AF10 full source V artifacts')
    path = d.BASE / 'source_selection.json'
    if path.exists():
        if d.read(path) != result:
            raise ValueError('Immutable source selection already differs; refusing overwrite')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        if d.read(path) != result:
            raise ValueError('Source selection readback mismatch')
    return result
