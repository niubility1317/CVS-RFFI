"""Reuse the exact M2 six-forward path; change only fixed As CE weights."""
from pathlib import Path
from types import MethodType

import torch


def _labeled_hook(self, **kwargs):
    from experiments.cvs_quality_risk70 import model as nominal
    if not self._quality_pool_bound or self._quality_method != 'QUALITY_STRAT':
        raise ValueError('Bind the existing QUALITY_STRAT pool before any bounded-risk update')
    before = len(self._quality_records)
    result = nominal._labeled_hook(self, **kwargs)
    if len(self._quality_records) != before + 1:
        raise ValueError('One original six-forward record per bounded-risk update required')
    record = self._quality_records[-1]
    reader = self._full_channel
    p, pi, v = (list(value) for value in (reader.nominal_p_g, reader.pi_g, reader.v_g))
    if list(record['p_g']) != pi:
        raise ValueError('The original hook must have used the fixed risk mixture')
    group_ce = record['group_CE']
    if any((p[g] > 0 or pi[g] > 0) and group_ce[g] is None for g in range(4)):
        raise ValueError('Every positive risk group requires an observed group mean')
    nominal_ce = sum(p[g] * group_ce[g] for g in range(4) if p[g] > 0)
    increment = .25 * sum((pi[g] - p[g]) * group_ce[g]
                          for g in range(4) if pi[g] > 0 or p[g] > 0)
    nominal_total = record['used_total_L'] - increment
    # P70 names its loss-weight argument p_g.  That temporary argument was pi;
    # restore the scientific nominal p in persisted records, and expose both.
    record.update(method='BOUNDED_RISK', allocation_method='QUALITY_STRAT',
        p_g=p, pi_g=pi, v_g=v, gamma=float(reader.gamma),
        nominal_M2_As_CE=nominal_ce, As_CE_nominal=nominal_ce,
        used_risk_As_CE=record['As_CE_weighted'],
        risk_vs_nominal_total_correction=increment, risk_increment_to_total=increment,
        nominal_M2_same_logits_total_L=nominal_total, nominal_same_logits_total_L=nominal_total,
        nominal_M2_is_same_logits_counterfactual=True,
        As_CE_coefficient=.25, As_KL_coefficient=.25,
        nominal_risk_changed=True, new_channel_calls=0)
    return result


_labeled_hook._accepts_labeled_inputs = True


def classifier(*args, **kwargs):
    from experiments.cvs_quality_risk70 import model as nominal
    return nominal.classifier(*args, **kwargs)


def source_domain_count(*args, **kwargs):
    from experiments.cvs_quality_risk70 import model as nominal
    return nominal.source_domain_count(*args, **kwargs)


def _attach(m, c):
    from . import design as d
    d.validate(c)
    if m._quality_method != 'QUALITY_STRAT' or m._quality_pool_bound or m._quality_records:
        raise ValueError('Start from the unchanged nominal M2 code before any update')
    m._source_labeled_feature_hook = MethodType(_labeled_hook, m)
    m._risk_records = m._quality_records
    m._feature_learning_report['bounded_risk71'] = dict(c['risk'],
        method=c['method'], allocation_method='QUALITY_STRAT',
        network_forward='Unchanged P70 M2 six L forwards/default head',
        parent='Original P43 As20 student and own EMA, not P70 final')
    return m


def install(m, c):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as nominal
    d.validate(c)
    return _attach(nominal.install(m, d.nominal_config()), c)


def build(c, device='cpu'):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as nominal
    d.validate(c)
    return _attach(nominal.build(d.nominal_config(), device), c)


def bind_pool(m, c, directory):
    from . import design as d, risk
    d.validate(c)
    if (m._quality_pool_bound or m._full_calls or m._quality_records
            or Path(directory).resolve() != Path(c['quality_pool']['root']).resolve()):
        raise ValueError('Bind the exact existing nominal pool once before any update')
    m._full_channel = risk.load(directory, seed=c['quality_pool']['allocation_seed'],
                                gamma=c['risk']['gamma'], v=c['risk']['v'])
    m._quality_pool_bound = True
    return m


def own_model(c, device='cpu'):
    from . import design as d
    from .source import parent_payload
    from experiments.cvs_quality_risk70.protocol import final_metadata, checkpoint
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
