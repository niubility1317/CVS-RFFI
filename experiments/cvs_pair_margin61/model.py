"""Reuse every original forward; add only a default-head paired margin loss."""
from pathlib import Path
from types import MethodType
import torch
from . import design as d
from .margin import captured_native_views, penalty
from experiments.cvs_noise_shape_factorial43 import model as original

classifier, source_domain_count = original.classifier, original.source_domain_count

def hook(self, **kw):
    head = self.id_backbone.encoder.core.id_backbone.cls_head
    if head.scale != 30. or tuple(head.weight.shape) != (6, 160):
        raise ValueError('Original default cosine head/scale30 required')
    with captured_native_views(self) as (z, observed):
        delta, info = original.hook(self, **kw)
    if len(z) != 5 or len(observed) != 1:
        raise ValueError('Original hook must execute ordinary1/mixed0/mixed1/raw0/raw1 and one channel call')
    mask = (torch.arange(128, device=kw['labels'].device) + kw['optimizer_step']).remainder(2) != 0
    clean = .5 * (z[3][mask] + z[4][mask])
    student = .5 * (z[1][mask] + z[2][mask])
    pair, record = penalty(clean, student, kw['labels'][mask], observed[0])
    native = self._full_records[-1]
    base = native['total_L']
    record.update(step=kw['optimizer_step'], cumulative_step=kw['optimizer_step'] + 4440,
                  base_L_before_pair=base, total_L=base + record['pair_weighted'])
    self._pair_margin_records.append(record)
    native.update(base_L_before_pair=base, pair_raw=record['pair_raw'],
                  pair_weighted=record['pair_weighted'], total_L=record['total_L'])
    info.update({'pair_margin/' + k: record[k] for k in ('pair_raw', 'pair_weighted', 'q_mean', 'clean_correct_count')})
    return delta + pair, info

hook._accepts_labeled_inputs = True

def attach(m, c):
    d.validate(c)
    m._pair_margin_records = []
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._feature_learning_report['pair_margin61'] = c['pair_margin']
    return m

def install(m, c):
    return attach(original.install(m, d.old_config(c['arm'])), c)

def build(c, device='cpu'):
    return attach(original.build(d.old_config(c['arm']), device), c)

def own_model(c, device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(c)
    parent = parent_payload(c)
    p = Path(c['output_root'])
    final_metadata(c, d.read(p / 'completion.json'), d.read(p / 'source_frozen.json'),
                   d.read(p / 'initialization.json'), d.read(p / 'source_contract.json'), d.read(d.SOURCE)['role_ids'])
    cp = torch.load(p / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(c, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA keys differ')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
