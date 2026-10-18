"""Unmodified P43 F/default cosine head and six-forward objective, both arms."""
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_noise_shape_factorial43 import model as original

classifier, source_domain_count = original.classifier, original.source_domain_count

def attach(m, c):
    d.validate(c)
    m._feature_learning_report['PA_waveform60'] = c['PA_waveform']
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
                   d.read(p / 'initialization.json'), d.read(p / 'source_contract.json'),
                   d.read(d.SOURCE)['role_ids'])
    cp = torch.load(p / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(c, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA keys differ')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
