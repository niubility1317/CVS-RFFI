"""Unchanged P43 F/default head and full labeled/native-U forward path."""
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_noise_shape_factorial43 import model as original

classifier, source_domain_count = original.classifier, original.source_domain_count


def install(m, c):
    d.validate(c)
    return original.install(m, d.old_config(c['arm']))


def build(c, device='cpu'):
    d.validate(c)
    return original.build(d.old_config(c['arm']), device)


def own_model(c, device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(c)
    parent = parent_payload(c)
    out = Path(c['output_root'])
    final_metadata(c, d.read(out / 'completion.json'), d.read(out / 'source_frozen.json'),
        d.read(out / 'initialization.json'), d.read(out / 'source_contract.json'), d.read(d.SOURCE)['role_ids'])
    cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(c, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Final default-head student/ownEMA state keys differ')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
