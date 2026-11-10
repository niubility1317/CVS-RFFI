"""Reuse the verified same-seed prefix and existing continuation machinery."""
from pathlib import Path

from experiments.pamf_physics_tri import source as protected
from experiments.pamf_physics_tri.plan import RUN as PROTECTED_RUN
from .plan import RUN, SEEDS
from .scope import installed

_protected_parent = protected.parent


def parent(c):
    seed = c['model_seed']
    root = Path(c['output_root'])
    if (seed not in SEEDS or c['run_id'] != RUN or c['row_id'] != f'tri-unprotected-s{seed}'
            or root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source')
            or c.get('algorithm_id') != 'pamf_tri_unprotected_v1'):
        raise ValueError('Unregistered unprotected continuation')
    row = f'tri-label-tangent-s{seed}'
    proxy = dict(c, run_id=PROTECTED_RUN, row_id=row, algorithm_id='pamf_tri_label_tangent_v1',
                 output_root=str(root.parents[3] / PROTECTED_RUN / 'F' / row / 'source'))
    return _protected_parent(proxy)


def train(c):
    old_parent, old_installed = protected.parent, protected.installed
    protected.parent, protected.installed = parent, installed
    try:
        return protected.train(c)
    finally:
        protected.parent, protected.installed = old_parent, old_installed
