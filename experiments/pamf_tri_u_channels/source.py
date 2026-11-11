from contextlib import contextmanager
from pathlib import Path
from experiments.pamf_tri_u_phase import objective, source as phase
from experiments.pamf_tri_u_phase.plan import RUN as PHASE_RUN
from .plan import RUN, SEEDS, ARMS
from .transform import channel_view

_phase_parent = phase.parent


def parent(c):
    seed, root, arm = c['model_seed'], Path(c['output_root']), c.get('algorithm_id')
    if (seed not in SEEDS or arm not in ARMS or c['run_id'] != RUN or c['row_id'] != f'{arm}-s{seed}'
        or root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source')
        or c.get('channel_perturbation') != ARMS[arm]):
        raise ValueError('Unregistered source-U channel continuation')
    row = f'tri-u-phase-s{seed}'
    proxy = dict(c, run_id=PHASE_RUN, row_id=row, algorithm_id='pamf_tri_u_phase_v1',
                 output_root=str(root.parents[3]/PHASE_RUN/'F'/row/'source'))
    return _phase_parent(proxy)


@contextmanager
def installed(c, holder):
    arm = c.get('algorithm_id')
    if arm not in ARMS or c.get('channel_perturbation') != ARMS[arm]:
        raise ValueError('Fixed channel view required')
    previous = objective.rotate_iq
    objective.rotate_iq = lambda x, angle: channel_view(x, angle, **ARMS[arm])
    try:
        with objective.installed(c, holder):
            yield
    finally:
        objective.rotate_iq = previous


def train(c):
    old = phase.parent, phase.installed
    phase.parent, phase.installed = parent, installed
    try:
        return phase.train(c)
    finally:
        phase.parent, phase.installed = old
