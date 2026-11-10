from contextlib import contextmanager
from experiments.pamf_tri_u_phase import objective as phase

_original_consistency=phase.branch_consistency


def subset_consistency(teacher,student,accepted,names):
    if not names or len(names)!=len(set(names)) or any(k not in teacher or k not in student for k in names):
        raise ValueError('Registered actual branch subset required')
    return _original_consistency({k:teacher[k] for k in names},{k:student[k] for k in names},accepted)


@contextmanager
def installed(c,holder):
    from .plan import ARMS
    names=tuple(c['phase_subset'])
    if c.get('algorithm_id') not in ARMS or names!=ARMS[c['algorithm_id']]:
        raise ValueError('Fixed branch subset must match registered arm')
    previous=phase.branch_consistency
    phase.branch_consistency=lambda teacher,student,accepted:subset_consistency(teacher,student,accepted,names)
    try:
        with phase.installed(c,holder):yield
    finally:phase.branch_consistency=previous
