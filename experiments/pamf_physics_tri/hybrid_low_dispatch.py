"""Dispatch the independently staged low-weight Hybrid queue."""
from experiments.cric_dura_comparison.io import read
from experiments.practical_full_2x2.cache import atomic
from . import concurrency_takeover as scheduler
from .hybrid_low_plan import RUN


def main():
    release = scheduler.PROJECT / 'releases' / (RUN + '-release')
    root = scheduler.PROJECT / 'runs' / RUN
    state = read(root / 'pipeline_state.json')
    if state['status'] != 'STAGED':
        raise ValueError('Only the independently staged low-weight queue may start')
    state['status'] = 'RUNNING'
    queue = dict(run=RUN, module='experiments.pamf_physics_tri.hybrid_low_runner', old_pid=None,
                 rt=read(release / 'runtime.json'), root=root, state=state, jobs=read(root / 'jobs.json'))
    scheduler.run([queue], read, atomic)


if __name__ == '__main__':
    main()
