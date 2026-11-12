"""Fresh source-development trajectories for branch-prototype tuning."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import time

from experiments.cric_dura_comparison.io import read, write
from . import source_development, source_tuning_plan as plan
from .practical_gate import installed as practical_scope
from .source_prototype import SourcePrototype


@contextmanager
def ssl_scope(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    @contextmanager
    def adapted(cfg, state, enabled, objective_factory):
        with paper_scope(cfg, state, ssl_enabled=enabled, objective_factory=objective_factory,
                         validate=plan.validate):
            yield

    with practical_scope(c, holder, base_scope=adapted, objective_factory=SourcePrototype), \
            source_development.installed(c):
        yield


def train(c):
    from experiments.cric_dura_comparison.f_train import train as segment
    from experiments.pamf_paper_completion.model import installed as paper_scope

    plan.validate(c)
    if any(c.get(key) for key in ('parent_source', 'shared_prefix', 'initial_checkpoint', 'target_inputs', 'truth')):
        raise ValueError('Source development training must start from scratch')
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    previous, segments, states = None, [], []
    start = time.monotonic()
    for epochs, offset in ((20, 0), (10, 20), (20, 30)):
        folder = root / f'segment{offset + epochs}'
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset * 222, parent_source=previous)
        holder = {}
        if offset == 0:
            with paper_scope(cfg, holder, ssl_enabled=False, validate=plan.validate), source_development.installed(cfg):
                segment(cfg)
        else:
            with ssl_scope(cfg, holder):
                segment(cfg)
        done = read(folder / 'complete.json')
        if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != epochs * 222
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']):
            raise ValueError('Source development segment incomplete or unauthorized')
        source_development.verify_parent_partition(c, done['config'])
        if offset:
            state = holder['objective'].state_dict()
            if (state['calls'] != epochs * 222 or state['EMA_updates'] or state['U_truth_read']
                    or state['student_views'] != 6 * state['seen']):
                raise ValueError('Source development SSL execution differs')
            write(folder / 'repl_state.json', state)
            states.append(state)
        segments.append(dict(folder=str(folder), **done))
        previous = str(folder)
        write(root / 'progress.json', dict(status='SEGMENT_COMPLETE', epoch=offset + epochs,
                                         updates=(offset + epochs) * 222), exclusive=False)
    shutil.copyfile(Path(previous) / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', read(c['source_contract']))
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100, config=c,
          segments=segments, initialization='same_contract_scratch20_own10_own20',
          ancestors=[s['folder'] for s in segments], source_contract=c['source_contract'],
          source_development_only=True, target_access=False, U_truth_read=False, head_fit_updates=0,
          actual_REPL={k: sum(s[k] for s in states) for k in ('calls', 'seen', 'accepted', 'student_views')},
          seconds=time.monotonic() - start, cumulative_training_seconds=sum(s['seconds'] for s in segments)))
