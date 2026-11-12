"""10-class source pilot for a three-of-five branch certificate."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time

from experiments.cric_dura_comparison.io import read, write
from . import source_tuning_plan as plan

RUN = '20261112-phase1-pamf-prototype-gate3-source-s2026092701-r01'
BASELINE_RUN = '20261112-phase1-pamf-branch-prototype-source-tuning-s2026092701-r01'
BASELINE_ROW = 'ten-class-prototype-temp-020-s2026092701'


def rows():
    return [dict(row_id='ten-class-prototype-gate3-s2026092701', setting='ten_class',
                 protocol_id='ManyRx', model_seed=2026092701, temperature=.2, required_votes=3)]


@contextmanager
def configured():
    original = plan.RUN, plan.candidate_rows, plan.configuration
    old_configuration = plan.configuration
    plan.RUN, plan.candidate_rows = RUN, rows

    def configuration(context, row, project, commit, partition):
        cfg = old_configuration(context, row, project, commit, partition)
        cfg['shared_prefix'] = str(Path(project) / 'runs' / BASELINE_RUN / 'F' / BASELINE_ROW / 'source' / 'segment20')
        plan.validate(cfg)
        return cfg

    plan.configuration = configuration
    try:
        yield
    finally:
        plan.RUN, plan.candidate_rows, plan.configuration = original


def parent(cfg):
    from experiments.pamf_branch_training.source import checked_e20_parent
    from .source_development import verify_parent_partition
    plan.validate(cfg)
    root = Path(cfg['output_root'])
    expected = root.parents[3] / BASELINE_RUN / 'F' / BASELINE_ROW / 'source' / 'segment20'
    if Path(cfg['shared_prefix']) != expected:
        raise ValueError('Only the matching source-development E20 prefix is permitted')
    evidence = checked_e20_parent(cfg)
    verify_parent_partition(cfg, evidence['config'])
    return evidence


def analyze(rt):
    from .source_development import verify_parent_partition
    root = Path(rt['run_root'])
    baseline = Path(rt['project']) / 'runs' / BASELINE_RUN
    row = rows()[0]
    source = root / 'F' / row['row_id'] / 'source'
    original_source = baseline / 'F' / BASELINE_ROW / 'source'
    verify_parent_partition(read(source / 'complete.json')['config'],
                            read(original_source / 'complete.json')['config'])
    new = read(root / 'source_scores' / row['row_id'] / 'result.json')
    old = read(baseline / 'source_scores' / BASELINE_ROW / 'result.json')
    if new['count'] != old['count']:
        raise ValueError('Matching source-development population required')
    delta = ((new['clean_accuracy'] + new['physical_accuracy']) -
             (old['clean_accuracy'] + old['physical_accuracy'])) / 2
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    write(out / 'complete.json', dict(status='SOURCE_SELECTION_COMPLETE', candidate=new, baseline=old,
          source_score_improvement=delta, source_pilot_pass=delta >= .0025,
          promotion_rule='source clean/physical equal mean improves >=0.25 percentage points over four votes',
          target_reads=0, target_feedback=False))


def dispatch(rt):
    from . import source_tuning_runner as runner
    baseline = Path(rt['project']) / 'runs' / BASELINE_RUN / 'pipeline_state.json'
    while True:
        state = read(baseline)
        if state['status'] == 'ANALYZED':
            break
        if state['status'].startswith(('FAILED', 'PARTIAL')) or not (Path('/proc') / str(state['pid'])).exists():
            raise RuntimeError('Source-development baseline did not complete; preserve queued candidate')
        print('WAITING_FOR_MATCHED_SOURCE_BASELINE', flush=True)
        time.sleep(30)
    old = runner.analyze
    runner.analyze = analyze
    try:
        return runner.dispatch(rt)
    finally:
        runner.analyze = old


def worker(cfg, kind):
    from . import source_tuning_runner as runner
    if kind != 'train':
        return runner.worker(cfg, kind)
    from experiments.pamf_repl_completion.runner import activate
    from .source import train_trajectory
    from .source_tuning import ssl_scope
    activate(cfg)
    return train_trajectory(cfg, parent(cfg), ssl_scope, ssl=True, physical_L=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with configured():
        config = read(args.config)
        dispatch(config) if args.action == 'dispatch' else worker(config, args.kind)
