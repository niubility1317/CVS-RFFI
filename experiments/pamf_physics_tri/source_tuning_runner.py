"""Source-only pilot; target scoring is deliberately not part of this run."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import source_tuning_plan as plan


def prepare(rt):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from .source_development import split_labeled_ids

    root, ctx = Path(rt['run_root']), read(rt['context'])
    rows = plan.candidate_rows()
    protocols = {row['protocol_id'] for row in rows}
    if len(protocols) != 1:
        raise ValueError('One source data configuration per pilot is required')
    handle = ctx['handles'][next(iter(protocols))]
    source = SourceDataset(handle['source_cache'], 'L_s', read(handle['source_contract']))
    train_ids, dev_ids = split_labeled_ids(source.metadata['ids'], source.metadata['y'], seed=plan.SPLIT_SEED)
    partition = dict(train_ids=train_ids, dev_ids=dev_ids, seed=plan.SPLIT_SEED)
    jobs, sources = [], {}
    for row in rows:
        cfg = plan.configuration(ctx, row, rt['project'], rt['code_commit'], partition)
        cfg['frozen_sources'] = rt['frozen_sources']
        rid, folder = row['row_id'], Path(cfg['output_root'])
        cp = root / 'configs' / f'train__{rid}.json'
        write(cp, cfg)
        result = root / 'source_scores' / rid
        score_config = root / 'configs' / f'source_score__{rid}.json'
        write(score_config, dict(run_id=plan.RUN, source_output=str(folder), output_root=str(result),
                                 frozen_sources=rt['frozen_sources']))
        jobs.append(dict(id=f'train__{rid}', kind='train', config=str(cp),
                         completion=str(folder / 'complete.json'), successful_statuses=['SOURCE_FROZEN'], dependencies=[]))
        jobs.append(dict(id=f'source_score__{rid}', kind='source_score', config=str(score_config),
                         completion=str(result / 'complete.json'), successful_statuses=['SOURCE_SCORE_COMPLETE'],
                         dependencies=[str(folder / 'complete.json')]))
        sources[rid] = dict(source_output=str(folder), source_score=str(result))
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs


def smoke(rt):
    from .full_matrix import EmptyCertifiedBatch
    from .full_matrix_plan import check_graph
    from .pair_runner import check_full_matrix
    from .runner import smoke as smoke_rows
    from .source_tuning import ssl_scope
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN

    root = Path(rt['run_root'])
    checks = []
    for row in plan.candidate_rows():
        cfg = read(root / 'configs' / f"train__{row['row_id']}.json")
        original = f"{row['setting']}-factorial-ssl-full-permuted-s{row['model_seed']}"
        parent = Path(rt['project']) / 'runs' / PAPER_RUN / 'F' / original / 'source' / 'segment20'

        def parent_check(c):
            # This historical checkpoint is for operational smoke only; it is never a training parent.
            return checked_e20_parent(dict(c, shared_prefix=str(parent)))

        for offset in range(10):
            try:
                item = smoke_rows(rt, rows=lambda: [row], configure=lambda *args: cfg,
                                  check_parent=parent_check, scope=ssl_scope, check_u=check_full_matrix,
                                  graph_check=check_graph, segment_output=True,
                                  labeled_batch_offset=offset, write_result=False)
            except EmptyCertifiedBatch:
                continue
            item[0]['source_labeled_batch_offset'] = offset
            checks.extend(item)
            break
        else:
            raise ValueError(f"No certified source-U packet in ten batches: {row['row_id']}")
    write(root / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          checkpoint_use='operational smoke only; excluded from development training', optimizer_updates=0,
          query_reads=0, U_truth_reads=0))


def worker(c, kind):
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        return smoke(c)
    if kind == 'train':
        from .source_tuning import train
        return train(c)
    if kind == 'source_score':
        from .source_tuning_selection import score_source
        out = Path(c['output_root'])
        if c['run_id'] != plan.RUN:
            raise ValueError('Only this registered source-development run is allowed')
        out.mkdir(parents=True, exist_ok=False)
        result = score_source(c['source_output'], 'cuda:0')
        write(out / 'result.json', result)
        write(out / 'complete.json', dict(status='SOURCE_SCORE_COMPLETE', target_reads=0))
        return
    raise ValueError('Only source training and source-development scoring allowed')


def analyze(rt):
    from .source_tuning_selection import select
    root = Path(rt['run_root'])
    sources = read(root / 'sources.json')
    results = [read(Path(sources[row['row_id']]['source_score']) / 'result.json') for row in plan.candidate_rows()]
    winner = select(results)
    scores = {r['row_id']: (r['clean_accuracy'] + r['physical_accuracy']) / 2 for r in results}
    original = next(r for r in plan.candidate_rows() if r['temperature'] == .2)
    improvement = scores[winner['row_id']] - scores[original['row_id']]
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    write(out / 'complete.json', dict(status='SOURCE_SELECTION_COMPLETE', winner=winner, results=results,
          source_score_improvement=improvement, source_pilot_pass=improvement >= .0025,
          promotion_rule='source clean/physical equal mean improves >=0.25 percentage points over temperature 0.2',
          target_reads=0, target_feedback=False))


def dispatch(rt):
    from experiments.pamf_branch_training import runner as base, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch

    def admits(gpu, active, required_mib):
        if any(item['gpu'] == gpu for _, item in active.values()):
            return False
        return physical_admits(gpu, active, required_mib, max_total=rt.get('max_total_processes_per_gpu', 1))

    old = preparation.prepare, base.admits, base.launch, base.analyze
    preparation.prepare, base.admits = prepare, admits
    base.launch = partial(launch, module=rt.get('worker_module', 'experiments.pamf_physics_tri.source_tuning_runner'))
    base.analyze = analyze
    try:
        return base.dispatch(rt)
    finally:
        preparation.prepare, base.admits, base.launch, base.analyze = old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    dispatch(config) if args.action == 'dispatch' else worker(config, args.kind)
