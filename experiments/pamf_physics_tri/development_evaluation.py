"""Test the three frozen development-trained models without retraining."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import development_evaluation_plan as plan


def prepare(rt):
    root = Path(rt['run_root'])
    jobs, sources, configs = plan.make_jobs(root, rt['project'], read(rt['context']), rt['frozen_sources'])
    for job in jobs:
        write(job['config'], configs[job['id']])
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs


def predict(c):
    from experiments.pamf_single_view_paper import evaluation as native
    from experiments.pamf_paper_completion import evaluation as paper
    old = native.READOUTS
    native.READOUTS = ('raw',)
    try:
        return paper.predict(c)
    finally:
        native.READOUTS = old


def smoke(rt):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_paper_completion.model import load_reader

    row = plan.rows()[0]
    source = plan.source_path(row, rt['project'])
    done = read(source / 'complete.json')
    c = done['config']
    if not done.get('source_development_only') or done.get('target_access') or done.get('U_truth_read'):
        raise ValueError('Frozen source-development model required for no-query smoke')
    contract = read(c['source_contract'])
    torch.cuda.reset_peak_memory_stats()
    reader, _ = load_reader(source, contract, c['paper_row'], 'cuda:0')
    labeled = SourceDataset(c['source_cache'], 'L_s', contract)
    train_ids = set(c['source_development']['train_ids'])
    indices = [i for i, value in enumerate(labeled.metadata['ids']) if value in train_ids][:2]
    if len(indices) != 2:
        raise ValueError('Two source training records required for no-query smoke')
    with torch.inference_mode():
        logits = reader.classify_features(reader.features(labeled.x[indices].cuda()))
    if logits.shape != (2, len(contract['classes'])) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite frozen default-head output required')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=[dict(
          peak_allocated_bytes=torch.cuda.max_memory_allocated())], real_checkpoint=True,
          source_records=2, query_reads=0, model_updates=0))


def worker(c, kind):
    if kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        return smoke(c)
    if kind == 'predict':
        return predict(c)
    raise ValueError('Only frozen prediction and independent scoring are allowed')


def dispatch(rt):
    from experiments.pamf_branch_training import runner as base, prepare as preparation
    from experiments.pamf_physics_branch.runner import admits as physical_admits
    from experiments.pamf_physics_route.runner import launch, analyze

    def admits(gpu, active, required_mib):
        if any(item['gpu'] == gpu for _, item in active.values()):
            return False
        return physical_admits(gpu, active, required_mib, max_total=2)

    old = preparation.prepare, base.admits, base.launch, base.analyze
    preparation.prepare, base.admits = prepare, admits
    base.launch = partial(launch, module='experiments.pamf_physics_tri.development_evaluation')
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
    c = read(args.config)
    dispatch(c) if args.action == 'dispatch' else worker(c, args.kind)
