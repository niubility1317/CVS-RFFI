"""Run the source-only pilot with the established balanced N607 dispatcher."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import pilot


def smoke(rt, *, fold_validator=pilot.registered_fold, setting=None):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_paper_completion.model import reader_scope

    root = Path(rt['run_root'])
    jobs = read(root/'jobs.json')
    for job in jobs:
        fold_validator(read(job['config']))
    # This exact-contract ordinary-PL E20 smoke never enters any fold or decision.
    setting = setting or ('receiver_split_2' if rt['run_id'] == pilot.SECOND_RUN else 'receiver_split_1')
    h = read(rt['context'])['handles'][pilot.paper.PROTOCOLS[setting][0]]
    folder = Path(rt['project'])/'runs'/pilot.paper.RUN/'F'/f'{setting}-factorial-ssl-full-permuted-s{pilot.SEED}'/'source'/'segment20'
    done = read(folder/'complete.json')
    c = done['config']
    contract = read(h['source_contract'])
    if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']
            or done['epoch'] != 20 or done['updates'] != 4440 or read(c['source_contract']) != contract):
        raise ValueError('Exact-contract original E20 required for the independent source smoke')
    payload = torch.load(folder/'final_ssdg.pth', map_location='cpu', weights_only=False)
    with reader_scope(c['paper_row']):
        model = reader_from_payload(c, payload, 'cuda:0')
    data = SourceDataset(h['source_cache'], 'L_s', contract, positions=np.arange(2))
    with torch.inference_mode():
        logits = model.classify_features(model.features(data.x.to('cuda:0')))
    if logits.shape != (2, c['num_classes']) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Original E20 failed actual-source no-query smoke')
    write(root/'smoke.json', dict(status='PASS', checks=[dict(peak_allocated_bytes=torch.cuda.max_memory_allocated())],
          real_checkpoint=True, folds=len(jobs), inherited_into_folds=False, query_reads=0, U_truth_reads=0))


def dispatch(rt):
    from experiments.pamf_branch_training import prepare, runner
    from experiments.pamf_physics_branch.runner import admits
    from experiments.pamf_physics_route.runner import launch

    saved = prepare.prepare, runner.admits, runner.launch, runner.analyze

    def capacity(gpu, active, mib):
        return admits(gpu, active, max(mib, 12288), max_total=2)

    prepare.prepare, runner.admits = pilot.prepare, capacity
    runner.launch = partial(launch, module='experiments.pamf_repl_source_adapt.runner')
    runner.analyze = pilot.analyze
    try:
        runner.dispatch(rt)
    finally:
        prepare.prepare, runner.admits, runner.launch, runner.analyze = saved


if __name__ == '__main__':
    from experiments.pamf_repl_completion.runner import activate

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'worker'))
    parser.add_argument('--kind', choices=('train', 'smoke'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if args.action == 'dispatch':
        dispatch(config)
    else:
        activate(config)
        if args.kind == 'smoke':
            smoke(config)
        elif args.kind == 'train':
            pilot.train_fold(config)
        else:
            parser.error('Worker requires --kind')
