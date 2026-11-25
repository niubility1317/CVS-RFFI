"""Release B/C/D on the proven train → predict → independent score pipeline."""

import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_wisig_branch_style import runner as matched
from experiments.pamf_wisig_branch_style.summary import analyze

from .plan import ARMS, RUN, row_id, source_config

MODULE = 'experiments.pamf_wisig_preamble_bcd.runner'


def dispatch(runtime):
    if runtime['run_id'] != RUN:
        raise ValueError('Expected the registered WiSig B/C/D release')
    return matched.dispatch(
        runtime, preparer=partial(matched.prepare, arms=ARMS, run_id=RUN, make_row=row_id,
                                  make_source=source_config),
        launcher=partial(matched.launch, worker_module=MODULE),
        analyzer=partial(analyze, arms=ARMS, baseline_arm='B'), shared_memory=True, max_per_gpu=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', choices=['smoke', 'train', 'predict', 'score'])
    args = parser.parse_args()
    config = read(args.config)
    if args.action == 'dispatch':
        dispatch(config)
    else:
        matched.worker(config, args.kind, source_module=__package__ + '.source')
