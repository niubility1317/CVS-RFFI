"""Frozen setting-specific recipes and two held-seed ten-class continuations."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_adaptive_kl import plan as adaptive
from experiments.pamf_repl_day_pairs import plan as days
from experiments.pamf_repl_two_setting import plan as two
from . import full_suite, receiver_balance as balance, transfer


RUN = '20261120-phase1-pamf-tri-setting-profiles-clean-m33-r01'
MODULE = 'experiments.pamf_repl_source_adapt.setting_profiles'
GPUS = full_suite.GPUS
paper = transfer.paper
TEN = 'ten_class_vote_hard1'
HYBRID = '20261112-phase1-pamf-tri-hybrid-contrast-m9-r01'
FUSION6 = ('receiver_split_1', 'labeled_5pct')


def matrix():
    for setting, (protocol, _, _) in paper.PROTOCOLS.items():
        for seed in paper.SEEDS:
            prior = {}
            method, weight = 'original_tri', 6.
            if setting in FUSION6:
                method = 'fusion_rx_balance'
                prior = dict(reuse_run=balance.RUN if seed == balance.SEED else balance.CONFIRM_RUN,
                             reuse_row=f'{setting}-tri-balance-joint-s{seed}')
            elif setting == 'receiver_split_2':
                method, weight = 'fusion_rx_balance', 3.
                prior = dict(reuse_run=balance.DOSE_RUN if seed == balance.SEED else balance.DOSE_CONFIRM_RUN,
                             reuse_row=f'{setting}-tri-fusion-w3-s{seed}')
            elif setting == 'ten_class':
                method, weight = TEN, .75
                if seed == adaptive.SEED:
                    prior = dict(reuse_run=adaptive.RUN, reuse_row=f'{setting}-repl-{TEN}-s{seed}')
            elif setting == 'source_rx_1':
                method, weight = 'hybrid_fusion_branch', None
                prior = dict(reuse_run=HYBRID, reuse_row=f'{setting}-tri-hybrid-fusion_branch-s{seed}')
            elif setting == 'source_rx_6':
                method, weight = 'raw_kl_same_day_fusion', .75
                prior = (dict(reuse_run=days.RUN, reuse_row=f'{setting}-repl-source_rx_6_same_date50-s{seed}')
                         if seed == adaptive.SEED else dict(reuse_run=two.RUN, reuse_row=f'{setting}-repl-six_same_date-s{seed}'))
            row = dict(row_id=f'{setting}-tri-profile-s{seed}', setting=setting, protocol_id=protocol,
                       model_seed=seed, method=method, weight=weight, views=['clean'],
                       reuses_original=method == 'original_tri', **prior)
            if method == TEN:
                row['recipe'] = TEN
            yield row


def candidate_rows():
    return [row for row in matrix() if not row['reuses_original'] and not row.get('reuse_run')]


@contextmanager
def scoped():
    from experiments.pamf_repl_priority import plan as priority

    old = priority.RUN, priority.SETTINGS, priority.RECIPES, priority.candidate_rows
    priority.RUN, priority.SETTINGS = RUN, {'ten_class': 'ManyRx'}
    priority.RECIPES, priority.candidate_rows = {TEN: adaptive.RECIPES[TEN]}, candidate_rows
    try:
        with priority.scoped():
            yield
    finally:
        priority.RUN, priority.SETTINGS, priority.RECIPES, priority.candidate_rows = old


def verify_recipe(row, config):
    if (config['run_id'], config['row_id']) != (row['reuse_run'], row['reuse_row']):
        raise ValueError('Readout row does not match the frozen profile')
    method = row['method']
    if method == 'fusion_rx_balance':
        match = (config['hard_pl']['threshold'] == .95
                 and config['hard_pl']['weight'] == config['hard_pl_gradient_guard']['weight'] == row['weight']
                 and config['source_adaptation'] in balance.rows(row['reuse_run'])
                 and config['source_adaptation']['route'] == 'joint')
    elif method == TEN:
        match = config['tuning_recipe'] == adaptive.RECIPES[TEN]
    elif method == 'raw_kl_same_day_fusion':
        match = config['tuning_recipe'] == two.RECIPES['six_same_date']
    else:
        match = (config['hybrid_contrast'] == dict(mode='fusion_branch', temperature=.2)
                 and config['practical_gate'] == dict(mode='branch', strength=.25, schedule='taper')
                 and config['fused_physics'] == dict(weight='original_joint', raw_detached=True)
                 and config['day_relation_weight'] == .025 and config.get('source_prototype') is None)
    if not match:
        raise ValueError(f"Historical recipe differs for {row['row_id']}")


def prepare(rt):
    from experiments.pamf_repl_component_guard import plan as base
    from experiments.pamf_repl_step_guard.runner import add_u_scoring

    root, project = Path(rt['run_root']), Path(rt['project'])
    context = read(rt['context'])
    native = read(project/'runs'/paper.RUN/'sources.json')
    rows, reused, catalogs = list(matrix()), {}, {}
    for row in rows:
        pid, seed = row['protocol_id'], row['model_seed']
        native_cfg = full_suite.frozen_source(native[transfer.original_row(row)], row, context)
        if native_cfg['hard_pl']['weight'] != 6.:
            raise ValueError('Unexpected original Tri U weight')
        full_suite.frozen_source(context['existing'][f'{pid}-ordinary_PL-s{seed}'], row, context)
        history = row.get('reuse_run')
        if history:
            if history not in catalogs:
                folder = project/'runs'/history
                if read(folder/'pipeline_state.json')['status'] != 'ANALYZED':
                    raise ValueError(f'Require completed history {history}')
                catalogs[history] = read(folder/'sources.json')
            artifact = catalogs[history][row['reuse_row']]
            verify_recipe(row, full_suite.frozen_source(artifact, row, context))
            reused[row['row_id']] = artifact
    jobs = transfer.prepare_jobs(rt, rows, base.configuration, reused_sources=reused)
    return add_u_scoring(rt, candidate_rows(), jobs)


def analyze(rt):
    transfer.analyze(rt, interpretation='Frozen setting profiles, incl. Hybrid for single source; historical clean target',
                     source_decision_run=None)
    folder = Path(rt['run_root'])/'analysis'
    rows = read(folder/'results.json')['rows']
    summary = full_suite.summarize(rows)
    for row in rows:
        summary[row['setting']].update(method=row['method'], U_weight=row['weight'])
    write(folder/'summary_by_setting.json', summary)


def main():
    from experiments.pamf_repl_adaptive_kl.objective import installed
    from experiments.pamf_repl_component_guard import runner as base
    from experiments.pamf_repl_priority import runner as priority

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'worker', 'publish'))
    parser.add_argument('--kind')
    for option in ('config', 'repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option)
    args = parser.parse_args()
    if args.action == 'publish':
        from .transfer_publish import publish
        publish(args.repo, args.output, args.context, args.ssh_config,
                run_id=RUN, module=MODULE, decision_run=None, gpus=GPUS)
        return
    config = read(args.config)
    with scoped():
        old = base.installed, priority.installed
        base.installed = priority.installed = installed
        try:
            if args.action == 'dispatch':
                transfer.dispatch(config, prepare_fn=prepare, analyze_fn=analyze, module=MODULE)
            else:
                priority.worker(config, args.kind)
        finally:
            base.installed, priority.installed = old


if __name__ == '__main__':
    main()
