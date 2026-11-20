"""Compare one fixed, batch-adaptive Tri policy with PL on the full clean matrix."""
import argparse
from functools import partial
from pathlib import Path
from statistics import mean

from experiments.cric_dura_comparison.io import read, write
from . import receiver_balance as balance, transfer


RUN = '20261120-phase1-pamf-tri-adaptive-clean-m33-r01'
GPUS = (0, 4, 5, 6)
MODULE = 'experiments.pamf_repl_source_adapt.full_suite'
paper = transfer.paper
IDENTITY_SETTINGS = ('source_rx_1', 'source_rx_2')


def rows():
    result = []
    for setting, (protocol, _, _) in paper.PROTOCOLS.items():
        for seed in paper.SEEDS:
            history = (balance.RUN if seed == balance.SEED else balance.CONFIRM_RUN) if setting in balance.SETTINGS else None
            result.append(dict(row_id=f'{setting}-tri-adaptive-full-s{seed}', setting=setting, protocol_id=protocol,
                               model_seed=seed, route='joint', weight=6., views=['clean'],
                               reuses_original=setting in IDENTITY_SETTINGS, reuse_run=history))
    return result


def new_rows():
    return [row for row in rows() if not row['reuses_original'] and row['reuse_run'] is None]


def validate(c):
    row = c.get('source_adaptation')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if (row not in new_rows() or c['run_id'] != RUN or c['row_id'] != row['row_id']
            or c['model_seed'] != row['model_seed'] or c['protocol_id'] != row['protocol_id']
            or c['paper_row']['row_id'] != transfer.original_row(row) or c['epochs'] != 50
            or c['optimizer_updates'] != 11100 or len(c['source_rxs']) < 3
            or c['hard_pl']['threshold'] != .95 or c['hard_pl']['weight'] != 6.
            or c['hard_pl_gradient_guard']['weight'] != 6.
            or any(c.get(k) for k in ('tests', 'target_inputs', 'target_truth', 'truth',
                                      'views_root', 'source_fold', 'checkpoint'))
            or root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source')):
        raise ValueError('Only predeclared, same-contract native Tri with fusion RX balancing is allowed')


def configure(context, row, project, commit):
    config = transfer.source_configuration(context, row, project, commit, RUN)
    validate(config)
    return config


parent = partial(transfer.parent, validate_config=validate)
installed = partial(balance.installed, validate_config=validate)


def frozen_source(item, row, context):
    done = read(Path(item['source_output']) / 'complete.json')
    cfg = done['config']
    if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
            or done['target_access'] or done['U_truth_read']
            or cfg['model_seed'] != row['model_seed'] or cfg['protocol_id'] != row['protocol_id']
            or read(cfg['source_contract']) != read(context['handles'][row['protocol_id']]['source_contract'])
            or read(Path(item['predictions']) / 'complete.json')['status'] != 'ALL_PREDICTIONS_COMPLETE'
            or read(Path(item['scores']) / 'complete.json')['status'] != 'SCORED_COMPLETE'):
        raise ValueError(f"Matched frozen training and clean artifacts required for {row['row_id']}")
    return cfg


def prepare(rt):
    project = Path(rt['project'])
    context = read(rt['context'])
    catalog = read(project/'runs'/paper.RUN/'sources.json')
    history = {}
    for run_id in (balance.RUN, balance.CONFIRM_RUN):
        folder = project/'runs'/run_id
        if read(folder/'pipeline_state.json')['status'] != 'ANALYZED':
            raise ValueError(f'Require completed frozen reuse run {run_id}')
        history[run_id] = read(folder/'sources.json')
    matrix, reuse = rows(), {}
    for row in matrix:
        origin = catalog[transfer.original_row(row)]
        original_cfg = frozen_source(origin, row, context)
        pl = context['existing'][f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]
        frozen_source(pl, row, context)
        if row['reuses_original']:
            expected = 1 if row['setting'] == 'source_rx_1' else 2
            if len(set(original_cfg['source_rxs'])) != expected or original_cfg['hard_pl']['weight'] != 6.:
                raise ValueError('Exact identity for one or two source receivers required')
        elif row['reuse_run']:
            key = f"{row['setting']}-tri-balance-joint-s{row['model_seed']}"
            item = history[row['reuse_run']][key]
            cfg = frozen_source(item, row, context)
            if (cfg['run_id'] != row['reuse_run'] or cfg['row_id'] != key
                    or cfg['source_adaptation'] not in balance.rows(row['reuse_run'])
                    or cfg['source_adaptation']['route'] != 'joint'
                    or cfg['hard_pl']['weight'] != 6. or cfg['hard_pl_gradient_guard']['weight'] != 6.
                    or cfg['hard_pl']['threshold'] != .95):
                raise ValueError('Exact fusion-balance E50 recipe required for reuse')
            balance.parent(cfg)
            reuse[row['row_id']] = item
    return transfer.prepare_jobs(rt, matrix, configure, reused_sources=reuse)


def smoke(rt):
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph
    from experiments.pamf_physics_tri.runner import smoke as source_smoke

    source_smoke(rt, rows=new_rows, configure=configure, check_parent=parent, scope=installed,
                 check_u=lambda state: dict(accepted=state['accepted']), graph_check=check_graph,
                 segment_output=True)


def summarize(results):
    result = {}
    for setting in paper.PROTOCOLS:
        subset = [row for row in results if row['setting'] == setting]
        if tuple(row['model_seed'] for row in subset) != paper.SEEDS:
            raise ValueError('Every setting requires three matched seeds')
        result[setting] = {
            'clean_mean': {arm: mean(row['scores'][arm]['accuracy'] for row in subset)
                           for arm in ('adaptive', 'tri', 'pl')},
            'clean_paired_net': {arm: [row['paired'][arm]['net'] for row in subset] for arm in ('tri', 'pl')},
            'cross_both_paired_net': {arm: [row['by_test'][arm]['cross_both']['net'] for row in subset]
                                      for arm in ('tri', 'pl')},
            'cross_both_mean': {arm: mean(row['scores'][arm]['per_test']['cross_both'] for row in subset)
                                for arm in ('adaptive', 'tri', 'pl')},
        }
    return result


def analyze(rt):
    transfer.analyze(rt, interpretation='Fixed batch-adaptive fusion rule across eleven existing clean settings',
                     source_decision_run=None)
    folder = Path(rt['run_root'])/'analysis'
    write(folder/'summary_by_setting.json', summarize(read(folder/'results.json')['rows']))


def main():
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
    if args.action == 'dispatch':
        transfer.dispatch(config, prepare_fn=prepare, analyze_fn=analyze, module=MODULE)
    elif args.kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        score(config)
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(config)
        if args.kind == 'smoke':
            smoke(config)
        elif args.kind == 'train':
            from experiments.pamf_physics_tri.source import train_trajectory
            train_trajectory(config, parent(config), installed, ssl=True, physical_L=False)
        elif args.kind == 'predict':
            from experiments.pamf_physics_tri.development_evaluation import predict
            predict(config)
        else:
            parser.error('Unknown registered stage')


if __name__ == '__main__':
    main()
