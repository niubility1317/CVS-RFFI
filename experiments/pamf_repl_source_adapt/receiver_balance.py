"""Use source RX support to reweight only one native Tri contrast route."""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin
from experiments.pamf_ssl_research import hard_pl_joint_objective as joint, hard_pl_objective as hard
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from . import transfer
from .pilot import paper


RUN = '20261119-phase1-pamf-repl-rxbalance-clean-m8-r01'
CONFIRM_RUN = '20261119-phase1-pamf-repl-rxbalance-seedcheck-clean-m8-r02'
DOSE_RUN = '20261119-phase1-pamf-repl-rxbalance-dose-clean-m12-r01'
DOSE_CONFIRM_RUN = '20261119-phase1-pamf-repl-rxbalance-dose-seedcheck-clean-m2-r02'
DOSE_RUNS = (DOSE_RUN, DOSE_CONFIRM_RUN)
DOSES = (5.25, 4.5, 3.)
MODULE = 'experiments.pamf_repl_source_adapt.receiver_balance'
SETTINGS = ('receiver_split_1', 'receiver_split_2', 'labeled_5pct', 'source_rx_3')
SEED = 2026092702
ROUTES = {'shallow': hard, 'joint': joint}


@torch.no_grad()
def positive_target(labels, rx):
    same_label = labels[:, None] == labels[None, :]
    same_rx = rx[:, None] == rx[None, :]
    positives = same_label & ~same_rx
    uniform = positives.float() / positives.sum(1, keepdim=True).clamp_min(1)
    # Each positive receives equal mass within its receiver and each supported
    # receiver gets equal total mass before mixing with the original target.
    receivers = positives.float() @ same_rx.float()
    per_rx = positives / receivers.clamp_min(1.)
    balanced = per_rx / per_rx.sum(1, keepdim=True).clamp_min(1.)
    mixed = .5 * (uniform + balanced)
    valid = positives.any(1) & (~same_label).any(1)
    deviation = .5 * (mixed - uniform).abs().sum(1)
    affected = valid & (deviation > 1e-6)
    valid_count = valid.sum().clamp_min(1)
    metrics = dict(affected=affected.sum()/valid_count, tv=(deviation*valid).sum()/valid_count)
    return (mixed if bool(affected.any()) else None), metrics


class BalancedREPL(GuardedReceiverExtrapolation):
    def loss(self, student, device):
        self.receiver_balance = None
        loss = super().loss(student, device)
        stats = self.receiver_balance or dict(affected=0., tv=0.)
        self.metrics.update({f'HPL_balance_{key}': value for key, value in stats.items()})
        self.receiver_balance = None
        return loss


class ObservedBalancedREPL(PseudoObservationMixin, BalancedREPL):
    pass


def rows(run_id=RUN):
    if run_id in DOSE_RUNS:
        settings, seeds, weights = (SETTINGS, (SEED,), DOSES) if run_id == DOSE_RUN else (
            ('receiver_split_2',), (2026092701, 2026092703), (3.,))
        return [dict(row_id=f"{name}-tri-fusion-w{format(weight, 'g').replace('.', 'p')}-s{seed}", setting=name,
                     protocol_id=paper.PROTOCOLS[name][0], model_seed=seed, weight=weight,
                     route='joint', views=['clean'], reuses_original=False)
                for name in settings for weight in weights for seed in seeds]
    seeds, routes = {RUN: ((SEED,), tuple(ROUTES)), CONFIRM_RUN: ((2026092701, 2026092703), ('joint',))}[run_id]
    return [dict(row_id=f'{name}-tri-balance-{route}-s{seed}', setting=name, protocol_id=paper.PROTOCOLS[name][0],
                 model_seed=seed, weight=6., route=route, views=['clean'], reuses_original=False)
            for name in SETTINGS for route in routes for seed in seeds]


def validate(c):
    row = c.get('source_adaptation')
    run_id = c['run_id']
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if (row not in rows(run_id) or c['row_id'] != row['row_id'] or c['model_seed'] != row['model_seed']
            or root.parts[-5:] != ('runs', run_id, 'F', row['row_id'], 'source')
            or c['protocol_id'] != row['protocol_id'] or c['paper_row']['row_id'] != transfer.original_row(row)
            or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['hard_pl']['weight'] != row['weight'] or c['hard_pl_gradient_guard']['weight'] != row['weight']
            or c['hard_pl']['threshold'] != .95
            or any(c.get(k) for k in ('tests', 'target_inputs', 'target_truth', 'truth', 'views_root', 'source_fold'))):
        raise ValueError('Expected native Tri with one registered RX balancing route')


def configure(ctx, row, project, commit, *, run_id=RUN):
    cfg = transfer.source_configuration(ctx, row, project, commit, run_id)
    validate(cfg)
    return cfg


@contextmanager
def installed(config, holder, *, validate_config=validate):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_repl_view_shuffle.source import installed as shuffled_scope

    module = ROUTES[config['source_adaptation']['route']]
    original = module.contrast

    def balance(maps, labels, receivers):
        distribution, stats = positive_target(labels, receivers)
        holder['objective'].receiver_balance = stats
        return original(maps, labels, receivers, positive_distribution=distribution)

    module.contrast = balance
    objective_class = ObservedBalancedREPL if config['run_id'] in DOSE_RUNS else BalancedREPL
    try:
        if config['source_adaptation']['setting'] == 'main_split':
            validate_config(config)
            scope = shuffled_scope(config, holder, True, objective_factory=objective_class)
        else:
            scope = paper_scope(config, holder, ssl_enabled=True, validate=validate_config,
                                objective_factory=objective_class)
        with scope:
            try:
                yield
            finally:
                obj = holder.get('objective')
                if isinstance(obj, ObservedBalancedREPL):
                    observed = obj.close_pseudo_observation()
                    if not config.get('hard_pl_smoke') and (observed['calls'], observed['seen'], observed['accepted']) != (obj.calls, obj.seen, obj.accepted):
                        raise ValueError('Blind U observation coverage must match native REPL')
    finally:
        module.contrast = original


parent = partial(transfer.parent, validate_config=validate)


def smoke(rt):
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph
    from experiments.pamf_physics_tri.runner import smoke as standard

    run_id = rt['run_id']
    standard(rt, rows=lambda: rows(run_id), configure=partial(configure, run_id=run_id), check_parent=parent, scope=installed,
             check_u=lambda state: dict(accepted=state['accepted']), graph_check=check_graph, segment_output=True)


def prepare(rt):
    run_id = rt['run_id']
    configured = rows(run_id)
    jobs = transfer.prepare_jobs(rt, configured, partial(configure, run_id=run_id))
    if run_id not in DOSE_RUNS:
        return jobs
    from experiments.pamf_repl_step_guard.runner import add_u_scoring

    root = Path(rt['run_root'])
    prior = Path(rt['project'])/'runs'/(RUN if run_id == DOSE_RUN else CONFIRM_RUN)
    if read(prior/'pipeline_state.json')['status'] != 'ANALYZED':
        raise ValueError('Require the completed matched fusion weight-six run')
    by_id = read(prior/'sources.json')
    controls = read(root/'controls.json')
    for row in configured:
        item = by_id[f"{row['setting']}-tri-balance-joint-s{row['model_seed']}"]
        record = read(Path(item['source_output'])/'complete.json')
        recipe = record['config']['source_adaptation']
        if (record['status'] != 'SOURCE_FROZEN' or record['epoch'] != 50 or record['updates'] != 11100
                or record['target_access'] or record['U_truth_read'] or recipe['route'] != 'joint' or recipe['weight'] != 6.
                or any(recipe[key] != row[key] for key in ('setting', 'protocol_id', 'model_seed'))):
            raise ValueError('Need an exact frozen fusion weight-six control')
        controls[row['row_id']]['fusion6'] = item
    write(root/'controls.json', controls, exclusive=False)
    return add_u_scoring(rt, configured, jobs)


def dispatch(rt):
    interpretation = {RUN: 'exploratory matched receiver balance within original loss',
        CONFIRM_RUN: 'fixed fusion receiver balance on two independent training seeds',
        DOSE_RUN: 'matched global U dose with fusion RX balance held fixed',
        DOSE_CONFIRM_RUN: 'fixed receiver-split-2 U weight 3 on two independent training seeds'}[rt['run_id']]
    analyze = partial(transfer.analyze, interpretation=interpretation, source_decision_run=None)
    transfer.dispatch(rt, prepare_fn=prepare, analyze_fn=analyze, module=MODULE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'worker'))
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = read(args.config)
    if args.action == 'dispatch':
        dispatch(c)
    elif args.kind == 'score' and c.get('diagnostic_type') == 'U_truth_after_freeze':
        from experiments.pamf_repl_step_guard.u_scoring import score
        score(c)
    elif args.kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        score(c)
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if args.kind == 'smoke':
            smoke(c)
        elif args.kind == 'train':
            from experiments.pamf_physics_tri.source import train_trajectory
            train_trajectory(c, parent(c), installed, ssl=True, physical_L=False)
        elif args.kind == 'predict':
            from experiments.pamf_physics_tri.development_evaluation import predict
            predict(c)
        else:
            parser.error('Unknown registered stage')
