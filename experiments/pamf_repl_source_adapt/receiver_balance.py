"""Use source RX support to reweight only one native Tri contrast route."""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research import hard_pl_joint_objective as joint, hard_pl_objective as hard
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from . import transfer
from .pilot import paper


RUN = '20261119-phase1-pamf-repl-rxbalance-clean-m8-r01'
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


def rows():
    return [dict(row_id=f'{name}-tri-balance-{route}-s{SEED}', setting=name, protocol_id=paper.PROTOCOLS[name][0],
                 model_seed=SEED, weight=6., route=route, views=['clean'], reuses_original=False)
            for name in SETTINGS for route in ROUTES]


def validate(c):
    row = c.get('source_adaptation')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if (row not in rows() or c['run_id'] != RUN or c['row_id'] != row['row_id'] or c['model_seed'] != SEED
            or root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source')
            or c['protocol_id'] != row['protocol_id'] or c['paper_row']['row_id'] != transfer.original_row(row)
            or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['hard_pl']['weight'] != 6. or c['hard_pl_gradient_guard']['weight'] != 6.
            or c['hard_pl']['threshold'] != .95
            or any(c.get(k) for k in ('tests', 'target_inputs', 'target_truth', 'truth', 'views_root', 'source_fold'))):
        raise ValueError('Expected native Tri with one registered RX balancing route')


def configure(ctx, row, project, commit):
    cfg = transfer.source_configuration(ctx, row, project, commit, RUN)
    validate(cfg)
    return cfg


@contextmanager
def installed(config, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    module = ROUTES[config['source_adaptation']['route']]
    original = module.contrast

    def balance(maps, labels, receivers):
        distribution, stats = positive_target(labels, receivers)
        holder['objective'].receiver_balance = stats
        return original(maps, labels, receivers, positive_distribution=distribution)

    module.contrast = balance
    try:
        with paper_scope(config, holder, ssl_enabled=True, validate=validate, objective_factory=BalancedREPL):
            yield
    finally:
        module.contrast = original


parent = partial(transfer.parent, validate_config=validate)


def smoke(rt):
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph
    from experiments.pamf_physics_tri.runner import smoke as standard

    standard(rt, rows=rows, configure=configure, check_parent=parent, scope=installed,
             check_u=lambda state: dict(accepted=state['accepted']), graph_check=check_graph, segment_output=True)


def prepare(rt):
    return transfer.prepare_jobs(rt, rows(), configure)


def dispatch(rt):
    analyze = partial(transfer.analyze, interpretation='exploratory matched receiver balance within original loss', source_decision_run=None)
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
