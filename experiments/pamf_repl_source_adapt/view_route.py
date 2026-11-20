"""Adapt U classification supervision to observed labeled cell coverage."""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin
from experiments.pamf_ssl_research import hard_pl_joint_objective as joint, hard_pl_objective as hard
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from . import full_suite, receiver_balance as balance, transfer


RUN = '20261121-phase1-pamf-tri-label01-viewroute-clean-m7-r01'
MODULE = 'experiments.pamf_repl_source_adapt.view_route'
SEED = 2026092701
ROUTES = (('fixed', .25), ('adaptive', .5), ('adaptive', 1.))


def rows():
    arms = [('none', 0., True)] + [(mode, strength, rx) for mode,strength in ROUTES for rx in (False,True)]
    return [dict(row_id=f'labeled_1pct-tri-route{index}-s{SEED}', setting='labeled_1pct', protocol_id='label01',
                 model_seed=SEED, weight=6., views=['clean'], reuses_original=False,
                 mode=mode, strength=strength, fusion_balance=rx)
            for index,(mode,strength,rx) in enumerate(arms)]


def validate(c):
    row = c.get('source_adaptation')
    folder = Path(c['output_root'])
    if folder.name in ('segment30','segment50'):
        folder = folder.parent
    if (row not in rows() or c['run_id'] != RUN or c['row_id'] != row['row_id']
            or c['protocol_id'] != 'label01' or c['model_seed'] != SEED
            or c['paper_row']['row_id'] != transfer.original_row(row)
            or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['hard_pl']['weight'] != 6. or c['hard_pl_gradient_guard']['weight'] != 6.
            or c['hard_pl']['threshold'] != .95
            or folder.parts[-5:] != ('runs',RUN,'F',row['row_id'],'source')
            or any(c.get(k) for k in ('target_truth','target_inputs','tests','truth','views_root','source_fold','checkpoint'))):
        raise ValueError('Only the registered source-only label01 arm is allowed')


def configure(ctx, row, project, commit):
    c = transfer.source_configuration(ctx,row,project,commit,RUN)
    validate(c)
    return c


parent = partial(transfer.parent, validate_config=validate)


@torch.no_grad()
def coefficients(count, raw_probability, mixed_probability, accepted, mode, strength):
    if mode == 'fixed':
        return accepted * strength
    scarcity = 14 / (count.float() + 14)
    drop = ((raw_probability-mixed_probability)*4).clamp(0,1)
    return (.5 * strength) * scarcity * drop * accepted


def transfer_ce(m0,m1,r0,r1,pseudo,accepted,count,mode,strength):
    index = pseudo[:,None]
    with torch.no_grad():
        raw = .5*(r0.float().softmax(1).gather(1,index)+r1.float().softmax(1).gather(1,index))
        mixed = .5*(m0.float().softmax(1).gather(1,index)+m1.float().softmax(1).gather(1,index))
        mass = coefficients(count,raw.squeeze(1),mixed.squeeze(1),accepted,mode,strength)

    def ce(a,b):
        return .5*(F.cross_entropy(a,pseudo,reduction='none',label_smoothing=.01)
                   +F.cross_entropy(b,pseudo,reduction='none',label_smoothing=.01))

    # Both augmented halves originally contribute .25 times their sub-batch
    # means, equivalent to .5 times an all-batch mean.
    delta = (mass * (ce(r0,r1)-ce(m0,m1))).mean()/1.25
    return delta, mass


class ViewRoute(PseudoObservationMixin, GuardedReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.route = c['source_adaptation']
        self.label_support = None
        self.support = None
        self.routed = 0
        if self.route['mode'] == 'adaptive':
            with np.load(Path(c['source_cache'])/'L_s_metadata.npz', allow_pickle=False) as f:
                axes = tuple(torch.tensor(f[k].tolist(),dtype=torch.long) for k in ('rx_i','day_i','y'))
            table = torch.zeros(max(c['source_rxs'])+1,max(c['source_days'])+1,c['num_classes'],dtype=torch.long)
            table.index_put_(axes,torch.ones_like(axes[0]),accumulate=True)
            self.label_support = table

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        if self.label_support is not None:
            self.label_support = self.label_support.to(x.device)
            self.support = self.label_support[metadata['rx_i'].long(),metadata['day_i'].long(),pseudo]
        return super().mixed_view(student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted)

    def state_dict(self):
        out = super().state_dict()
        if self.support is not None or self.routed != self.calls:
            raise ValueError('Each native U view batch must be routed once')
        out['U_view_route'] = dict(calls=self.routed,recipe=self.route)
        return out


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    orig_loss, orig_contrast = hard.mechanism_loss, joint.contrast
    recipe = c['source_adaptation']

    def route(o0,o1,m0,m1,r0,r1,pseudo,accepted,physical,aux):
        base, details = orig_loss(o0,o1,m0,m1,r0,r1,pseudo,accepted,physical,aux)
        obj = holder['objective']
        obj.routed += 1
        if recipe['mode'] == 'none':
            return base, details
        support, obj.support = obj.support, None
        correction, mass = transfer_ce(m0,m1,r0,r1,pseudo,accepted,support,recipe['mode'],recipe['strength'])
        details.update(route_mass=mass.sum()/accepted.sum().clamp_min(1),
                       route_active=(mass>0).sum(), route_delta=correction.detach())
        return base+correction, details

    def reweight(maps,labels,rx):
        distribution, stats = balance.positive_target(labels,rx)
        holder['objective'].metrics.update({f'HPL_balance_{k}':v for k,v in stats.items()})
        return orig_contrast(maps,labels,rx,positive_distribution=distribution)

    hard.mechanism_loss = route
    if recipe['fusion_balance']:
        joint.contrast = reweight
    try:
        with paper_scope(c, holder, ssl_enabled=True, validate=validate, objective_factory=ViewRoute):
            yield
            obj = holder['objective']
            if not c.get('hard_pl_smoke'):
                obs = obj.close_pseudo_observation()
                if (obs['calls'],obs['seen'],obs['accepted']) != (obj.calls,obj.seen,obj.accepted):
                    raise ValueError('Blind U observations must match training')
    finally:
        hard.mechanism_loss, joint.contrast = orig_loss, orig_contrast


def prepare(rt):
    from experiments.pamf_repl_step_guard.runner import add_u_scoring
    candidates = rows()
    return add_u_scoring(rt,candidates,transfer.prepare_jobs(rt,candidates,configure))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('publish','dispatch','worker'))
    parser.add_argument('--kind')
    for name in ('repo','output','context','ssh-config','config'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.action == 'publish':
        from .transfer_publish import publish
        publish(args.repo,args.output,args.context,args.ssh_config,run_id=RUN,module=MODULE,
                decision_run=None,gpus=full_suite.GPUS)
        return
    c = read(args.config)
    if args.action == 'dispatch':
        analyze = partial(transfer.analyze, interpretation='Fixed label01 exploration on historical clean',source_decision_run=None)
        transfer.dispatch(c, prepare_fn=prepare, analyze_fn=analyze, module=MODULE)
    elif args.kind == 'score':
        from experiments.pamf_repl_priority.runner import worker
        worker(c,'score')
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if args.kind == 'smoke':
            from experiments.pamf_physics_tri.full_matrix_plan import check_graph
            from experiments.pamf_physics_tri.runner import smoke
            smoke(c,rows=rows,configure=configure,check_parent=parent,scope=installed,graph_check=check_graph,
                  check_u=lambda state:state['U_view_route'],segment_output=True)
        elif args.kind == 'train':
            from experiments.pamf_physics_tri.source import train_trajectory
            train_trajectory(c,parent(c),installed,ssl=True,physical_L=False)
        elif args.kind == 'predict':
            from experiments.pamf_physics_tri.development_evaluation import predict
            predict(c)
        else:
            parser.error('Unknown stage')


if __name__ == '__main__':
    main()
