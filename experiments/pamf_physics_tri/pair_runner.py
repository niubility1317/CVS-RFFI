"""Fixed semi-supervised six-row run, followed by truth-last practical scoring."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_route import prepare as routes
from experiments.pamf_physics_route import runner as dispatching
from . import pair_plan, partial_plan, practical_plan, recovery_plan, scheduled_plan, weak_practical_plan
from .pairing import installed as pairing_scope
from .practical_gate import installed as practical_scope
from .recovery import installed as recovery_scope
from .runner import smoke, worker as existing_worker
from .source import train_trajectory


def check_pairing(state):
    pair = state['U_branch_pairing']
    if (pair['calls'] != 1 or pair['extra_eval_forwards'] != 2 or pair['probed'] <= 0
            or pair['changed_vs_random'] <= 0 or (pair['beta'] > 0 and pair['mean_eligible_discount'] <= 0)):
        raise ValueError('Actual U branch probes and eligible partner replacements required')
    return dict(U_branch_pairing=pair)


def check_recovery(state):
    recovery = state['U_recovery']
    if recovery['calls'] != 1 or recovery['extra_eval_forwards'] != 2 or recovery['probed'] <= 0:
        raise ValueError('Actual rejected-U source probe required')
    return dict(U_recovery=recovery)


def check_partial(state):
    details = check_recovery(state)
    recovery = details['U_recovery']
    if recovery['used'] <= 0 or recovery['candidate_class_slots'] <= 0:
        raise ValueError('Actual rejected-U partial labels must reach the existing raw passes')
    return details


def check_practical(state):
    data = state['U_practical_gate']
    if data['calls'] != 1 or data['offered'] <= 0 or not 0 <= data['used'] <= data['offered']:
        raise ValueError('Actual source-channel U routing must be executed')
    return dict(U_practical_gate=data)


METHODS = {pair_plan.RUN: (pair_plan, pairing_scope, check_pairing),
           recovery_plan.RUN: (recovery_plan, recovery_scope, check_recovery),
           partial_plan.RUN: (partial_plan, recovery_scope, check_partial),
           practical_plan.RUN: (practical_plan, practical_scope, check_practical),
           weak_practical_plan.RUN: (weak_practical_plan, practical_scope, check_practical),
           scheduled_plan.RUN: (scheduled_plan, practical_scope, check_practical)}


def worker(c, kind):
    if kind in ('predict', 'score'):
        return existing_worker(c, kind)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    plan, installed, check_u = METHODS[c['run_id']]
    if kind == 'train':
        return train_trajectory(c, plan.parent(c), installed, ssl=True, physical_L=False)
    if kind == 'smoke':
        return smoke(c, rows=plan.candidate_rows, configure=plan.configuration, check_parent=plan.parent,
                     scope=installed, check_u=check_u)
    raise ValueError('Only source and clean/practical evaluation stages are allowed')


def dispatch(rt):
    plan, _, _ = METHODS[rt['run_id']]
    prepare = partial(routes.prepare, rows_for_run=lambda _: plan.candidate_rows(), configure=plan.configuration,
                      source_diagnostic=False, tri_dura=True,
                      extra_controls=scheduled_plan.extra_controls if plan is scheduled_plan else None)
    launch = partial(dispatching.launch, module='experiments.pamf_physics_tri.pair_runner')
    dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
