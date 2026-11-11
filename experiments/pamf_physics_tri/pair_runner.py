"""Fixed semi-supervised six-row run, followed by truth-last practical scoring."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_route import prepare as routes
from experiments.pamf_physics_route import runner as dispatching
from . import branchguard_plan, certified_day_plan, curriculum_plan, pair_plan, partial_plan, practical_plan, recovery_plan, scheduled_plan, single_rx_plan, validation_plan, weak_practical_plan
from .branch_guard import installed as branchguard_scope
from .pairing import installed as pairing_scope
from .practical_gate import installed as practical_scope
from .recovery import installed as recovery_scope
from .validation_scope import installed as validation_scope
from .runner import smoke, worker as existing_worker
from .source import train_trajectory
from .single_rx import installed as single_rx_scope
from .certified_day import installed as certified_day_scope
from .curriculum import installed as curriculum_scope


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


def check_branch_guard(state):
    details = check_practical(state)
    guard = state['U_branch_gradient_guard']
    if guard['calls'] != 1 or not any(guard['last'][name]['U_norm'] > 0
                                       for name in ('time', 'frequency', 'pa', 'signature', 'reference')):
        raise ValueError('Actual U gradients must enter the named branch guard')
    return dict(details, U_branch_gradient_guard=guard)


def check_single_rx(state):
    details = check_branch_guard(state)
    day = state['U_cross_day']
    if day['calls'] != 1 or day['branch_anchors'] <= 0 or state['U_receiver_extrapolation']['eligible_packet_views'] <= 0:
        raise ValueError('Real source-day branch and fusion relations must execute')
    return dict(details, U_cross_day=day)


def check_certified_day(state):
    details = check_single_rx(state)
    cert = state['U_relation_certificate']
    if cert['calls'] != 1 or not 0 < cert['relation_used'] <= cert['offered'] or not cert['raw_pseudo_CE_mask_unchanged']:
        raise ValueError('Actual source-certified day relation must execute')
    return dict(details, U_relation_certificate=cert)


def check_curriculum(state):
    return check_certified_day(state) if 'U_relation_certificate' in state else check_single_rx(state)


METHODS = {pair_plan.RUN: (pair_plan, pairing_scope, check_pairing),
           recovery_plan.RUN: (recovery_plan, recovery_scope, check_recovery),
           partial_plan.RUN: (partial_plan, recovery_scope, check_partial),
           practical_plan.RUN: (practical_plan, practical_scope, check_practical),
           weak_practical_plan.RUN: (weak_practical_plan, practical_scope, check_practical),
           scheduled_plan.RUN: (scheduled_plan, practical_scope, check_practical),
           validation_plan.RUN: (validation_plan, validation_scope, check_practical),
           branchguard_plan.RUN: (branchguard_plan, branchguard_scope, check_branch_guard),
           single_rx_plan.RUN: (single_rx_plan, single_rx_scope, check_single_rx),
           certified_day_plan.RUN: (certified_day_plan, certified_day_scope, check_certified_day),
           curriculum_plan.RUN: (curriculum_plan, curriculum_scope, check_curriculum)}


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
                     scope=installed, check_u=check_u,
                     graph_check=plan.check_graph if plan in (validation_plan, branchguard_plan, single_rx_plan, certified_day_plan, curriculum_plan) else None)
    raise ValueError('Only source and clean/practical evaluation stages are allowed')


def dispatch(rt):
    plan, _, _ = METHODS[rt['run_id']]
    prepare = partial(routes.prepare, rows_for_run=lambda _: plan.candidate_rows(), configure=plan.configuration,
                      source_diagnostic=False, tri_dura=True,
                      extra_controls=(scheduled_plan.extra_controls if plan is scheduled_plan else
                                      branchguard_plan.extra_controls if plan is branchguard_plan else
                                      certified_day_plan.extra_controls if plan is certified_day_plan else
                                      curriculum_plan.extra_controls if plan is curriculum_plan else None),
                      dynamic_protocol=plan in (validation_plan, branchguard_plan, single_rx_plan, certified_day_plan, curriculum_plan))
    launch = partial(dispatching.launch, module='experiments.pamf_physics_tri.pair_runner')
    from experiments.pamf_physics_branch import runner as physical
    old_admits = physical.admits
    if rt.get('max_processes_per_gpu', 2) == 4:
        physical.admits = partial(old_admits, max_total=4)
    try:
        dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)
    finally:
        physical.admits = old_admits


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
