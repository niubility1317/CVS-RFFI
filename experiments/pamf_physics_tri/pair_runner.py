"""Fixed semi-supervised six-row run, followed by truth-last practical scoring."""
import argparse
from functools import partial

from experiments.cric_dura_comparison.io import read
from experiments.pamf_physics_route import prepare as routes
from experiments.pamf_physics_route import runner as dispatching
from . import branchguard_plan, certified_day_plan, curriculum_plan, fused_physics_plan, hybrid_contrast_plan, multilevel_physics_plan, packet_contrast_plan, pair_plan, partial_plan, practical_plan, recovery_plan, scheduled_plan, single_rx_plan, soft_channel_plan, validation_plan, weak_practical_plan
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
from .fused_physics import installed as fused_physics_scope
from .multilevel_physics import installed as multilevel_physics_scope
from .soft_channel import installed as soft_channel_scope
from .packet_contrast import installed as packet_contrast_scope
from .hybrid_contrast import installed as hybrid_contrast_scope


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


def check_fused_physics(state):
    details = check_single_rx(state)
    fused = state['U_fused_physics']
    if (fused['calls'] != 1 or fused['selected'] <= 0 or fused['extra_student_forwards'] != 0
            or min(fused['smoke_gradients'].values()) <= 0):
        raise ValueError('Actual same-packet fusion loss must train signature and reference paths')
    return dict(details, U_fused_physics=fused)


def check_multilevel_physics(state):
    details = check_single_rx(state)
    level = state['U_multilevel_physics']
    expected = {key for key, enabled in level['recipe'].items() if enabled}
    if (level['calls'] != 1 or level['selected'] <= 0 or level['extra_student_forwards'] != 0
            or set(level['smoke_gradients']) != expected
            or any(min(norms.values()) <= 0 for norms in level['smoke_gradients'].values())):
        raise ValueError('Every new physical objective must train signature and reference branches')
    return dict(details, U_multilevel_physics=level)


def check_soft_channel(state):
    details = check_single_rx(state)
    soft = state['U_soft_channel']
    if (soft['calls'] != 1 or soft['selected'] <= 0 or not 0 < soft['relative_strength_sum'] <= soft['selected']
            or not soft['hard_pseudo_mask_unchanged']):
        raise ValueError('Bounded sample-specific source-channel assignment must execute')
    return dict(details, U_soft_channel=soft)


def check_packet_contrast(state):
    fused = state['U_fused_physics']
    details = check_fused_physics(state) if fused['cosine_enabled'] else check_single_rx(state)
    packet = state['U_packet_contrast']
    if (fused['calls'] != 1 or fused['selected'] <= 0 or packet['calls'] != 1
            or packet['anchors'] <= 0 or packet['extra_student_forwards'] != 0
            or min(packet['smoke_gradients'].values()) <= 0):
        raise ValueError('Actual same-packet contrast must reach signature and reference paths')
    return dict(details, U_fused_physics=fused, U_packet_contrast=packet)


def check_hybrid_contrast(state):
    details = check_single_rx(state)
    fused, hybrid = state['U_fused_physics'], state['U_hybrid_contrast']
    expected = {'symmetric': {'fusion', 'reverse'}, 'branch': {'branch'},
                'fusion_branch': {'fusion', 'branch'}}[hybrid['recipe']['mode']]
    if (fused['cosine_enabled'] or fused['selected'] <= 0 or hybrid['calls'] != 1 or hybrid['anchors'] <= 0
            or hybrid['extra_student_forwards'] != 0 or set(hybrid['smoke_gradients']) != expected
            or any(min(v.values()) <= 0 for v in hybrid['smoke_gradients'].values())):
        raise ValueError('Every enabled physical contrast must reach signature and reference paths')
    return dict(details, U_fused_physics=fused, U_hybrid_contrast=hybrid)


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
           curriculum_plan.RUN: (curriculum_plan, curriculum_scope, check_curriculum),
           fused_physics_plan.RUN: (fused_physics_plan, fused_physics_scope, check_fused_physics),
           multilevel_physics_plan.RUN: (multilevel_physics_plan, multilevel_physics_scope, check_multilevel_physics),
           soft_channel_plan.RUN: (soft_channel_plan, soft_channel_scope, check_soft_channel),
           packet_contrast_plan.RUN: (packet_contrast_plan, packet_contrast_scope, check_packet_contrast),
           hybrid_contrast_plan.RUN: (hybrid_contrast_plan, hybrid_contrast_scope, check_hybrid_contrast)}


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
                     graph_check=plan.check_graph if plan in (validation_plan, branchguard_plan, single_rx_plan, certified_day_plan, curriculum_plan, fused_physics_plan, multilevel_physics_plan, soft_channel_plan, packet_contrast_plan, hybrid_contrast_plan) else None)
    raise ValueError('Only source and clean/practical evaluation stages are allowed')


def dispatch(rt):
    plan, _, _ = METHODS[rt['run_id']]
    prepare = partial(routes.prepare, rows_for_run=lambda _: plan.candidate_rows(), configure=plan.configuration,
                      source_diagnostic=False, tri_dura=True,
                      extra_controls=(scheduled_plan.extra_controls if plan is scheduled_plan else
                                      branchguard_plan.extra_controls if plan is branchguard_plan else
                                      certified_day_plan.extra_controls if plan is certified_day_plan else
                                      curriculum_plan.extra_controls if plan is curriculum_plan else
                                      fused_physics_plan.extra_controls if plan is fused_physics_plan else
                                      multilevel_physics_plan.extra_controls if plan is multilevel_physics_plan else
                                      soft_channel_plan.extra_controls if plan is soft_channel_plan else
                                      packet_contrast_plan.extra_controls if plan is packet_contrast_plan else
                                      hybrid_contrast_plan.extra_controls if plan is hybrid_contrast_plan else None),
                      dynamic_protocol=plan in (validation_plan, branchguard_plan, single_rx_plan, certified_day_plan, curriculum_plan, fused_physics_plan, multilevel_physics_plan, soft_channel_plan, packet_contrast_plan, hybrid_contrast_plan))
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
