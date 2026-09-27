"""Source-only qualification from the actual complete training-loss graph.

The main forward/augmentation/teacher/pseudo decisions are reused, not replayed
with an approximate objective. Only disposable optimizer/model copies are stepped.
"""
from copy import deepcopy
from collections import defaultdict
import math
import random

import torch

from .replay_audit import capture_rng_state, restore_rng_state, freeze, grouped_source_risk
from .training import routed_response_gradients


def paired_actual_optimizer_update(model, auxiliary, optimizer, scaler, *,
        baseline_loss, identity_loss, response_loss, decision_loss, cross_loss,
        roles, lambda_resp, lambda_dec, lambda_cross, gradient_cap, max_grad_norm,
        evaluate, response_group_losses=None, capture_tensors=False):
    """Initial/A/B student risk at identical actual pre-optimizer state.

    EMA/prototype/pseudo inputs have already entered the supplied loss graph.
    They are neither recomputed nor updated by this audit. The risk is explicitly
    the student classifier after one AdamW step, not a multi-step/EMA rollout.
    The scaled-gradient addition/unscale/clip/step order matches response_backward.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('actual AdamW optimizer required')
    params = [p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in params}) != len(params):
        raise ValueError('optimizer parameters must be unique')
    if not math.isfinite(float(max_grad_norm)) or max_grad_norm < 0:
        raise ValueError('finite nonnegative actual clipping threshold required')
    original_rng = capture_rng_state()
    scale = float(scaler.get_scale())
    # Initializing a lazy scaler on a copy avoids changing live scaler state.
    base_objective = baseline_loss + float(lambda_dec)*decision_loss + float(lambda_cross)*cross_loss
    finite_loss = bool(torch.isfinite(base_objective.detach()))
    base_scaled = torch.autograd.grad(base_objective * scale, params, allow_unused=True, retain_graph=True)
    contributions, routing = routed_response_gradients(identity_loss=identity_loss,
        response_loss=response_loss, roles=roles, scale=scale, lambda_resp=lambda_resp,
        joint_open=True, gradient_cap=gradient_cap, response_group_losses=response_group_losses)
    contribution_by_id = {id(p): g for p, g in contributions}
    joint_scaled = []
    for p, g in zip(params, base_scaled):
        value = g.detach().clone() if g is not None else None
        extra = contribution_by_id.get(id(p))
        if extra is not None:
            if value is None:
                value = extra * scale
            else:
                value.add_(extra, alpha=scale)
        joint_scaled.append(value)
    initial_parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    result = {}
    try:
        for key, gradients in (('initial', None), ('base', base_scaled), ('base_response', joint_scaled)):
            # Copy the entire graph once to preserve optimizer/model/aux aliases.
            m, aux, opt, sc = deepcopy((model, auxiliary, optimizer, scaler))
            copied_params = [p for group in opt.param_groups for p in group['params']]
            step_applied = False
            if gradients is not None:
                sc.scale(torch.zeros((), device=params[0].device))
                for p, g in zip(copied_params, gradients):
                    p.grad = g.detach().clone() if g is not None else None
                present = [p for p in copied_params if p.grad is not None]
                sc.unscale_(opt)
                finite = all(bool(torch.isfinite(p.grad).all()) for p in present)
                branch_loss_finite = finite_loss and (key == 'base' or bool(torch.isfinite(response_loss.detach())))
                if finite and branch_loss_finite and present:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(copied_params, max_grad_norm, error_if_nonfinite=False)
                    sc.step(opt)
                    step_applied = True
                if present:
                    sc.update()
            updates = {}
            for name, p in m.named_parameters():
                delta = p.detach().cpu() - initial_parameters[name]
                updates[name] = dict(l2=float(delta.double().norm()), max_abs=float(delta.abs().max()))
            row = dict(step_applied=step_applied, parameter_updates=updates,
                       scaler=deepcopy(sc.state_dict()))
            if capture_tensors:
                row.update(model_state=freeze(m.state_dict()), auxiliary_state=freeze(aux.state_dict()),
                           optimizer_state=freeze(opt.state_dict()))
            restore_rng_state(original_rng)
            m.eval()
            with torch.no_grad():
                row['risk'] = freeze(evaluate(m))
            result[key] = row
            del m, aux, opt, sc, copied_params
    finally:
        restore_rng_state(original_rng)
    result['metadata'] = dict(baseline='actual_complete_training_loss_graph',
        risk_model='student_after_one_adamw_update', augmentation='same_actual_forward_graph',
        persistent_inputs='actual_teacher_EMA_prototype_pseudo_inputs_already_in_loss_graph',
        main_state_modified=False, branches_promoted=False, scale=scale,
        max_grad_norm=max_grad_norm, routing=routing,
        baseline_loss=float(baseline_loss.detach()), matched_base_loss=float(base_objective.detach()))
    return result


def joint_objective_contract(config, scope):
    """Scientific objective identity; sampling strategy is deliberately excluded."""
    keys = ('implementation_version', 'response_enabled', 'decision_enabled', 'identity_interaction_enabled',
            'head_only', 'permanent_detach', 'P', 'Q', 'K', 'view_policy', 'target_family', 'target_dim',
            'statistics_epsilon', 'readout_dim', 'predictor_mode', 'rank', 'lambda_resp', 'lambda_dec',
            'lambda_cross', 'decision_delta', 'decision_min_records', 'interaction_weight', 'role_rotation',
            'normalization_policy', 'mixstyle_role_policy', 'gradient_cap', 'response_routing',
            'response_decomposition', 'decision_mode', 'decision_calibration', 'interaction_mode')
    return dict(base_method='matched_CORE90', scope=scope,
                parameters={key: deepcopy(config.get(key)) for key in keys})


def source_prediction_measurements(report):
    """Match every pretrained/shuffled prediction to the same fitted-head task."""
    if report.get('status') != 'VERIFIED' or report.get('source_role') != 'source_validation':
        raise ValueError('source prediction measurements unavailable')
    fits = report.get('matched_fits', {})
    for key in ('rx_only', 'tx_only', 'head_only'):
        if not fits.get(key, {}).get('errors'):
            raise ValueError('missing matched source baseline: ' + key)
    values = defaultdict(list)
    physical = set()
    index = 0
    for entry in report['reuse'].values():
        for task in entry['tasks']:
            physical.update(task['query_physical_ids'])
            for branch in task['branches']:
                if branch.get('shuffled_tx') is None:
                    raise ValueError('TX necessity requires a nontrivial fixed TX permutation')
                for key, error in (('full_error', branch['pretrained_full']),
                                   ('constant_error', branch['constant_frozen_source_location']),
                                   ('shuffled_tx_error', branch['shuffled_tx']),
                                   ('rx_only_error', fits['rx_only']['errors'][index]),
                                   ('tx_only_error', fits['tx_only']['errors'][index]),
                                   ('head_only_error', fits['head_only']['errors'][index])):
                    value = float(error['mse'])
                    if not math.isfinite(value):
                        raise ValueError('nonfinite source prediction measurement')
                    values[key].append(value)
                index += 1
    if not index or any(len(fits[key]['errors']) != index for key in ('rx_only', 'tx_only', 'head_only')):
        raise ValueError('unmatched source prediction tasks')
    means = {key: math.fsum(rows)/len(rows) for key, rows in values.items()}
    return dict(capability=dict(sample_count=len(physical), full_error=means['full_error'],
                    baseline_error=min(means[key] for key in ('constant_error', 'rx_only_error', 'tx_only_error'))),
                necessity=dict(sample_count=len(physical), **{key: means[key] for key in
                    ('full_error', 'shuffled_tx_error', 'rx_only_error', 'head_only_error')}),
                prediction_details=dict(means, matched_tasks=index,
                    unique_query_physical_records=len(physical), head_only_scope='matched_fit_on_frozen_current_features'))


def prepare_source_query(dataset, records, *, per_cell, seed, num_classes, device,
                         domain_label_map, args, batch_size):
    """Balanced independent V records; one physical query shared by all A/B views."""
    from .source_eval import fixed_source_eval
    from cvsrffi.eval import apply_sat_channel_for_scenario
    cells = defaultdict(list)
    for record in records:
        cells[record.tx_id, record.rx_id, record.day_id, record.condition_id].append(record)
    if not cells or any(len(rows) < per_cell for rows in cells.values()):
        raise ValueError('insufficient source V records for the frozen per-cell query budget')
    if getattr(dataset, 'transform', None) is not None or getattr(getattr(dataset, 'base', dataset), 'transform', None) is not None:
        raise ValueError('source query requires unaugmented received inputs')
    chosen = []
    rng = random.Random(seed)
    for key in sorted(cells, key=repr):
        rows = sorted(cells[key], key=lambda r: repr(r.physical_sample_id))
        chosen.extend(rng.sample(rows, per_cell))
    loaded = [dataset[r.index] for r in chosen]
    x = torch.stack([torch.as_tensor(row[0]) for row in loaded]).to(device)
    y = torch.tensor([int(row[1]) for row in loaded], device=device)
    if set(y.tolist()) != set(range(num_classes)):
        raise ValueError('source identity probe must evaluate every registered TX class')
    domains = torch.tensor([domain_label_map.get(int(row[2]), -1) for row in loaded], device=device)
    if (domains < 0).any():
        raise ValueError('unknown source domain in identity probe')
    generator = torch.Generator(device=device).manual_seed(seed)
    views = {'clean': x}
    for scenario in ('leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak'):
        views[scenario], _ = apply_sat_channel_for_scenario(x, scenario, args, gen=generator)
    receivers = [r.rx_id for r in chosen]
    conditions = [f'day={r.day_id}/condition={r.condition_id}' for r in chosen]
    def evaluate(model):
        risks = {}
        with fixed_source_eval((model,), seed):
            for scenario, iq in views.items():
                logits = []
                for start in range(0, len(y), batch_size):
                    output = model(iq[start:start+batch_size], y_tx=None, grl_lambda=0., return_aux=True,
                                   domain_labels=domains[start:start+batch_size])
                    logits.append(output['tx_logits'].detach())
                all_logits = torch.cat(logits)
                if not torch.isfinite(all_logits).all():
                    raise ValueError('nonfinite full-class source identity risk')
                risks[scenario] = grouped_source_risk(all_logits, y, receivers, conditions)
        return risks
    return evaluate, [r.physical_sample_id for r in chosen], dict(
        unique_query_records=len(chosen), views=4, model_input_records=3*4*len(chosen),
        backbone_forwards=3*4*math.ceil(len(chosen)/batch_size), query_seed=seed,
        risk='full_class_student_CE', groups='every_source_RX_day_condition')


def aggregate_source_observation(measurements, *, scope, source_freeze_id):
    """Paired-batch uncertainty is separate from unique physical support counts."""
    if len(measurements) < 2 or any(row['scope'] != scope for row in measurements):
        raise ValueError('multiple same-scope paired training batches required')
    def average(values):
        values = list(values)
        if not values or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('finite paired source measurements required')
        return math.fsum(values)/len(values)
    result = dict(source_role='source_validation', target_used=False, scope=scope,
        source_freeze_id=source_freeze_id,
        observation_id='steps:' + ','.join(str(row['step']) for row in measurements))
    for kind in ('capability', 'necessity'):
        result[kind] = {key: average(row[kind][key] for row in measurements)
                        for key in measurements[0][kind] if key != 'sample_count'}
        result[kind]['sample_count'] = min(row[kind]['sample_count'] for row in measurements)
    train_ids = set().union(*(set(row['training_physical_ids']) for row in measurements))
    query_ids = set().union(*(set(row['query_physical_ids']) for row in measurements))
    if not train_ids or not query_ids or train_ids & query_ids:
        raise ValueError('independent source training/query records required')
    update = dict(sample_count=min(len(set(row['query_physical_ids'])) for row in measurements),
        base_risk=average(row['risks']['base']['clean']['overall']['ce'] for row in measurements),
        joint_risk=average(row['risks']['base_response']['clean']['overall']['ce'] for row in measurements),
        training_physical_ids=sorted(train_ids, key=repr), query_physical_ids=sorted(query_ids, key=repr),
        hard_groups={}, leo_groups={})
    for scene in ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak'):
        group_keys = set(measurements[0]['risks']['base'][scene]['groups'])
        if any(set(row['risks'][branch][scene]['groups']) != group_keys
               for row in measurements for branch in ('base', 'base_response')):
            raise ValueError('paired source group coverage changed')
        for group in sorted(group_keys):
            values = [row['risks'] for row in measurements]
            evidence = dict(sample_count=min(r['base'][scene]['groups'][group]['n'] for r in values),
                base_risk=average(r['base'][scene]['groups'][group]['ce'] for r in values),
                joint_risk=average(r['base_response'][scene]['groups'][group]['ce'] for r in values))
            update['hard_groups' if scene == 'clean' else 'leo_groups'][scene + '/' + group] = evidence
    differences = [row['risks']['base_response']['clean']['overall']['ce']
                   - row['risks']['base']['clean']['overall']['ce'] for row in measurements]
    mean = average(differences)
    variance = math.fsum((x-mean)**2 for x in differences)/(len(differences)-1)
    result['update_value'] = update
    result['uncertainty'] = dict(unit='paired_actual_training_batches_not_training_seeds',
        pairs=len(differences), clean_joint_minus_base=differences, mean=mean,
        standard_error=math.sqrt(variance/len(differences)))
    return result
