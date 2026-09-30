"""Derive complete FCR8 training diagnostics from an independently verified pilot.

No fitting, scoring, optimizer verification or query/result-trace access.
Optional SSH transports this standalone script through stdin for read-only stdout.
Full coordinates stay in the original NPZ artifacts; outputs contain scalars/refs.
"""
import argparse
from collections import Counter, OrderedDict
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

INPUT_STATUS = 'COMPLETE_FCR8_PROBE_VERIFIED'
STATUS = 'COMPLETE_FCR8_TRAINING_DIAGNOSTICS_DERIVED'
MODES = {'B_FCR8': 'B', 'C_FCR8_seq': 'C_seq', 'C_reset_init': 'C_reset_init'}
COORDS = ('row_id', 'split_id', 'scope', 'fold', 'outer_trial', 'state', 'train_k')
EXACT = dict(episodes=160, k1_episodes=40, oof_episodes=120, proxy_anchor_count=1400,
    sequence_paths=1760, baseline_head_fit_count=3168, fcr_preparation_count=3168,
    fcr_stage_count=4576, diagnostic_fit_count=0)
EXPECTED_STAGES = EXACT['fcr_stage_count']
COUNTERS = ('optimizer_steps', 'optimizer_iterations', 'inner_objective_evaluation_count',
    'inner_head_fit_count', 'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count',
    'derivative_triangular_solve_count', 'task_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count',
    'backward_evaluation_count', 'fcr_forward_evaluation_count', 'accepted_trial_count', 'rejected_trial_count',
    'trial_count', 'trial_attempt_count', 'nonzero_projected_update_count', 'fcr_stage_count', 'factorization_count')
LOSS_FIELDS = ('loss_task', 'loss_keep', 'loss_data', 'loss_proximal', 'loss_total', 'inner_training_accuracy',
    'inner_training_margin_mean')
EVENT_FIELDS = ('gradient_norm', 'keep_gradient_norm', 'direction_norm', 'guard_active', 'guard_dot_before',
    'guard_dot_after', 'guard_bound', 'step_size', 'update_norm', 'loss_before', 'loss_after', 'armijo_rhs',
    'armijo_tolerance', 'objective_acceptance_bound', 'armijo_pass', 'objective_nonincrease_pass', 'keep_pass',
    'accepted', 'rejection_reason', 'keep_risk_trial', 'keep_limit', 'keep_tolerance', 'keep_violation',
    'learning_rate', 'step_seconds')
COST_FIELDS = ('fit_seconds', 'score_seconds', 'baseline_binding_seconds', 'head_state_bytes', 'adapter_state_bytes',
    'lineage_state_bytes', 'persistent_state_bytes', 'optimizer_state_bytes', 'trainable_parameter_count',
    'retained_vector_record_bytes', 'maximum_trainable_parameter_count')
FUNCTION_FIELDS = ('pre_tangent_displacement_mean_squared', 'pre_tangent_displacement_rms',
    'coordinate_squared_norm', 'function_coordinate_reconstruction_error')
MECHANISM_STATS = ('block_angle_radians', 'tangent_over_kappa',
    'joint_distance_relative_change', 'adapted_distance_relative_change')
META_FIELDS = {'status', 'scope', 'run_id', 'release_commit', 'coverage', 'algorithm', 'resources',
    'training_stage_count', 'raw_training_sources', 'state_archives', 'state_archive_file_count',
    'state_archive_file_bytes', 'query_rows_used', 'source_rows_used'}


def require(value, message):
    if not value: raise ValueError(message)


def _skip_value(text, start):
    """Skip unneeded top-level JSON values without deserializing held metrics."""
    position = start; stack = []; quoted = False; escaped = False
    while position < len(text):
        char = text[position]
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char in '[{': stack.append(char)
        elif char in ']}':
            if not stack: break
            stack.pop()
        elif char == ',' and not stack: break
        position += 1
    require(not quoted and not stack, 'Truncated summary metadata source')
    return position


def read_metadata(path):
    text = Path(path).read_text(encoding='utf-8'); decoder = json.JSONDecoder(); position = 0; result = {}
    while position < len(text) and text[position].isspace(): position += 1
    require(text[position:position+1] == '{', 'Summary must be a JSON object'); position += 1
    while True:
        while position < len(text) and (text[position].isspace() or text[position] == ','): position += 1
        if text[position:position+1] == '}': break
        key, position = decoder.raw_decode(text, position)
        while text[position].isspace(): position += 1
        require(text[position] == ':', 'Malformed summary metadata'); position += 1
        while text[position].isspace(): position += 1
        if key in META_FIELDS: result[key], position = decoder.raw_decode(text, position)
        else: position = _skip_value(text, position)
    return result


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            require(bool(line.strip()), f'Blank structured record {path}:{index}')
            yield json.loads(line)


def key_for(row, row_id=None, event=False):
    return tuple((row_id or row.get('row_id')) if name == 'row_id' else
        row.get('outer_trial') if event and name == 'outer_trial' else
        row.get('trial') if not event and name == 'outer_trial' else row.get(name) for name in COORDS)


def identity(key): return dict(zip(COORDS, key))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def statistics(values):
    known = [value for value in values if finite(value)]
    return dict(count=len(known), missing_count=len(values)-len(known),
        mean=math.fsum(known)/len(known) if known else None, minimum=min(known) if known else None,
        maximum=max(known) if known else None, sum=math.fsum(known) if known else None)


class Arrays:
    """Small read-only cache; the prior summary owns mathematical certification."""
    def __init__(self, lane): self.lane = Path(lane).resolve(); self.cache = OrderedDict(); self.read_paths = set()
    def __call__(self, ref):
        relative = Path(ref['path']); path = (self.lane/relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(self.lane/'state_arrays'), 'NPZ reference outside training archive')
        if ref['path'] not in self.cache:
            with np.load(path, allow_pickle=False) as source:
                self.cache[ref['path']] = {name: np.array(source[name], copy=True) for name in source.files}
            for value in self.cache[ref['path']].values():
                require(value.dtype.kind in 'fbiu' and np.isfinite(value).all(), 'Nonnumeric/nonfinite diagnostic source array')
                value.setflags(write=False)
            self.read_paths.add(ref['path'])
        self.cache.move_to_end(ref['path'])
        result = self.cache[ref['path']]
        if len(self.cache) > 8: self.cache.popitem(last=False)
        return result


def vector_statistics(value, prefix):
    return {prefix+'_norm': float(np.linalg.norm(value)),
        prefix+'_min': float(np.min(value)) if value.size else None,
        prefix+'_max': float(np.max(value)) if value.size else None,
        prefix+'_coordinate_count': int(value.size), prefix+'_nonzero_coordinates': int(np.count_nonzero(value))}


def counters(value):
    """Preserve actual scalar counters, including future measured count fields."""
    return {name: item for name, item in value.items() if type(item) is int and
        (name in COUNTERS or name.endswith('_count'))}


def parameter_metrics(ref, arrays, anchor):
    value = arrays(ref); U, Z, W = value['U'], value['Z'], value['W']
    result = dict(U_anchor_distance=float(np.linalg.norm(U-anchor['anchor_U'])),
        Z_initial_distance=float(np.linalg.norm(Z-anchor['Z'])), latent_rank=int(Z.shape[1]),
        actual_trainable_parameter_count=int(Z.size))
    for name, item in (('U', U), ('Z', Z), ('W', W)): result.update(vector_statistics(item, name))
    return result


def gradient_metrics(event, arrays):
    value = arrays(event['state_ref']); g, keep, Z = value['g_Z'], value['keep_g_Z'], value['Z']
    task = g-keep-Z
    result = dict(task_gradient_reconstruction='total_minus_keep_minus_Z',
        proximal_gradient_rule='Z_without_physical_count_division',
        task_gradient_norm=float(np.linalg.norm(task)), Z_proximal_gradient_norm=float(np.linalg.norm(Z)))
    for name, item in (('Z_gradient', g), ('Z_keep_gradient', keep),
                       ('Z_task_gradient', task), ('Z_direction', value['d_Z'])):
        result.update(vector_statistics(item, name))
    an = float(np.linalg.norm(keep))
    for name, left in (('total_vs_keep', g), ('task_vs_keep', task)):
        dot = float(np.sum(left*keep)); norm = float(np.linalg.norm(left))
        cosine = dot/(norm*an) if norm and an else None
        result.update({name+'_dot': dot, name+'_cosine': cosine,
            name+'_conflict': dot < 0 if norm and an else None})
    gn = float(np.linalg.norm(g))
    result['total_descent_keep_slope'] = -result['total_vs_keep_dot']/gn if gn else None
    result['guard_direction_change_norm'] = float(np.linalg.norm(value['d_Z']+g/gn)) if gn else None
    result['guard_direction_retained_norm'] = float(np.linalg.norm(value['d_Z']))
    return result


def measured_fold(value):
    """Retain measured scalars/stats, excluding labels, scores and bulk arrays."""
    result = {}
    for name, item in value.items():
        if item is None or type(item) in (str, bool, int, float): result[name] = item
        elif isinstance(item, dict) and set(item).issubset({'count', 'minimum', 'mean', 'maximum'}):
            result[name] = item
    return result


def coordinate_metrics(stage, final, arrays):
    prep = final['preparation']; ref = stage['coordinate_state_ref']; value = arrays(ref)
    result = {name: prep.get(name) for name in ('latent_rank', 'rank_estimated', 'rank_energy_threshold',
        'whitening_residual', 'whitening_tolerance', 'dictionary_rms', 'no_information_reason')}
    result.update(coordinate_state_ref=ref, coordinate_physical_count=int(len(value['H'])),
        optimizer_coordinate_scope=prep.get('optimizer_coordinate_scope'), fixed_dictionary_trainable=False)
    for name in ('H', 'W'): result.update(vector_statistics(value[name], name))
    spectrum = prep.get('singular_values')
    result['singular_values'] = spectrum
    known = bool(prep.get('rank_estimated')) and spectrum is not None
    maximum = max(spectrum) if known and spectrum else 0.
    result['singular_value_ratios'] = [s/maximum for s in spectrum] if maximum else ([0.]*len(spectrum) if known else None)
    result['spectrum_unmeasured_reason'] = None if known else 'RANK_NOT_ESTIMATED_'+str(prep.get('no_information_reason'))
    result['whitening_unmeasured_reason'] = None if prep.get('whitening_residual') is not None else result['spectrum_unmeasured_reason']
    return result


def objective_metrics(value):
    result = {name: value.get(name) if value else None for name in LOSS_FIELDS+FUNCTION_FIELDS}
    if not value: return result
    result.update({name: item for name, item in value.items() if item is None or type(item) in (str, bool, int, float)})
    folds = value.get('inner_folds', []); count = sum(fold['held_physical_count'] for fold in folds)
    result['inner_training_physical_count'] = count or None
    result['inner_training_correct_count'] = sum(fold['held_training_correct_count'] for fold in folds) if folds else None
    result['inner_training_accuracy'] = result['inner_training_correct_count']/count if count else None
    margins = [fold.get('held_training_margin_mean') for fold in folds]
    result['inner_training_margin_mean'] = math.fsum(m*fold['held_physical_count'] for m, fold in zip(margins, folds))/count if count and all(finite(m) for m in margins) else None
    result['objective_seconds'] = value.get('objective_seconds')
    result['inner_forward_seconds'] = math.fsum(fold.get('forward_seconds', 0.) for fold in folds if fold.get('head_fit_count'))
    for channel in ('task', 'keep'):
        result['inner_'+channel+'_adjoint_seconds'] = math.fsum(fold.get(channel+'_adjoint_seconds', 0.) for fold in folds)
    result['inner_fold_mechanisms'] = folds
    # Pool already measured distribution summaries by their measured counts.
    names = sorted(set(MECHANISM_STATS) | {name for fold in folds for name, item in fold.items() if isinstance(item, dict)})
    for name in names:
        values = [fold[name] for fold in folds if isinstance(fold.get(name), dict)]
        measured = [item for item in values if item.get('count', 0) and finite(item.get('mean'))]
        n = sum(item['count'] for item in measured)
        result[name+'_count'] = n
        result[name+'_mean'] = math.fsum(item['mean']*item['count'] for item in measured)/n if n else None
        result[name+'_min'] = min(item['minimum'] for item in measured) if n else None
        result[name+'_max'] = max(item['maximum'] for item in measured) if n else None
        result[name+'_unmeasured_fold_count'] = sum(not isinstance(fold.get(name), dict) for fold in folds)
    for name in sorted({name for fold in folds for name, item in fold.items() if finite(item)}):
        values = [fold.get(name) for fold in folds]
        result['fold_'+name+'_mean'] = statistics(values)['mean']
    for name in ('held_winner_change_count', 'original_zero_distance_pair_count'):
        values = [fold.get(name) for fold in folds]
        result[name] = sum(values) if values and all(finite(v) for v in values) else None
    return result


def teacher_metrics(event, row_id, arrays):
    teacher = event['teacher']; ref = teacher.get('state_ref'); result = dict(identity(key_for(event, row_id, True)),
        inner_fold=event['inner_fold'], source=teacher.get('source'), available=teacher['available'], state_ref=ref,
        teacher_exact_wrong_count=None, teacher_error_reason='Held truth labels are not in this training archive; zero q includes negative or tied margin')
    if ref is None:
        result.update(physical_count=0, unavailable_reason=teacher.get('reason')); return result
    value = arrays(ref); q, score = value['teacher_q'], value['teacher_scores']; n = len(q)
    zeros = q == 0; unique_max = np.sum(score == np.max(score, axis=1, keepdims=True), axis=1) == 1 if n else np.zeros(0, dtype=bool)
    result['forward_mechanisms'] = measured_fold(teacher.get('head_audit', {}))
    result.update(physical_count=n, q_mean=float(np.mean(q)) if n else None,
        q_min=float(np.min(q)) if n else None, q_max=float(np.max(q)) if n else None,
        positive_margin_correct_count=int(np.sum(q > 0)), wrong_or_tied_zero_q_count=int(np.sum(zeros)),
        guaranteed_wrong_unique_winner_count=int(np.sum(zeros & unique_max)),
        ambiguous_zero_q_tied_winner_count=int(np.sum(zeros & ~unique_max)),
        zero_q_count=int(np.sum(zeros)), low_positive_q_count=int(np.sum((q > 0) & (q < .25))),
        q_025_to_05_count=int(np.sum((q >= .25) & (q < .5))), q_05_to_075_count=int(np.sum((q >= .5) & (q < .75))),
        q_075_to_1_count=int(np.sum((q >= .75) & (q < 1))), unit_q_count=int(np.sum(q == 1)))
    return result


def thin_objective(value):
    if value is None: return None
    result = {name: item for name, item in value.items() if item is None or type(item) in (str, bool, int, float)}
    result['inner_folds'] = [measured_fold(fold) for fold in value.get('inner_folds', [])]
    return result


def thin_event(value):
    """Discard repeated preparations/solver arrays; preserve measured event data."""
    names = set(COORDS+COUNTERS+COST_FIELDS+EVENT_FIELDS+FUNCTION_FIELDS+('event', 'mode', 'trial', 'iteration', 'step',
        'state_ref', 'gradient_state_ref', 'initialization_state_ref', 'final_state_ref',
        'train_physical_count', 'training_physical_ids', 'classes', 'no_information', 'stop_reason',
        'keep_available', 'keep_anchor_risk', 'keep_slack', 'keep_limit', 'latent_rank'))
    dynamic = counters(value)
    result = {name: item for name, item in value.items() if name in names or name in dynamic or
        item is None or type(item) in (str, bool, int, float)}
    if 'preparation' in value:
        prep = value['preparation']; result['preparation'] = measured_fold(prep)
        for name in ('coordinate_state_ref', 'singular_values'): result['preparation'][name] = prep.get(name)
    if 'final_fit' in value: result['final_forward_mechanisms'] = measured_fold(value['final_fit'])
    for name in ('objective', 'final_objective'):
        if name in value: result[name] = thin_objective(value[name])
    return result


def stage_diagnostic(stage, stream, arrays):
    final = stream['final']; init_ref = stage['initialization_state_ref']; anchor = arrays(init_ref)
    init_event = stream['initial']; initial_obj = init_event['objective'] if init_event else None
    final_obj = final['final_objective']; key = key_for(stage); mode = MODES[stage['state']]
    info = not final['no_information']; updated = final['optimizer_steps'] > 0
    initial = objective_metrics(initial_obj); initial.update(parameter_metrics(init_ref, arrays, anchor))
    last = objective_metrics(final_obj); last.update(parameter_metrics(final['final_state_ref'], arrays, anchor))
    last.update({name: final[name] for name in FUNCTION_FIELDS if name in final})
    limit = final.get('keep_limit'); curve = []; gradient_by_ref = {}
    def point(kind, event, objective, ref, extra=None):
        measures = objective_metrics(objective); measures.update(parameter_metrics(ref, arrays, anchor))
        if extra: measures.update(extra)
        for name in EVENT_FIELDS:
            if name in event: measures[name] = event[name]
        risk = measures.get('loss_keep')
        measures['keep_remaining'] = limit-risk if limit is not None and finite(risk) else None
        measures['keep_excess'] = max(0., risk-limit) if limit is not None and finite(risk) else None
        curve.append(dict(identity(key), mode=mode, kind=kind, iteration=0 if kind == 'initial' else event.get('iteration'), trial=event.get('trial'),
            accepted=event.get('accepted'), state_ref=ref, metrics=measures,
            event_scalars={name: item for name, item in event.items() if item is None or type(item) in (str, bool, int, float)}))
    if init_event: point('initial', init_event, initial_obj, init_ref)
    for event in stream['events']:
        kind = event['event']
        if kind == 'FCR_GRADIENT':
            extra = gradient_metrics(event, arrays); gradient_by_ref[event['state_ref']['path']] = event
            point('gradient', event, event['objective'], event['state_ref'], extra)
        elif kind == 'FCR_TRIAL':
            grad = gradient_by_ref[event['gradient_state_ref']['path']]
            nominal = event['step_size']*grad['direction_norm']
            before = arrays(event['gradient_state_ref']); after = arrays(event['state_ref'])
            extra = dict(nominal_function_coordinate_update_norm=nominal,
                function_coordinate_update_ratio=event['update_norm']/nominal if nominal else None,
                Z_actual_update_norm=float(np.linalg.norm(after['Z']-before['Z'])),
                U_actual_update_norm=float(np.linalg.norm(after['U']-before['U'])),
                armijo_slack=event['armijo_rhs']+event['armijo_tolerance']-event['loss_after'],
                objective_nonincrease_slack=event['loss_before']+event['armijo_tolerance']-event['loss_after'])
            point('trial', event, event['objective'], event['state_ref'], extra)
        elif kind == 'FCR_STEP':
            point('accepted_step', event, event['objective'], event['state_ref'])
    point('final_cached' if info else 'no_information', final, final_obj, final['final_state_ref'],
        {name: final[name] for name in FUNCTION_FIELDS if name in final})
    for value in (initial, last):
        value['keep_remaining'] = limit-value['loss_keep'] if limit is not None and finite(value['loss_keep']) else None
        value['keep_excess'] = max(0., value['loss_keep']-limit) if limit is not None and finite(value['loss_keep']) else None
    scalar_fields = set(LOSS_FIELDS+FUNCTION_FIELDS) | {name for name in set(initial) | set(last)
        if finite(initial.get(name)) or finite(last.get(name))}
    delta = {field: last[field]-initial[field] if finite(last.get(field)) and finite(initial.get(field)) else None for field in sorted(scalar_fields)}
    all_event_counters = counters(final); prep_counters = counters(final['preparation'])
    # FINAL starts with the preparation payload. Shared preparation-only counts
    # are descriptive event fields, not work repeated by each consuming stage.
    counts = {name: item for name, item in all_event_counters.items() if name in COUNTERS or name not in prep_counters}
    trials = [p for p in curve if p['kind'] == 'trial']; gradients = [p for p in curve if p['kind'] == 'gradient']
    current = arrays(final['final_state_ref'])
    result = dict(identity(key), mode=mode, parent_k=stage['k'], new_count=stage['new_count'],
        information_stage=info, updated_stage=updated, zero_update_information_stage=info and not updated,
        stop_reason=final['stop_reason'], initial=initial, final=last, final_minus_initial=delta, counts=counts,
        all_event_counters=all_event_counters,
        final_event_scalars={name: item for name, item in final.items() if item is None or type(item) in (str, bool, int, float)},
        keep_available=final['keep_available'], keep_anchor_risk=final['keep_anchor_risk'], keep_slack=final['keep_slack'], keep_limit=limit,
        costs={field: stage.get(field, final.get(field)) for field in COST_FIELDS},
        initialization_state_ref=init_ref, final_state_ref=final['final_state_ref'],
        U_changed_coordinates=int(np.count_nonzero(current['U'] != anchor['U'])),
        Z_changed_coordinates=int(np.count_nonzero(current['Z'] != anchor['Z'])),
        coordinates=coordinate_metrics(stage, final, arrays),
        final_forward_mechanisms=final.get('final_forward_mechanisms', {}),
        function_coordinate_path_bound=.5, function_displacement_scope='PRE_TANGENT_RESIDUAL_ON_CURRENT_SUPPORT',
        accepted_Z_path_length=sum(event['update_norm'] for event in stream['events'] if event['event']=='FCR_STEP'),
        guard_active_count=sum(p['metrics']['guard_active'] for p in gradients),
        total_vs_keep_conflict_count=sum(p['metrics']['total_vs_keep_conflict'] is True for p in gradients),
        task_vs_keep_conflict_count=sum(p['metrics']['task_vs_keep_conflict'] is True for p in gradients),
        rejection_reasons=dict(Counter(p['metrics']['rejection_reason'] for p in trials if not p['accepted'])),
        rejected_keep_violation_count=sum(p['metrics']['keep_pass'] is False for p in trials),
        accepted_keep_violation_count=sum(p['accepted'] is True and p['metrics']['keep_pass'] is False for p in trials),
        accepted_objective_increase_count=sum(p['accepted'] is True and p['metrics']['objective_nonincrease_pass'] is False for p in trials),
        source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        prediction_change_count=None, evidence_scope='INNER_SUPPORT_TRAINING_NOT_INDEPENDENT_VALIDATION')
    binding = (stage['row_id'], stage['scope'], stage['k'], stage['train_k'], tuple(final['classes']), tuple(final['training_physical_ids']))
    return result, curve, binding


def aggregate(stages, curves):
    groups = {}
    for row in stages:
        key = (row['mode'], row['train_k']); group = groups.setdefault(key, []) ; group.append(row)
    rows = []
    for (mode, k), values in sorted(groups.items()):
        info = [row for row in values if row['information_stage']]
        row = dict(mode=mode, train_k=k, actual_stages=len(values), information_stages=len(info),
            updated_stages=sum(v['updated_stage'] for v in values), zero_update_information_stages=sum(v['zero_update_information_stage'] for v in values),
            stop_counts=dict(Counter(v['stop_reason'] for v in values)),
            rejection_counts=dict(sum((Counter(v['rejection_reasons']) for v in values), Counter())),
            counts={name: sum(v['counts'].get(name, 0) for v in values) for name in sorted({name for v in values for name in v['counts']})},
            latent_ranks=dict(Counter(str(v['coordinates']['latent_rank']) for v in values)),
            snapshots={point: {field: statistics([v[point].get(field) for v in info]) for field in LOSS_FIELDS+FUNCTION_FIELDS+('keep_remaining', 'U_norm', 'U_anchor_distance', 'Z_norm', 'W_norm')}
                       for point in ('initial', 'final', 'final_minus_initial')},
            resources={field: statistics([v['costs'].get(field) for v in values]) for field in COST_FIELDS})
        rows.append(row)
    curve_groups = {}
    for row in curves:
        key = (row['mode'], row['train_k'], row['kind'], row['iteration'], row['trial'], row['accepted'])
        curve_groups.setdefault(key, []).append(row['metrics'])
    measured = []
    for key, values in curve_groups.items():
        names = sorted({name for value in values for name, item in value.items() if finite(item)})
        for name in names:
            measured.append(dict(zip(('mode', 'train_k', 'kind', 'iteration', 'trial', 'accepted'), key), metric=name,
                **statistics([v.get(name) for v in values])))
    return dict(by_mode_train_k=rows, curve_statistics=measured)


def collect(summary_root, run_root=None):
    root = Path(summary_root); metadata = read_metadata(root/'summary.json'); coverage = metadata.get('coverage', {})
    require(metadata.get('status') == INPUT_STATUS and all(coverage.get(k) == v for k, v in EXACT.items())
        and metadata.get('training_stage_count') == EXPECTED_STAGES and metadata.get('query_rows_used') == metadata.get('source_rows_used') == 0,
        'A complete independently verified 160-parent FCR8 summary is required before training analysis')
    sources = metadata['raw_training_sources']; require(len(sources) == 4 and len({s['row_id'] for s in sources}) == 4, 'Missing four verified training sources')
    stages = list(jsonlines(root/'training_objectives.jsonl'))
    require(len(stages) == EXPECTED_STAGES and len({key_for(s) for s in stages}) == len(stages), 'Missing/duplicate derived training stages')
    streams = {}; teachers = []; lane_sources = []; event_counts = Counter(); read_paths = {}; arrays_by_row = {}
    for source in sources:
        row_id = source['row_id']; lane = Path(run_root)/row_id/'probe' if run_root else Path(source['full_training_events']).parent
        path = lane/'training_events.jsonl'; arrays = Arrays(lane); arrays_by_row[row_id] = arrays; count = 0
        for event in jsonlines(path):
            count += 1; kind = event['event']; event_counts[kind] += 1
            if kind == 'FCR_INNER_PREPARED': teachers.append(teacher_metrics(event, row_id, arrays)); continue
            require(kind in ('FCR_INITIAL', 'FCR_GRADIENT', 'FCR_TRIAL', 'FCR_STEP', 'FCR_FINAL'), 'Unknown FCR8 training event schema')
            event = thin_event(event)
            key = key_for(event, row_id, True); stream = streams.setdefault(key, dict(initial=None, final=None, events=[]))
            if kind == 'FCR_INITIAL': require(stream['initial'] is None, 'Duplicate initial event'); stream['initial'] = event
            elif kind == 'FCR_FINAL': require(stream['final'] is None, 'Duplicate final event'); stream['final'] = event
            else: stream['events'].append(event)
        lane_sources.append(dict(row_id=row_id, full_training_events=str(path), complete_event_records=count))
    require(set(streams) == {key_for(stage) for stage in stages}, 'Training stage/event coverage differs from verified summary')
    output_stages = []; curves = []; bindings = {}; preparations = {}
    for stage in stages:
        key = key_for(stage); stream = streams[key]; require(stream['final'] is not None, 'Missing final training event')
        final = stream['final']; require(bool(stream['initial']) is (not final['no_information']), 'Missing initial information-stage event')
        # Completeness cross-check only: mathematical acceptance/projection verification belongs to the input summary.
        for kind, field in (('FCR_GRADIENT', 'gradients'), ('FCR_TRIAL', 'trials'), ('FCR_STEP', 'steps')):
            require(sum(e['event'] == kind for e in stream['events']) == len(stage[field]), 'Missing complete training curve events')
        diagnostic, points, binding = stage_diagnostic(stage, stream, arrays_by_row[stage['row_id']])
        output_stages.append(diagnostic); curves.extend(points)
        if diagnostic['mode'] == 'B': bindings.setdefault(binding, []).append(diagnostic)
        prep = final['preparation']; prep_key = (stage['row_id'], prep['coordinate_state_ref']['path'])
        if prep_key not in preparations:
            prep_identity = dict(identity(key), state='B_prepare' if diagnostic['mode'] == 'B' else 'C_prepare')
            preparations[prep_key] = dict(prep_identity, counts=counters(prep),
                measurements={name: item for name, item in prep.items() if name not in counters(prep)},
                actual_consuming_stages=0, consuming_modes=[])
        preparations[prep_key]['actual_consuming_stages'] += 1
        preparations[prep_key]['consuming_modes'].append(diagnostic['mode'])
    for row_id, arrays in arrays_by_row.items(): read_paths[row_id] = len(arrays.read_paths)
    dedup = [values[0] for values in bindings.values()]
    totals = dict(actual_stages=len(output_stages), information_stages=sum(s['information_stage'] for s in output_stages),
        updated_stages=sum(s['updated_stage'] for s in output_stages),
        zero_update_information_stages=sum(s['zero_update_information_stage'] for s in output_stages),
        curve_records=len(curves), teacher_fold_records=len(teachers), preparation_records=len(preparations),
        unique_B_physical_bindings=len(bindings), repeated_B_actual_contexts=sum(len(v)-1 for v in bindings.values()))
    return dict(status=STATUS, run_id=metadata['run_id'], release_commit=metadata['release_commit'], coverage=coverage,
        totals=totals, algorithm=metadata['algorithm'], sources=lane_sources, source_summary=str(root/'summary.json'),
        source_training_objectives=str(root/'training_objectives.jsonl'), event_counts=dict(event_counts),
        state_archives=metadata['state_archives'], state_archive_file_count=metadata['state_archive_file_count'],
        state_archive_file_bytes=metadata['state_archive_file_bytes'], NPZ_files_read_for_diagnostics=read_paths,
        resources=metadata['resources'], stages=output_stages, curves=curves, teachers=teachers,
        preparations=list(preparations.values()),
        preparation_count_totals={name: sum(v['counts'].get(name, 0) for v in preparations.values())
            for name in sorted({name for v in preparations.values() for name in v['counts']})},
        B_binding_groups=[dict(representative={name: values[0][name] for name in COORDS}, actual_contexts=len(values)) for values in bindings.values()],
        deduplicated_B_statistics=aggregate(dedup, []), **aggregate(output_stages, curves), limitations=[
            'Only complete independently verified training products are analyzed. Parent result traces, outer scores, query, registry and history are never opened or deserialized.',
            'The prior summary certifies training bindings, full coordinates and optimizer mathematics. This collector derives descriptive metrics without repeating that audit, fitting or selecting parameters.',
            'All initial, gradient, accepted/rejected trial, accepted-step and final events are consumed. Accepted-step rows repeat their trial objective; resource totals are taken from verified counts, not curve-row sums.',
            'Information and updated stages differ. Initial forward/backward work remains charged for zero-update information stages; K1 loss fields are N/A.',
            'Task function-coordinate gradients are g_Z minus keep_g_Z minus Z. The proximal gradient is Z without division by N. total_vs_keep and task_vs_keep are separate; cosine is N/A when either norm is zero.',
            'Z has 736 times the retained latent rank coordinates, including exact empty rank-zero arrays. U stays in the original 736-by-8 coordinates and V is fixed; no V gradient or parameter-ball diagnostic is fabricated.',
            'H/W and the full singular-value array retain original NPZ references. Spectrum/rank/whitening are measured preparation fields; unestimated rank and legitimate original tau=None remain N/A.',
            'Forward mechanism statistics use existing folds only: pre-tangent displacement, block angles, tangent/kappa, joint/adapted distances, kernel changes and held-winner transitions. Zero-bandwidth adapted-distance bypass remains N/A with its measured reason.',
            'Preparation records are deduplicated by actual preparation context because C_seq and C_reset_init share one preparation. Stage counters and complete run coverage retain every actual measured dynamic counter.',
            'q=0 includes negative or tied teacher margin. Zero-q with a unique score winner gives a guaranteed wrong-teacher lower bound; exact wrong/tied classification is N/A without held labels.',
            'B physical bindings may repeat across new-count parents. Actual resources retain repetitions; deduplicated B statistics are separately labeled and do not create independent evidence.',
            'Inner losses/accuracy and keep constraints are supervised training diagnostics, not outer performance or a guarantee of old-class retention.',
            'Array paths preserve all coordinates in original NPZ files. No vectors, member hashes or raw source event copies are embedded in derived reports.',
            'Per-stage outer score time is N/A when unavailable in the pre-scoring FCR_FINAL event; verified summary resource totals retain actual resource measurements.',
            'Source validation, query prediction transitions and unmeasured deployment/transfer/hardware metrics stay N/A. Held-winner changes refer only to supervised inner training. Stage timing sums are work, not concurrent elapsed; nested archive timings are not added twice.'])


def csv_value(value):
    if value is None: return 'N/A'
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return str(value)


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: csv_value(row.get(key)) for key in fields} for row in rows)


def flat_stage(row):
    result = {k: v for k, v in row.items() if k not in ('initial', 'final', 'final_minus_initial', 'counts', 'costs', 'coordinates')}
    for prefix in ('initial', 'final', 'final_minus_initial', 'counts', 'costs', 'coordinates'):
        result.update({prefix+'_'+k: v for k, v in row[prefix].items()})
    return result


def write_outputs(output, value):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    compact = {key: item for key, item in value.items() if key not in ('stages', 'curves', 'teachers', 'preparations', 'curve_statistics')}
    compact['complete_derived_streams'] = dict(stages='stages.jsonl', curves='curves.jsonl', teachers='teachers.jsonl', preparations='preparations.jsonl', curve_statistics='curve_statistics.csv')
    with (out/'summary.json').open('x', encoding='utf-8') as stream: json.dump(compact, stream, ensure_ascii=False, allow_nan=False)
    for key in ('stages', 'curves', 'teachers', 'preparations'):
        with (out/(key+'.jsonl')).open('x', encoding='utf-8') as stream:
            for row in value[key]: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    write_csv(out/'stages.csv', [flat_stage(row) for row in value['stages']])
    write_csv(out/'curves.csv', [{**{k:v for k,v in row.items() if k != 'metrics'}, **row['metrics']} for row in value['curves']])
    write_csv(out/'teachers.csv', value['teachers']); write_csv(out/'preparations.csv', value['preparations'])
    write_csv(out/'by_mode_train_k.csv', value['by_mode_train_k'])
    write_csv(out/'curve_statistics.csv', value['curve_statistics']); write_csv(out/'B_binding_groups.csv', value['B_binding_groups'])
    archive_rows = [dict(row_id=source['row_id'], phase=phase, **metrics) for source in value['state_archives'] for phase, metrics in source['by_phase'].items()]
    write_csv(out/'archive_by_phase.csv', archive_rows)
    lines = ['# FCR8 完整训练机制诊断', '', '状态：'+value['status']+'；run：'+value['run_id']+'。', '',
        '全部训练事件已读取。本文只解释监督训练机制，不报告外层评分，不调整冻结方法。', '',
        '| 模式 | train K | 阶段 | 有信息 | 有更新 | 零更新有信息 | task 初→末 | keep 初→末 | proximal 初→末 | total 初→末 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['by_mode_train_k']:
        cells = []
        for field in ('loss_task', 'loss_keep', 'loss_proximal', 'loss_total'):
            a = row['snapshots']['initial'][field]['mean']; b = row['snapshots']['final'][field]['mean']
            cells.append('N/A' if a is None or b is None else f'{a:.6f}→{b:.6f}')
        lines.append('| '+' | '.join([row['mode'], str(row['train_k'])]+[str(row[k]) for k in
            ('actual_stages', 'information_stages', 'updated_stages', 'zero_update_information_stages')]+cells)+' |')
    lines += ['', f"实际状态 {value['totals']['actual_stages']} 个，完整曲线位置 {value['totals']['curve_records']} 个；教师折记录 {value['totals']['teacher_fold_records']} 个。",
        f"原始 NPZ 文件 {value['state_archive_file_count']} 个，共 {value['state_archive_file_bytes']} B；阶段分层见 archive_by_phase.csv。",
        '', 'teacher q 分布、错误下界及无法区分的并列见 teachers.csv；保持余量/违反、Z 梯度冲突、guard、三项接受条件、拒因与已有实测前向机制见完整 curves.csv/jsonl。', '',
        'task_vs_keep 使用 g_Z − keep_g_Z − Z，不除以 N。动态秩、H 谱与实测白化见 stages.csv；共享准备计数见 preparations.csv。V 固定，函数坐标没有参数球投影。', '',
        'teacher q=0 不等于精确错误数；inner held 准确率及 winner change 均为监督训练诊断。accepted_step 与其 trial 是重复展示，不能将所有曲线行相加计费。', '']
    lines += ['- '+text for text in value['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-root', required=True); parser.add_argument('--run-root'); parser.add_argument('--output')
    parser.add_argument('--ssh-host'); parser.add_argument('--ssh-config'); parser.add_argument('--remote-python')
    parser.add_argument('--collect-stdout', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.collect_stdout:
        require(not args.ssh_host and not args.output, 'Read-only stdout mode does not write outputs')
        print(json.dumps(collect(args.summary_root, args.run_root), ensure_ascii=False, allow_nan=False)); return
    require(bool(args.output), '--output is required')
    if args.ssh_host:
        require(bool(args.ssh_config) and bool(args.remote_python), 'Explicit SSH config and NumPy-capable remote Python required')
        command = [args.remote_python, '-', '--summary-root', args.summary_root, '--collect-stdout']
        if args.run_root: command += ['--run-root', args.run_root]
        result = subprocess.run(['ssh', '-F', args.ssh_config, args.ssh_host, shlex.join(command)],
            input=Path(__file__).read_text(encoding='utf-8'), text=True, encoding='utf-8', capture_output=True, timeout=600)
        require(result.returncode == 0, 'Read-only remote collection failed: '+result.stderr[-2000:])
        result = json.loads(result.stdout)
    else: result = collect(args.summary_root, args.run_root)
    write_outputs(args.output, result)
    print(json.dumps(dict(status=result['status'], totals=result['totals'], state_archive_file_count=result['state_archive_file_count']), allow_nan=False))


if __name__ == '__main__': main()
