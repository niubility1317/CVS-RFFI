"""Describe complete, independently verified AJLR training without fitting.

Snapshot reads permitted summary metadata and all scalar training streams. Extract
reads only referenced training NPZ arrays. Neither phase opens fit_trace, outer
features/scores, query, history or a registry. Remote phases use SSH stdin/stdout;
all new exports are local, exclusive files. Original arrays remain in place.
"""
import argparse
from collections import Counter, OrderedDict
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np

INPUT_STATUS = 'COMPLETE_ANCHOR_JOINT_PROBE_VERIFIED'
SNAPSHOT_STATUS = 'COMPLETE_AJLR_TRAINING_READONLY_SNAPSHOT'
STATUS = 'COMPLETE_AJLR_TRAINING_DIAGNOSTICS_DERIVED'
MODES = {'B_AJLR': 'B', 'C_AJLR_seq': 'C_seq'}
EXACT = dict(episodes=160, k1_episodes=40, oof_episodes=120,
    proxy_anchor_count=1400, sequence_paths=1800, baseline_head_fit_count=3240,
    ajlr_preparation_count=3240, ajlr_stage_count=3240)
EXPECTED_STAGES = 3240
COORDS = ('row_id', 'split_id', 'scope', 'fold', 'outer_trial', 'state', 'train_k')
PREPARATION_COUNTERS = ('ajlr_preparation_count', 'latent_svd_count',
    'dictionary_physical_evaluation_count', 'prepared_distance_evaluation_count',
    'original_distance_pair_count', 'prior_head_fit_count', 'prior_factorization_count',
    'prior_triangular_solve_count', 'prior_score_evaluation_count',
    'prior_score_physical_count', 'reference_distance_evaluation_count',
    'reference_distance_pair_count', 'raw_distance_evaluation_count',
    'raw_distance_pair_count', 'kernel_evaluation_count', 'kernel_pair_count',
    'adapter_physical_evaluation_count')
STAGE_COUNTERS = ('ajlr_stage_count', 'optimizer_steps', 'optimizer_iterations',
    'inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count',
    'final_head_fit_count', 'final_factorization_count', 'head_triangular_solve_count',
    'derivative_triangular_solve_count', 'ce_adjoint_solve_count',
    'backward_evaluation_count', 'accepted_trial_count', 'rejected_trial_count',
    'trial_count', 'trial_attempt_count', 'ajlr_forward_evaluation_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'raw_distance_evaluation_count', 'raw_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count', 'adapter_physical_evaluation_count')
FUNCTION_FIELDS = ('pre_tangent_displacement_mean_squared', 'pre_tangent_displacement_rms',
    'coordinate_squared_norm', 'function_coordinate_reconstruction_error')
LOSS_FIELDS = ('RMSCE', 'loss_ce', 'loss_task', 'loss_proximal', 'loss_total',
    'inner_training_accuracy', 'inner_training_margin_mean', 'old_CE_RMS', 'new_CE_RMS',
    'old_training_accuracy', 'new_training_accuracy')
META_FIELDS = {'status', 'scope', 'run_id', 'release_commit', 'coverage', 'algorithm',
    'resources', 'resource_statistics', 'training_stage_count', 'raw_training_sources',
    'state_archives', 'state_archive_file_count', 'state_archive_file_bytes',
    'query_rows_used', 'source_rows_used'}
ID_FIELDS = {'event', 'state', 'split_id', 'scope', 'fold', 'outer_trial', 'train_k',
    'training_physical_ids', 'classes', 'old_classes', 'prior_folds', 'final_problem'}


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def statistics(values):
    known = [x for x in values if finite(x)]
    return dict(count=len(known), missing_count=len(values)-len(known),
        mean=math.fsum(known)/len(known) if known else None,
        minimum=min(known) if known else None, maximum=max(known) if known else None,
        sum=math.fsum(known) if known else None)


def _skip_value(text, start):
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
    require(not quoted and not stack, 'Truncated metadata source')
    return position


def selected_json(text, allowed):
    """Skip forbidden top-level result values without deserializing them."""
    decoder = json.JSONDecoder(); position = 0; result = {}
    while position < len(text) and text[position].isspace(): position += 1
    require(text[position:position+1] == '{', 'Metadata must be an object'); position += 1
    while True:
        while position < len(text) and (text[position].isspace() or text[position] == ','): position += 1
        if text[position:position+1] == '}': break
        require(position < len(text), 'Truncated metadata object')
        key, position = decoder.raw_decode(text, position)
        while position < len(text) and text[position].isspace(): position += 1
        require(text[position:position+1] == ':', 'Malformed metadata'); position += 1
        while position < len(text) and text[position].isspace(): position += 1
        if key in allowed: result[key], position = decoder.raw_decode(text, position)
        else: position = _skip_value(text, position)
    return result


def read_metadata(path):
    return selected_json(Path(path).read_text(encoding='utf-8'), META_FIELDS)


def jsonlines(path, allowed=None):
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            require(bool(line.strip()), f'Blank structured record {path}:{index}')
            yield selected_json(line, allowed) if allowed is not None else json.loads(line)


def key_for(row, row_id=None, event=False):
    return tuple((row_id or row.get('row_id')) if name == 'row_id' else
        row.get('outer_trial') if event and name == 'outer_trial' else
        row.get('trial') if not event and name == 'outer_trial' else row.get(name) for name in COORDS)


def identity(key):
    return dict(zip(COORDS, key))


def scalars(value):
    return {key: item for key, item in value.items()
        if item is None or type(item) in (str, bool, int, float)}


def counters(value):
    """Retain current and future actual integer work fields."""
    return {key: item for key, item in value.items() if type(item) is int and
        (key in STAGE_COUNTERS or key in PREPARATION_COUNTERS or
            key.endswith(('_count','_calls','_solves')))}


def validate_metadata(metadata):
    coverage = metadata.get('coverage', {})
    require(metadata.get('status') == INPUT_STATUS and
        all(coverage.get(key) == value for key, value in EXACT.items()) and
        metadata.get('training_stage_count') == EXPECTED_STAGES and
        metadata.get('query_rows_used') == metadata.get('source_rows_used') == 0,
        'A complete independently verified 160-parent AJLR summary is required')
    sources = metadata.get('raw_training_sources', [])
    require(len(sources) == 4 and len({s['row_id'] for s in sources}) == 4,
        'Missing four verified training sources')


def lane_path(source, run_root=None):
    return Path(run_root)/source['row_id']/'probe' if run_root else Path(source['compact_training_events']).parent


def measured_supplement(compact, full):
    """Restore measured summary statistics omitted by the entry compactor."""
    if not isinstance(full,dict) or not isinstance(compact,dict): return
    for key,item in full.items():
        if isinstance(item,dict) and set(item).issubset({'count','minimum','mean','maximum'}):
            compact[key]=item
        elif key in ('residual_sample_mean','fitted_old_reference_mean'):
            compact[key]=item
    for key in ('objective','final_objective','final_fit'):
        if key in compact: measured_supplement(compact[key],full.get(key))
    for small,large in zip(compact.get('inner_folds',[]),full.get('inner_folds',[])):
        measured_supplement(small,large)


def snapshot(summary_root, run_root=None):
    """First read-only phase, no NPZ reads and no raw parent result reads."""
    root = Path(summary_root); metadata = read_metadata(root/'summary.json')
    validate_metadata(metadata)  # Gate before opening any training stream.
    stages = list(jsonlines(root/'training_objectives.jsonl'))
    require(len(stages) == EXPECTED_STAGES and len({key_for(s) for s in stages}) == len(stages),
        'Missing/duplicate summary training stages')
    lanes = []
    for source in metadata['raw_training_sources']:
        lane = lane_path(source, run_root); events = list(jsonlines(lane/'training_events_compact.jsonl'))
        logs = list(jsonlines(lane/'fit_stages.jsonl'))
        # Compact events omit identifiers, prior metadata and measured distribution
        # summaries (including core block angles). Select TRAINING metadata only.
        # Objectives contain refs/scalars, never materialized numeric score arrays.
        final_ids=[]; prepared_metadata=[]; full_count=0
        allowed=ID_FIELDS|{'objective','final_objective','final_fit'}
        for index,row in enumerate(jsonlines(lane/'training_events.jsonl',allowed)):
            require(index<len(events) and row['event']==events[index]['event'],
                'Full/compact training stream coverage mismatch')
            measured_supplement(events[index],row);full_count+=1
            if row['event']=='AJLR_FINAL':
                final_ids.append({key:item for key,item in row.items() if key in ID_FIELDS and
                    key not in ('prior_folds','final_problem')})
            elif row['event']=='AJLR_PREPARED':
                prepared_metadata.append({key:item for key,item in row.items() if key in ID_FIELDS})
        require(full_count==len(events),'Full/compact training stream coverage mismatch')
        require(len(final_ids) == sum(e['event'] == 'AJLR_FINAL' for e in events),
            'Missing final training identity metadata')
        lanes.append(dict(row_id=source['row_id'], lane=str(lane), events=events,
            stage_logs=logs, final_training_identities=final_ids,
            prepared_training_metadata=prepared_metadata,
            compact_training_events=str(lane/'training_events_compact.jsonl'),
            scalar_stage_logs=str(lane/'fit_stages.jsonl'),
            training_identity_source=str(lane/'training_events.jsonl')))
    return dict(status=SNAPSHOT_STATUS, metadata=metadata, lanes=lanes,
        summary_stages=stages, source_summary=str(root/'summary.json'),
        source_training_objectives=str(root/'training_objectives.jsonl'),
        boundary='TRAINING_ONLY_NO_FIT_TRACE_OUTER_FEATURES_SCORES_QUERY_OR_HISTORY')


class Arrays:
    """Read-only bounded cache of referenced TRAINING archives, no manifest audit."""
    def __init__(self, lane):
        self.lane = Path(lane).resolve(); self.cache = OrderedDict(); self.read_paths = set()

    def __call__(self, ref):
        relative = Path(ref['path']); path = (self.lane/relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(self.lane/'state_arrays'),
            'NPZ reference outside training archive')
        if ref['path'] not in self.cache:
            with np.load(path, allow_pickle=False) as source:
                values = {name: np.array(source[name], copy=True) for name in source.files}
            for value in values.values():
                require(value.dtype.kind in 'fbiu' and np.isfinite(value).all(),
                    'Nonnumeric/nonfinite training archive')
                value.setflags(write=False)
            self.cache[ref['path']] = values; self.read_paths.add(ref['path'])
        self.cache.move_to_end(ref['path']); result = self.cache[ref['path']]
        if len(self.cache) > 12: self.cache.popitem(last=False)
        return result


def vector_statistics(value, prefix):
    return {prefix+'_norm': float(np.linalg.norm(value)),
        prefix+'_min': float(np.min(value)) if value.size else None,
        prefix+'_max': float(np.max(value)) if value.size else None,
        prefix+'_coordinate_count': int(value.size),
        prefix+'_nonzero_coordinates': int(np.count_nonzero(value))}


def scalar_array(value):
    return float(value.reshape(-1)[0]) if value.size else None


def parameter_metrics(ref, arrays, initial):
    value = arrays(ref); U, Z, W = value['U'], value['Z'], value['W']
    result = dict(U_anchor_distance=float(np.linalg.norm(U-initial['anchor_U'])),
        Z_initial_distance=float(np.linalg.norm(Z-initial['Z'])),
        latent_rank=int(W.shape[1]), coordinate_parameter_count=int(Z.size))
    for name in ('U', 'Z', 'W'): result.update(vector_statistics(value[name], name))
    return result


def gradient_metrics(event, arrays):
    data = arrays(event['state_ref']); g, Z = data['g_Z'], data['Z']; ce = g-Z
    cn = float(np.linalg.norm(ce)); zn = float(np.linalg.norm(Z)); gn = float(np.linalg.norm(g))
    dot = float(np.sum(ce*Z))
    result = dict(CE_gradient_reconstruction='g_Z_minus_Z',
        proximal_gradient_rule='Z_without_physical_count_division',
        CE_gradient_norm=cn, proximal_gradient_norm=zn,
        CE_vs_proximal_dot=dot, CE_vs_proximal_cosine=dot/(cn*zn) if cn and zn else None,
        CE_vs_proximal_conflict=dot < 0 if cn and zn else None,
        CE_descent_proximal_slope=-dot/cn if cn else None,
        total_descent_CE_slope=-float(np.sum(g*ce))/gn if gn else None,
        total_descent_proximal_slope=-float(np.sum(g*Z))/gn if gn else None,
        normalized_direction_deviation_norm=float(np.linalg.norm(data['d_Z']+g/gn)) if gn else None)
    for name, value in (('Z_gradient', g), ('Z_CE_gradient', ce), ('Z_proximal_gradient', Z),
                        ('Z_direction', data['d_Z'])):
        result.update(vector_statistics(value, name))
    return result


def coordinate_metrics(prep, arrays):
    ref = prep['prepared_state_ref']; data = arrays(ref); spectrum = data['singular_values'].tolist()
    result = {key: prep.get(key) for key in ('latent_rank', 'rank_estimated',
        'rank_energy_threshold', 'whitening_residual', 'whitening_tolerance',
        'dictionary_rms', 'no_information_reason')}
    top = max(spectrum) if spectrum else 0.; retained = int(data['W'].shape[1])
    result.update(prepared_state_ref=ref, coordinate_physical_count=int(len(data['H'])),
        singular_values=spectrum, singular_value_ratios=[s/top for s in spectrum] if top else [0.]*len(spectrum),
        retained_condition_number=top/spectrum[retained-1] if retained and spectrum[retained-1] else None,
        coordinate_parameter_count=int(data['anchor_U'].shape[0]*retained),
        fixed_dictionary_trainable=False, function_displacement_scope='PRE_TANGENT_ON_CURRENT_TRAINING_SUPPORT')
    for name in ('H', 'W', 'anchor_U'): result.update(vector_statistics(data[name], name))
    return result


def score_statistics(scores, labels, classes, old_classes, means=None, counts=None):
    """Descriptive supervised score statistics; never evaluate an outer input."""
    labels = np.asarray(labels, dtype=int); n = len(labels); c = len(classes)
    old_columns = np.asarray([name in old_classes for name in classes], dtype=bool)
    order = np.asarray(sorted(range(c),key=lambda i:classes[i]),dtype=int)
    prediction = order[np.argmax(scores[:,order], axis=1)] if n else np.empty(0, dtype=int)
    correct = prediction == labels
    margin = np.empty(n)
    if n and c > 1:
        wrong = np.array(scores, copy=True); wrong[np.arange(n), labels] = -np.inf
        margin = scores[np.arange(n), labels]-wrong.max(axis=1)
    else: margin = np.empty(0)
    result = dict(inner_training_physical_count=n, inner_training_correct_count=int(correct.sum()),
        inner_training_accuracy=float(correct.mean()) if n else None,
        inner_training_margin_mean=float(margin.mean()) if margin.size else None,
        score_rms=float(np.linalg.norm(scores)/math.sqrt(scores.size)) if scores.size else None,
        exact_tied_winner_count=int(np.sum(np.sum(scores == scores.max(axis=1, keepdims=True), axis=1)>1)) if n else 0,
        class_ce_means=means.tolist() if means is not None else None,
        class_ce_counts=counts.tolist() if counts is not None else None)
    for group, columns in (('old', old_columns), ('new', ~old_columns)):
        members = columns[labels] if n else np.zeros(0, dtype=bool); k = int(members.sum())
        result.update({group+'_training_physical_count': k,
            group+'_training_correct_count': int(correct[members].sum()),
            group+'_training_accuracy': float(correct[members].mean()) if k else None,
            group+'_training_margin_mean': float(margin[members].mean()) if k and margin.size else None,
            group+'_class_count': int(columns.sum()),
            group+'_CE_RMS': float(np.linalg.norm(means[columns])/math.sqrt(int(columns.sum())))
                if means is not None and np.any(columns) else None,
            group+'_CE_macro_mean': float(means[columns].mean()) if means is not None and np.any(columns) else None,
            group+'_CE_physical_mean': float(np.sum(means[columns]*counts[columns])/counts[columns].sum())
                if means is not None and counts is not None and counts[columns].sum() else None})
    if n and old_columns.any():
        mask = old_columns[labels]; indices = order[old_columns[order]]
        old_prediction = indices[np.argmax(scores[:, indices], axis=1)]
        result['old_columns_training_accuracy'] = float(np.mean(old_prediction[mask] == labels[mask])) if mask.any() else None
    else: result['old_columns_training_accuracy'] = None
    return result


def score_change(before, after, labels, classes, old_classes):
    labels = np.asarray(labels, dtype=int); n = len(labels)
    order = np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]),dtype=int)
    a = order[np.argmax(before[:,order], axis=1)] if n else np.empty(0, dtype=int)
    b = order[np.argmax(after[:,order], axis=1)] if n else np.empty(0, dtype=int)
    result = dict(score_change_rms=float(np.linalg.norm(after-before)/math.sqrt(after.size)) if after.size else None,
        held_winner_change_count=int(np.sum(a != b)), physical_count=n,
        wrong_to_correct_count=int(np.sum((a != labels)&(b == labels))),
        correct_to_wrong_count=int(np.sum((a == labels)&(b != labels))))
    old = np.asarray([name in old_classes for name in classes], dtype=bool)
    for name, members in (('old', old[labels] if n else np.zeros(0, dtype=bool)),
                          ('new', ~old[labels] if n else np.zeros(0, dtype=bool))):
        result[name+'_winner_change_count'] = int(np.sum((a != b)&members))
        result[name+'_wrong_to_correct_count'] = int(np.sum((a != labels)&(b == labels)&members))
        result[name+'_correct_to_wrong_count'] = int(np.sum((a == labels)&(b != labels)&members))
    return result


def relative_change(current, initial):
    delta = float(np.linalg.norm(current-initial)); norm = float(np.linalg.norm(initial))
    return dict(absolute=delta, relative=delta/norm if norm else None,
        relative_unmeasured_reason=None if norm else 'ZERO_REFERENCE_NORM')


def head_metrics(ref, audit, arrays, initial_ref=None):
    data = arrays(ref); before = arrays(initial_ref) if initial_ref else data
    result = dict(scalars(audit), head_ref=ref, initial_head_ref=initial_ref,
        tau=scalar_array(data['tau']), gamma=scalar_array(data['gamma']), s0=scalar_array(data['s0']),
        actual_trace=scalar_array(data['actual_trace']), old_reference_count=int(np.count_nonzero(data['q'])),
        reference_measure_sum=float(data['q'].sum()), reference_measure_mean=float(data['q'].mean()) if data['q'].size else None,
        fixed_reference_measure_change_norm=float(np.linalg.norm(data['q']-before['q'])) if initial_ref else None,
        prior_held_score_norm=float(np.linalg.norm(data['M_held'])),
        prior_held_score_change_norm=float(np.linalg.norm(data['M_held']-before['M_held'])) if initial_ref else None,
        residual_held_score_norm=float(np.linalg.norm(data['scores']-data['M_held'])),
        residual_sample_mean_norm=float(np.linalg.norm(data['E'].mean(axis=0))) if len(data['E']) else None,
        fitted_old_reference_mean_norm=float(np.linalg.norm(data['q']@data['train_scores'])),
        old_reference_target_minus_prior_mean=(data['q']@data['E']).tolist(),
        old_reference_target_minus_prior_mean_norm=float(np.linalg.norm(data['q']@data['E'])),
        initial_comparison_unmeasured_reason=None if initial_ref else 'NO_INITIAL_HEAD_REFERENCE')
    for key in ('block_angle_radians', 'residual_sample_mean', 'fitted_old_reference_mean', 'prior_ref'):
        if key in audit: result[key] = audit[key]
    for key in ('K', 'L', 'alpha', 'E', 'M_train', 'M_held', 'center_mean'):
        result[key+'_norm'] = float(np.linalg.norm(data[key]))
    for key in ('tau', 'gamma', 's0', 'actual_trace'):
        original = scalar_array(before[key]) if initial_ref else None; current = scalar_array(data[key])
        result[key+'_initial'] = original
        result[key+'_change_from_initial'] = current-original if current is not None and original is not None else None
    gamma = result['gamma']; tau = result['tau']
    result['nuisance_unmeasured_reason'] = 'NO_OLD_KERNEL_INFORMATION' if gamma is None else None
    result['adapted_distance_unmeasured_reason'] = 'ZERO_BANDWIDTH_USES_ORIGINAL_EQUIVALENCE' if tau == 0 else None
    # Arrays contain the actual mixed distance used by the head. No new distance
    # function or adapter forward is called to create an unrecorded measurement.
    for key in ('distance', 'cross_distance', 'K', 'L', 'center_mean'):
        result[key+'_change_from_initial'] = relative_change(data[key],before[key]) if initial_ref else None
    result['held_score_change_from_initial'] = relative_change(data['scores'],before['scores']) if initial_ref else None
    for side in ('train', 'held'):
        changed = math.fsum(float(np.sum((data['adapted_'+side+'_'+part]-before['adapted_'+side+'_'+part])**2)) for part in ('b','a'))
        count = len(data['adapted_'+side+'_b'])
        result['adapted_'+side+'_geometry_change_rms'] = math.sqrt(changed/count) if count and initial_ref else None
    result['tangent_over_kappa'] = None
    result['tangent_unmeasured_reason'] = 'AJLR_ARCHIVE_DOES_NOT_STORE_TANGENT_OR_TANGENT_STATISTICS'
    result['adapted_only_distance_change'] = None
    result['adapted_only_distance_unmeasured_reason'] = 'ONLY_MIXED_USED_DISTANCE_IS_ARCHIVED'
    return result


def objective_metrics(value, ref, arrays, initial_ref, classes, old_classes):
    data = arrays(ref); initial = arrays(initial_ref)
    result = {name: value.get(name) if value else None for name in LOSS_FIELDS+FUNCTION_FIELDS}
    if not value:
        result.update(inner_training_physical_count=0, objective_unmeasured_reason='NO_ADAPTER_SUPERVISION')
        return result
    result.update(scalars(value))
    result.update(score_statistics(data['scores'], data['labels'], classes, old_classes,
        data['class_ce_means'], data['class_ce_counts']))
    result['class_ce_sums'] = data['class_ce_sums'].tolist()
    result['score_change_from_initial'] = score_change(initial['scores'], data['scores'], data['labels'], classes, old_classes)
    folds = value.get('inner_folds', []); initial_heads = value.get('_initial_head_refs', [])
    result['inner_fold_mechanisms'] = [head_metrics(fold['head_ref'], fold, arrays,
        initial_heads[i] if i < len(initial_heads) else fold['head_ref']) for i, fold in enumerate(folds)]
    result['inner_head_forward_seconds_display_sum'] = math.fsum(f.get('forward_seconds',0.) for f in folds)
    result['inner_ce_adjoint_seconds'] = math.fsum(f.get('adjoint_seconds',0.) for f in folds)
    result['forward_seconds_display_scope'] = 'CACHED_FOLD_TIMES_REPEAT_ON_GRADIENT_STEP_AND_FINAL_DO_NOT_SUM'
    for name in ('actual_trace', 'tau', 'gamma', 's0', 'prior_held_score_norm', 'residual_held_score_norm'):
        result['fold_'+name] = statistics([f.get(name) for f in result['inner_fold_mechanisms']])
    return result


def registration_metrics(initial_event, arrays, classes, old_classes, mode):
    if mode != 'C_seq': return dict(available=False, reason='B_HAS_ZERO_PRIOR_NOT_REGISTRATION')
    objective = initial_event.get('objective')
    if not objective:
        return dict(available=False, reason='NO_INNER_HEAD_Z0_ARCHIVE_FOR_NO_INFORMATION_STAGE',
            full_support_Z0_head_effect=None)
    pieces = [arrays(f['head_ref']) for f in objective['inner_folds']]
    prior = np.concatenate([p['M_held'] for p in pieces]); scores = np.concatenate([p['scores'] for p in pieces])
    labels = np.concatenate([p['held_labels'] for p in pieces])
    return dict(available=True, scope='SAME_INNER_TRAINING_HELD_PRIOR_VS_Z0_FITTED_RESIDUAL',
        prior=score_statistics(prior, labels, classes, old_classes),
        Z0_head=score_statistics(scores, labels, classes, old_classes),
        Z0_residual_effect=score_change(prior, scores, labels, classes, old_classes),
        prior_refs=[f.get('prior_ref') for f in objective['inner_folds']],
        full_support_Z0_head_effect=None,
        full_support_Z0_head_unmeasured_reason='FULL_FINAL_C_HEAD_AT_Z0_IS_NOT_SAVED_NO_REFIT_ALLOWED',
        causality='Within the fixed inner problem this is a paired descriptive head change; it is not an isolated experiment or independent generalization evidence')


def stage_diagnostic(stage, stream, prep, stage_log, arrays, ids):
    final = stream['final']; initial_event = stream['initial']; key = key_for(stage)
    initial_ref = stage['initialization_state_ref']; initial_data = arrays(initial_ref)
    initial_objective = initial_event.get('objective'); final_objective = final.get('final_objective')
    initial_heads = [f['head_ref'] for f in (initial_objective or {}).get('inner_folds', [])]
    classes, old = final['classes'], final['old_classes']; mode = MODES[stage['state']]
    def measures(obj, ref):
        if obj: obj = dict(obj, _initial_head_refs=initial_heads)
        result = objective_metrics(obj, ref, arrays, initial_ref, classes, old)
        result.update(parameter_metrics(ref, arrays, initial_data)); return result
    initial = measures(initial_objective, initial_ref)
    final_obj_ref = final.get('final_objective_state_ref') or final['final_state_ref']
    last = measures(final_objective, final_obj_ref)
    last.update({name: final[name] for name in FUNCTION_FIELDS if name in final})
    curves = []; gradient_by_ref = {}
    def point(kind, event, obj, ref, extra=None):
        metrics = measures(obj, ref)
        if extra: metrics.update(extra)
        curves.append(dict(identity(key), mode=mode, parent_k=stage['k'], new_count=stage['new_count'],
            model_seed=stage['model_seed'], cohort=stage['cohort'], receiver=stage['receiver'], scenario=stage['scenario'],
            kind=kind, iteration=event.get('iteration'), trial=event.get('trial'), accepted=event.get('accepted'),
            state_ref=ref, metrics=metrics, event_scalars=scalars(event)))
    point('initial' if initial_objective else 'no_information_initial', initial_event, initial_objective, initial_ref)
    for event in stream['events']:
        kind = event['event']; extra = {}
        if kind == 'AJLR_GRADIENT':
            extra = gradient_metrics(event, arrays); gradient_by_ref[event['state_ref']['path']] = event
        elif kind == 'AJLR_TRIAL':
            gradient = gradient_by_ref[event['gradient_state_ref']['path']]
            before, after = arrays(event['gradient_state_ref']), arrays(event['state_ref'])
            bound = event['loss_before']+1e-4*event['gradient_dot_delta']
            extra = dict(Z_actual_update_norm=float(np.linalg.norm(after['Z']-before['Z'])),
                U_actual_update_norm=float(np.linalg.norm(after['U']-before['U'])),
                nominal_function_coordinate_update_norm=event['step_size']*gradient['direction_norm'],
                armijo_rhs=bound, armijo_slack=bound+event['comparison_tolerance']-event['loss_after'],
                objective_nonincrease_slack=event['loss_before']+event['comparison_tolerance']-event['loss_after'],
                rejection_reason=None if event['accepted'] else
                    'ARMIJO_AND_OBJECTIVE_INCREASE' if not event['armijo_pass'] and not event['objective_nonincrease_pass'] else
                    'ARMIJO' if not event['armijo_pass'] else 'OBJECTIVE_INCREASE')
        point({'AJLR_GRADIENT':'gradient','AJLR_TRIAL':'trial','AJLR_STEP':'accepted_step'}[kind],
            event, event['objective'], event['state_ref'], extra)
    point('final_cached' if final_objective else 'no_information_final', final, final_objective, final_obj_ref,
        {name:final[name] for name in FUNCTION_FIELDS if name in final})
    trials = [p for p in curves if p['kind']=='trial']; gradients = [p for p in curves if p['kind']=='gradient']
    information = not final['no_information']; updated = final['optimizer_steps'] > 0
    # FINAL begins with preparation audit, then stage counters override shared
    # names. Exclude preparation-only counters; preserve the full event separately.
    prep_counts = counters(prep); all_counts = counters(final)
    work = {name:value for name,value in all_counts.items() if name in STAGE_COUNTERS or name not in prep_counts}
    names = sorted({name for name in set(initial)|set(last) if finite(initial.get(name)) or finite(last.get(name))})
    delta = {name:last[name]-initial[name] if finite(last.get(name)) and finite(initial.get(name)) else None for name in names}
    stage_metrics = dict(identity(key), mode=mode, parent_k=stage['k'], new_count=stage['new_count'],
        **{name:stage[name] for name in ('model_seed','cohort','receiver','scenario')},
        information_stage=information, updated_stage=updated, zero_update_information_stage=information and not updated,
        stop_reason=final['stop_reason'], initial=initial, final=last, final_minus_initial=delta,
        counts=work, all_event_counters=all_counts, final_event_scalars=scalars(final),
        stage_log_scalars=scalars(stage_log), coordinates=coordinate_metrics(prep, arrays),
        preparation_state_ref=prep['prepared_state_ref'], initialization_state_ref=initial_ref,
        final_state_ref=final['final_state_ref'], final_objective_state_ref=final.get('final_objective_state_ref'),
        accepted_Z_path_length=math.fsum(e['update_norm'] for e in stream['events'] if e['event']=='AJLR_STEP'),
        function_coordinate_path_bound=.5, function_coordinate_path_bound_scope='PRE_TANGENT_CURRENT_SUPPORT_4_TIMES_1_OVER_8',
        CE_vs_proximal_conflict_count=sum(p['metrics']['CE_vs_proximal_conflict'] is True for p in gradients),
        rejection_reasons=dict(Counter(p['metrics']['rejection_reason'] for p in trials if not p['accepted'])),
        accepted_objective_increase_count=sum(p['accepted'] is True and not p['event_scalars']['objective_nonincrease_pass'] for p in trials),
        registration_Z0_inner_effect=registration_metrics(initial_event, arrays, classes, old, mode),
        later_adapter_inner_effect=last.get('score_change_from_initial'),
        final_full_support_head=head_metrics(final['final_state_ref'], final['final_fit'], arrays),
        costs={name:value for name,value in scalars(stage_log).items() if name.endswith(('_bytes','_seconds'))},
        outer_inference_work=None,
        outer_inference_work_reason='PER_STAGE_COMPACT_LOG_OMITS_NESTED_WORKLOAD_USE_VERIFIED_RESOURCE_TOTALS_AND_STRATA',
        training_physical_ids=ids['training_physical_ids'], classes=classes, old_classes=old,
        source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        evidence_scope='INNER_SUPPORT_TRAINING_NOT_INDEPENDENT_VALIDATION')
    binding = (stage['row_id'], stage['scope'], stage['k'], stage['train_k'],
        tuple(classes), tuple(ids['training_physical_ids']))
    return stage_metrics, curves, binding


def aggregate(stages, dimensions):
    groups = {}
    for row in stages: groups.setdefault(tuple(row[name] for name in dimensions), []).append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda pair:repr(pair[0])):
        information = [v for v in values if v['information_stage']]
        result.append(dict(zip(dimensions,key), actual_stages=len(values),
            information_stages=len(information), updated_stages=sum(v['updated_stage'] for v in values),
            zero_update_information_stages=sum(v['zero_update_information_stage'] for v in values),
            stop_counts=dict(Counter(v['stop_reason'] for v in values)),
            rejection_counts=dict(sum((Counter(v['rejection_reasons']) for v in values),Counter())),
            latent_ranks=dict(Counter(str(v['coordinates']['latent_rank']) for v in values)),
            counts={name:sum(v['counts'].get(name,0) for v in values) for name in sorted({n for v in values for n in v['counts']})},
            snapshots={point:{field:statistics([v[point].get(field) for v in information])
                for field in LOSS_FIELDS+FUNCTION_FIELDS+('Z_norm','U_norm','U_anchor_distance')}
                for point in ('initial','final','final_minus_initial')},
            costs={name:statistics([v['costs'].get(name) for v in values]) for name in sorted({n for v in values for n in v['costs']})}))
    return result


def extract(value, run_root=None):
    """Second read-only phase. Derive scalar statistics, never fit or score."""
    require(value.get('status') == SNAPSHOT_STATUS, 'Complete training snapshot required')
    metadata = value['metadata']; validate_metadata(metadata)
    summary_stages = value['summary_stages']
    require(len(summary_stages)==EXPECTED_STAGES and len({key_for(s) for s in summary_stages})==EXPECTED_STAGES,
        'Missing/duplicate snapshot summary stages')
    streams = {}; preps = {}; stage_logs = {}; ids = {}; arrays_by_row = {}; event_counts = Counter(); sources = []
    prepared_metadata = {}
    for source in value['lanes']:
        row_id = source['row_id']; lane = Path(run_root)/row_id/'probe' if run_root else Path(source['lane'])
        require(row_id not in arrays_by_row, 'Duplicate snapshot row')
        arrays_by_row[row_id] = Arrays(lane)
        for row in source['prepared_training_metadata']:
            name = row['state'].removesuffix('_prepare')
            require(name in ('B','C'),'Unknown preparation metadata')
            key = key_for(dict(row,state='B_AJLR' if name=='B' else 'C_AJLR_seq'),row_id,True)
            require(key not in prepared_metadata,'Duplicate preparation metadata'); prepared_metadata[key] = row
        for row in source['final_training_identities']:
            key = key_for(row,row_id,True); require(key not in ids,'Duplicate final training identity'); ids[key] = row
        for row in source['stage_logs']:
            if row['event']=='CANDIDATE_FIT':
                key = key_for(row,row_id); require(key not in stage_logs,'Duplicate candidate scalar log'); stage_logs[key] = row
        for row in source['events']:
            kind = row['event']; event_counts[kind] += 1
            require(kind in ('AJLR_PREPARED','AJLR_INITIAL','AJLR_GRADIENT','AJLR_TRIAL','AJLR_STEP','AJLR_FINAL'),
                'Unknown AJLR training event schema')
            if kind == 'AJLR_PREPARED':
                name = row['state'].removesuffix('_prepare'); candidate = 'B_AJLR' if name=='B' else 'C_AJLR_seq'
                require(name in ('B','C'), 'Unknown preparation stage')
                key = key_for(dict(row,state=candidate),row_id,True)
                require(key not in preps,'Duplicate preparation event'); preps[key] = row; continue
            key = key_for(row,row_id,True); stream = streams.setdefault(key,dict(initial=None,final=None,events=[]))
            if kind in ('AJLR_INITIAL','AJLR_FINAL'):
                name = 'initial' if kind=='AJLR_INITIAL' else 'final'
                require(stream[name] is None,'Duplicate '+name+' training event'); stream[name] = row
            else: stream['events'].append(row)
        sources.append({key:item for key,item in source.items() if key not in
            ('events','stage_logs','final_training_identities','prepared_training_metadata')})
    expected = {key_for(stage) for stage in summary_stages}
    require(set(streams)==set(preps)==set(prepared_metadata)==set(stage_logs)==set(ids)==expected,
        'Training stage/preparation/log/identity coverage differs from complete summary')
    output_stages = []; curves = []; preparations = []; prior_heads = []; bindings = {}; prep_keys = set(); counts_by_log_kind = {}
    for source in value['lanes']:
        for log in source['stage_logs']:
            group = counts_by_log_kind.setdefault(log['event'],Counter()); group.update(counters(log))
    for stage in summary_stages:
        key = key_for(stage); stream = streams[key]
        require(stream['initial'] is not None and stream['final'] is not None,'Missing initial/final training event')
        for kind,field in (('AJLR_GRADIENT','gradients'),('AJLR_TRIAL','trials'),('AJLR_STEP','steps')):
            require(sum(e['event']==kind for e in stream['events'])==len(stage[field]),'Incomplete training curve events')
        arrays = arrays_by_row[stage['row_id']]; prep = preps[key]; prep_key = (stage['row_id'],prep['prepared_state_ref']['path'])
        require(prep_key not in prep_keys,'Repeated AJLR preparation archive context'); prep_keys.add(prep_key)
        diagnostic, points, binding = stage_diagnostic(stage,stream,prep,stage_logs[key],arrays,ids[key])
        output_stages.append(diagnostic); curves.extend(points)
        preparations.append(dict(identity(key), mode=diagnostic['mode'], parent_k=stage['k'],new_count=stage['new_count'],
            **{name:stage[name] for name in ('model_seed','cohort','receiver','scenario')},
            counts=counters(prep), scalar_measurements=scalars(prep), coordinates=diagnostic['coordinates'],
            prepared_state_ref=prep['prepared_state_ref'], actual_consuming_stages=1,
            final_problem=prepared_metadata[key].get('final_problem'),
            prior_folds=prepared_metadata[key].get('prior_folds',[])))
        for fold in prepared_metadata[key].get('prior_folds',[]):
            ref = fold['head_state_ref']; data = arrays(ref)
            prior_heads.append(dict(identity(key),inner_fold=fold['inner_fold'],
                source='FROZEN_ACTUAL_B_U_OLD_INNER_TRAIN_HEAD',head_ref=ref,
                forward_mechanisms=head_metrics(ref,fold.get('final_fit',{}),arrays),
                supervised_scores=score_statistics(data['scores'],data['held_labels'],
                    diagnostic['old_classes'],diagnostic['old_classes'])))
        if diagnostic['mode']=='B': bindings.setdefault(binding,[]).append(diagnostic)
    totals = dict(actual_stages=len(output_stages), preparation_records=len(preparations),
        information_stages=sum(s['information_stage'] for s in output_stages), updated_stages=sum(s['updated_stage'] for s in output_stages),
        zero_update_information_stages=sum(s['zero_update_information_stage'] for s in output_stages),
        curve_records=len(curves), unique_B_physical_bindings=len(bindings),
        prior_head_records=len(prior_heads),
        repeated_B_actual_contexts=sum(len(v)-1 for v in bindings.values()),
        scalar_stage_log_records=sum(len(l['stage_logs']) for l in value['lanes']),
        compact_training_event_records=sum(len(l['events']) for l in value['lanes']))
    strata_dimensions = dict(by_mode_train_k=('mode','train_k'),
        by_scope_k_new_count=('mode','scope','parent_k','new_count'),
        by_receiver_scene=('mode','cohort','receiver','scenario','parent_k','new_count'),
        by_model_cohort=('mode','model_seed','cohort','parent_k','new_count'),
        by_row=('mode','row_id','scope','parent_k','new_count'))
    strata = {name:aggregate(output_stages,dimensions) for name,dimensions in strata_dimensions.items()}
    curve_groups = {}
    for row in curves:
        key = (row['mode'],row['train_k'],row['kind'],row['iteration'],row['trial'],row['accepted'])
        curve_groups.setdefault(key,[]).append(row['metrics'])
    curve_stats = []
    for key, metrics in curve_groups.items():
        for name in sorted({n for row in metrics for n,item in row.items() if finite(item)}):
            curve_stats.append(dict(zip(('mode','train_k','kind','iteration','trial','accepted'),key),
                metric=name,**statistics([row.get(name) for row in metrics])))
    return dict(status=STATUS,run_id=metadata['run_id'],release_commit=metadata['release_commit'],
        source_summary=value['source_summary'],source_training_objectives=value['source_training_objectives'],
        sources=sources,coverage=metadata['coverage'],algorithm=metadata['algorithm'],totals=totals,
        event_counts=dict(event_counts),all_scalar_log_counter_totals={k:dict(v) for k,v in counts_by_log_kind.items()},
        stage_count_totals=dict(sum((Counter(s['counts']) for s in output_stages),Counter())),
        preparation_count_totals=dict(sum((Counter(p['counts']) for p in preparations),Counter())),
        resources=metadata['resources'],resource_statistics=metadata.get('resource_statistics',{}),
        state_archives=metadata['state_archives'],state_archive_file_count=metadata['state_archive_file_count'],
        state_archive_file_bytes=metadata['state_archive_file_bytes'],
        NPZ_files_read_for_diagnostics={row:len(arrays.read_paths) for row,arrays in arrays_by_row.items()},
        stages=output_stages,curves=curves,preparations=preparations,prior_heads=prior_heads,
        curve_statistics=curve_stats,strata=strata,
        B_binding_groups=[dict(representative=identity(tuple(v[0][name] for name in COORDS)),
            actual_contexts=len(v),physical_id_count=len(v[0]['training_physical_ids'])) for v in bindings.values()],
        deduplicated_B_statistics=aggregate([v[0] for v in bindings.values()],('mode','scope','parent_k','train_k')),
        limitations=[
            'Complete independently verified summary is the mathematical certificate. This collector derives descriptions without a second optimizer/head audit, fit, parameter selection or extra hashes/receipts.',
            'All compact training events, scalar fit-stage logs and summary training objectives are scanned. Full training events supplement identifiers, prior/final-problem preparation metadata and measured distribution summaries omitted by compact_event; raw parent traces and outer feature/score arrays are never opened.',
            'Class CE means pool all folds by physical class counts BEFORE RMS. Old/new CE and correctness are supervised diagnostics; inner-held labels trained the adapter and are not independent validation.',
            'CE gradient is g_Z-Z; proximal gradient is Z with no division by physical N. V is fixed, U remains original coordinates and Z uses actual retained rank, including empty coordinates.',
            'Displacement and the 0.5 accepted-path bound are PRE-TANGENT on current training support. They do not measure real score retention or imply a parameter ball.',
            'Fixed tau/gamma/s0 and q can be compared per inner head. Actual kernel trace and moving reference centers are measured separately. Null old scale and exact tau=0 are not replaced by epsilon.',
            'Angles are measured in the core. Mixed distances, kernels, scores and reference means are archived. Tangent/kappa and adapted-only distances are absent and stay null; no new forward is run to invent them.',
            'C inner Z0 score minus M_held measures registration residual effect at inherited U_B; final inner minus initial measures subsequent adapter/closed-head change on the same inner problem. The full final C head at Z0 is not archived, so its isolated full-support registration effect is N/A.',
            'Final full-support head retains actual B prior and residual measurements, but no independent outer score transitions are calculated. These paired inner changes do not isolate causality or establish generalization.',
            'Accepted STEP repeats its accepted TRIAL; cached GRADIENT/final displays reuse old head forward times. Resource costs come from final stage/preparation counters and verified resource totals, not sums over curve displays.',
            'Prior preparation/solve/scoring, student original solves, CE adjoints and baseline primal/EDF work remain separate. Reference pair counters are subsets of raw distance work and must not be added again.',
            'Per-stage nested outer score_workload is absent from compact logs. Already verified summary resources and resource_statistics preserve outer residual/prior inference work, timings and strata without opening scores.',
            'Real/proxy K1 still fits final closed heads but has no supervised adapter objective; Nnew=0 reuses B and creates no C stage. Actual B repeats are charged; exact physical-binding dedup is descriptive only and does not create independent evidence.',
            'Resident numeric bytes, minimum deployment numeric bytes, preparation/cache bytes and compressed NPZ bytes differ. Overlapping component bytes/times are not additive peak/work measurements. Package, transfer, GPU, source validation and unmeasured deployment values stay N/A.'])


def collect(summary_root, run_root=None):
    return extract(snapshot(summary_root,run_root))


def csv_value(value):
    if value is None: return 'N/A'
    if isinstance(value,(dict,list)): return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
    return str(value)


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({key:csv_value(row.get(key)) for key in fields} for row in rows)


def flat_stage(row):
    names = ('initial','final','final_minus_initial','counts','costs','coordinates')
    result = {key:item for key,item in row.items() if key not in names}
    for name in names: result.update({name+'_'+key:item for key,item in row[name].items()})
    return result


def write_json(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False); stream.write('\n')


def write_outputs(output, value):
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    collections = ('stages','curves','preparations','prior_heads','curve_statistics')
    compact = {key:item for key,item in value.items() if key not in collections}
    compact['complete_derived_streams'] = {name:name+'.jsonl' for name in collections}
    write_json(out/'summary.json',compact)
    for name in collections:
        with (out/(name+'.jsonl')).open('x',encoding='utf-8',newline='\n') as stream:
            for row in value[name]: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(out/'stages.csv',[flat_stage(row) for row in value['stages']])
    write_csv(out/'curves.csv',[dict({k:v for k,v in row.items() if k!='metrics'},**row['metrics']) for row in value['curves']])
    write_csv(out/'preparations.csv',value['preparations']); write_csv(out/'curve_statistics.csv',value['curve_statistics'])
    write_csv(out/'prior_heads.csv',value['prior_heads'])
    write_csv(out/'B_binding_groups.csv',value['B_binding_groups'])
    for name,rows in value['strata'].items(): write_csv(out/(name+'.csv'),rows)
    for name,rows in value['resource_statistics'].items(): write_csv(out/('resources_'+name+'.csv'),rows)
    write_csv(out/'archive_by_phase.csv',[dict(row_id=source['row_id'],phase=phase,**metrics)
        for source in value['state_archives'] for phase,metrics in source['by_phase'].items()])
    lines = ['# AJLR 完整训练机制诊断','', '状态：'+value['status']+'；run：'+value['run_id']+'。','',
        '全部训练流已读取，仅报告监督训练机制与实际工作量。','',
        '| 模式 | train K | 阶段 | 有信息 | 更新 | RMSCE 初→末 | prox 初→末 | total 初→末 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['strata']['by_mode_train_k']:
        cells = []
        for field in ('RMSCE','loss_proximal','loss_total'):
            a=row['snapshots']['initial'][field]['mean']; b=row['snapshots']['final'][field]['mean']
            cells.append('N/A' if a is None or b is None else f'{a:.6f}→{b:.6f}')
        lines.append('| '+' | '.join([row['mode'],str(row['train_k']),str(row['actual_stages']),
            str(row['information_stages']),str(row['updated_stages'])]+cells)+' |')
    lines += ['',f"实际阶段 {value['totals']['actual_stages']} 个；曲线记录 {value['totals']['curve_records']} 个；准备 {value['totals']['preparation_records']} 个。",'',
        '完整分层见 by_*.csv。C 的内层注册残差与后续 adapter 变化分列；全 support 的 Z=0 final C head 未保存，不能补拟合。',
        '梯度分解为 g_Z−Z；近端不除以 N。固定 nuisance、实际 trace、参考测度与缓存费用分别保留。',
        '实际成本、outer prior/residual 推理仅引用已验证的资源字段，不读取 outer scores。','']
    lines += ['- '+text for text in value['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def remote_operation(args, operation, request=None):
    require(bool(args.ssh_config) and bool(args.remote_python),'Explicit SSH config and NumPy-capable Python required')
    command = [args.remote_python,'-',operation]
    if args.summary_root: command += ['--summary-root',args.summary_root]
    if args.run_root: command += ['--run-root',args.run_root]
    source = Path(__file__).read_text(encoding='utf-8')
    if request is not None:
        import base64
        import gzip
        encoded = base64.b64encode(gzip.compress(
            json.dumps(request,ensure_ascii=False,allow_nan=False).encode('utf-8'),mtime=0)).decode('ascii')
        source = 'import base64, gzip, json\nSNAPSHOT_REQUEST = json.loads(gzip.decompress(base64.b64decode('+repr(encoded)+')).decode(\'utf-8\'))\n'+source
    result = subprocess.run(['ssh','-F',args.ssh_config,args.ssh_host,shlex.join(command)],
        input=source,text=True,encoding='utf-8',capture_output=True,timeout=1800)
    require(result.returncode==0,'Read-only remote '+operation+' failed: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-root'); parser.add_argument('--run-root'); parser.add_argument('--output')
    parser.add_argument('--snapshot-output'); parser.add_argument('--snapshot')
    parser.add_argument('--ssh-host'); parser.add_argument('--ssh-config'); parser.add_argument('--remote-python')
    parser.add_argument('--snapshot-stdout',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--extract-stdout',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.snapshot_stdout or args.extract_stdout:
        require(not args.ssh_host and not args.output and not args.snapshot_output,'Read-only stdout cannot write outputs')
        if args.snapshot_stdout:
            require(bool(args.summary_root) and not args.extract_stdout,'Snapshot requires verified summary')
            value=snapshot(args.summary_root,args.run_root)
        else:
            require('SNAPSHOT_REQUEST' in globals(),'Extraction requires captured training-only snapshot')
            value=extract(globals()['SNAPSHOT_REQUEST'],args.run_root)
        print(json.dumps(value,ensure_ascii=False,allow_nan=False)); return
    if args.snapshot_output:
        require(bool(args.summary_root) and not args.snapshot and not args.output,'Snapshot capture is a separate phase')
        require(not Path(args.snapshot_output).exists(),'Snapshot destination already exists')
        value=remote_operation(args,'--snapshot-stdout') if args.ssh_host else snapshot(args.summary_root,args.run_root)
        require(value.get('status')==SNAPSHOT_STATUS,'Remote snapshot status mismatch')
        write_json(args.snapshot_output,value)
        print(json.dumps(dict(status=value['status'],snapshot=args.snapshot_output,stages=len(value['summary_stages'])))); return
    require(bool(args.output) and not Path(args.output).exists(),'New exclusive --output is required')
    if args.snapshot:
        request=json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
        value=remote_operation(args,'--extract-stdout',request) if args.ssh_host else extract(request,args.run_root)
    else:
        require(bool(args.summary_root) and not args.ssh_host,'Remote collection requires the two explicit snapshot/extract phases')
        value=collect(args.summary_root,args.run_root)
    require(value.get('status')==STATUS,'Derived diagnostics status mismatch')
    write_outputs(args.output,value)
    print(json.dumps(dict(status=value['status'],totals=value['totals']),allow_nan=False))


if __name__=='__main__': main()
