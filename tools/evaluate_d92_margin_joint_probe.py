"""Draft MarginJointLocalRidge support paths with complete numeric evidence.

The independent candidate is awaiting root validation and publication.  This
entry accepts support caches only; it never opens query or source samples.
"""
import argparse
from collections.abc import Mapping
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import fit_branch_local_ridge
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from cvsrffi.d92_margin_joint_local_ridge import (
    FROZEN_CONFIG, prepare_margin_joint_training, fit_margin_joint_local_ridge,
    PREPARATION_COUNTERS, STAGE_COUNTERS, AUDIT_COUNTERS,
)
from evaluate_d92_registration_diagnostic import (
    CHANNEL, SCENARIOS, BRANCHES, split_identity,
    diagnose_evidence, measured_metrics as baseline_metrics, _safe,
)
from evaluate_d92_branch_support_probe import load_support, read, write, check, scalars, csv_record, peak_rss
from evaluate_d92_branch_local_ridge_probe import stage_csv_from_jsonl
from run_d92_margin_joint_probe import validate_selection, selected_tasks, validate_qp_resources

STATUS = 'MARGIN_JOINT_PROBE_COMPLETE'
SCOPE = 'SUPPORT_ONLY_MARGIN_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
SCHEMA = 'd92_margin_joint_local_ridge_v1'
METHOD = 'D92-MarginJointLocalRidge-v1'
IMPLEMENTATION_STATUS = 'DRAFT_AWAITING_ROOT_VALIDATION_AND_PUBLICATION'
PATHS = ('R0', 'R_MARGIN_seq')
PROBE_CONFIG = deepcopy(FROZEN_CONFIG)
PEAK_COUNTERS = frozenset(('margin_qp_peak_factor_buffer_bytes',
    'margin_qp_peak_explicit_solve_temporary_bytes'))
COUNTERS = tuple(dict.fromkeys(('episodes', 'k1_episodes', 'oof_episodes', 'proxy_anchor_count',
    'sequence_paths', 'baseline_head_fit_count', 'baseline_factorization_count',
    'baseline_head_triangular_solve_count', 'baseline_effective_df_triangular_solve_count', 'baseline_triangular_solve_count',
    'head_fit_count', 'factorization_count', 'trained_margin_stage_count', 'diagnostic_fit_count',
    'candidate_preparation_count', 'candidate_stage_count',
    'final_score_evaluation_count', 'final_score_physical_count')+tuple(AUDIT_COUNTERS)+tuple(sorted(PEAK_COUNTERS))))
METRICS = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy',
    'C_old_accuracy', 'C_new_accuracy', 'C_h', 'adaptation_gain_B_minus_A', 'support_adaptation_B_minus_B0',
    'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap',
    'C_new_minus_old', 'C_old_minus_R0', 'C_new_minus_R0', 'old_order_recovery_fraction',
    'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction') + tuple(
        'correctness_B_Cold_C_'+bits for bits in ('000', '100', '010', '110', '011', '111'))
MARGIN_FIELDS = ('margin_mean', 'margin_minus_R0', 'winner_changed_from_R0_fraction',
    'R0_correct_to_wrong_fraction', 'R0_wrong_to_correct_fraction')
METRICS += tuple(stage+'_'+field for stage in ('B_old', 'C_old', 'C_new') for field in MARGIN_FIELDS)



class StateArchive:
    """Exclusive numeric archives: JSON references retain every array coordinate."""
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root/'state_arrays'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = []

    def __call__(self, key, arrays):
        return self._save(key, arrays, allow_nonfinite=False)

    def failure(self, key, arrays):
        """Retain failed numeric states, including nonfinite coordinates, as NPZ."""
        return self._save(key, arrays, allow_nonfinite=True)

    def _save(self, key, arrays, *, allow_nonfinite):
        check(isinstance(key, str) and bool(key), 'State archive key must be nonempty text')
        check(isinstance(arrays, Mapping) and bool(arrays), 'State archive requires numeric arrays')
        numeric, metadata, summaries = {}, {}, {}
        for name, value in arrays.items():
            check(isinstance(name, str) and name and '/' not in name, 'Invalid state array name')
            array = np.asarray(value)
            check(array.dtype.kind in 'fbiu', 'Nonnumeric archived state')
            finite = bool(np.isfinite(array).all())
            check(allow_nonfinite or finite, 'Nonfinite archived successful state')
            check(array.dtype.kind != 'f' or array.dtype == np.dtype('float64'), 'State floats must be float64')
            numeric[name] = np.array(array, copy=True)
            metadata[name] = dict(shape=list(array.shape), dtype=str(array.dtype), nbytes=int(array.nbytes),
                all_finite=finite, nonfinite_count=int(np.sum(~np.isfinite(array))))
            norm = float(np.linalg.norm(array.reshape(-1))) if finite else None
            summaries[name] = dict(norm=norm if norm is not None and math.isfinite(norm) else None,
                minimum=float(np.min(array)) if finite and array.size else None,
                maximum=float(np.max(array)) if finite and array.size else None)
        relative = 'state_arrays/'+str(len(self.files)).zfill(8)+'.npz'
        path = self.root/relative
        tick = time.perf_counter()
        with path.open('xb') as stream:
            np.savez_compressed(stream, **numeric)
        ref = json_native(dict(key=key.rsplit('/', 1)[-1], namespace=key.rsplit('/', 1)[0] if '/' in key else None,
            path=relative, arrays=metadata, array_summaries=summaries, file_bytes=path.stat().st_size,
            archive_seconds=time.perf_counter()-tick, failed_numeric_state=allow_nonfinite))
        self.files.append(ref)
        return deepcopy(ref)

    def finalize(self, status):
        phases = {}
        for ref in self.files:
            phase = json.loads(ref['namespace'])['state'] if ref['namespace'] else 'unscoped'
            group = phases.setdefault(phase, dict(file_count=0, file_bytes=0, numeric_array_bytes=0, archive_seconds=0.))
            group['file_count'] += 1; group['file_bytes'] += ref['file_bytes']
            group['numeric_array_bytes'] += sum(a['nbytes'] for a in ref['arrays'].values())
            group['archive_seconds'] += ref['archive_seconds']
        manifest = dict(schema='d92_margin_joint_state_archive_v1', method=METHOD,
            prediction_formula='actual_B_prior_plus_full_support_margin_constrained_affine_residual', status=status, files=self.files,
            file_count=len(self.files), total_file_bytes=sum(ref['file_bytes'] for ref in self.files),
            numeric_array_bytes=sum(a['nbytes'] for ref in self.files for a in ref['arrays'].values()),
            archive_seconds=sum(ref['archive_seconds'] for ref in self.files), by_phase=phases)
        with (self.root/'state_manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(json_native(manifest), stream, ensure_ascii=False, allow_nan=False)
        return manifest


def json_native(value, path='$'):
    """Convert only NumPy/container data types at the public output boundary."""
    if isinstance(value, np.generic): return json_native(value.item(), path)
    if isinstance(value, np.ndarray): return json_native(value.tolist(), path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            native_key = json_native(key, path+'<key>')
            if not isinstance(native_key, str): raise TypeError(path+': JSON object key must be a string')
            result[native_key] = json_native(item, path+'.'+native_key)
        return result
    if isinstance(value, (list, tuple)):
        return [json_native(item, path+'['+str(index)+']') for index, item in enumerate(value)]
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float:
        if not math.isfinite(value): raise ValueError(path+': nonfinite JSON number')
        return value
    raise TypeError(path+': unsupported JSON value '+type(value).__module__+'.'+type(value).__name__)


def held_observations(ids, classes, scores, labels):
    """Post-fit outer-held diagnostics only; never called by the fitting API."""
    values = np.asarray(scores, dtype=np.float64)
    check(values.shape == (len(ids), len(classes)), 'Held score shape mismatch')
    check(np.isfinite(values).all(), 'Nonfinite held scores')
    result = {}
    for pid, row in zip(ids, values):
        target = classes.index(labels[pid]); pred = min(classes[j] for j in np.flatnonzero(row == np.max(row)))
        wrong = [value for j, value in enumerate(row) if j != target]
        result[pid] = dict(predicted_class=pred, correct=pred == labels[pid],
            true_class_margin=float(row[target]-max(wrong)) if wrong else None)
    return result


def held_comparisons(evidence, reference, old):
    result = {}; metrics = {}
    for stage, ids, classes, scores, ref_scores in (
        ('B_old', evidence['b_ids'], evidence['b_classes'], evidence['b_scores'], reference['b_scores']),
        ('C_old', [pid for pid in evidence['c_ids'] if evidence['held_labels'][pid] in old],
            evidence['c_classes'], evidence['c_scores'], reference['c_scores']),
        ('C_new', [pid for pid in evidence['c_ids'] if evidence['held_labels'][pid] not in old],
            evidence['c_classes'], evidence['c_scores'], reference['c_scores'])):
        all_ids = evidence['b_ids'] if stage == 'B_old' else evidence['c_ids']
        actual = held_observations(all_ids, classes, scores, evidence['held_labels'])
        baseline = held_observations(all_ids, classes, ref_scores, evidence['held_labels'])
        pairs = [dict(physical_id=pid, actual=actual[pid], R0=baseline[pid]) for pid in ids]
        result[stage] = dict(evidence_scope='OUTER_HELD_AFTER_ALL_UPDATES', records=pairs)
        n = len(pairs); margins = [p['actual']['true_class_margin'] for p in pairs]
        refs = [p['R0']['true_class_margin'] for p in pairs]
        known = n > 0 and all(v is not None for v in margins+refs)
        metrics.update({stage+'_margin_mean': sum(margins)/n if known else None,
            stage+'_margin_minus_R0': sum(a-b for a, b in zip(margins, refs))/n if known else None,
            stage+'_winner_changed_from_R0_fraction': sum(p['actual']['predicted_class'] != p['R0']['predicted_class'] for p in pairs)/n if n else None,
            stage+'_R0_correct_to_wrong_fraction': sum(p['R0']['correct'] and not p['actual']['correct'] for p in pairs)/n if n else None,
            stage+'_R0_wrong_to_correct_fraction': sum(not p['R0']['correct'] and p['actual']['correct'] for p in pairs)/n if n else None})
    return result, metrics


def path_metrics(diagnostic, reference):
    old, new = diagnostic['C_full_acc'], diagnostic['Cnew_acc']; base = baseline_metrics(diagnostic)
    result = dict(A_old_accuracy=None, B0_old_accuracy=reference['B_acc'], B_old_accuracy=diagnostic['B_acc'],
        C_old_columns_accuracy=diagnostic['C_old_columns_acc'], C_old_accuracy=old, C_new_accuracy=new,
        C_h=diagnostic['C_harmonic_mean'], adaptation_gain_B_minus_A=None,
        support_adaptation_B_minus_B0=diagnostic['B_acc']-reference['B_acc'],
        C_abs_new_old_gap=None if new is None else abs(new-old), C_new_minus_old=None if new is None else new-old,
        C_old_minus_R0=old-reference['C_full_acc'], C_new_minus_R0=None if new is None else new-reference['Cnew_acc'],
        **{key: base[key] for key in ('old_order_change', 'new_competition_loss', 'total_old_accuracy_drop',
            'old_order_recovery_fraction', 'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction')})
    for bits in ('000', '100', '010', '110', '011', '111'):
        result['correctness_B_Cold_C_'+bits] = sum(''.join(str(int(row[key])) for key in
            ('B_correct', 'C_old_columns_correct', 'C_full_correct')) == bits for row in diagnostic['old_records'])/diagnostic['old_held_count']
    return result


def assess_paths(evidence, old):
    diagnostics = {name: diagnose_evidence(value, old) for name, value in evidence.items()}
    result = {}
    for name, value in diagnostics.items():
        comparisons, margins = held_comparisons(evidence[name], evidence['R0'], old)
        result[name] = dict(diagnostic=value, held_comparisons=comparisons,
            metrics=dict(path_metrics(value, diagnostics['R0']), **margins))
    return result


def parent_mean(trials):
    return {name: {metric: None if trials[0]['paths'][name]['metrics'][metric] is None else
        sum(t['paths'][name]['metrics'][metric] for t in trials)/len(trials) for metric in METRICS} for name in PATHS}


def pooled_assess(entries, labels, classes, old):
    evidence = {}
    for name in PATHS:
        brows, crows = {}, {}
        for entry in entries:
            check(not set(brows).intersection(entry['b_ids']) and not set(crows).intersection(entry['c_ids']), 'Duplicate physical held ID')
            brows.update(zip(entry['b_ids'], entry['paths'][name]['b_scores']))
            crows.update(zip(entry['c_ids'], entry['paths'][name]['c_scores']))
        evidence[name] = dict(b_ids=sorted(brows), b_classes=old, b_scores=[brows[i] for i in sorted(brows)],
            c_ids=sorted(crows), c_classes=classes, c_scores=[crows[i] for i in sorted(crows)], held_labels=labels)
    return assess_paths(evidence, old)


_write_native_json = write


def write(path, value):
    return _write_native_json(path, json_native(value))


def compact_event(event):
    """Keep measured scalars and complete NPZ references without copying matrices."""
    event = json_native(event)
    result = scalars(event)
    for key, value in event.items():
        if key in ('state_ref', 'head_ref', 'head_refs', 'prior_ref') or key.endswith('_state_ref') or key.endswith('_head_ref'):
            result[key] = value
    for key in ('objective', 'initial_objective', 'final_objective', 'final_fit'):
        value = event.get(key)
        if isinstance(value, dict):
            result[key] = compact_event(value)
    for key in ('class_ce_means', 'class_ce_sums', 'class_ce_counts', 'held_ce_sums', 'held_ce_counts', 'classes', 'old_classes'):
        if key in event: result[key] = event[key]
    if isinstance(event.get('inner_folds'), list):
        result['inner_folds'] = [compact_event(value) for value in event['inner_folds']]
    return result


def validate_spec(spec):
    from run_d92_margin_joint_probe import validate_spec as validate
    return validate(spec)


def verify_marker(path, spec, row):
    from run_d92_margin_joint_probe import verify_marker as verify
    return verify(path, spec, row)


def _account(result, audit, keys):
    for key in keys:
        check(type(audit[key]) is int and audit[key] >= 0, 'Invalid core workload: '+key)
        result[key] = max(result[key], audit[key]) if key in PEAK_COUNTERS else result[key]+audit[key]


def _totals(result):
    result['candidate_preparation_count'] = result['ajlr_preparation_count']
    result['candidate_stage_count'] = result['ajlr_stage_count']
    result['head_fit_count'] = sum(result[key] for key in
        ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'prior_head_fit_count'))
    result['factorization_count'] = sum(result[key] for key in
        ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'prior_factorization_count'))


def probe_margin_joint(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                       classes, old_classes, max_transitions, max_factor_buffer_bytes,
                       log_callback=None, event_callback=None, context=None, state_callback=None):
    started = time.perf_counter()
    qp_resources = dict(max_transitions=max_transitions, max_factor_buffer_bytes=max_factor_buffer_bytes)
    validate_qp_resources(qp_resources)
    context = dict(context or {})
    # A direct synthetic/in-memory invocation creates its own inheritance scope.
    # The executable cache entry requires explicit launch-owner run/row binding.
    token = uuid.uuid4().hex
    context.setdefault('run_id', 'in_memory_support_probe_'+token)
    context.setdefault('row_id', 'in_memory_support_parent_'+token)
    check(all(isinstance(context[key], str) and bool(context[key]) for key in ('run_id', 'row_id')),
        'Explicit nonempty run/row inheritance binding required')
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    check(bool(old), 'Old registry required'); classes, old = list(canonical), sorted(old)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {key: np.asarray(value)[order] for key, value in zip(BRANCHES, (z_id, fft, t_emb, f_emb, pa_local))}
    label_by_id = {pid: classes[int(y)] for pid, y in zip(ids, labels)}
    is_old = np.asarray([label_by_id[pid] in old for pid in ids]); folds = 0 if k == 1 else min(k, 3)
    result = dict(schema=SCHEMA, method=METHOD, inheritance_binding=dict(context), qp_resources=qp_resources,
        classes=classes, old_classes=old, support_count=len(ids), old_class_count=len(old),
        new_class_count=len(classes)-len(old), fold_count=folds, physical_fold_assignment=[],
        numerical=interaction._diagonal_stats(b, a), folds=[], oof=None, oneshot_proxy=None, full_support=None,
        **dict.fromkeys(COUNTERS[4:], 0), persistent_state_bytes=0,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    positions = [np.flatnonzero(labels == i) for i in range(len(classes))]
    assignments = np.full(len(ids), -1, dtype=int)
    if folds:
        for indices in positions: assignments[indices] = np.arange(k) % folds
        result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=label_by_id[pid], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    completed = []

    def path(keep, scope, index):
        reuse = classes == old; has_held = bool((~keep).any())
        entry = dict(scope=scope, fold=index if scope == 'support_oof' else None,
            trial=index if scope == 'support_oneshot_proxy' else None, parent_k=k,
            train_k=int(keep.sum())//len(classes), held_k=int((~keep).sum())//len(classes),
            b_training_ids=[pid for i, pid in enumerate(ids) if keep[i] and is_old[i]],
            c_training_ids=[pid for i, pid in enumerate(ids) if keep[i]],
            b_ids=[pid for i, pid in enumerate(ids) if not keep[i] and is_old[i]],
            c_ids=[pid for i, pid in enumerate(ids) if not keep[i]], b_classes=old, c_classes=classes,
            held_labels={pid: label_by_id[pid] for i, pid in enumerate(ids) if not keep[i]},
            stages=[], preparations=[], candidate_stages=[], training_events=[], c_reuses_b0=reuse,
            c_reuses_b_candidates=reuse, paths={}, outer_features_state_ref=None)
        coords = dict(context or {}, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        bases, prepared, candidates = {}, {}, {}
        def emit_event(row, stage):
            value = dict(row); value.update({key: item for key, item in coords.items() if key != 'trial'},
                schema=SCHEMA, method=METHOD, outer_trial=coords['trial'], state=stage, source_validation=None,
                source_validation_reason='SOURCE_ACCESS_FORBIDDEN', objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
            entry['training_events'].append(value)
            if event_callback: event_callback(value)
        def archive_callback(stage):
            if state_callback is None: return None
            namespace = json.dumps(json_native(dict(coords, state=stage)), sort_keys=True, separators=(',', ':'))
            return lambda key, arrays: state_callback(namespace+'/'+key, arrays)
        try:
            outer_archive = archive_callback('OUTER_SUPPORT_HELD')
            if outer_archive is not None and has_held:
                entry['outer_features_state_ref'] = outer_archive('features', {key: np.asarray(value[~keep], dtype=np.float64) for key, value in raw.items()})
            for name, train, held, registry in (('B0', keep & is_old, ~keep & is_old, old), ('C0', keep, ~keep, classes)):
                if name == 'C0' and reuse: bases[name] = bases['B0']; continue
                tids = [pid for i, pid in enumerate(ids) if train[i]]
                ys = np.asarray([registry.index(label_by_id[pid]) for pid in tids], dtype=np.int64)
                tick = time.perf_counter()
                state = fit_branch_local_ridge(**{key: value[train] for key, value in raw.items()},
                    support_labels=ys, support_ids=tids, classes=registry, old_classes=old, arm='local_ridge')
                top = state.audit_dict(); audit = top['final_fit']
                audit.update(coords, state=name, training_physical_ids=tids, all_states_estimated_from_trainfold_only=True,
                    held_physical_count=int(held.sum()), class_count=len(registry), learning_rate=None,
                    learning_rate_reason='Closed-form baseline; no optimizer', source_validation=None,
                    source_validation_reason='SOURCE_ACCESS_FORBIDDEN', score_seconds=0., persistent_state_bytes=top['persistent_state_bytes'])
                ref = archive_callback(name)
                audit['final_state_ref'] = None if ref is None else ref('baseline', dict(
                    original_train_b=state.support_background, original_train_a=state.support_auxiliary,
                    alpha=state.alpha, reference_kernel=state.reference_kernel,
                    reference_self=np.asarray(state.reference_self), center_mean=state.center_mean,
                    center_grand=np.asarray(state.center_grand),
                    tau=np.asarray([] if state.bandwidth_tau is None else [state.bandwidth_tau], dtype=np.float64),
                    gamma=np.asarray([] if state.trace_scale is None else [state.trace_scale], dtype=np.float64)))
                entry['stages'].append(audit); result['baseline_head_fit_count'] += 1
                result['baseline_factorization_count'] += audit['factorization_calls']
                result['baseline_head_triangular_solve_count'] += 2*audit['factorization_calls']
                result['baseline_effective_df_triangular_solve_count'] += audit['effective_degrees_of_freedom_extra_triangular_solves']
                result['baseline_triangular_solve_count'] = result['baseline_head_triangular_solve_count']+result['baseline_effective_df_triangular_solve_count']
                bases[name] = dict(state=state, train=train, held=held, registry=registry, labels=ys, ids=tids, scores=np.empty((0, len(registry))))
                if held.any():
                    score_start = time.perf_counter(); result['final_score_evaluation_count'] += 1
                    result['final_score_physical_count'] += int(held.sum())
                    bases[name]['scores'] = state.score(**{key: value[held] for key, value in raw.items()})
                    audit['score_seconds'] = time.perf_counter()-score_start
                audit['fit_and_score_seconds'] = time.perf_counter()-tick
                if log_callback: log_callback(dict(event='BASE_FIT', **audit))
            for prep_name, name, mode in (('B', 'B_MARGIN', 'B'), ('C', 'C_MARGIN_seq', 'C_seq')):
                if prep_name == 'C' and reuse: candidates[name] = candidates['B_MARGIN']; continue
                base = bases[prep_name+'0']
                prepared[prep_name] = prepare_margin_joint_training(**{key: value[base['train']] for key, value in raw.items()},
                    support_labels=base['labels'], support_ids=base['ids'], classes=base['registry'], old_classes=old,
                    inherited=None if prep_name == 'B' else candidates['B_MARGIN']['state'], context=dict(coords, stage=prep_name),
                    **qp_resources,
                    log_callback=lambda row, current=prep_name: emit_event(row, current+'_prepare'),
                    state_callback=archive_callback(prep_name+'_prepare'))
                prep = prepared[prep_name].audit_dict()
                prep.update(coords, state=prep_name, training_physical_ids=base['ids'],
                    train_physical_count=len(base['ids']), class_count=len(base['registry']),
                    inherited_adapter_from=None if prep_name == 'B' else 'B_MARGIN', objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                entry['preparations'].append(prep); _account(result, prep, PREPARATION_COUNTERS)
                if log_callback: log_callback(dict(event='MARGIN_PREPARATION', **prep))
                state = fit_margin_joint_local_ridge(prepared[prep_name], mode=mode, baseline_state=base['state'],
                    log_callback=lambda row, current=name: emit_event(row, current), state_callback=archive_callback(name))
                audit = state.audit_dict()
                audit['persistent_state_bytes'] = audit['resident_numeric_state_bytes']
                audit.update(coords, state=name, preparation_ref=prep_name, mode=mode,
                    training_physical_ids=base['ids'], train_physical_count=len(base['ids']), class_count=len(base['registry']),
                    held_physical_count=int(base['held'].sum()), source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
                    score_seconds=0., score_workload=None, score_workload_unavailable_reason='NO_OUTER_HELD_INFERENCE' if not base['held'].any() else None,
                    objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                entry['candidate_stages'].append(audit); candidates[name] = dict(state=state, scores=np.empty((0, len(base['registry']))))
                _account(result, audit, STAGE_COUNTERS)
                _account(result, audit, tuple(sorted(PEAK_COUNTERS)))
                result['trained_margin_stage_count'] += int(audit['optimizer_steps'] > 0)
                if base['held'].any():
                    score_start = time.perf_counter(); result['final_score_evaluation_count'] += 1
                    result['final_score_physical_count'] += int(base['held'].sum())
                    candidates[name]['scores'], audit['score_workload'] = state.score_with_audit(**{key: value[base['held']] for key, value in raw.items()})
                    audit['score_seconds'] = time.perf_counter()-score_start
                if log_callback: log_callback(dict(event='CANDIDATE_FIT', **audit))
            common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
            evidence = dict(R0=dict(common, b_scores=bases['B0']['scores'].tolist(), c_scores=bases['C0']['scores'].tolist()),
                R_MARGIN_seq=dict(common, b_scores=candidates['B_MARGIN']['scores'].tolist(), c_scores=candidates['C_MARGIN_seq']['scores'].tolist()))
            entry['prediction_status'] = 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN' if has_held else 'NO_HELD_PREDICTIONS'
            if has_held:
                fresh = assess_paths(evidence, old)
                entry['paths'] = {name: dict(b_scores=value['b_scores'], c_scores=value['c_scores'], **fresh[name]) for name, value in evidence.items()}
            else:
                entry['paths'] = {name: dict(b_scores=[], c_scores=[], diagnostic=None, held_comparisons=None,
                    metrics=dict.fromkeys(METRICS), evidence_scope='FULL_SUPPORT_HEAD_NO_INDEPENDENT_HELD') for name in PATHS}
            final_audit = candidates['C_MARGIN_seq']['state'].audit_dict()
            entry['deployment_C_state_bytes'] = dict(R_MARGIN_seq=final_audit['resident_numeric_state_bytes'])
            entry['minimum_deployment_numeric_state_bytes'] = dict(R_MARGIN_seq=final_audit['deployment_numeric_state_bytes'])
            result['persistent_state_bytes'] = max(result['persistent_state_bytes'], entry['deployment_C_state_bytes']['R_MARGIN_seq'])
        except Exception as exc:
            _totals(result)
            audit_method = getattr(exc, 'audit_dict', None)
            failed_fit = audit_method() if callable(audit_method) else getattr(exc, 'audit', None)
            arrays = getattr(exc, 'arrays', None)
            arrays_method = getattr(exc, 'to_arrays', None)
            if arrays is None and callable(arrays_method):
                arrays = arrays_method()
            failed_arrays_ref = failed_fit.get('failure_state_ref') if isinstance(failed_fit, Mapping) else None
            if failed_arrays_ref is None and isinstance(arrays, Mapping) and arrays and state_callback is not None:
                saver = getattr(state_callback, 'failure', state_callback)
                namespace = json.dumps(json_native(dict(coords, state='FAILED_FIT')), sort_keys=True, separators=(',', ':'))
                failed_arrays_ref = saver(namespace+'/failed_numeric_state', arrays)
            measured_failed_counters = {key: failed_fit[key] for key in tuple(AUDIT_COUNTERS)+tuple(sorted(PEAK_COUNTERS))
                if isinstance(failed_fit, Mapping) and key in failed_fit}
            exc.registration_context = _safe(dict(current_path=entry, failed_fit=failed_fit,
                failed_fit_audit_unavailable_reason=None if failed_fit is not None else 'EXCEPTION_HAS_NO_MEASURED_FIT_AUDIT',
                failed_numeric_state_ref=failed_arrays_ref, failed_fit_counters=measured_failed_counters or None,
                counters_scope='COMPLETED_BASELINES_PREPARATIONS_AND_CANDIDATE_STAGES_BEFORE_FAILURE',
                failed_fit_counters_scope='SEPARATE_PARTIAL_FIT_AUDIT_NOT_ADDED_TO_COMPLETED_COUNTERS',
                workload_complete=False, qp_resources=qp_resources,
                completed_preparations={name: value.audit_dict() for name, value in prepared.items()},
                completed_candidate_states={name: value['state'].audit_dict() for name, value in candidates.items()},
                completed_paths=completed, counters={key: result[key] for key in COUNTERS[4:]}))
            raise
        result['sequence_paths'] += 1; _totals(result); completed.append(entry)
        return entry

    if k == 1:
        result['full_support'] = path(np.ones(len(ids), dtype=bool), 'support_full_k1', None)
    else:
        for fold in range(folds): result['folds'].append(path(assignments != fold, 'support_oof', fold))
        result['oof'] = dict(paths=pooled_assess(result['folds'], label_by_id, classes, old), aggregation='one_record_per_physical_held_id')
        trials = []
        for trial in range(k):
            keep = np.zeros(len(ids), dtype=bool)
            for indices in positions: keep[indices[trial]] = True
            trials.append(path(keep, 'support_oneshot_proxy', trial))
        result['oneshot_proxy'] = dict(trials=trials, trial_count=k, parent_mean_metrics=parent_mean(trials),
            aggregation='all_anchors_mean_within_parent_then_equal_parent', proxy_train_k=1)
    result['fit_seconds'] = time.perf_counter()-started
    return result


def compact_record(record):
    keys = ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'new_count', 'support_count',
        'old_class_count', 'new_class_count', 'fold_count', 'fit_seconds', 'persistent_state_bytes', 'heldout_unavailable_reason')+COUNTERS[4:]
    return json_native(dict({key: record[key] for key in keys},
        oof=None if record['oof'] is None else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if record['oneshot_proxy'] is None else record['oneshot_proxy']['parent_mean_metrics'],
        full_support_head_fitted=record['full_support'] is not None,
        proxy_anchor_count=0 if record['oneshot_proxy'] is None else record['oneshot_proxy']['trial_count'],
        schema=SCHEMA, method=METHOD, scope=SCOPE, query_rows_used=0, source_rows_used=0))


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed, run_id, row_id):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    check(set(config) == {'algorithm', 'producer_matrix', 'selection', 'qp_resources'} and config['algorithm'] == PROBE_CONFIG == FROZEN_CONFIG,
        'Fixed MARGIN_JOINT code/config mismatch')
    validate_qp_resources(config['qp_resources'])
    check(all(isinstance(value, str) and bool(value) for value in (run_id, row_id)),
        'Explicit nonempty run/row inheritance binding required')
    validate_selection(config['selection']); manifest = read(Path(capsule)/'manifest.json')
    check(all(manifest.get('channel', {}).get(key) == value for key, value in CHANNEL.items())
        and manifest.get('scenarios') == SCENARIOS, 'Practical residual channel mismatch')
    begin = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['producer_matrix']))
    check(len(old) == 6, 'Fixed six-old-class pilot required'); chosen = selected_tasks(tasks, config['selection'], old)
    binding = dict(run_id=run_id, row_id=row_id, capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=expected_model_seed, channel=deepcopy(manifest['channel']), scenarios=manifest['scenarios'])
    payload = dict(feature_cache_reused=True, checkpoint_loaded=False, native_physical_forward_count_this_run=0,
        new_source_payload_bytes=0, new_ground_statistics_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=time.perf_counter()-begin,
        deployment_package_bytes=None, incremental_transfer_bytes=None, unmeasured_reason='No deployment package serialized or transmitted')
    startup = dict(binding, schema=SCHEMA, method=METHOD, implementation_status=IMPLEMENTATION_STATUS,
        qp_resources=deepcopy(config['qp_resources']),
        scope=SCOPE, config=config, episodes=len(chosen), producer_episodes=len(tasks),
        support_features=str(support_features), provenance=provenance, payload_audit=payload,
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        checkpoint_loaded=False, encoder_updated=False, adapted_state_inherited=True,
        actual_A=None, actual_A_unavailable_reason='No matched legal ground A predictor in support pilot',
        source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION',
        argv=sys.argv, pid=os.getpid(), python=sys.executable,
        hardware=dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(), dtype='float64', gpu_use=False),
        triangular_rhs_count_scope='sum_per_actual_solve_columns_r; elements=n*r; dense_work_units=n*n*r_not_measured_FLOPs',
        blas_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    out.mkdir(parents=True, exist_ok=False); write(out/'startup.json', startup)
    startup = json_native(startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    totals = dict.fromkeys(COUNTERS, 0); peak_state = 0
    archive = StateArchive(out); archive_status = 'INCOMPLETE'
    try:
        with (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
             (out/'training_events.jsonl').open('x', encoding='utf-8') as events, \
             (out/'training_events_compact.jsonl').open('x', encoding='utf-8') as compactevents, \
             (out/'training.log').open('x', encoding='utf-8') as textlog:
            textlog.write('STARTUP '+json.dumps(startup, allow_nan=False)+'\n'); textlog.flush()
            writer = None
            for split, positions, labels in chosen:
                def log(stage):
                    row = dict(compact_event(stage), schema=SCHEMA, method=METHOD, split_id=split['split_id'])
                    line = json.dumps(row, allow_nan=False)
                    stages.write(line+'\n'); stages.flush(); textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                def event(row):
                    full = json_native(dict(row, split_id=split['split_id']))
                    events.write(json.dumps(full, allow_nan=False)+'\n'); events.flush()
                    small = compact_event(full); line = json.dumps(small, allow_nan=False)
                    compactevents.write(line+'\n'); compactevents.flush()
                    textlog.write('MARGIN_JOINT_TRAINING '+line+'\n'); textlog.flush(); print('MARGIN_JOINT_TRAINING '+line, flush=True)
                try:
                    audit = probe_margin_joint(**{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'], classes=split['registered_classes'], old_classes=old,
                        **config['qp_resources'],
                        log_callback=log, event_callback=event, context=dict(run_id=run_id, row_id=row_id, split_id=split['split_id']),
                        state_callback=archive)
                except Exception as exc:
                    write(out/'probe_failed.json', _safe(dict(status='MARGIN_JOINT_PROBE_FAILED', **binding,
                        split_identity=split_identity(split, old), error_type=type(exc).__name__, error=str(exc),
                        failure_context=getattr(exc, 'registration_context', {}), completed_episodes=totals['episodes'],
                        completed_episode_counters=totals, counters_scope='COMPLETED_EPISODES_ONLY', workload_complete=False,
                        qp_resources=config['qp_resources'],
                        query_rows_used=0, source_rows_used=0)))
                    raise
                record = json_native(dict(audit, **split_identity(split, old), scope=SCOPE, query_rows_used=0, source_rows_used=0))
                trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush(); small = compact_record(record)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                if writer is None: writer = csv.DictWriter(csvfile, fieldnames=list(small)); writer.writeheader()
                writer.writerow(csv_record(small)); csvfile.flush()
                for key in COUNTERS[4:]:
                    totals[key] = max(totals[key], audit[key]) if key in PEAK_COUNTERS else totals[key]+audit[key]
                totals['episodes'] += 1; totals['k1_episodes'] += int(split['k'] == 1)
                totals['oof_episodes'] += int(split['k'] > 1); totals['proxy_anchor_count'] += small['proxy_anchor_count']
                peak_state = max(peak_state, audit['persistent_state_bytes'])
                print(json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', **small, completed=totals['episodes']), allow_nan=False), flush=True)
            archive_status = 'COMPLETE'
    finally:
        for name in ('fit_stages', 'training_events_compact'):
            if (out/(name+'.jsonl')).exists(): stage_csv_from_jsonl(out/(name+'.jsonl'), out/(name+'.csv'))
        state_manifest = archive.finalize(archive_status)
    marker = dict(binding, **totals, status=STATUS, schema=SCHEMA, method=METHOD,
        qp_resources=deepcopy(config['qp_resources']), counters_scope='ALL_COMPLETED_EPISODES', workload_complete=True,
        implementation_status=IMPLEMENTATION_STATUS, scope=SCOPE, algorithm=PROBE_CONFIG,
        selection=config['selection'], producer_matrix=config['producer_matrix'], payload_audit=payload,
        query_rows_used=0, source_rows_used=0, truth_read=False, persistent_state_bytes=peak_state,
        persistent_state_scope='maximum_retained_C_resident_numeric_state_including_actual_B_not_minimal_deployment',
        triangular_rhs_count_scope='sum_per_actual_solve_columns_r; elements=n*r; dense_work_units=n*n*r_not_measured_FLOPs',
        state_archive_file_count=state_manifest['file_count'], state_archive_file_bytes=state_manifest['total_file_bytes'],
        state_archive_numeric_bytes=state_manifest['numeric_array_bytes'], state_archive_seconds=state_manifest['archive_seconds'],
        state_manifest='state_manifest.json',
        peak_gpu_memory_bytes=None, gpu_memory_reason='CPU only', wall_seconds=time.perf_counter()-begin,
        peak_process_rss_bytes=peak_rss())
    marker = json_native(marker)
    marker['artifact_manifest'] = 'artifact_manifest.json'
    write(out/'probe_complete.json', marker)
    write(out/'artifact_manifest.json', dict(schema='d92_margin_joint_artifacts_v1', method=METHOD, status=STATUS,
        files=[dict(path=path.relative_to(out).as_posix(), file_bytes=path.stat().st_size)
               for path in sorted(out.rglob('*')) if path.is_file()],
        inventory_excludes='artifact_manifest.json itself'))
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256', 'run-id', 'row-id'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); evaluate(**args)

if __name__ == '__main__': main()
