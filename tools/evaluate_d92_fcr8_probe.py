"""Three frozen FCR8 support paths with lossless numeric state logging."""
import argparse
from collections.abc import Mapping
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import platform
from pathlib import PurePosixPath
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import fit_branch_local_ridge
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from cvsrffi.d92_function_coordinate_residual8_local_ridge import FROZEN_CONFIG, prepare_function_coordinate_training, fit_function_coordinate_local_ridge
from evaluate_d92_registration_diagnostic import (
    CHANNEL, SCENARIOS, BRANCHES, split_identity, selected_tasks, validate_selection,
    diagnose_evidence, measured_metrics as baseline_metrics, _safe,
)
from evaluate_d92_branch_support_probe import load_support, read, write, check, scalars, csv_record, peak_rss
from evaluate_d92_branch_local_ridge_probe import stage_csv_from_jsonl
PROBE_CONFIG = deepcopy(FROZEN_CONFIG)
STATUS = 'FCR8_PROBE_COMPLETE'


class StateArchive:
    """Exclusive numeric archives: JSON references retain every array coordinate."""
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root/'state_arrays'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = []

    def __call__(self, key, arrays):
        check(isinstance(key, str) and bool(key), 'State archive key must be nonempty text')
        check(isinstance(arrays, Mapping) and bool(arrays), 'State archive requires numeric arrays')
        numeric, metadata, summaries = {}, {}, {}
        for name, value in arrays.items():
            check(isinstance(name, str) and name and '/' not in name, 'Invalid state array name')
            array = np.asarray(value)
            check(array.dtype.kind in 'fbiu' and np.isfinite(array).all(), 'Nonfinite or nonnumeric archived state')
            check(array.dtype.kind != 'f' or array.dtype == np.dtype('float64'), 'State floats must be float64')
            numeric[name] = np.array(array, copy=True)
            metadata[name] = dict(shape=list(array.shape), dtype=str(array.dtype), nbytes=int(array.nbytes))
            summaries[name] = dict(norm=float(np.linalg.norm(array.reshape(-1))), minimum=float(np.min(array)) if array.size else None,
                maximum=float(np.max(array)) if array.size else None)
        relative = 'state_arrays/'+str(len(self.files)).zfill(8)+'.npz'
        path = self.root/relative
        tick = time.perf_counter()
        with path.open('xb') as stream:
            np.savez_compressed(stream, **numeric)
        ref = json_native(dict(key=key.rsplit('/', 1)[-1], namespace=key.rsplit('/', 1)[0] if '/' in key else None,
            path=relative, arrays=metadata, array_summaries=summaries, file_bytes=path.stat().st_size,
            archive_seconds=time.perf_counter()-tick))
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
        manifest = dict(schema='d92_fcr8_state_archive_v1', status=status, files=self.files,
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


_write_native_json = write


def write(path, value):
    return _write_native_json(path, json_native(value))

SCOPE = 'SUPPORT_ONLY_FCR8_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
PATHS = ('R0', 'R_FCR8_seq', 'R_FCR8_reset_init')
COUNTERS = ('episodes', 'k1_episodes', 'oof_episodes', 'proxy_anchor_count', 'sequence_paths',
    'baseline_head_fit_count', 'baseline_factorization_count', 'fcr_preparation_count', 'fcr_stage_count',
    'trained_fcr_stage_count', 'optimizer_steps', 'inner_objective_evaluation_count', 'inner_head_fit_count',
    'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count',
    'head_fit_count', 'factorization_count', 'diagnostic_fit_count', 'derivative_triangular_solve_count',
    'backward_evaluation_count', 'accepted_trial_count', 'rejected_trial_count', 'trial_count', 'trial_attempt_count', 'optimizer_iterations',
    'prepared_distance_evaluation_count',
    'fcr_forward_evaluation_count', 'teacher_head_fit_count', 'teacher_factorization_count',
    'teacher_score_evaluation_count', 'teacher_score_physical_count',
    'initial_inner_head_fit_count', 'initial_inner_factorization_count',
    'task_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count',
    'final_score_evaluation_count', 'final_score_physical_count',
    'latent_svd_count', 'dictionary_physical_evaluation_count')
STAGE_COUNTERS = ('optimizer_steps', 'inner_objective_evaluation_count', 'inner_head_fit_count',
    'inner_factorization_count', 'final_head_fit_count', 'final_factorization_count', 'derivative_triangular_solve_count',
    'backward_evaluation_count', 'accepted_trial_count', 'rejected_trial_count', 'trial_count', 'trial_attempt_count',
    'optimizer_iterations', 'fcr_forward_evaluation_count',
    'task_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count')
PREPARATION_COUNTERS = ('inner_head_fit_count', 'inner_factorization_count',
    'prepared_distance_evaluation_count', 'fcr_forward_evaluation_count',
    'teacher_head_fit_count', 'teacher_factorization_count', 'teacher_score_evaluation_count', 'teacher_score_physical_count',
    'initial_inner_head_fit_count', 'initial_inner_factorization_count',
    'latent_svd_count', 'dictionary_physical_evaluation_count')
EXACT_COUNTS = dict(episodes=160, k1_episodes=40, oof_episodes=120,
    proxy_anchor_count=1400, sequence_paths=1760, baseline_head_fit_count=3168,
    fcr_preparation_count=3168, fcr_stage_count=4576, diagnostic_fit_count=0)
MAX_COUNTS = dict(trained_fcr_stage_count=936, optimizer_steps=3744,
    inner_objective_evaluation_count=12168, inner_head_fit_count=36504,
    inner_factorization_count=36504, final_head_fit_count=936,
    final_factorization_count=936, head_fit_count=42336, factorization_count=42336,
    baseline_factorization_count=3168, derivative_triangular_solve_count=44928,
    teacher_head_fit_count=1728, teacher_factorization_count=1728)


def validate_spec(spec):
    """Validate the same fixed support-cache pilot, without starting any work."""
    import run_d92_registration_diagnostic as baseline
    probe = spec['probe']
    check(probe['algorithm'] == PROBE_CONFIG and probe['channel'] == CHANNEL,
        'Frozen FCR8 algorithm/channel mismatch')
    check(probe['exact_counts'] == EXACT_COUNTS and probe['maximum_counts'] == MAX_COUNTS,
        'Frozen FCR8 pilot budget mismatch')
    check(probe['expected_head_fits'] == MAX_COUNTS['head_fit_count'], 'Head upper bound mismatch')
    check(probe['candidate'] == 'R_FCR8_seq' and probe['controls'] == ['R0', 'R_FCR8_reset_init'],
        'FCR8 candidate/control mismatch')
    check(probe['interpretation'] == 'support_joint_pilot_no_direct_promotion', 'Pilot interpretation mismatch')
    check(spec['permissions'].get('adapted_state_reuse') ==
        'within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse', 'Sequential inheritance scope mismatch')
    view = deepcopy(spec)
    view['probe'].update(algorithm=baseline.DIAGNOSTIC_CONFIG, expected_head_fits=3168)
    for name, co in view['probe']['cohorts'].items():
        check(probe['cohorts'][name]['evaluation_config'] == str(PurePosixPath(spec['code']['cwd'])/
            f'configs/d92_fcr8_support_{name}_20260930.json'), 'FCR8 config escaped release')
        co['evaluation_config'] = str(PurePosixPath(spec['code']['cwd'])/baseline.CONFIG_NAMES[name])
    baseline.validate_spec(view)


def verify_marker(path, spec, row):
    marker = read(path); co = spec['probe']['cohorts'][row['cohort']]
    expected = dict(status=STATUS, capsule_id=co['capsule_id'], checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'], algorithm=PROBE_CONFIG, selection=co['selection'], producer_matrix=co['matrix'],
        query_rows_used=0, source_rows_used=0, **{key: value//4 for key, value in EXACT_COUNTS.items()})
    check(all(marker.get(key) == value for key, value in expected.items()), 'Incomplete or incorrectly bound FCR8 pilot')
    for key in COUNTERS:
        check(type(marker.get(key)) is int and marker[key] >= 0, 'Invalid actual counter: '+key)
    for key, limit in MAX_COUNTS.items(): check(marker[key] <= limit//4, 'FCR8 upper bound exceeded: '+key)
    check(marker['head_fit_count'] == sum(marker[key] for key in
        ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'teacher_head_fit_count')), 'Head accounting mismatch')
    check(marker['factorization_count'] == sum(marker[key] for key in
        ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'teacher_factorization_count')), 'Factorization accounting mismatch')
    check(marker['derivative_triangular_solve_count'] == marker['task_derivative_triangular_solve_count']+
        marker['keep_derivative_triangular_solve_count'], 'Two-channel adjoint accounting mismatch')
    check(marker.get('truth_read') is False, 'Query truth access forbidden')
    return marker
METRICS = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy',
    'C_old_accuracy', 'C_new_accuracy', 'C_h', 'adaptation_gain_B_minus_A', 'support_adaptation_B_minus_B0',
    'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap',
    'C_new_minus_old', 'C_old_minus_R0', 'C_new_minus_R0', 'old_order_recovery_fraction',
    'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction') + tuple(
        'correctness_B_Cold_C_'+bits for bits in ('000', '100', '010', '110', '011', '111'))
MARGIN_FIELDS = ('margin_mean', 'margin_minus_R0', 'winner_changed_from_R0_fraction',
    'R0_correct_to_wrong_fraction', 'R0_wrong_to_correct_fraction')
METRICS += tuple(stage+'_'+field for stage in ('B_old', 'C_old', 'C_new') for field in MARGIN_FIELDS)


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


def compact_event(event):
    """Scalar diagnostics plus full-coordinate NPZ references; no vector copying."""
    event = json_native(event)
    result = scalars(event)
    for key, value in event.items():
        if key == 'state_ref' or key.endswith('_state_ref'): result[key] = value
    for prefix, objective in (('', event), ('objective_', event.get('objective')), ('final_', event.get('final_objective'))):
        if prefix and isinstance(objective, dict):
            result.update({prefix+key: value for key, value in scalars(objective).items()})
        if isinstance(objective, dict) and isinstance(objective.get('inner_folds'), list):
            folds = objective['inner_folds']
            mechanism_keys = ('block_angle_radians', 'tangent_over_kappa', 'joint_distance_relative_change',
                'adapted_distance_relative_change', 'original_zero_distance_pair_count', 'kernel_frobenius_norm',
                'kernel_change_from_initial', 'kernel_relative_change_from_initial', 'kernel_change_unmeasured_reason',
                'held_score_change_rms', 'held_winner_change_count')
            result[prefix+'inner_mechanisms'] = [dict(inner_fold=fold.get('inner_fold'),
                **{key: fold[key] for key in mechanism_keys if key in fold}) for fold in folds]
            if folds and all('held_training_correct_count' in fold and 'held_physical_count' in fold for fold in folds):
                n = sum(fold['held_physical_count'] for fold in folds)
                result[prefix+'inner_training_accuracy'] = sum(fold['held_training_correct_count'] for fold in folds)/n if n else None
                result[prefix+'inner_training_physical_count'] = n
                margins = [fold.get('held_training_margin_mean') for fold in folds]
                minima = [fold.get('held_training_margin_min') for fold in folds]
                result[prefix+'inner_training_margin_mean'] = sum(v*fold['held_physical_count'] for v, fold in zip(margins, folds))/n if n and all(v is not None for v in margins) else None
                result[prefix+'inner_training_margin_min'] = min(minima) if minima and all(v is not None for v in minima) else None
    return result


def probe_fcr8(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                         classes, old_classes, log_callback=None, event_callback=None, context=None, state_callback=None):
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    check(bool(old), 'Old registry required'); classes, old = list(canonical), sorted(old)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {key: np.asarray(value)[order] for key, value in zip(BRANCHES, (z_id, fft, t_emb, f_emb, pa_local))}
    label_by_id = {pid: classes[int(y)] for pid, y in zip(ids, labels)}
    is_old = np.asarray([label_by_id[pid] in old for pid in ids]); folds = 0 if k == 1 else min(k, 3)
    result = dict(classes=classes, old_classes=old, support_count=len(ids), old_class_count=len(old),
        new_class_count=len(classes)-len(old), fold_count=folds, physical_fold_assignment=[],
        numerical=interaction._diagonal_stats(b, a), folds=[], oof=None, oneshot_proxy=None,
        **dict.fromkeys(COUNTERS[4:], 0), persistent_state_bytes=0,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    positions = [np.flatnonzero(labels == i) for i in range(len(classes))]
    assignments = np.full(len(ids), -1, dtype=int)
    for indices in positions: assignments[indices] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=label_by_id[pid], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    completed = []

    def path(keep, scope, index):
        reuse = classes == old
        entry = dict(scope=scope, fold=index if scope == 'support_oof' else None,
            trial=index if scope == 'support_oneshot_proxy' else None, parent_k=k,
            train_k=int(keep.sum())//len(classes), held_k=int((~keep).sum())//len(classes),
            b_training_ids=[pid for i, pid in enumerate(ids) if keep[i] and is_old[i]],
            c_training_ids=[pid for i, pid in enumerate(ids) if keep[i]],
            b_ids=[pid for i, pid in enumerate(ids) if not keep[i] and is_old[i]],
            c_ids=[pid for i, pid in enumerate(ids) if not keep[i]], b_classes=old, c_classes=classes,
            held_labels={pid: label_by_id[pid] for i, pid in enumerate(ids) if not keep[i]},
            stages=[], preparations=[], candidate_stages=[], training_events=[], c_reuses_b0=reuse,
            c_reuses_b_candidates=reuse, paths={})
        coords = dict(context or {}, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        bases, prepared, candidates = {}, {}, {}
        def emit_event(row, stage):
            # Core trial is the Armijo index; the outer proxy anchor is separate.
            value = dict(row); value.update({key: item for key, item in coords.items() if key != 'trial'},
                outer_trial=coords['trial'], state=stage, source_validation=None,
                source_validation_reason='SOURCE_ACCESS_FORBIDDEN', objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
            entry['training_events'].append(value)
            if event_callback: event_callback(value)
        def archive_callback(stage):
            if state_callback is None: return None
            namespace = json.dumps(json_native(dict(coords, state=stage)), sort_keys=True, separators=(',', ':'))
            return lambda key, arrays: state_callback(namespace+'/'+key, arrays)
        try:
            for name, train, held, registry in (('B0', keep & is_old, ~keep & is_old, old), ('C0', keep, ~keep, classes)):
                if name == 'C0' and reuse:
                    bases[name] = bases['B0']; continue
                tids = [pid for i, pid in enumerate(ids) if train[i]]
                ys = np.asarray([registry.index(label_by_id[pid]) for pid in tids], dtype=np.int64)
                tick = time.perf_counter()
                state = fit_branch_local_ridge(**{key: value[train] for key, value in raw.items()},
                    support_labels=ys, support_ids=tids, classes=registry, old_classes=old, arm='local_ridge')
                top = state.audit_dict(); audit = top['final_fit']
                audit.update(coords, state=name, training_physical_ids=tids, all_states_estimated_from_trainfold_only=True,
                    held_physical_count=int(held.sum()), class_count=len(registry), learning_rate=None,
                    learning_rate_reason='Closed-form baseline; no optimizer', source_validation=None,
                    source_validation_reason='SOURCE_ACCESS_FORBIDDEN', score_seconds=None, persistent_state_bytes=top['persistent_state_bytes'])
                entry['stages'].append(audit)
                result['baseline_head_fit_count'] += 1; result['head_fit_count'] += 1
                result['baseline_factorization_count'] += audit['factorization_calls']; result['factorization_count'] += audit['factorization_calls']
                bases[name] = dict(state=state, train=train, held=held, registry=registry, labels=ys, ids=tids, scores=None)
                score_start = time.perf_counter()
                result['final_score_evaluation_count'] += 1; result['final_score_physical_count'] += int(held.sum())
                scores = state.score(**{key: value[held] for key, value in raw.items()})
                scores.setflags(write=False); bases[name]['scores'] = scores
                audit.update(score_seconds=time.perf_counter()-score_start, fit_and_score_seconds=time.perf_counter()-tick)
                if log_callback: log_callback(dict(event='BASE_FIT', **audit))
            for preparation_name, stages in (('B', [('B_FCR8', 'B')]),
                                              ('C', [('C_FCR8_seq', 'C_seq'), ('C_reset_init', 'C_reset_init')])):
                if preparation_name == 'C' and reuse:
                    candidates['C_FCR8_seq'] = candidates['B_FCR8']; candidates['C_reset_init'] = candidates['B_FCR8']
                    continue
                base = bases[preparation_name+'0']
                prepared[preparation_name] = prepare_function_coordinate_training(**{key: value[base['train']] for key, value in raw.items()},
                    support_labels=base['labels'], support_ids=base['ids'], classes=base['registry'], old_classes=old,
                    inherited=None if preparation_name == 'B' else candidates['B_FCR8']['state'],
                    context=dict(coords, stage=preparation_name),
                    log_callback=lambda row, name=preparation_name: emit_event(row, name+'_prepare'),
                    state_callback=archive_callback(preparation_name+'_prepare'))
                prep = prepared[preparation_name].audit_dict()
                prep.update(coords, state=preparation_name, training_physical_ids=base['ids'],
                    train_physical_count=len(base['ids']), class_count=len(base['registry']),
                    inherited_adapter_from=None if preparation_name == 'B' else 'B_FCR8',
                    objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                entry['preparations'].append(prep); result['fcr_preparation_count'] += 1
                for key in PREPARATION_COUNTERS: result[key] += prep[key]
                result['head_fit_count'] += prep['inner_head_fit_count']+prep['teacher_head_fit_count']
                result['factorization_count'] += prep['inner_factorization_count']+prep['teacher_factorization_count']
                if log_callback: log_callback(dict(event='FCR_PREPARATION', **prep))
                for name, mode in stages:
                    state = fit_function_coordinate_local_ridge(prepared[preparation_name], mode=mode, baseline_state=base['state'],
                        log_callback=lambda row, current=name: emit_event(row, current), state_callback=archive_callback(name))
                    audit = state.audit_dict()
                    audit.update(coords, state=name, preparation_ref=preparation_name, mode=mode,
                        training_physical_ids=base['ids'], train_physical_count=len(base['ids']), class_count=len(base['registry']),
                        held_physical_count=int(base['held'].sum()), source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
                        score_seconds=None, objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                    # Successful fitting is accounted before external-held inference.
                    entry['candidate_stages'].append(audit); candidates[name] = dict(state=state, scores=None)
                    result['fcr_stage_count'] += 1
                    result['trained_fcr_stage_count'] += int(audit['optimizer_steps'] > 0)
                    for key in STAGE_COUNTERS:
                        result[key] += audit[key]
                    result['head_fit_count'] += audit['inner_head_fit_count']+audit['final_head_fit_count']
                    result['factorization_count'] += audit['inner_factorization_count']+audit['final_factorization_count']
                    score_start = time.perf_counter()
                    result['final_score_evaluation_count'] += 1; result['final_score_physical_count'] += int(base['held'].sum())
                    scores = state.score(**{key: value[base['held']] for key, value in raw.items()})
                    candidates[name]['scores'] = scores; audit['score_seconds'] = time.perf_counter()-score_start
                    if log_callback: log_callback(dict(event='CANDIDATE_FIT', **audit))
            common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
            evidence = dict(R0=dict(common, b_scores=bases['B0']['scores'].tolist(), c_scores=bases['C0']['scores'].tolist()))
            for name, bname, cname in (('R_FCR8_seq', 'B_FCR8', 'C_FCR8_seq'), ('R_FCR8_reset_init', 'B_FCR8', 'C_reset_init')):
                evidence[name] = dict(common, b_scores=candidates[bname]['scores'].tolist(), c_scores=candidates[cname]['scores'].tolist())
            fresh = assess_paths(evidence, old)
            entry['paths'] = {name: dict(b_scores=value['b_scores'], c_scores=value['c_scores'], **fresh[name]) for name, value in evidence.items()}
            entry['deployment_C_state_bytes'] = {name: candidates[cname]['state'].audit_dict()['persistent_state_bytes']
                for name, cname in (('R_FCR8_seq', 'C_FCR8_seq'), ('R_FCR8_reset_init', 'C_reset_init'))}
            result['persistent_state_bytes'] = max(result['persistent_state_bytes'], *entry['deployment_C_state_bytes'].values())
        except Exception as exc:
            exc.registration_context = _safe(dict(current_path=entry,
                failed_fit=exc.audit_dict() if callable(getattr(exc, 'audit_dict', None)) else {},
                completed_preparations={name: value.audit_dict() for name, value in prepared.items()},
                completed_candidate_states={name: dict(audit=value['state'].audit_dict()) for name, value in candidates.items()},
                completed_paths=completed, counters={key: result[key] for key in COUNTERS[4:]}))
            raise
        result['sequence_paths'] += 1; completed.append(entry)
        return entry

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
        proxy_anchor_count=0 if record['oneshot_proxy'] is None else record['oneshot_proxy']['trial_count'],
        scope=SCOPE, query_rows_used=0, source_rows_used=0))


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    check(set(config) == {'algorithm', 'producer_matrix', 'selection'} and config['algorithm'] == PROBE_CONFIG == FROZEN_CONFIG,
        'Frozen FCR8 code/config mismatch')
    validate_selection(config['selection']); manifest = read(Path(capsule)/'manifest.json')
    check(all(manifest.get('channel', {}).get(key) == value for key, value in CHANNEL.items())
        and manifest.get('scenarios') == SCENARIOS, 'Practical residual channel mismatch')
    begin = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['producer_matrix']))
    check(len(old) == 6, 'Fixed six-old-class pilot required'); chosen = selected_tasks(tasks, config['selection'], old)
    binding = dict(capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=expected_model_seed, channel=deepcopy(manifest['channel']), scenarios=manifest['scenarios'])
    payload = dict(feature_cache_reused=True, checkpoint_loaded=False, native_physical_forward_count_this_run=0,
        new_source_payload_bytes=0, new_ground_statistics_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=time.perf_counter()-begin,
        deployment_package_bytes=None, incremental_transfer_bytes=None, unmeasured_reason='No deployment package serialized or transmitted')
    startup = dict(binding, scope=SCOPE, config=config, episodes=len(chosen), producer_episodes=len(tasks),
        support_features=str(support_features), provenance=provenance, payload_audit=payload,
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        checkpoint_loaded=False, encoder_updated=False, adapted_state_inherited=True,
        actual_A=None, actual_A_unavailable_reason='No matched legal ground A predictor in support pilot',
        source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION',
        argv=sys.argv, pid=os.getpid(), python=sys.executable,
        hardware=dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(), dtype='float64', gpu_use=False),
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
                    row = dict(compact_event(stage), split_id=split['split_id'])
                    line = json.dumps(row, allow_nan=False)
                    stages.write(line+'\n'); stages.flush(); textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                def event(row):
                    full = json_native(dict(row, split_id=split['split_id']))
                    events.write(json.dumps(full, allow_nan=False)+'\n'); events.flush()
                    small = compact_event(full); line = json.dumps(small, allow_nan=False)
                    compactevents.write(line+'\n'); compactevents.flush()
                    textlog.write('FCR8_TRAINING '+line+'\n'); textlog.flush(); print('FCR8_TRAINING '+line, flush=True)
                try:
                    audit = probe_fcr8(**{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'], classes=split['registered_classes'], old_classes=old,
                        log_callback=log, event_callback=event, context=dict(row_id=out.parent.name, split_id=split['split_id']),
                        state_callback=archive)
                except Exception as exc:
                    write(out/'probe_failed.json', _safe(dict(status='FCR8_PROBE_FAILED', **binding,
                        split_identity=split_identity(split, old), error_type=type(exc).__name__, error=str(exc),
                        failure_context=getattr(exc, 'registration_context', {}), completed_episodes=totals['episodes'],
                        query_rows_used=0, source_rows_used=0)))
                    raise
                record = json_native(dict(audit, **split_identity(split, old), scope=SCOPE, query_rows_used=0, source_rows_used=0))
                trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush(); small = compact_record(record)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                if writer is None: writer = csv.DictWriter(csvfile, fieldnames=list(small)); writer.writeheader()
                writer.writerow(csv_record(small)); csvfile.flush()
                for key in COUNTERS[4:]: totals[key] += audit[key]
                totals['episodes'] += 1; totals['k1_episodes'] += int(split['k'] == 1)
                totals['oof_episodes'] += int(split['k'] > 1); totals['proxy_anchor_count'] += small['proxy_anchor_count']
                peak_state = max(peak_state, audit['persistent_state_bytes'])
                print(json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', **small, completed=totals['episodes']), allow_nan=False), flush=True)
            archive_status = 'COMPLETE'
    finally:
        for name in ('fit_stages', 'training_events_compact'):
            if (out/(name+'.jsonl')).exists(): stage_csv_from_jsonl(out/(name+'.jsonl'), out/(name+'.csv'))
        state_manifest = archive.finalize(archive_status)
    marker = dict(binding, **totals, status=STATUS, scope=SCOPE, algorithm=PROBE_CONFIG,
        selection=config['selection'], producer_matrix=config['producer_matrix'], payload_audit=payload,
        query_rows_used=0, source_rows_used=0, truth_read=False, persistent_state_bytes=peak_state,
        persistent_state_scope='maximum_one_deployable_C_head_with_fcr8_and_inheritance_binding',
        state_archive_file_count=state_manifest['file_count'], state_archive_file_bytes=state_manifest['total_file_bytes'],
        state_archive_numeric_bytes=state_manifest['numeric_array_bytes'], state_archive_seconds=state_manifest['archive_seconds'],
        state_manifest='state_manifest.json',
        peak_gpu_memory_bytes=None, gpu_memory_reason='CPU only', wall_seconds=time.perf_counter()-begin,
        peak_process_rss_bytes=peak_rss())
    marker = json_native(marker)
    marker['artifact_manifest'] = 'artifact_manifest.json'
    write(out/'probe_complete.json', marker)
    write(out/'artifact_manifest.json', dict(schema='d92_fcr8_artifacts_v1', status=STATUS,
        files=[dict(path=path.relative_to(out).as_posix(), file_bytes=path.stat().st_size)
               for path in sorted(out.rglob('*')) if path.is_file()],
        inventory_excludes='artifact_manifest.json itself'))
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); evaluate(**args)


if __name__ == '__main__': main()
