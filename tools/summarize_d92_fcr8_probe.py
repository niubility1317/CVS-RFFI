"""Verify all four pilot rows before recomputing fixed support comparisons."""
import argparse
import csv
from collections import OrderedDict
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_fcr8_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, PROBE_CONFIG, PATHS, METRICS, COUNTERS,
    check, read, scalars, csv_record, split_identity, selected_tasks, compact_record, compact_event,
    assess_paths, pooled_assess, parent_mean, baseline_metrics, STAGE_COUNTERS, PREPARATION_COUNTERS,
)
from evaluate_d92_fcr8_probe import EXACT_COUNTS, MAX_COUNTS, validate_spec, verify_marker
import evaluate_d92_registration_diagnostic as baseline
from summarize_d92_registration_diagnostic import verify_record as verify_baseline, _statistics, write_json
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close

SUMMARY_STATUS = 'COMPLETE_FCR8_PROBE_VERIFIED'


class StateResolver:
    """Check archive content, metadata and inventory before using any coordinates."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read(self.root/'state_manifest.json')
        check(self.manifest['schema'] == 'd92_fcr8_state_archive_v1' and
            self.manifest['status'] == 'COMPLETE', 'Incomplete state archive')
        files = self.manifest['files']
        self.refs = {ref['path']: ref for ref in files}
        check(len(files) == len(self.refs) == self.manifest['file_count'], 'Duplicate/missing state archive inventory')
        check(self.manifest['total_file_bytes'] == sum(ref['file_bytes'] for ref in files), 'Archive byte inventory mismatch')
        self.used, self.verified, self.cache = set(), set(), OrderedDict()

    def __call__(self, ref):
        check(isinstance(ref, dict) and ref.get('path') in self.refs and ref == self.refs[ref['path']],
            'State reference missing from exact archive manifest')
        relative = Path(ref['path'])
        check(not relative.is_absolute() and relative.parts and relative.parts[0] == 'state_arrays', 'State path escaped archive')
        path = (self.root/relative).resolve()
        check(path.is_relative_to(self.root/'state_arrays') and path.suffix == '.npz', 'State path escaped archive')
        self.used.add(ref['path'])
        if ref['path'] in self.cache:
            self.cache.move_to_end(ref['path']); return self.cache[ref['path']]
        check(path.stat().st_size == ref['file_bytes'], 'Archived state file size mismatch')
        with np.load(path, allow_pickle=False) as source:
            check(set(source.files) == set(ref['arrays']), 'Archived array inventory mismatch')
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
        for name, value in arrays.items():
            meta = ref['arrays'][name]
            summary = ref['array_summaries'][name]
            check(value.dtype.kind in 'fbiu' and np.isfinite(value).all() and list(value.shape) == meta['shape']
                and str(value.dtype) == meta['dtype'] and value.nbytes == meta['nbytes'], 'Archived array shape/type/finite mismatch')
            check(value.dtype.kind != 'f' or value.dtype == np.dtype('float64'), 'Archived state precision mismatch')
            close(summary['norm'], float(np.linalg.norm(value.reshape(-1))), 'Archived norm summary mismatch')
            check(summary['minimum'] == (float(np.min(value)) if value.size else None) and
                summary['maximum'] == (float(np.max(value)) if value.size else None), 'Archived coordinate range mismatch')
            value.setflags(write=False)
        self.verified.add(ref['path']); self.cache[ref['path']] = arrays
        if len(self.cache) > 8: self.cache.popitem(last=False)
        return arrays

    def verify_tree(self, value):
        if isinstance(value, dict):
            if {'path', 'arrays', 'array_summaries'} <= set(value): self(value)
            else:
                for child in value.values(): self.verify_tree(child)
        elif isinstance(value, list):
            for child in value: self.verify_tree(child)

    def finalize(self):
        actual = {p.relative_to(self.root).as_posix() for p in (self.root/'state_arrays').glob('*.npz')}
        check(actual == set(self.refs) == self.used, 'Unreferenced/missing/extra full-coordinate archive')
        check(self.manifest['numeric_array_bytes'] == sum(meta['nbytes'] for ref in self.refs.values()
            for meta in ref['arrays'].values()), 'Numeric state byte inventory mismatch')
        phases = {}
        for ref in self.refs.values():
            phase = json.loads(ref['namespace'])['state'] if ref['namespace'] else 'unscoped'
            value = phases.setdefault(phase, dict(file_count=0, file_bytes=0, numeric_array_bytes=0, archive_seconds=0.))
            value['file_count'] += 1; value['file_bytes'] += ref['file_bytes']
            value['numeric_array_bytes'] += sum(a['nbytes'] for a in ref['arrays'].values())
            value['archive_seconds'] += ref['archive_seconds']
        check(phases == self.manifest['by_phase'], 'Phase archive inventory mismatch')


def verify_artifact_inventory(root, marker):
    root = Path(root).resolve(); value = read(root/'artifact_manifest.json')
    check(value['schema'] == 'd92_fcr8_artifacts_v1' and value['status'] == STATUS
        and marker['artifact_manifest'] == 'artifact_manifest.json', 'Missing complete artifact inventory')
    listed = {row['path']: row['file_bytes'] for row in value['files']}
    actual = {path.relative_to(root).as_posix(): path.stat().st_size for path in root.rglob('*')
              if path.is_file() and path.name != 'artifact_manifest.json'}
    check(len(listed) == len(value['files']) and listed == actual, 'Artifact file inventory/size mismatch')


def coordinate_state(ref, resolver, rank):
    arrays = resolver(ref)
    Z, U = arrays["Z"], arrays["U"]
    check(Z.shape == (736, rank) and U.shape == (736, 8), "FCR coordinate state shape mismatch")
    return Z, U


def dct_initial():
    k = np.arange(736, dtype=np.float64)
    return np.asarray([np.full(736, 1/math.sqrt(736))]+[
        math.sqrt(2/736)*np.cos(math.pi*j*(k+.5)/736) for j in range(1, 8)])


def verify_coordinate_map(Z, U, anchor, W):
    expected = anchor.copy() if not np.any(Z) else anchor+Z@W.T
    check(np.array_equal(U, expected), "FCR original-U coordinate reconstruction mismatch")


def verify_latent_coordinates(prep, resolver):
    arrays = resolver(prep["coordinate_state_ref"])
    H, W = arrays["H"], arrays["W"]
    n, rank = prep["train_physical_count"], prep["latent_rank"]
    check(H.shape == (n, 8) and W.shape == (8, rank) and 0 <= rank <= 8,
          "FCR dictionary/coordinate shape mismatch")
    eta = 128*sys.float_info.epsilon*max(n, 8)
    close(prep["rank_energy_threshold"], eta, "FCR frozen numerical rank threshold mismatch")
    check(prep["dictionary_physical_evaluation_count"] == n
          and prep["optimizer_coordinate_ids"] == prep["training_physical_ids"]
          and prep['optimizer_coordinate_scope'] == 'ALL_CURRENT_OUTER_TRAIN_UNLABELLED'
          and prep['fixed_dictionary_trainable'] is False,
          "FCR optimizer coordinate physical binding mismatch")
    if not prep['rank_estimated']:
        check(rank == prep['latent_svd_count'] == 0 and prep['singular_values'] is None
            and prep['whitening_residual'] is prep['whitening_tolerance'] is None
            and np.array_equal(arrays['singular_values'], np.zeros(8)) and prep['no_information'],
            'FCR invented skipped coordinate decomposition')
        close(prep['dictionary_rms'], float(np.linalg.norm(H))/math.sqrt(n), 'FCR dictionary RMS mismatch')
        return H, W
    scale = float(np.max(np.abs(H)))
    values = np.zeros(8)
    if scale:
        _, spectrum, vt = np.linalg.svd((H/scale)/math.sqrt(n), full_matrices=False)
        values[:len(spectrum)] = spectrum*scale
        retained = spectrum/spectrum[0] > math.sqrt(eta)
        expected_rank = int(retained.sum())
        tolerance = float(eta*max(1., spectrum[0]/spectrum[retained][-1])) if expected_rank else 0.
    else:
        expected_rank, tolerance = 0, 0.
    check(rank == expected_rank and prep["latent_svd_count"] == int(scale > 0),
          "FCR latent numerical rank/SVD count mismatch")
    check(np.allclose(values, prep["singular_values"], rtol=eta, atol=0.),
          "FCR full dictionary singular spectrum mismatch")
    check(np.array_equal(arrays['singular_values'], np.asarray(prep['singular_values'])),
          'FCR archived/reported singular spectrum mismatch')
    if rank:
        expected_W = (vt[retained].T/spectrum[retained])/scale
        check(np.allclose(W@W.T, expected_W@expected_W.T, rtol=tolerance,
            atol=tolerance*float(np.linalg.norm(expected_W@expected_W.T))),
            'FCR retained SVD coordinate subspace mismatch')
    white = (H@W)/math.sqrt(n)
    error = float(np.linalg.norm(white.T@white-np.eye(rank), ord=2)) if rank else 0.
    check(error <= tolerance, "FCR dictionary whitening residual exceeded")
    close(prep["whitening_tolerance"], tolerance, "FCR whitening tolerance mismatch")
    close(prep["whitening_residual"], error, "FCR whitening residual mismatch")
    close(prep["dictionary_rms"], float(np.linalg.norm(H))/math.sqrt(n), "FCR dictionary RMS mismatch")
    check(prep["dictionary_physical_evaluation_count"] == n
          and prep["optimizer_coordinate_ids"] == prep["training_physical_ids"],
          "FCR optimizer coordinate physical binding mismatch")
    return H, W


def baseline_view(record):
    def view(entry):
        result = dict(entry, **entry['paths']['R0'])
        result['metrics'] = baseline_metrics(result['diagnostic']); return result
    entries = record['folds']+([] if record['oneshot_proxy'] is None else record['oneshot_proxy']['trials'])
    result = dict(record, scope=baseline.SCOPE, persistent_state_bytes=0, optimizer_steps=0,
        head_fit_count=record['baseline_head_fit_count'],
        factorization_count=sum(stage['factorization_calls'] for entry in entries for stage in entry['stages']),
        folds=[view(entry) for entry in record['folds']])
    if record['oof'] is not None:
        value = record['oof']['paths']['R0']['diagnostic']
        result['oof'] = dict(diagnostic=value, metrics=baseline_metrics(value), aggregation='one_record_per_physical_held_id')
        trials = [view(entry) for entry in record['oneshot_proxy']['trials']]
        result['oneshot_proxy'] = dict(record['oneshot_proxy'], trials=trials, parent_mean_metrics=baseline.parent_mean(trials))
    return result


# FCR8 validation routines


def verify_preparation(prep, entry, labels, old, resolver):
    b = prep['state'] == 'B'; ids = entry['b_training_ids' if b else 'c_training_ids']
    classes = entry['b_classes' if b else 'c_classes']; k = entry['train_k']; folds = prep['inner_folds']
    basic_noinfo = k == 1 or len(classes) == 1
    verify_latent_coordinates(prep, resolver)
    noinfo = prep['no_information']; reason = prep['no_information_reason']
    if basic_noinfo:
        check(noinfo and reason == ('PHYSICAL_K1' if k == 1 else 'SINGLE_REGISTERED_CLASS'), 'FCR basic no-information mismatch')
    elif noinfo:
        check(reason in ('ZERO_DICTIONARY', 'ALL_INNER_GEOMETRY_DEGENERATE'), 'FCR unknown structural no-information reason')
        if reason == 'ZERO_DICTIONARY': check(prep['rank_estimated'] and prep['latent_rank'] == 0, 'False zero dictionary stop')
        else:
            original = prep['original_inner_geometry']
            check(len(original) == min(k, 3) and all(v['original_interaction_centered_trace'] == 0 or v['original_bandwidth_tau'] == 0 for v in original),
                  'False all-inner-geometry-degenerate stop')
    else: check(reason is None and prep['latent_rank'] > 0, 'FCR information/rank mismatch')
    check(prep['training_physical_ids'] == ids and prep['classes'] == classes and sorted(prep['old_classes']) == old
        and prep['train_physical_count'] == len(ids) and prep['class_count'] == len(classes)
        and prep['fcr_preparation_count'] == 1 and prep['inherited_state'] is (not b)
        and prep['inherited_adapter_from'] == (None if b else 'B_FCR8'), 'FCR8 preparation physical/lineage mismatch')
    nfolds = min(k, 3) if not noinfo else 0
    check(len(folds) == len(prep['teacher_folds']) == nfolds and prep['prepare_seconds'] >= 0
        and prep['no_information'] is noinfo, 'FCR8 preparation fold/no-information mismatch')
    available = len(old) > 1
    check(prep['keep_available'] is available, 'Teacher availability mismatch')
    groups = {cls: sorted(pid for pid in ids if labels[pid] == cls) for cls in classes}
    assignment = {pid: index % nfolds for values in groups.values() for index, pid in enumerate(values)} if nfolds else {}
    seen = set(); head_count = factor_count = score_count = score_physical = 0
    for index, (fold, teacher) in enumerate(zip(folds, prep['teacher_folds'])):
        held = {pid for pid in ids if assignment[pid] == index}; train = set(ids)-held
        check(fold['inner_fold'] == index and set(fold['training_physical_ids']) == train
            and len(fold['training_physical_ids']) == len(train) and set(fold['held_physical_ids']) == held
            and len(fold['held_physical_ids']) == len(held) and not seen.intersection(held)
            and not (train | held).intersection(entry['c_ids']), 'FCR8 inner/outer physical isolation mismatch')
        check(fold['train_physical_count'] == len(train) and fold['held_physical_count'] == len(held)
            and fold['all_head_statistics_from_inner_train_only'] is True
            and fold['original_interaction_centered_trace'] >= 0, 'FCR8 inner train-only statistic mismatch')
        seen.update(held)
        check(teacher['available'] is available and teacher['old_classes'] == old, 'Teacher class registry mismatch')
        check(fold['teacher'] == teacher, 'Inner fold/teacher record mismatch')
        if not available:
            check(teacher['state_ref'] is None and teacher['reason'] == 'SINGLE_TEACHER_CLASS_NO_WRONG_MARGIN', 'Invented single-class teacher')
            continue
        old_train = [pid for pid in fold['training_physical_ids'] if labels[pid] in old]
        old_held = [pid for pid in fold['held_physical_ids'] if labels[pid] in old]
        check(teacher['training_physical_ids'] == old_train and teacher['held_physical_ids'] == old_held
            and teacher['old_inner_train_only'] is True and teacher['source'] ==
            ('INITIAL_R0_CACHE' if b else 'FROZEN_B_ADAPTER_OLD_INNER_HEAD'), 'Teacher fold/head/lineage isolation mismatch')
        data = resolver(teacher['state_ref']); scores = data['teacher_scores']; q = data['teacher_q']
        mask = data['held_old_mask']
        check(scores.shape == (len(old_held), len(old)) and q.shape == (len(old_held),)
            and np.array_equal(mask, np.asarray([labels[pid] in old for pid in fold['held_physical_ids']], dtype=float)),
            'Teacher score/q/mask shape or physical binding mismatch')
        margins = []
        for pid, score in zip(old_held, scores):
            target = old.index(labels[pid]); margins.append(min(1., max(0., float(score[target]-max(np.delete(score, target))))))
        check(np.array_equal(q, np.asarray(margins)), 'Teacher clipped margin mismatch')
        head = teacher['head_audit']; certify_head(head)
        head_count += head['head_fit_count']; factor_count += head['factorization_count']
        score_count += 1; score_physical += len(old_held)
    check(not folds or seen == set(ids), 'FCR8 inner held physical coverage mismatch')
    check(prep['inner_head_fit_count'] == prep['initial_inner_head_fit_count'] == (head_count if b else 0)
        and prep['inner_factorization_count'] == prep['initial_inner_factorization_count'] == (factor_count if b else 0)
        and prep['teacher_head_fit_count'] == (0 if b else head_count)
        and prep['teacher_factorization_count'] == (0 if b else factor_count)
        and prep['teacher_score_evaluation_count'] == score_count
        and prep['teacher_score_physical_count'] == score_physical and prep['fcr_forward_evaluation_count'] == head_count,
        'Prepaid initial/teacher actual cost mismatch')
    original_fold_count = 0 if basic_noinfo else min(k, 3)
    check(prep['prepared_distance_evaluation_count'] == 2*(original_fold_count+1)+ (2*nfolds if not b and available else 0),
        'Student/teacher/full prepared geometry cost mismatch')


def certify_head(value):
    check(0 <= value['normal_equation_residual'] <= value['numerical_tolerance']
        and 0 <= value['trace_relative_error'] <= value['numerical_tolerance'], 'Uncertified FCR8 ridge head')


def verify_objective(value, Z, U, anchor_U, prep):
    folds = value['inner_folds']; n = prep['train_physical_count']; classes = prep['classes']
    check(value['loss_scope'] == 'CLASS_RMS_TASK_PLUS_OLD_CLASS_RMS_TEACHER_KEEP_PLUS_PROXIMAL'
        and len(folds) == len(prep['inner_folds']) and value['classes'] == classes, 'FCR8 objective scope/folds mismatch')
    sums = np.zeros(len(classes)); keep = np.zeros(len(classes)); counts = np.zeros(len(classes), dtype=int)
    for result, prepared in zip(folds, prep['inner_folds']):
        check(result['training_physical_ids'] == prepared['training_physical_ids']
            and result['held_physical_ids'] == prepared['held_physical_ids']
            and result['held_physical_count'] == prepared['held_physical_count']
            and 0 <= result['held_training_correct_count'] <= result['held_physical_count'], 'FCR8 objective fold binding mismatch')
        certify_head(result)
        check(result['original_bandwidth_tau'] == prepared['original_bandwidth_tau'],
            'FCR original fold bandwidth binding mismatch')
        for name in ('block_angle_radians', 'tangent_over_kappa', 'joint_distance_relative_change', 'adapted_distance_relative_change'):
            stats = result[name]
            if name == 'adapted_distance_relative_change':
                bypass = prepared['original_bandwidth_tau'] == 0 and bool(np.any(U != 0))
                check((stats is None) is bypass
                    and result['adapted_distance_unmeasured_reason'] ==
                    ('ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE' if bypass else None),
                    'FCR adapted-distance measurement/bypass binding mismatch')
                if bypass:
                    continue
            check(type(stats['count']) is int and stats['count'] >= 0, 'FCR invalid measured mechanism count')
            if stats['count']:
                check(stats['minimum'] <= stats['mean'] <= stats['maximum'], 'FCR invalid measured mechanism range')
            else: check(stats['minimum'] is stats['mean'] is stats['maximum'] is None, 'Invented empty mechanism statistic')
        check(type(result['original_zero_distance_pair_count']) is int and result['original_zero_distance_pair_count'] >= 0
            and result['kernel_change_from_initial'] is not None and result['kernel_change_from_initial'] >= 0
            and result['held_winner_change_count'] <= result['held_physical_count'], 'FCR missing measured forward mechanism')
        for field in ('class_physical_counts', 'class_task_loss_sums', 'class_keep_loss_sums'):
            check(len(result[field]) == len(classes) and all(v >= 0 for v in result[field]), 'Invalid FCR8 class risk evidence')
        check(all(type(v) is int for v in result['class_physical_counts']), 'Noninteger physical risk count')
        counts += result['class_physical_counts']; sums += result['class_task_loss_sums']; keep += result['class_keep_loss_sums']
        check(result['derivative_triangular_solve_count'] == result['task_derivative_triangular_solve_count']+
            result['keep_derivative_triangular_solve_count'], 'Per-fold two-channel adjoint mismatch')
    check(sum(counts) == n and all(counts == prep['train_k']) and value['class_held_counts'] == counts.tolist(), 'FCR8 pooled physical coverage mismatch')
    taskmeans, keepmeans = sums/counts, keep/counts
    old_indices = [i for i, cls in enumerate(classes) if cls in prep['old_classes']]
    check(value['old_class_indices'] == old_indices and value['keep_available'] is prep['keep_available'], 'Keep class binding mismatch')
    for field, expected in (('class_task_loss_means', taskmeans), ('class_keep_loss_means', keepmeans)):
        check(len(value[field]) == len(expected), 'FCR8 pooled risk dimension mismatch')
        for actual, target in zip(value[field], expected): close(actual, float(target), 'FCR8 pooled class risk mismatch')
    close(value['loss_task'], float(np.linalg.norm(taskmeans))/math.sqrt(len(classes)), 'FCR8 task class RMS mismatch')
    expected_keep = float(np.linalg.norm(keepmeans[old_indices]))/math.sqrt(len(old_indices)) if prep['keep_available'] else 0.
    close(value['loss_keep'], expected_keep, 'FCR8 old-class keep RMS mismatch')
    check(all(keepmeans[i] == 0 for i in range(len(classes)) if i not in old_indices), 'Keep objective charged new classes')
    close(value['loss_data'], value['loss_task']+value['loss_keep'], 'FCR8 data components mismatch')
    close(value["loss_proximal"], .5*float(np.linalg.norm(Z))**2, "FCR function proximal scaling mismatch")
    close(value['loss_total'], value['loss_data']+value['loss_proximal'], 'FCR8 total objective mismatch')
    for total, field in (('inner_head_fit_count', 'head_fit_count'), ('inner_factorization_count', 'factorization_count'),
                        ('fcr_forward_evaluation_count', 'fcr_forward_evaluation_count'),
                        ('derivative_triangular_solve_count', 'derivative_triangular_solve_count'),
                        ('task_derivative_triangular_solve_count', 'task_derivative_triangular_solve_count'),
                        ('keep_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count')):
        check(value[total] == sum(fold[field] for fold in folds), 'FCR8 objective per-fold actual cost mismatch: '+total)
    check(value['inner_objective_evaluation_count'] == int(not value['forward_cache_reused'] or value['initial_teacher_cache_reused']),
        'FCR8 objective logical-forward/cache counter mismatch')
    if value['forward_cache_reused']:
        check(value['inner_head_fit_count'] == value['inner_factorization_count'] == value['fcr_forward_evaluation_count'] == 0,
            'FCR8 cached objective refitted a head')


def verify_candidate(stage, prep, entry, b_fcr8, resolver):
    mode = stage['mode']; rank = prep['latent_rank']
    arrays = resolver(prep['coordinate_state_ref']); H, W = arrays['H'], arrays['W']
    Z, U = coordinate_state(stage['initialization_state_ref'], resolver, rank)
    anchor = resolver(b_fcr8['final_state_ref'])['U'] if mode == 'C_seq' else np.zeros((736, 8))
    initial = resolver(stage['initialization_state_ref'])
    check(np.array_equal(Z, np.zeros((736, rank))) and np.array_equal(U, anchor)
        and np.array_equal(initial['anchor_U'], anchor), 'FCR exact zero-Z/original-U inheritance mismatch')
    check(stage['preparation_ref'] == prep['state'] and stage['training_physical_ids'] == prep['training_physical_ids']
        and stage['preparation']['inner_folds'] == prep['inner_folds'] and stage['preparation']['teacher_folds'] == prep['teacher_folds']
        and stage['preparation']['coordinate_state_ref'] == prep['coordinate_state_ref']
        and stage['config'] == PROBE_CONFIG and stage['source_validation'] is None
        and stage['status'] == 'FCR_STAGE_COMPLETE' and stage['no_information'] is prep['no_information'],
        'FCR stage preparation mismatch')
    def check_state(ref):
        data = resolver(ref); z, u = coordinate_state(ref, resolver, rank)
        check(np.array_equal(data['anchor_U'], anchor) and np.array_equal(data['W'], W)
            and np.array_equal(data['singular_values'], arrays['singular_values']), 'FCR stage coordinate/anchor binding mismatch')
        verify_coordinate_map(z, u, anchor, W)
        return z, u
    def objective(value, z, u):
        verify_objective(value, z, u, anchor, prep)
        displacement = np.stack([np.sum((u-anchor)*h[None, :], axis=1) for h in H])
        actual = float((np.linalg.norm(displacement)/math.sqrt(len(H)))**2)
        expected = float(np.linalg.norm(z))**2
        close(value['pre_tangent_displacement_mean_squared'], actual, 'FCR measured physical function displacement mismatch')
        close(value['pre_tangent_displacement_rms'], math.sqrt(actual), 'FCR measured function RMS mismatch')
        close(value['coordinate_squared_norm'], expected, 'FCR coordinate squared norm mismatch')
        close(value['function_coordinate_reconstruction_error'], abs(actual-expected), 'FCR function rounding residual mismatch')
        if prep['rank_estimated']:
            e = prep['whitening_residual']
            # Independently accumulated physical outputs include original-U addition rounding.
            tol = 128*sys.float_info.epsilon*max(1, u.size)*max(1., actual, expected)
            check(abs(actual-expected) <= e*expected+tol, 'FCR physical function/coordinate equality mismatch')
    check_state(stage['initialization_state_ref'])
    steps, trials, gradients = stage['steps'], stage['trials'], stage['gradients']
    check(stage['optimizer_steps'] == len(steps) <= 4 and stage['trial_count'] == len(trials) <= 12
        and stage['trial_attempt_count'] == len(trials) and stage['optimizer_iterations'] == len(gradients)
        and stage['accepted_trial_count'] == len(steps) and stage['rejected_trial_count'] == len(trials)-len(steps)
        and stage['backward_evaluation_count'] == len(gradients) <= 4, 'FCR actual optimizer counts mismatch')
    objectives = []; current_loss = None; path_length = 0.
    if prep['no_information']:
        check(not steps and not trials and not gradients and stage['initial_objective'] is None and stage['final_objective'] is None
            and stage['stop_reason'] == prep['no_information_reason'] and stage['keep_limit'] is None, 'Invented FCR no-information update')
    else:
        obj = stage['initial_objective']; objective(obj, Z, U); objectives.append(obj); current_loss = obj['loss_total']
        check(obj['backward_evaluation_count'] == 0 and obj['initial_teacher_cache_reused'] is
            (mode == 'B' and prep['initial_inner_head_fit_count'] > 0), 'Initial R0 teacher cache reuse mismatch')
        slack = 1/(2*entry['train_k']*math.sqrt(len(prep['old_classes']))) if prep['keep_available'] else None
        check(stage['keep_available'] is prep['keep_available'], 'Stage keep availability mismatch')
        if slack is None: check(stage['keep_slack'] is stage['keep_limit'] is None, 'Invented single-class keep limit')
        else:
            close(stage['keep_slack'], slack, 'Frozen physical keep slack mismatch')
            close(stage['keep_anchor_risk'], obj['loss_keep'], 'Keep anchor risk mismatch')
            close(stage['keep_limit'], obj['loss_keep']+slack, 'Keep stage limit mismatch')
    accepted = []; used_trials = 0
    for iteration, event in enumerate(gradients, 1):
        data = resolver(event['state_ref']); gz, ku, dz = data['g_Z'], data['keep_g_Z'], data['d_Z']
        ez, eu = check_state(event['state_ref'])
        check(event['iteration'] == iteration and np.array_equal(ez, Z) and np.array_equal(eu, U),
            'FCR gradient accepted-cache binding mismatch')
        check(all(value.shape == Z.shape for value in (gz, ku, dz)), 'FCR gradient coordinate shape mismatch')
        gn, kn = float(np.linalg.norm(gz)), float(np.linalg.norm(ku))
        close(event['gradient_norm'], gn, 'FCR total gradient norm mismatch')
        close(event['keep_gradient_norm'], kn, 'FCR keep gradient norm mismatch')
        obj = event['objective']; objective(obj, Z, U)
        check(obj['forward_cache_reused'] is True and obj['backward_evaluation_count'] == 1, 'FCR gradient refitted accepted cache')
        close(obj['loss_total'], current_loss, 'FCR gradient cache objective mismatch')
        bound = max(0., stage['keep_limit']-obj['loss_keep'])/.125 if prep['keep_available'] else 0.
        d0 = -gz/gn if gn else np.zeros_like(gz)
        dot = float(np.sum(ku*d0)); active = kn > 0 and dot > bound
        expected = d0-(ku/kn)*((dot-bound)/kn) if active else d0
        tolerance = 128*sys.float_info.epsilon*max(1, gz.size)
        check(np.allclose(dz, expected, rtol=0., atol=tolerance), 'FCR keep halfspace direction mismatch')
        check(event['guard_active'] is active, 'FCR halfspace activation mismatch')
        for field, expected in (('guard_bound', bound), ('guard_dot_before', dot), ('guard_dot_after', float(np.sum(ku*dz))),
                                ('direction_norm', float(np.linalg.norm(dz)))):
            close(event[field], expected, 'FCR direction diagnostic mismatch: '+field)
        check(float(np.linalg.norm(dz)) <= 1+tolerance, 'FCR guard direction renormalized or expanded')
        group = [trial for trial in trials if trial['iteration'] == iteration]
        check(len(group) <= 3 and (gn > 0 and np.linalg.norm(dz) > 0 or not group), 'FCR trial after zero direction/gradient')
        winner = None
        for number, trial in enumerate(group, 1):
            check(winner is None and trial['trial'] == number and trial['step_size'] == .125/(2**(number-1))
                and trial['gradient_state_ref'] == event['state_ref'], 'FCR fixed trial sequence mismatch')
            tz, tu = check_state(trial['state_ref'])
            check(np.array_equal(tz, Z+trial['step_size']*dz), 'FCR actual function-coordinate trial mismatch')
            close(trial['update_norm'], float(np.linalg.norm(tz-Z)), 'FCR actual coordinate update norm mismatch')
            objective(trial['objective'], tz, tu)
            check(trial['objective']['forward_cache_reused'] is False and trial['objective']['backward_evaluation_count'] == 0,
                'FCR trial differentiated/reused cache')
            close(trial['loss_before'], current_loss, 'FCR trial accepted-cache loss mismatch')
            after = trial['objective']['loss_total']; rhs = current_loss+1e-4*float(np.sum(gz*(tz-Z)))
            tol = 128*sys.float_info.epsilon*max(1., abs(current_loss), abs(after), abs(rhs))
            close(trial['loss_after'], after, 'FCR trial total loss mismatch'); close(trial['armijo_rhs'], rhs, 'FCR Armijo RHS mismatch')
            close(trial['objective_acceptance_bound'], min(current_loss, rhs), 'FCR objective nonincrease bound mismatch')
            check(trial['armijo_tolerance'] == tol, 'FCR Armijo tolerance mismatch')
            risk = trial['objective']['loss_keep']; limit = stage['keep_limit']
            ktol = 128*sys.float_info.epsilon*max(1., abs(risk), abs(limit)) if limit is not None else 0.
            armijo = after <= rhs+tol; nonincrease = after <= current_loss+tol
            keepok = limit is None or risk <= limit+ktol; accept = armijo and nonincrease and keepok
            close(trial['keep_risk_trial'], risk, 'FCR actual trial keep risk mismatch')
            check(trial['keep_limit'] == limit and trial['armijo_accepted'] is armijo and trial['keep_accepted'] is keepok
                and trial['armijo_pass'] is armijo and trial['objective_nonincrease_pass'] is nonincrease
                and trial['keep_pass'] is keepok and trial['accepted'] is accept, 'FCR actual three-condition acceptance mismatch')
            check(trial['keep_tolerance'] == ktol, 'FCR keep tolerance mismatch')
            close(trial['keep_violation'], max(0., risk-limit) if limit is not None else 0., 'FCR trial keep violation mismatch')
            reason = None if accept else '_AND_'.join(name for name, passed in
                (('ARMIJO', armijo), ('OBJECTIVE_INCREASE', nonincrease), ('KEEP', keepok)) if not passed)
            check(trial['reject_reason'] == trial['rejection_reason'] == reason, 'FCR rejection reason mismatch')
            objectives.append(trial['objective']); used_trials += 1
            if accept: winner = trial
        if winner:
            step = steps[len(accepted)]
            check(step['step'] == len(accepted)+1 and step['iteration'] == iteration and step['trial'] == winner['trial']
                and step['state_ref'] == winner['state_ref'] and step['gradient_state_ref'] == event['state_ref']
                and step['objective'] == winner['objective'], 'FCR accepted step/trial mismatch')
            close(step['loss_after'], winner['loss_after'], 'FCR accepted step loss mismatch')
            path_length += winner['update_norm']
            Z, U = check_state(winner['state_ref']); current_loss = winner['loss_after']; accepted.append(step)
        else: check(iteration == len(gradients), 'FCR optimizer continued after failed/zero update')
    final = resolver(stage['final_state_ref']); fz, fu = check_state(stage['final_state_ref'])
    check(used_trials == len(trials) and accepted == steps and np.array_equal(fz, Z) and np.array_equal(fu, U),
        'FCR final state is not last accepted cache')
    tolerance = 128*sys.float_info.epsilon*max(1, Z.size)
    check(np.linalg.norm(Z) <= path_length+tolerance and path_length <= .5+tolerance, 'FCR function-coordinate path bound mismatch')
    if not prep['no_information']:
        objective(stage['final_objective'], Z, U)
        check(stage['final_objective']['forward_cache_reused'] is True and stage['final_objective']['backward_evaluation_count'] == 0
            and stage['final_objective']['derivative_triangular_solve_count'] == 0, 'FCR final cached objective did extra backward')
        close(stage['final_objective']['loss_total'], current_loss, 'FCR final objective is not accepted loss')
        reason = stage['stop_reason']
        check(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'ZERO_GUARDED_DIRECTION', 'ZERO_PROJECTED_STEP',
            'ARMIJO_OR_KEEP_BUDGET_EXHAUSTED'), 'Unknown FCR solver stop')
        if reason == 'MAX_ITERATIONS': check(len(gradients) == len(steps) == 4, 'Premature FCR iteration stop')
        if reason == 'ZERO_GRADIENT': check(gradients and gradients[-1]['gradient_norm'] == 0, 'False FCR zero gradient stop')
        if reason == 'ZERO_GUARDED_DIRECTION': check(gradients and gradients[-1]['direction_norm'] == 0, 'False FCR guarded direction stop')
        if reason == 'ARMIJO_OR_KEEP_BUDGET_EXHAUSTED':
            check(gradients and len([t for t in trials if t['iteration'] == len(gradients)]) == 3 and not trials[-1]['accepted'],
                'False FCR exhausted budget stop')
        if reason == 'ZERO_PROJECTED_STEP':
            last = gradients[-1]; data = resolver(last['state_ref'])
            number = len([t for t in trials if t['iteration'] == last['iteration']])+1
            check(1 <= number <= 3 and last['gradient_norm'] > 0 and last['direction_norm'] > 0
                and np.array_equal(Z+.125/(2**(number-1))*data['d_Z'], Z), 'False FCR zero coordinate displacement stop')
    for key in ('inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count'):
        check(stage[key] == sum(value[key] for value in objectives), 'FCR actual forward cost mismatch: '+key)
    check(stage['fcr_forward_evaluation_count'] == sum(value['fcr_forward_evaluation_count'] for value in objectives)+stage['final_head_fit_count'],
        'FCR inner/final forward count mismatch')
    for key in ('derivative_triangular_solve_count', 'task_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count'):
        check(stage[key] == sum(g['objective'][key] for g in gradients), 'FCR actual cached adjoint count mismatch')
    changed = not np.array_equal(U, anchor)
    check(stage['parameter_changed_from_anchor'] is stage['u_changed_from_anchor'] is changed
        and stage['nonzero_projected_update_count'] == len(steps), 'FCR state update evidence mismatch')
    close(stage['u_update_norm'], float(np.linalg.norm(U-anchor)), 'FCR original-U anchor distance mismatch')
    close(stage['parameter_U_norm'], float(np.linalg.norm(U)), 'FCR U norm mismatch')
    close(stage['coordinate_norm'], float(np.linalg.norm(Z)), 'FCR Z norm mismatch')
    check(stage['parameter_V_delta_norm'] == 0 and stage['maximum_trainable_parameter_count'] == 5888
        and stage['trainable_parameter_count'] == 736*rank, 'FCR fixed dictionary/active parameter count mismatch')
    check(stage['persistent_state_bytes'] == stage['head_state_bytes']+stage['lineage_state_bytes']+stage['adapter_state_bytes']
        and stage['adapter_state_bytes'] == 94208 and stage['fit_seconds'] >= 0 and stage['score_seconds'] >= 0,
        'FCR complete state/parameter/cost mismatch')
    certify_head(stage['final_fit'])
    baseline_stage = entry['stages'][0 if prep['state'] == 'B' else -1]
    check(stage['final_fit']['interaction_centered_trace'] == baseline_stage['interaction_centered_trace'], 'FCR original trace target changed')


def verify_record(record, split, old, resolver=None):
    finite_tree(record); check(record['scope'] == SCOPE and record['query_rows_used'] == record['source_rows_used'] == 0, 'Forbidden parent access')
    verify_baseline(baseline_view(record), split, old)
    if split['k'] == 1:
        check(all(record[key] == 0 for key in COUNTERS[4:]) and record['persistent_state_bytes'] == 0, 'True K1 fabricated training')
        return [], [], dict(oof=None, proxy=None), []
    check(resolver is not None, 'Complete FCR8 coordinate archive resolver required')
    old = sorted(old); classes = sorted(split['registered_classes'])
    labels = {pid: split['registered_classes'][label] for pid, label in zip(split['support_ids'], split['support_labels'])}
    counts = dict.fromkeys(COUNTERS[4:], 0); logs = []; events = []; training = []; maximum = 0
    for entry in record['folds']+record['oneshot_proxy']['trials']:
        check(set(entry['paths']) == set(PATHS), 'Three fixed paths required')
        common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
        evidence = {name: dict(common, b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
        fresh = assess_paths(evidence, old)
        for name in PATHS:
            check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]), 'Fixed-score paired result mismatch')
        reuse = classes == old
        check(entry['c_reuses_b_candidates'] is reuse and entry['paths']['R_FCR8_seq']['b_scores'] == entry['paths']['R_FCR8_reset_init']['b_scores'], 'Shared B/N0 mismatch')
        if reuse:
            for name in PATHS: check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'N0 must reuse B scores exactly')
        if entry['train_k'] == 1:
            check(all(entry['paths'][name] == entry['paths']['R0'] for name in PATHS), 'Proxy trainK1 must exactly equal R0')
        check([p['state'] for p in entry['preparations']] == (['B'] if reuse else ['B', 'C']), 'Preparation sharing mismatch')
        expected_stages = ['B_FCR8']+([] if reuse else ['C_FCR8_seq', 'C_reset_init'])
        check([s['state'] for s in entry['candidate_stages']] == expected_stages, 'Candidate stage coverage/order mismatch')
        by_stage = {stage['state']: stage for stage in entry['candidate_stages']}; by_prep = {prep['state']: prep for prep in entry['preparations']}
        for base in entry['stages']:
            logs.append(dict(event='BASE_FIT', **base)); counts['baseline_head_fit_count'] += 1; counts['head_fit_count'] += 1
            counts['baseline_factorization_count'] += base['factorization_calls']; counts['factorization_count'] += base['factorization_calls']
            counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += base['held_physical_count']
        for prep in entry['preparations']:
            verify_preparation(prep, entry, labels, old, resolver); logs.append(dict(event='FCR_PREPARATION', **prep)); counts['fcr_preparation_count'] += 1
            for key in PREPARATION_COUNTERS: counts[key] += prep[key]
            counts['head_fit_count'] += prep['inner_head_fit_count']+prep['teacher_head_fit_count']
            counts['factorization_count'] += prep['inner_factorization_count']+prep['teacher_factorization_count']
            for stage in (s for s in entry['candidate_stages'] if s['preparation_ref'] == prep['state']):
                verify_candidate(stage, prep, entry, by_stage['B_FCR8'], resolver); logs.append(dict(event='CANDIDATE_FIT', **stage))
                counts['fcr_stage_count'] += 1
                counts['trained_fcr_stage_count'] += int(stage['optimizer_steps'] > 0)
                for key in STAGE_COUNTERS: counts[key] += stage[key]
                counts['head_fit_count'] += stage['inner_head_fit_count']+stage['final_head_fit_count']
                counts['factorization_count'] += stage['inner_factorization_count']+stage['final_factorization_count']
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += stage['held_physical_count']
                small_training = compact_event(dict(final_state_ref=stage['final_state_ref'], initialization_state_ref=stage['initialization_state_ref'],
                    no_information=stage['no_information'], optimizer_steps=stage['optimizer_steps'],
                    nonzero_projected_update_count=stage['nonzero_projected_update_count'],
                    u_changed_from_anchor=stage['u_changed_from_anchor'], u_update_norm=stage['u_update_norm'],
                    final_objective=stage['final_objective']))
                training.append(dict(split_id=record['split_id'], scope=entry['scope'], fold=entry['fold'], trial=entry['trial'],
                    k=record['k'], new_count=record['new_count'], state=stage['state'], train_k=entry['train_k'],
                    **small_training, stop_reason=stage['stop_reason'],
                    initial_objective=None if stage['initial_objective'] is None else compact_event(stage['initial_objective']),
                    coordinate_state_ref=prep['coordinate_state_ref'], latent_rank=prep['latent_rank'],
                    trainable_parameter_count=stage['trainable_parameter_count'], coordinate_norm=stage['coordinate_norm'],
                    pre_tangent_displacement_rms=stage['pre_tangent_displacement_rms'],
                    steps=[compact_event(step) for step in stage['steps']],
                    gradients=[compact_event(value) for value in stage['gradients']],
                    trials=[compact_event(value) for value in stage['trials']],
                    evidence_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION',
                    full_vectors_source='Referenced row state_arrays/*.npz and state_manifest.json; all coordinates are verified'))
        per_stage = {name: dict(steps=0, trials=0, gradients=0, initial=0, final=0) for name in by_stage}
        prepared_seen = set()
        for event in entry['training_events']:
            check(event['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION' and event['source_validation'] is None,
                'Invented validation event')
            check(event['scope'] == entry['scope'] and event['fold'] == entry['fold']
                and event['outer_trial'] == entry['trial'], 'Training event outer-parent coordinates mismatch')
            name = event['state']; kind = event['event']
            if kind == 'FCR_INNER_PREPARED':
                prep_name = name.removesuffix('_prepare'); index = event['inner_fold']; key = (prep_name, index)
                check(prep_name in by_prep and key not in prepared_seen
                    and 0 <= index < len(by_prep[prep_name]['inner_folds']), 'Unknown/duplicate inner preparation event')
                expected_inner = by_prep[prep_name]['inner_folds'][index]
                check(all(event.get(field) == value for field, value in expected_inner.items()), 'Inner preparation binding mismatch')
                prepared_seen.add(key); continue
            check(name in by_stage, 'Unknown training event stage'); stage = by_stage[name]; seen = per_stage[name]
            mapping = {'FCR_GRADIENT': 'gradients', 'FCR_TRIAL': 'trials', 'FCR_STEP': 'steps'}
            if kind in mapping:
                field = mapping[kind]; index = seen[field]
                check(index < len(stage[field]), 'Training event exceeds full solver trace length')
                mismatched = [key for key, value in stage[field][index].items() if event.get(key) != value]
                check(not mismatched, 'Training event/full solver trace mismatch: '+name+'/'+kind+'/'+str(index)+
                    ' fields='+','.join(mismatched))
                seen[field] += 1
            elif kind == 'FCR_INITIAL':
                check(seen['initial'] == 0 and event['state_ref'] == stage['initialization_state_ref']
                    and event['objective'] == stage['initial_objective'], 'Initial accepted-cache event mismatch')
                seen['initial'] += 1
            elif kind == 'FCR_FINAL':
                check(seen['final'] == 0 and event['final_state_ref'] == stage['final_state_ref'] and event['final_objective'] == stage['final_objective']
                    and event['stop_reason'] == stage['stop_reason'], 'Final solver event mismatch')
                for key in STAGE_COUNTERS: check(event[key] == stage[key], 'Final event actual cost mismatch')
                seen['final'] += 1
            else: raise ValueError('Unexpected FCR8 training event kind')
        for name, stage in by_stage.items():
            seen = per_stage[name]
            check(seen['final'] == 1 and seen['initial'] == int(not stage['no_information'])
                and all(seen[field] == len(stage[field]) for field in ('steps', 'trials', 'gradients')),
                'Full solver event coverage mismatch')
        check(prepared_seen == {(p['state'], index) for p in entry['preparations'] for index in range(len(p['inner_folds']))},
            'Inner preparation event coverage mismatch')
        events.extend(entry['training_events']); counts['sequence_paths'] += 1
        for name, stage_name in (('R_FCR8_seq', 'B_FCR8' if reuse else 'C_FCR8_seq'), ('R_FCR8_reset_init', 'B_FCR8' if reuse else 'C_reset_init')):
            check(entry['deployment_C_state_bytes'][name] == by_stage[stage_name]['persistent_state_bytes'], 'Deployment state bytes mismatch')
        maximum = max(maximum, *entry['deployment_C_state_bytes'].values())
    check(all(record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == maximum, 'Actual parent totals mismatch')
    check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'), 'OOF physical pooling mismatch')
    check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy parent-first mean mismatch')
    resolver.verify_tree(record)
    return logs, events, dict(oof={name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=record['oneshot_proxy']['parent_mean_metrics']), training


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.)+record['fit_seconds']
    for row in logs:
        group = row['event'].lower()
        for key, value in row.items():
            if key.endswith('_seconds') and key != 'preparation_seconds' and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group+'_'+key+'_sum'
                resources[name] = resources.get(name, 0.)+value
        if row['event'] == 'CANDIDATE_FIT':
            name = 'training_steps_seconds_sum'
            resources[name] = resources.get(name, 0.)+sum(step['step_seconds'] for step in row['steps'])
            for field in ('baseline_binding_distance_evaluation_count', 'baseline_binding_factorization_count'):
                name = 'candidate_fit_'+field+'_sum'; resources[name] = resources.get(name, 0)+row[field]
            for field, values in (('objective_forward_seconds_sum',
                    ([] if row['initial_objective'] is None else [row['initial_objective']])+[t['objective'] for t in row['trials']]),
                    ('objective_backward_seconds_sum', [g['objective'] for g in row['gradients']])):
                resources[field] = resources.get(field, 0.)+sum(value['objective_seconds'] for value in values)
            forward = ([] if row['initial_objective'] is None else [row['initial_objective']])+[t['objective'] for t in row['trials']]
            resources['inner_forward_seconds_sum'] = resources.get('inner_forward_seconds_sum', 0.)+sum(
                fold['forward_seconds'] for value in forward for fold in value['inner_folds'])
            for channel in ('task', 'keep'):
                key = 'inner_'+channel+'_adjoint_seconds_sum'
                resources[key] = resources.get(key, 0.)+sum(fold.get(channel+'_adjoint_seconds', 0.)
                    for value in row['gradients'] for fold in value['objective']['inner_folds'])
            resources['final_head_forward_seconds_sum'] = resources.get('final_head_forward_seconds_sum', 0.)+(
                row['final_fit']['forward_seconds'] if row['final_head_fit_count'] else 0.)
        if row['event'] == 'FCR_PREPARATION':
            for name, meta in row['coordinate_state_ref']['arrays'].items():
                key = 'fcr_preparation_maximum_'+name+'_bytes'
                resources[key] = max(resources.get(key, 0), meta['nbytes'])
            resources['teacher_or_prepaid_initial_forward_seconds_sum'] = resources.get('teacher_or_prepaid_initial_forward_seconds_sum', 0.)+sum(
                value.get('head_audit', {}).get('forward_seconds', 0.) for value in row['teacher_folds'])
        for key in ('persistent_state_bytes', 'adapter_state_bytes', 'head_state_bytes', 'lineage_state_bytes', 'prepared_numeric_state_bytes', 'optimizer_state_bytes', 'transient_distance_bytes', 'initial_forward_cache_numeric_bytes', 'teacher_record_numeric_bytes', 'retained_vector_record_bytes'):
            if key in row: resources[group+'_maximum_'+key] = max(resources.get(group+'_maximum_'+key, 0), row[key])


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec; validate_spec(spec)
    root = Path(run_root or spec['execution']['remote_run_root']); out = Path(output)
    if out.exists(): raise FileExistsError(out)
    launch, complete, state = [read(root/name) for name in ('startup.json', 'complete.json', 'state.json')]
    check(launch['spec'] == spec and complete['status'] == STATUS and complete['model_rows'] == complete['completed_rows'] == 4
        and complete['episodes'] == 160 and complete['commit'] == launch['commit'], 'Full four-row pilot incomplete')
    check(set(state) == {row['row_id'] for row in spec['rows']} and all(v['status'] == STATUS for v in state.values()), 'Incomplete row state')
    check(all(v.get('query_access') is False and v.get('source_sample_access') is False for v in (launch, complete)), 'Forbidden run access')
    lanes = []
    # Verify ALL completion/source bindings before opening any outer-held score trace.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root/row['row_id']/'probe'
        marker = verify_marker(lane/'probe_complete.json', spec, row); startup = read(lane/'startup.json'); check_bind(startup, row, co)
        check(startup['config'] == dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
            and startup['scope'] == marker['scope'] == SCOPE and startup['episodes'] == 40
            and startup['producer_episodes'] == co['expected_split_count'], 'Startup/config mismatch')
        for value in (startup, marker):
            check(all(value.get('channel', {}).get(key) == expected for key, expected in CHANNEL.items())
                and value['scenarios'] == SCENARIOS and value['query_rows_used'] == value['source_rows_used'] == 0
                and value['truth_read'] is False, 'Channel or forbidden access mismatch')
            check(value['payload_audit']['new_source_payload_bytes'] == value['payload_audit']['new_ground_statistics_bytes'] == 0,
                'Unexpected additional ground data payload')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
            and startup['actual_A'] is None and startup['adapted_state_inherited'] is True
            and startup['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION', 'Invented or forbidden adaptation state')
        feature_root = Path(startup['support_features']); check(feature_root == Path(row['support_features']), 'Unexpected support cache reference')
        feature, plan, provenance = [read(feature_root/name) for name in ('features_complete.json', 'support_splits.json', 'checkpoint_provenance.json')]
        for value in (feature, plan):
            check(value['capsule_id'] == co['capsule_id'] and value['checkpoint_sha256'] == row['expected_checkpoint_sha256'], 'Producer binding mismatch')
        check_bind(feature, row, co)
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == co['expected_split_count']
            and provenance == startup['provenance'] and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH'
            and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance'] == [], 'Producer provenance mismatch')
        old = feature['classes']; check(len(old) == 6, 'Fixed six-old-class pilot required')
        chosen = selected_tasks([(s, None, None) for s in plan['splits']], co['selection'], old)
        lanes.append((row, lane, marker, {s['split_id']: s for s, _, _ in chosen}, old, startup))
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; training = []; archive_sources = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    resource_strata = {name: {} for name in ('by_k_new_count', 'by_receiver_scene', 'by_model_cohort', 'by_row')}
    for row, lane, marker, expected, old, startup in lanes:
        verify_artifact_inventory(lane, marker); resolver = StateResolver(lane)
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stages = iter(jsonlines(lane/'fit_stages.jsonl'))
        event_stream = iter(jsonlines(lane/'training_events.jsonl')); small_events = iter(jsonlines(lane/'training_events_compact.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact length mismatch'); sid = record['split_id']
            check(sid in expected and sid not in seen, 'Unexpected/duplicate parent'); seen.add(sid)
            logs, events, measurements, train = verify_record(record, expected[sid], old, resolver); accumulate_resources(resources, record, logs)
            resource_keys = dict(by_k_new_count=(record['k'], record['new_count']),
                by_receiver_scene=(row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']),
                by_model_cohort=(row['seeds']['model'], row['cohort'], record['k'], record['new_count']), by_row=(row['row_id'],))
            for group_name, group_key in resource_keys.items():
                cell = resource_strata[group_name].setdefault(group_key, {})
                accumulate_resources(cell, record, logs); cell['parents'] = cell.get('parents', 0)+1
                for key in COUNTERS[4:]: cell[key] = cell.get(key, 0)+record[key]
            check(small == compact_record(record), 'Compact evidence mismatch')
            for value in logs: check(next(stages, None) == dict(compact_event(value), split_id=sid), 'Stage stream mismatch')
            for value in events:
                full = dict(value, split_id=sid)
                check(next(event_stream, None) == full and next(small_events, None) == compact_event(full), 'Full/compact training event mismatch')
            training.extend(dict(row_id=row['row_id'], model_seed=row['seeds']['model'], cohort=row['cohort'],
                receiver=record['receiver'], scenario=record['scenario'], **value) for value in train)
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1)
            counts['oof_episodes'] += int(record['k'] > 1); counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for key in COUNTERS[4:]: counts[key] += record[key]
            for diagnostic, paths in measurements.items():
                if paths is None: paths = {name: dict.fromkeys(METRICS) for name in PATHS}
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                for name, metrics in paths.items():
                    _add(strata['overall'], (diagnostic, name, population), metrics)
                    _add(strata['by_k_new_count'], (diagnostic, name, record['k'], record['new_count']), metrics)
                    _add(strata['by_receiver_scene'], (diagnostic, name, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), metrics)
                    _add(strata['by_model_cohort'], (diagnostic, name, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), metrics)
        check(seen == set(expected) and next(stages, None) is next(event_stream, None) is next(small_events, None) is None, 'Missing parents or extra events')
        resolver.finalize(); manifest = resolver.manifest
        check(marker['state_archive_file_count'] == manifest['file_count'] and marker['state_archive_file_bytes'] == manifest['total_file_bytes']
            and marker['state_archive_numeric_bytes'] == manifest['numeric_array_bytes'], 'Marker/archive actual byte mismatch')
        archive_sources.append(dict(row_id=row['row_id'], root=str(lane), manifest=str(lane/'state_manifest.json'),
            file_count=manifest['file_count'], file_bytes=manifest['total_file_bytes'], numeric_array_bytes=manifest['numeric_array_bytes'],
            archive_seconds=manifest['archive_seconds'], by_phase=manifest['by_phase']))
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row actual totals mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
    check(all(coverage[key] == complete[key] for key in COUNTERS), 'Run totals mismatch')
    check(all(coverage[key] == value for key, value in EXACT_COUNTS.items())
        and all(coverage[key] <= value for key, value in MAX_COUNTS.items()), 'Fixed pilot actual budget mismatch')
    check(complete['finished'] >= launch['started'], 'Run wall-clock bound mismatch')
    resources.update(run_wall_seconds=complete['finished']-launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _ in lanes
            if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        additional_ground_data_payload_bytes=0, additional_ground_statistics_bytes=0,
        remote_code_release_archive_bytes=None, remote_code_release_archive_reason='Release transport artifact is separate from method data payload and is not measured by this evaluator',
        separate_inner_kernel_seconds=None,
        unmeasured_reason='CPU only; no deployment package/transfer measured; RSS null if runtime unavailable; objective wall times include their measured forward or cached backward work; standalone kernel timing is not measured',
        fitted_state_scope='Full numeric U and fixed V0 adapter, LocalRidge head, original/adapted support, raw/labels retained for inheritance; excludes frozen Phase1 delivery, shared feature archive, teacher/training caches, Python and serialization overhead',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment']) for row, _, _, _, _, startup in lanes],
        interpretation='Stage duration sums measure work, not concurrent elapsed time. Shared preparation is charged once. Final scoring timings are measured on the stated CPU/thread configuration.')
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    resource_dimensions = dict(by_k_new_count=('k', 'new_count'),
        by_receiver_scene=('cohort', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_cohort=('model_seed', 'cohort', 'k', 'new_count'), by_row=('row_id',))
    resource_tables = {name: [dict(zip(resource_dimensions[name], key), **value) for key, value in sorted(groups.items())]
                       for name, groups in resource_strata.items()}
    summary = dict(status=SUMMARY_STATUS, scope=SCOPE, run_id=spec['run_id'], release_commit=complete['commit'],
        coverage=coverage, resources=resources, resource_statistics=resource_tables,
        algorithm=PROBE_CONFIG, channel=CHANNEL, statistics=tables,
        old_class_count=6, actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        query_rows_used=0, source_rows_used=0, training_stage_count=len(training),
        raw_training_sources=[dict(row_id=row['row_id'], fit_trace=str(lane/'fit_trace.jsonl'),
            full_training_events=str(lane/'training_events.jsonl'), compact_training_events=str(lane/'training_events_compact.jsonl'))
            for row, lane, _, _, _, _ in lanes],
        state_archives=archive_sources, state_archive_file_count=sum(v['file_count'] for v in archive_sources),
        state_archive_file_bytes=sum(v['file_bytes'] for v in archive_sources),
        interpretation=[
            'R0 remains original LocalRidge. R_FCR8_seq is the fixed main candidate; R_FCR8_reset_init changes C initialization, proximal anchor and the resulting stage keep limit to its own initial state, while retaining the same B teacher.',
            'Sequential and reset share one trained B and one C preparation/teacher. Every teacher old head and student head uses its own physical inner-train. C sequential exactly inherits original B U and builds fresh W with Z=0; reset_init uses U=0. Fixed DCT V0 is shared and never trained.',
            'Task class RMS and previous-class teacher-deficit RMS pool physical losses across folds before reduction; both coefficients are one. Inner-held labels supervise function-coordinate Z/original U and are not independent validation.',
            'Four iterations and three trials are fixed. The normalized function-coordinate total gradient is projected into the keep halfspace without a parameter ball. Each actual trial must pass Armijo, total-objective nonincrease and real keep-risk checks. Every rejection and bounded stop is verified; final parameters use the last accepted cache.',
            'True K1 has no fitting or held score. Proxy trainK1 uses exact R0 identity forward for every path; no K1 benefit is claimed.',
            'A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.',
            'OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.',
            'Old-only reuse and repeated old support across new-count rows are not independent observations.',
            'At most 5888 U parameters are trainable (736 times actual latent rank); U and materialized fixed V0 together occupy 94208 bytes; this is not total model size or proof of lower training cost. Baseline, prepaid B initial heads, C teacher heads, student trial heads, both adjoint channels and full deployed state are separately counted.',
            'Every parameter/gradient/direction coordinate and teacher score/q is preserved in exclusive NPZ archives, referenced by complete native finite JSON and checked against exact inventories/content. Compact logs retain Z/U/W and gradient array summaries plus measured forward mechanisms; bulk archives remain original artifacts and are not duplicated in reports.',
            '10/1/3 percentage-point ideal directions are descriptive and impose no automatic promotion gate.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out/'summary.json', summary)
    for name, rows in tables.items():
        with (out/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, rows in resource_tables.items():
        fields = sorted({key for row in rows for key in row})
        with (out/('resources_'+name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in rows)
    with (out/'training_objectives.jsonl').open('x', encoding='utf-8') as stream:
        for row in training: stream.write(json.dumps(row, allow_nan=False)+'\n')
    with (out/'training_objectives.csv').open('x', encoding='utf-8', newline='') as stream:
        compact_training = [compact_event(row) for row in training]
        fields = sorted({key for row in compact_training for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in compact_training)
    lines = ['# FCR8-LocalRidge support pilot', '',
        '完整160 parent、4 row、三路径均已核验；旧类固定6个。A与B−A为N/A。R_FCR8_seq为预声明顺序主线，R_FCR8_reset_init仅作继承对照。', '',
        '| 诊断 | 路径 | K | 新类数 | A旧 | B0旧 | B旧 | C旧类列 | C旧 | C新 | H | B−B0 | B−C旧类列 | 新竞争损失 | 注册下降 | 新旧差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cells = {}
    for row in tables['by_k_new_count']:
        cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    display = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy', 'C_old_accuracy', 'C_new_accuracy',
        'C_h', 'support_adaptation_B_minus_B0', 'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    for key, values in sorted(cells.items()):
        lines.append('| '+' | '.join([str(value) for value in key]+['N/A' if values[m] is None else f'{100*values[m]:.3f}' for m in display])+' |')
    lines += ['', '准确率为百分数，差值为百分点。全部分层、相对R0变化与六类正确性转换见CSV。内部训练目标、参数、梯度、全部回溯试探和停止原因见training_objectives，不把内部训练准确率称为独立验证。', '',
        f"实际有效训练阶段{coverage['trained_fcr_stage_count']}个、函数坐标更新{coverage['optimizer_steps']}次、内层head拟合{coverage['inner_head_fit_count']}次、新增最终head拟合{coverage['final_head_fit_count']}次。",
        f"实测运行墙钟{resources['run_wall_seconds']:.3f} s；完整适配器和分类头状态、线程与分项工作量见summary.json。", '']
    lines += ['- '+line for line in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8'); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
