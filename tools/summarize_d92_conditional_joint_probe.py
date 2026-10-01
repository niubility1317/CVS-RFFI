"""Independently certify ConditionalJoint support archives, traces and resources.

No candidate fit, forward, score or adjoint is called by this analyzer. Saved
Gaussian geometry, frozen actual B and complete KKT equations are its evidence.
"""
import argparse
from collections import OrderedDict
import csv
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
from evaluate_d92_conditional_joint_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, SCHEMA, METHOD, PATHS, METRICS, BRANCHES,
    PROBE_CONFIG, COUNTERS, PREPARATION_COUNTERS, STAGE_COUNTERS,
    read, check, csv_record, split_identity, selected_tasks, compact_event,
    assess_paths, pooled_assess, parent_mean, compact_record, json_native,
)
from run_d92_conditional_joint_probe import validate_spec, verify_marker, budget_for_spec, command
import summarize_d92_affine_joint_probe as affine_analysis
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close
from summarize_d92_registration_diagnostic import _statistics, write_json
from cvsrffi.d92_branch_local_ridge import _distances
from d92_affine_analysis_math import center_kernel_vjp
from d92_conditional_analysis_math import certify_positive_kernel_head, positive_kernel_head_vjp

SUMMARY_STATUS = 'COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED'
SUMMARY_SCHEMA = 'd92_conditional_joint_support_summary_v1'
PREFIXES = ('projection', 'residual', 'projection_adjoint', 'residual_adjoint')
SUFFIXES = ('factorization_count', 'triangular_solve_count', 'triangular_rhs_count',
            'triangular_rhs_element_count', 'triangular_dense_work_unit_count')
KERNEL_COUNTERS = tuple(p + '_' + s for p in PREFIXES for s in SUFFIXES) + (
    'spectral_diagnostic_count', 'completed_factorization_count')
_array_close = affine_analysis._array_close
_scalar = affine_analysis._scalar
adapted_blocks = affine_analysis.adapted_blocks
dct_initial = affine_analysis.dct_initial


class StateResolver(affine_analysis.StateResolver):
    """Reuse numeric inventory/cache checks; require the independent schema."""
    def __init__(self, root, cache_budget_bytes=64 * 1024 * 1024, max_entries=64):
        self.root = Path(root).resolve()
        self.manifest = read(self.root / 'state_manifest.json')
        check(self.manifest['schema'] == 'd92_conditional_joint_state_archive_v1'
              and self.manifest['method'] == METHOD and self.manifest['status'] == 'COMPLETE'
              and self.manifest['prediction_formula'] == 'actual_B_prior_plus_old_anchor_zero_conditional_affine_residual',
              'Incomplete Conditional state archive')
        files = self.manifest['files']; self.refs = {ref['path']: ref for ref in files}
        check(len(files) == len(self.refs) == self.manifest['file_count'], 'Duplicate/missing numeric archive')
        check(self.manifest['total_file_bytes'] == sum(ref['file_bytes'] for ref in files), 'Archive byte inventory mismatch')
        check(type(cache_budget_bytes) is int and cache_budget_bytes >= 0
              and type(max_entries) is int and max_entries >= 0, 'Invalid archive cache budget')
        self.used, self.verified, self.cache = set(), set(), OrderedDict()
        self.cache_budget_bytes, self.max_entries = cache_budget_bytes, max_entries
        self.cache_numeric_bytes = 0; self.cache_sizes = {}; self.file_mtimes = {}
        self.cache_counts = dict(load_count=0, hit_count=0, eviction_count=0, loaded_numeric_bytes=0,
            evicted_numeric_bytes=0, oversized_load_count=0, uncached_load_count=0, clear_count=0,
            cleared_entry_count=0, cleared_numeric_bytes=0, peak_numeric_bytes=0, peak_entries=0)
        self.analysis_work = dict(independent_general_solve_count=0, independent_general_rhs_column_count=0,
            independent_general_rhs_element_count=0, independent_dense_cubic_work_unit_count=0,
            independent_dense_rhs_work_unit_count=0, independent_spectral_diagnostic_count=0)


def verify_artifact_inventory(root, marker):
    root = Path(root).resolve(); value = read(root / 'artifact_manifest.json')
    check(value['schema'] == 'd92_conditional_joint_artifacts_v1' and value['method'] == METHOD
          and value['status'] == STATUS and marker['artifact_manifest'] == 'artifact_manifest.json', 'Incomplete artifact inventory')
    listed = {item['path']: item['file_bytes'] for item in value['files']}
    actual = {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob('*')
              if p.is_file() and p.name != 'artifact_manifest.json'}
    check(len(listed) == len(value['files']) and listed == actual, 'Artifact inventory mismatch')


def _raw(b, a):
    full = np.column_stack((b, a))
    return dict(zip(BRANCHES, (full[:, :160], full[:, 160:256], full[:, 256:416],
                              full[:, 416:576], full[:, 576:736])))


def _radial(distance, tau, gamma, minus_one=False):
    if gamma is None: return np.zeros_like(distance)
    if tau == 0: return np.asarray(distance == 0, dtype=np.float64) - float(minus_one)
    check(tau is not None and tau > 0 and gamma > 0, 'Undefined raw Gaussian scale')
    return np.expm1(-distance / tau) if minus_one else np.exp(-distance / tau)


def head_score(data, raw, *, prefix=''):
    """Independent fixed single-record inference for B, C and the R0 baseline."""
    part = {key[len(prefix):]: value for key, value in data.items() if key.startswith(prefix)} if prefix else data
    # Saved original geometry is the normalized five-branch representation,
    # not a concatenation of the received raw vectors.  The zero adapter is an
    # independent way to rebuild that same fixed representation for every head.
    ob, oa = adapted_blocks(raw, np.zeros((736, 8)))
    if 'U' not in part:
        tau, gamma = _scalar(part, 'tau'), _scalar(part, 'gamma'); n = len(part['alpha']); q = np.full(n, 1 / n)
        train = _radial(_distances(part['original_train_b'], part['original_train_a']), tau, gamma, True)
        scores = np.empty((len(ob), part['alpha'].shape[1]))
        for i in range(len(ob)):
            cross = _radial(_distances(ob[i:i + 1], oa[i:i + 1], part['original_train_b'], part['original_train_a']), tau, gamma, True)
            _, L = affine_analysis._center(train, cross, q, gamma); scores[i] = (L @ part['alpha'])[0]
        _array_close(affine_analysis.head_score(part, raw), scores, 'R0 deployment gauge differs from independent centered function')
        return scores
    b, a = adapted_blocks(raw, part['U']); tau, gamma = _scalar(part, 'tau'), _scalar(part, 'gamma')
    c = part['alpha'].shape[1]; scores = np.empty((len(ob), c))
    for i in range(len(ob)):
        d0 = _distances(ob[i:i + 1], oa[i:i + 1], part['original_train_b'], part['original_train_a'])
        d = d0 if tau == 0 or not np.any(part['U']) else .5 * (d0 + _distances(
            b[i:i + 1], a[i:i + 1], part['adapted_train_b'], part['adapted_train_a']))
        if 'intercept' in part:
            if gamma is None: row = part['intercept'].copy()
            else:
                train = _radial(part['distance'], tau, gamma, True)
                cross = _radial(d, tau, gamma, True)
                _, L = affine_analysis._center(train, cross, part['q'], gamma)
                row = (L @ part['alpha'])[0] + part['intercept']
        else:
            if gamma is None: row = np.zeros(c)
            else:
                raw_kernel = gamma * _radial(d, tau, gamma)
                row = raw_kernel[0, part['new_indices']] @ part['alpha'] - raw_kernel[0, part['old_representative_indices']] @ part['beta'] + part['v']
        scores[i] = row
    if not prefix and 'prior_old_class_indices' in part:
        prior = head_score(part, raw, prefix='prior_B_')
        indices = part['prior_old_class_indices']
        check(indices.dtype.kind in 'iu' and indices.shape == (prior.shape[1],)
              and len(set(indices.tolist())) == len(indices), 'Invalid actual B class map')
        scores[:, indices] += prior
    check(np.isfinite(scores).all(), 'Nonfinite independent scores')
    return scores


def _factor(factor, matrix, message):
    check(factor.shape == matrix.shape and np.all(np.diag(factor) > 0)
          and np.array_equal(factor, np.tril(factor)), message + ' shape/triangular/positive diagonal')
    _array_close(factor @ factor.T, matrix, message, scale=np.linalg.norm(factor) ** 2)


def _expected_system(prefix, order, width, factor):
    return {prefix + '_' + key: value for key, value in zip(SUFFIXES,
        (factor, 2 * factor, 2 * factor * width, 2 * factor * order * width, 2 * factor * order * order * width))}


def _head_work(data, audit):
    n, h, c = len(data['train_labels']), len(data['held_labels']), data['Y'].shape[1]
    positive = int(_scalar(data, 'gamma') is not None)
    b = audit['mode'] == 'B'; m, p = len(data['old_representative_indices']), len(data['new_indices'])
    expected = _expected_system('projection', m, p + 1, positive * int(not b))
    expected.update(_expected_system('residual', n if b else p, c + 1 if b else c, positive))
    expected.update(spectral_diagnostic_count=3 * positive * int(not b),
                    completed_factorization_count=positive * (1 if b else 2),
                    factorization_count=positive * (1 if b else 2), head_fit_count=1, ajlr_forward_evaluation_count=1)
    for suffix in SUFFIXES[1:]: expected['head_' + suffix] = expected['projection_' + suffix] + expected['residual_' + suffix]
    expected.update(intercept_fit_count=1, intercept_addition_count=(n + h) * c)
    identity = not np.any(data['U']) or _scalar(data, 'tau') == 0; old_count = len(data['old_indices'])
    expected.update(raw_distance_evaluation_count=(1 + int(h > 0)) * int(not identity),
        raw_distance_pair_count=(n * (n - 1) // 2 + h * n) * int(not identity),
        reference_distance_evaluation_count=(1 + int(h > 0)) * int(not identity),
        reference_distance_pair_count=(old_count * (old_count - 1) // 2 + old_count * (n - old_count) + h * old_count) * int(not identity),
        kernel_evaluation_count=(1 + int(h > 0)) * positive, kernel_pair_count=(n * n + h * n) * positive,
        adapter_physical_evaluation_count=n + h)
    for key, value in expected.items(): check(type(audit[key]) is int and audit[key] == value, 'Actual head work mismatch: ' + key)


def verify_head(ref, audit, resolver, prior=None):
    """Reconstruct raw geometry and validate complete equations, without fitting."""
    data = resolver(ref); ids, held, classes = audit['training_physical_ids'], audit['held_physical_ids'], audit['classes']
    n, h, c = len(ids), len(held), len(classes)
    check(audit['schema'] == SCHEMA and audit['method'] == METHOD and audit['mode'] in ('B', 'C_seq'), 'Head schema/mode mismatch')
    y, yh = data['train_labels'], data['held_labels']
    check(y.dtype.kind in 'iu' and yh.dtype.kind in 'iu' and y.shape == (n,) and yh.shape == (h,)
          and np.all((y >= 0) & (y < c)) and np.all((yh >= 0) & (yh < c)), 'Head physical label shape mismatch')
    old = audit['old_classes']; oi = np.array([i for i, value in enumerate(y) if classes[int(value)] in old], dtype=np.int64)
    ni = np.array([i for i in range(n) if i not in set(oi)], dtype=np.int64)
    check(audit['old_reference_physical_ids'] == [ids[i] for i in oi] and len(set(ids + held)) == n + h,
          'Old reference/physical ID binding mismatch')
    _array_close(data['old_indices'], oi, 'Physical old index mismatch'); _array_close(data['new_indices'], ni, 'Physical new index mismatch')
    q = np.zeros(n); q[oi] = 1 / len(oi); _array_close(data['q'], q, 'Frozen old reference measure mismatch')
    ob, oa, hb, ha = [data[k] for k in ('original_train_b', 'original_train_a', 'original_held_b', 'original_held_a')]
    ub, ua = adapted_blocks(_raw(ob, oa), data['U']); uhb, uha = adapted_blocks(_raw(hb, ha), data['U'])
    for key, expected in (('adapted_train_b', ub), ('adapted_train_a', ua), ('adapted_held_b', uhb), ('adapted_held_a', uha)):
        _array_close(data[key], expected, 'Independent actual adapter mapping mismatch: ' + key)
    d0, dh0 = _distances(ob, oa), _distances(hb, ha, ob, oa)
    tau, gamma, s0 = (_scalar(data, key) for key in ('tau', 'gamma', 's0'))
    d = d0 if tau == 0 or not np.any(data['U']) else .5 * (d0 + _distances(ub, ua))
    dh = dh0 if tau == 0 or not np.any(data['U']) else .5 * (dh0 + _distances(uhb, uha, ub, ua))
    for key, expected in (('original_distance', d0), ('original_cross_distance', dh0), ('distance', d), ('cross_distance', dh)):
        _array_close(data[key], expected, 'Independent raw distance mismatch: ' + key)
    old_d = d0[np.ix_(oi, oi)]; oy = y[oi]
    expected_s0 = float(np.sum(old_d[np.triu_indices(len(oi), 1)]) / len(oi))
    expected_tau = None if len(set(oy.tolist())) == 1 else float(np.median(np.min(np.where(oy[:, None] != oy[None, :], old_d, np.inf), axis=1)))
    close(s0, expected_s0, 'Original R0 trace binding mismatch'); close(tau, expected_tau, 'Original R0 bandwidth binding mismatch')
    expected_gamma = None if expected_tau is None or expected_s0 == 0 else expected_s0 / float(-2 * np.sum(
        _radial(old_d, tau, 1.0, True)[np.triu_indices(len(oi), 1)]) / len(oi))
    close(gamma, expected_gamma, 'Original R0 gamma binding mismatch')
    radial, cross = _radial(d, tau, gamma), _radial(dh, tau, gamma)
    _array_close(data['raw_train'], radial, 'Raw Gaussian train kernel mismatch'); _array_close(data['raw_cross'], cross, 'Raw Gaussian held kernel mismatch')
    Y = np.eye(c)[y] - 1 / c; _array_close(data['Y'], Y, 'Unchanged row-centered targets mismatch')
    M, Mh = data['M_train'], data['M_held']
    check(M.shape == (n, c) and Mh.shape == (h, c) and data['train_scores'].shape == (n, c)
          and data['scores'].shape == (h, c), 'Every declared physical row/class column is required')
    if prior is None:
        check(audit['mode'] == 'B', 'C missing frozen actual B')
        _array_close(M, np.zeros((n, c)), 'B train prior is nonzero'); _array_close(Mh, np.zeros((h, c)), 'B held prior is nonzero')
    else:
        pdata, pclasses = prior
        check(audit['mode'] == 'C_seq' and pclasses == old, 'Prior class/mode binding mismatch')
        _array_close(pdata['original_train_b'], ob[oi], 'Actual B old raw geometry mismatch')
        _array_close(pdata['original_train_a'], oa[oi], 'Actual B old raw auxiliary mismatch')
        for raw, expected, message in ((_raw(ob, oa), M, 'train'), (_raw(hb, ha), Mh, 'held')):
            values = np.zeros(expected.shape); values[:, [classes.index(v) for v in pclasses]] = head_score(pdata, raw)
            _array_close(expected, values, 'Frozen actual B ' + message + ' prior mismatch')
    target = Y - M; _array_close(data['residual_target'], target, 'Residual target was altered/recentered')
    if audit['mode'] == 'B':
        augmented = dict(data, E=target, raw_train_minus_one=_radial(d, tau, gamma, True),
                         raw_cross_minus_one=_radial(dh, tau, gamma, True), actual_trace=np.asarray(float(np.trace(data['K']))))
        affine_analysis.verify_head(ref, audit, lambda unused: augmented)
        check(np.array_equal(data['old_representative_indices'], oi) and np.array_equal(data['old_inverse_groups'], np.arange(len(oi))), 'B invented old compression')
    else:
        reps, groups = [], []
        for index in oi:
            group = next((j for j, rep in enumerate(reps) if np.array_equal(np.r_[ob[index], oa[index]], np.r_[ob[rep], oa[rep]])
                and (tau == 0 or (np.array_equal(ub[index], ub[rep]) and np.array_equal(ua[index], ua[rep])))), None)
            if group is None: group = len(reps); reps.append(int(index))
            else: check(np.array_equal(radial[index], radial[reps[group]]) and np.array_equal(cross[:, index], cross[:, reps[group]]), 'Exact old kernel group inconsistent')
            groups.append(group)
        reps = np.asarray(reps, dtype=np.int64); groups = np.asarray(groups, dtype=np.int64)
        check(np.array_equal(data['old_representative_indices'], reps) and np.array_equal(data['old_inverse_groups'], groups), 'Old exact lossless compression mismatch')
        if gamma is None:
            zeros = dict(alpha=np.zeros((len(ni), c)), beta=np.zeros((len(reps), c)), v=np.zeros(c),
                         K_perp=np.zeros((len(ni), len(ni))), L_perp=np.zeros((h, len(ni))), old_residual=np.zeros((len(reps), c)))
            for key, expected in zeros.items(): _array_close(data[key], expected, 'Zero kernel C invented residual: ' + key)
            _array_close(data['train_scores'], M, 'Zero kernel C train prior mismatch'); _array_close(data['scores'], Mh, 'Zero kernel C held prior mismatch')
            check(audit['status'] == 'ZERO_KERNEL_CONSTRAINT_FORCES_ZERO_RESIDUAL', 'Zero-kernel status mismatch')
        else:
            raw = gamma * radial; rc = gamma * cross
            blocks = dict(A=raw[np.ix_(reps, reps)], B=raw[np.ix_(reps, ni)], D=raw[np.ix_(ni, ni)], F=rc[:, reps], E=rc[:, ni],
                          M_O=M[reps], M_N=M[ni], M_H=Mh, R_N=target[ni])
            for key, expected in blocks.items(): _array_close(data[key], expected, 'Independent conditional raw/prior block mismatch: ' + key)
            for key in ('A', 'D'): _array_close(data['raw_' + key], blocks[key], 'Lossless raw block mismatch: ' + key)
            certificate = certify_positive_kernel_head(**blocks, alpha=data['alpha'], beta=data['beta'], v=data['v'],
                old_scores=data['old_scores'], train_new_scores=data['train_new_scores'], held_scores=data['held_scores'])
            check(certificate.audit['positive_kernel_checked'] is False, 'Residual certificate falsely claims independent PSD diagnosis')
            _factor(data['chol_A'], blocks['A'], 'Saved projection factor residual')
            _array_close(blocks['A'] @ data['J'], blocks['B'], 'Projection J solve residual')
            _array_close(blocks['A'] @ data['z'], np.ones(len(reps)), 'Projection constant solve residual')
            s = float(data['z'].sum()); close(float(data['s']), s, 'Projection constant s mismatch'); check(s > 0, 'Nonpositive projection s')
            cn, ch = np.ones(len(ni)) - blocks['B'].T @ data['z'], np.ones(h) - blocks['F'] @ data['z']
            for key, expected in (('c_N', cn), ('c_H', ch), ('projection_rhs', np.column_stack((blocks['B'], np.ones(len(reps)))))):
                _array_close(data[key], expected, 'Complete constant projection mismatch: ' + key)
            K = blocks['D'] - blocks['B'].T @ data['J'] + np.outer(cn, cn) / s
            L = blocks['E'] - blocks['F'] @ data['J'] + np.outer(ch, cn) / s
            _array_close(data['K_perp'], K, 'Complete conditional kernel/rank-one mismatch')
            _array_close(data['L_perp'], L, 'Complete conditional cross/rank-one mismatch')
            _factor(data['chol_ridge'], K + np.eye(len(ni)), 'Saved residual factor residual')
            all_scores = M + raw[:, ni] @ data['alpha'] - raw[:, reps] @ data['beta'] + data['v']
            _array_close(data['train_scores'], all_scores, 'Complete conditional physical train score mismatch')
            _array_close(data['scores'], certificate.arrays['held_scores'], 'Complete conditional held score mismatch')
            _array_close(data['old_residual'], certificate.arrays['old_residual'], 'Real old residual mismatch')
            close(audit['head_training_loss_ridge'], .5 * certificate.audit['residual_rkhs_norm_squared'], 'Conditional raw RKHS norm audit mismatch')
        _array_close(data['train_scores'][oi], M[oi], 'Old physical residual is nonzero in registered columns')
    _head_work(data, audit)
    close(audit['head_training_loss_data'], .5 * float(np.sum((data['train_scores'] - Y) ** 2)), 'Actual head SSE mismatch')
    close(audit['head_training_loss_total'], audit['head_training_loss_data'] + audit['head_training_loss_ridge'], 'Head loss decomposition mismatch')
    return data


def verify_preparation(prep, entry, labels, old, resolver):
    b = prep['state'] == 'B'; ids = entry['b_training_ids' if b else 'c_training_ids']; classes = entry['b_classes' if b else 'c_classes']
    k = entry['train_k']; folds = 0 if k == 1 else min(k, 3)
    check(prep['schema'] == SCHEMA and prep['method'] == METHOD and prep['training_physical_ids'] == ids
          and prep['classes'] == classes and prep['old_classes'] == old and prep['train_k'] == k
          and prep['train_physical_count'] == len(ids) and prep['ajlr_preparation_count'] == 1
          and prep['inherited_state'] is (not b) and prep['inherited_adapter_from'] == (None if b else 'B_CONDITIONAL')
          and prep['source_inputs'] is False and prep['query_fit'] is False, 'Preparation physical/current-B binding mismatch')
    affine_analysis.verify_latent_coordinates(prep, resolver)
    check(len(prep['inner_folds']) == folds and len(prep['prior_folds']) == (0 if b else folds), 'Inner/prior fold coverage mismatch')
    assignment = {pid: j % folds for cls in classes for j, pid in enumerate(sorted(pid for pid in ids if labels[pid] == cls))} if folds else {}
    prior_arrays = []; seen = set(); original_pairs = 0
    for j, fold in enumerate(prep['inner_folds']):
        held = [pid for pid in ids if assignment[pid] == j]; train = [pid for pid in ids if assignment[pid] != j]
        check(fold['inner_fold'] == j and fold['training_physical_ids'] == train and fold['held_physical_ids'] == held
              and fold['classes'] == classes and fold['old_classes'] == old
              and fold['old_reference_physical_ids'] == [pid for pid in train if labels[pid] in old]
              and fold['all_head_statistics_from_old_inner_train_only'] is True
              and not set(train + held).intersection(entry['c_ids']), 'Inner/outer physical isolation mismatch')
        check(not seen.intersection(held), 'Duplicate inner held physical ID'); seen.update(held)
        n, h = len(train), len(held); original_pairs += n * (n - 1) // 2 + h * n
        if b: check(fold['prior_source'] == 'ZERO' and fold['prior_ref'] is None, 'B invented prior')
        else:
            prior = prep['prior_folds'][j]
            check(prior['training_physical_ids'] == [pid for pid in train if labels[pid] in old]
                  and prior['held_physical_ids'] == [pid for pid in held if labels[pid] in old]
                  and prior['classes'] == old and fold['prior_source'] == 'FROZEN_CURRENT_ACTUAL_B'
                  and fold['prior_ref'] == prior['head_state_ref'], 'Inner actual-B prior physical binding mismatch')
            pdata = verify_head(prior['head_state_ref'], prior['final_fit'], resolver)
            check(np.array_equal(pdata['U'], resolver(prep['prepared_state_ref'])['anchor_U']), 'Inner prior mapping is not frozen actual B U')
            prior_arrays.append(pdata)
            pn, ph = len(prior['training_physical_ids']), len(prior['held_physical_ids'])
            original_pairs += pn * (pn - 1) // 2 + ph * pn
    check(not folds or seen == set(ids), 'Incomplete physical inner-held coverage')
    original_pairs += len(ids) * (len(ids) - 1) // 2
    check(prep['prepared_distance_evaluation_count'] == (2 * folds + 1 if b else 4 * folds + 1)
          and prep['original_distance_pair_count'] == original_pairs, 'Actual original preparation distance work mismatch')
    factors = sum(int(_scalar(value, 'gamma') is not None) for value in prior_arrays)
    check(prep['prior_head_fit_count'] == len(prior_arrays) and prep['prior_factorization_count'] == factors
          and prep['prior_triangular_solve_count'] == 2 * factors
          and prep['prior_score_evaluation_count'] == (0 if b else 2 * folds + 1)
          and prep['prior_score_physical_count'] == (0 if b else (folds + 1) * len(ids)), 'Actual prior preparation work mismatch')
    for suffix, power in (('rhs_count', 0), ('rhs_element_count', 1), ('dense_work_unit_count', 2)):
        expected = sum(2 * len(value['alpha']) ** power * (len(old) + 1) for value in prior_arrays if _scalar(value, 'gamma') is not None)
        check(prep['prior_triangular_' + suffix] == expected, 'Prior C+1 RHS work mismatch: ' + suffix)
    reason = 'PHYSICAL_K1' if k == 1 else 'ZERO_DICTIONARY_RANK' if prep['latent_rank'] == 0 else \
        'NO_OLD_KERNEL_INFORMATION' if prep['final_problem']['trace_scale'] is None else \
        'ALL_INNER_GEOMETRY_DEGENERATE' if not any(f['trace_scale'] is not None and f['bandwidth_tau'] is not None and f['bandwidth_tau'] > 0 for f in prep['inner_folds']) else None
    check(prep['no_information_reason'] == reason and prep['no_information'] is (reason is not None), 'No-information/no-update reason mismatch')


def _kernel_vjp(data, G, fold, enabled, analysis_work=None):
    """Independent affine/KKT adjoints followed by full raw/distance/adapter VJP."""
    n, c = data['Y'].shape; h = len(G); tau, gamma = _scalar(data, 'tau'), _scalar(data, 'gamma')
    active = enabled and gamma is not None and tau is not None and tau > 0 and h > 0
    is_b = fold['mode'] == 'B'; m, p = len(data['old_representative_indices']), len(data['new_indices'])
    expected = _expected_system('projection_adjoint', m, c, int(active and not is_b))
    expected.update(_expected_system('residual_adjoint', n if is_b else p, c, int(active)))
    for prefix in ('projection_adjoint', 'residual_adjoint'): expected[prefix + '_factorization_count'] = 0
    for suffix in SUFFIXES[1:]: expected['derivative_' + suffix] = expected['projection_adjoint_' + suffix] + expected['residual_adjoint_' + suffix]
    expected['ce_adjoint_solve_count'] = int(active)
    for key, value in expected.items(): check(fold[key] == value, 'Complete adjoint real work mismatch: ' + key)
    if not active: return np.zeros((736, 8))
    if analysis_work is not None:
        order = n + 1 if is_b else m + p + 1
        for key, amount in (('independent_general_solve_count', 1), ('independent_general_rhs_column_count', c),
            ('independent_general_rhs_element_count', order * c), ('independent_dense_cubic_work_unit_count', order ** 3),
            ('independent_dense_rhs_work_unit_count', order * order * c)):
            analysis_work[key] += amount
    _array_close(data['adjoint_G'], G, 'Actual held CE upstream mismatch')
    _array_close(data['adjoint_g_b'], G.sum(0), 'Nonzero constant adjoint omitted')
    if is_b:
        matrix = np.block([[data['K'] + np.eye(n), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
        rhs = np.vstack((data['L'].T @ G, G.sum(0)[None, :]))
        solution = np.linalg.solve(matrix, rhs); T, eta = solution[:n], solution[-1]
        for key, value in (('T', T), ('eta', eta), ('rhs', rhs[:n])): _array_close(data['adjoint_' + key], value, 'Independent affine companion mismatch: ' + key)
        barL = G @ data['alpha'].T; barK = -.5 * (T @ data['alpha'].T + data['alpha'] @ T.T)
        barR, barQ = center_kernel_vjp(barK, barL, data['q'], gamma)
    else:
        inputs = {key: data[key] for key in ('A', 'B', 'D', 'F', 'E', 'M_O', 'M_N', 'M_H', 'R_N')}
        certificate = certify_positive_kernel_head(**inputs, **{key: data[key] for key in ('alpha', 'beta', 'v', 'old_scores', 'train_new_scores', 'held_scores')})
        independent = positive_kernel_head_vjp(certificate, G)
        for key in ('A', 'B', 'D', 'F', 'E'): _array_close(data['adjoint_' + key], independent.arrays[key], 'Independent full KKT kernel VJP mismatch: ' + key)
        T, lam = data['adjoint_T'], data['adjoint_Lambda']
        _array_close((data['K_perp'] + np.eye(p)) @ T, data['L_perp'].T @ G, 'Residual adjoint equation mismatch')
        _array_close(data['A'] @ lam[:m] + lam[-1], data['F'].T @ G, 'Projection adjoint first equation mismatch')
        _array_close(lam[:m].sum(0), G.sum(0), 'Projection adjoint constant equation mismatch')
        _array_close(data['adjoint_projection_adjoint_rhs'], data['F'].T @ G, 'Projection adjoint RHS mismatch')
        _array_close(data['adjoint_X_alpha'], np.vstack((data['beta'], -data['v'])), 'Complete X_alpha mismatch')
        xt = data['adjoint_X_T']
        _array_close(data['A'] @ xt[:m] + xt[-1], data['B'] @ T, 'X_T first saddle equation mismatch')
        _array_close(xt[:m].sum(0), T.sum(0), 'X_T constant saddle equation mismatch')
        oi, ni = data['old_representative_indices'], data['new_indices']; v = independent.arrays
        barR, barQ = np.zeros((n, n)), np.zeros((h, n))
        barR[np.ix_(oi, oi)] = v['A']; barR[np.ix_(ni, ni)] = v['D']
        barR[np.ix_(oi, ni)] = .5 * v['B']; barR[np.ix_(ni, oi)] = .5 * v['B'].T
        barQ[:, oi] = v['F']; barQ[:, ni] = v['E']; barR *= gamma; barQ *= gamma
    dd, dc = -.5 * barR * data['raw_train'] / tau, -.5 * barQ * data['raw_cross'] / tau
    for key, expected_array in (('raw_train_upstream', barR), ('raw_cross_upstream', barQ),
                                ('adapted_distance_upstream', dd), ('adapted_cross_distance_upstream', dc)):
        _array_close(data['adjoint_' + key], expected_array, 'Full raw/distance upstream mismatch: ' + key)
    b, a, hb, ha = [data[key] for key in ('adapted_train_b', 'adapted_train_a', 'adapted_held_b', 'adapted_held_a')]
    gb, ga, rb, ra = affine_analysis._distance_reverse(b, a, b, a, dd)
    hgb, hga, tgb, tga = affine_analysis._distance_reverse(hb, ha, b, a, dc)
    gU = affine_analysis._adapter_reverse(data['original_train_b'], data['original_train_a'], data['U'], np.column_stack((gb + rb + tgb, ga + ra + tga)))
    gU += affine_analysis._adapter_reverse(data['original_held_b'], data['original_held_a'], data['U'], np.column_stack((hgb, hga)))
    _array_close(data['adjoint_g_U'], gU, 'Independent complete adapter gradient mismatch', scale=np.linalg.norm(gU))
    return gU


def verify_objective(value, Z, U, prep, resolver, aggregate_ref):
    folds = value['inner_folds']; c = len(prep['classes']); sums = np.zeros(c); counts = np.zeros(c, dtype=np.int64); heads = []
    check(len(folds) == len(prep['inner_folds']) and value['temperature'] == 1., 'Objective inner fold/temperature mismatch')
    aggregate = resolver(aggregate_ref); gradient = 'g_Z' in aggregate
    for j, (fold, physical) in enumerate(zip(folds, prep['inner_folds'])):
        check(all(fold[key] == physical[key] for key in ('classes', 'training_physical_ids', 'held_physical_ids', 'old_reference_physical_ids', 'bandwidth_tau', 'trace_scale', 'prior_ref')),
              'Objective physical/current-B binding mismatch')
        prior = None if prep['state'] == 'B' else (resolver(prep['prior_folds'][j]['head_state_ref']), prep['old_classes'])
        data = verify_head(fold['head_state_ref'], fold, resolver, prior); heads.append(data)
        check(np.array_equal(data['U'], U), 'Objective head mapping changed')
        f, y = data['scores'], data['held_labels']; top = f.max(1)
        ce = np.log(np.exp(f - top[:, None]).sum(1)) + top - f[np.arange(len(y)), y]
        fs, fc = np.bincount(y, weights=ce, minlength=c), np.bincount(y, minlength=c)
        _array_close(fold['held_ce_sums'], fs, 'Physical fold CE sum mismatch')
        check(fold['held_ce_counts'] == fc.tolist() and fold['held_correct_count'] == int(np.sum(np.argmax(f, axis=1) == y)), 'Fold CE counts/accuracy mismatch')
        sums += fs; counts += fc
    check(np.all(counts == prep['train_k']), 'CE did not pool every physical inner-held row exactly once')
    means = sums / counts; risk = float(np.linalg.norm(means)) / math.sqrt(c)
    for key, expected in (('class_ce_sums', sums), ('class_ce_counts', counts), ('class_ce_means', means)):
        _array_close(value[key], expected, 'Pooled RMS class CE mismatch: ' + key)
        _array_close(aggregate[key], expected, 'Aggregate class CE coordinate mismatch: ' + key)
    for key in ('loss_ce', 'loss_task', 'RMSCE', 'loss_total'): close(value[key], risk, 'CE-only objective mismatch: ' + key)
    close(value['loss_proximal'], 0., 'CE-only objective added proximal loss')
    _array_close(aggregate['prox'], np.asarray(0.), 'CE-only aggregate added proximal loss')
    _array_close(aggregate['RMSCE'], np.asarray(risk), 'Aggregate RMSCE mismatch')
    check(value['coordinate_ball_radius'] == .5 and value['coordinate_ball_feasible'] is True and np.linalg.norm(Z) <= .5 + 128 * np.finfo(float).eps,
          'Coordinate ball contract mismatch')
    reused = value['forward_cache_reused']; check(type(reused) is bool and value['backward_evaluation_count'] == int(gradient), 'Cached objective/gradient flag mismatch')
    for key in KERNEL_COUNTERS + tuple('head_' + s for s in SUFFIXES[1:]) + ('intercept_fit_count', 'intercept_addition_count'):
        expected = sum(f[key] for f in folds) if 'adjoint' in key else (0 if reused else sum(f[key] for f in folds))
        check(value[key] == expected, 'Objective actual work/cached forward mismatch: ' + key)
    for key in ('ce_adjoint_solve_count',) + tuple('derivative_' + s for s in SUFFIXES[1:]):
        check(value[key] == sum(f[key] for f in folds), 'Objective complete adjoint count mismatch: ' + key)
    check(value['inner_head_fit_count'] == (0 if reused else len(folds))
          and value['inner_factorization_count'] == (0 if reused else sum(f['factorization_count'] for f in folds))
          and value['inner_objective_evaluation_count'] == int(not reused), 'Objective actual head/factor/cache mismatch')
    if gradient:
        gU = np.zeros((736, 8))
        for data, fold in zip(heads, folds):
            f, y = data['scores'], data['held_labels']; ex = np.exp(f - f.max(1)[:, None]); G = ex / ex.sum(1)[:, None]
            G[np.arange(len(y)), y] -= 1
            if risk: G *= (means[y] / (c * risk * counts[y]))[:, None]
            else: G.fill(0.)
            gU += _kernel_vjp(data, G, fold, risk > 0, getattr(resolver, 'analysis_work', None))
        W = resolver(prep['prepared_state_ref'])['W']
        _array_close(aggregate['g_Z'], gU @ W, 'Independent CE-only coordinate gradient mismatch', scale=np.linalg.norm(gU) * np.linalg.norm(W))
    else:
        for data, fold in zip(heads, folds): _kernel_vjp(data, np.zeros_like(data['scores']), fold, False)
    return risk


def _dictionary(raw):
    from scipy.special import erf
    full = np.column_stack(tuple(raw[key] for key in BRANCHES)); unit = np.zeros_like(full)
    for sl in (slice(0, 160), slice(160, 256), slice(256, 416), slice(416, 576), slice(576, 736)):
        for i, row in enumerate(full):
            norm = float(np.linalg.norm(row[sl]))
            if norm: unit[i, sl] = row[sl] / norm
    z = np.asarray([np.sum(dct_initial() * (row / math.sqrt(5))[None, :], axis=1) for row in unit])
    return .5 * z * (1 + erf(z / math.sqrt(2)))


def verify_candidate(stage, prep, entry, b_stage, resolver):
    coordinates = resolver(prep['prepared_state_ref']); W, H = coordinates['W'], coordinates['H']; rank = prep['latent_rank']
    anchor = np.zeros((736, 8)) if stage['mode'] == 'B' else resolver(b_stage['final_state_ref'])['U']
    check(stage['status'] == 'COMPLETED' and stage['schema'] == SCHEMA and stage['method'] == METHOD and stage['config'] == PROBE_CONFIG
          and stage['preparation']['prepared_state_ref'] == prep['prepared_state_ref']
          and stage['preparation']['inner_folds'] == prep['inner_folds'] and stage['preparation']['prior_folds'] == prep['prior_folds']
          and stage['no_information'] is prep['no_information'], 'Candidate preparation/schema/frozen structure mismatch')
    for key in ('run_id', 'row_id', 'split_id', 'scope', 'fold', 'parent_k', 'train_k'):
        check(stage['preparation'][key] == prep[key], 'Candidate current-B lineage mismatch: ' + key)
    def state(ref):
        data = resolver(ref); Z, U = data['Z'], data['U']
        check(Z.shape == (736, rank) and U.shape == (736, 8) and np.array_equal(data['W'], W)
              and np.array_equal(data['anchor_U'], anchor), 'Coordinate/actual-B anchor binding mismatch')
        affine_analysis.verify_coordinate_map(Z, U, anchor, W)
        check(np.linalg.norm(Z) <= .5 + 128 * np.finfo(float).eps, 'Coordinate ball exceeded')
        return data, Z, U
    initial, Z, U = state(stage['initialization_state_ref'])
    check(np.array_equal(Z, np.zeros((736, rank))) and np.array_equal(U, anchor), 'Exact B-to-C initialization mismatch')
    steps, gradients, trials = stage['steps'], stage['gradients'], stage['trials']; objectives = []
    check(stage['optimizer_steps'] == stage['accepted_trial_count'] == len(steps) <= 4
          and stage['optimizer_iterations'] == stage['backward_evaluation_count'] == len(gradients) <= 4
          and stage['trial_count'] == stage['trial_attempt_count'] == len(trials) <= 48
          and stage['rejected_trial_count'] == len(trials) - len(steps), 'Fixed 4x12 actual solver counts mismatch')
    if prep['no_information']:
        check(not steps and not gradients and not trials and stage.get('initial_objective') is None and stage.get('final_objective') is None
              and stage['stop_reason'] == prep['no_information_reason'], 'Invented no-information adapter training')
        current = None
    else:
        current = verify_objective(stage['initial_objective'], Z, U, prep, resolver, stage['initialization_state_ref'])
        objectives.append(stage['initial_objective'])
    used = 0; accepted = []; path_length = 0.
    for iteration, event in enumerate(gradients, 1):
        data, gz, gu = state(event['state_ref']); g, direction = data['g_Z'], data['d_Z']; norm = float(np.linalg.norm(g))
        check(event['iteration'] == iteration and np.array_equal(gz, Z) and np.array_equal(gu, U), 'Gradient cache changed current coordinates')
        _array_close(direction, -g / norm if norm else np.zeros_like(g), 'Normalized negative CE gradient mismatch')
        close(event['gradient_norm'], norm, 'Actual gradient norm mismatch'); close(event['direction_norm'], float(np.linalg.norm(direction)), 'Direction norm mismatch')
        loss = verify_objective(event['objective'], Z, U, prep, resolver, event['state_ref']); close(loss, current, 'Accepted gradient cache changed RMSCE')
        check(event['objective']['forward_cache_reused'] is True, 'Gradient refitted accepted heads'); objectives.append(event['objective'])
        group = [trial for trial in trials if trial['iteration'] == iteration]; check(len(group) <= 12 and (norm > 0 or not group), 'Trial after zero gradient/budget exceeded')
        winner = None
        for j, trial in enumerate(group, 1):
            check(winner is None and trial['trial'] == j and trial['step_size'] == .125 * .5 ** (j - 1)
                  and trial['gradient_state_ref'] == event['state_ref'], 'Fixed first-acceptable trial ordering mismatch')
            td, tz, tu = state(trial['state_ref']); proposal = Z + trial['step_size'] * direction; pn = float(np.linalg.norm(proposal))
            expected = proposal if pn <= .5 else proposal * (.5 / pn); delta = tz - Z
            check(np.array_equal(tz, expected) and np.array_equal(td['delta_Z'], delta) and np.any(delta), 'Projected actual delta mismatch')
            after = verify_objective(trial['objective'], tz, tu, prep, resolver, trial['state_ref']); objectives.append(trial['objective'])
            dot = float(np.sum(g * delta)); bound = current + 1e-4 * dot
            tol = 128 * np.finfo(float).eps * max(1., abs(current), abs(after), abs(bound))
            armijo, nonincrease = bool(after <= bound + tol), bool(after <= current + tol)
            check(trial['armijo_pass'] is armijo and trial['objective_nonincrease_pass'] is nonincrease
                  and trial['accepted'] is (armijo and nonincrease), 'Actual-delta CE-only Armijo mismatch')
            for key, expected_scalar in (('comparison_tolerance', tol), ('gradient_dot_delta', dot), ('update_norm', float(np.linalg.norm(delta))), ('loss_before', current), ('loss_after', after)):
                close(trial[key], expected_scalar, 'Trial scalar mismatch: ' + key)
            used += 1
            if trial['accepted']: winner = trial
        if winner is not None:
            step = steps[len(accepted)]
            check(step['iteration'] == iteration and step['trial'] == winner['trial'] and step['state_ref'] == winner['state_ref']
                  and step['objective'] == winner['objective'], 'Accepted step/trial binding mismatch')
            accepted.append(step); path_length += winner['update_norm']; _, Z, U = state(winner['state_ref']); current = winner['loss_after']
        else: check(iteration == len(gradients), 'Continued after failed/zero direction')
    final, fz, fu = state(stage['final_state_ref'])
    check(used == len(trials) and accepted == steps and np.array_equal(fz, Z) and np.array_equal(fu, U), 'Final state is not last accepted state')
    check(path_length <= .5 + 128 * np.finfo(float).eps * max(1, Z.size), 'Four-update path-length budget exceeded')
    if not prep['no_information']:
        final_loss = verify_objective(stage['final_objective'], Z, U, prep, resolver, stage['final_objective_state_ref'])
        close(final_loss, current, 'Final objective selected a different training step')
        check(stage['final_objective']['forward_cache_reused'] is True and stage['final_objective']['backward_evaluation_count'] == 0, 'Final cache was charged as new work')
        reason = stage['stop_reason']; check(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'TRIAL_BUDGET_EXHAUSTED', 'ZERO_FEASIBLE_DISPLACEMENT'), 'Unknown bounded stop reason')
        if reason == 'MAX_ITERATIONS': check(len(steps) == len(gradients) == 4, 'Premature max-iteration stop')
        if reason == 'ZERO_GRADIENT': check(gradients[-1]['gradient_norm'] == 0, 'False zero-gradient stop')
        if reason == 'TRIAL_BUDGET_EXHAUSTED': check(len(group) == 12 and not group[-1]['accepted'], 'False trial-budget exhaustion')
        if reason == 'ZERO_FEASIBLE_DISPLACEMENT':
            proposed = Z + .125 * .5 ** len(group) * direction; length = np.linalg.norm(proposed)
            projected = proposed if length <= .5 else proposed * (.5 / length)
            check(np.array_equal(projected, Z), 'False zero feasible displacement')
    prior = None
    if stage['mode'] == 'C_seq':
        bdata = resolver(b_stage['final_state_ref'])
        for key, value in bdata.items(): check(np.array_equal(final['prior_B_' + key], value), 'Final C did not inherit exact actual B state: ' + key)
        check(np.array_equal(final['prior_old_class_indices'], [prep['classes'].index(value) for value in prep['old_classes']]), 'Actual B class-column map changed')
        prior = (bdata, prep['old_classes'])
    verify_head(stage['final_state_ref'], stage['final_fit'], resolver, prior)
    raw = {key: final[key] for key in BRANCHES}
    _array_close(H, _dictionary(raw), 'Prepared dictionary not bound to actual physical support')
    _array_close(final['V0'], dct_initial(), 'Fixed DCT dictionary changed')
    check(np.array_equal(final['singular_values'], coordinates['singular_values']), 'Final coordinate spectrum changed')
    if prior is not None:
        positions = [i for i, pid in enumerate(prep['training_physical_ids']) if pid in b_stage['training_physical_ids']]
        for key in BRANCHES: check(np.array_equal(final[key][positions], bdata[key]), 'Inherited old physical raw input changed: ' + key)
    close(stage['coordinate_norm'], float(np.linalg.norm(Z)), 'Final coordinate norm mismatch')
    check(stage['declared_coordinate_scalar_count'] == 736 * rank
          and stage['trainable_parameter_count'] == stage['optimizer_parameter_count'] == (0 if prep['no_information'] else 736 * rank)
          and stage['active_parameter_count'] == stage['trained_parameter_count'] == (736 * rank if steps else 0), 'Real active/trainable parameter accounting mismatch')
    analytic = sum(final[key].size for key in ('alpha', 'beta', 'v', 'intercept') if key in final)
    check(stage['analytic_head_scalar_count'] == analytic and stage['final_head_fit_count'] == 1
          and stage['final_factorization_count'] == stage['final_fit']['factorization_count'], 'Final analytic head/resource accounting mismatch')
    for key in KERNEL_COUNTERS + tuple('head_' + s for s in SUFFIXES[1:]) + ('intercept_fit_count', 'intercept_addition_count', 'ajlr_forward_evaluation_count'):
        check(stage[key] == sum(obj[key] for obj in objectives) + stage['final_fit'][key], 'Full rejected-trial/final resource accounting mismatch: ' + key)
    for key in ('inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count', 'ce_adjoint_solve_count') + tuple('derivative_' + s for s in SUFFIXES[1:]):
        check(stage[key] == sum(obj[key] for obj in objectives), 'Full objective resource accounting mismatch: ' + key)
    return final


def verify_score_workload(value, final, n):
    check(isinstance(value, dict) and value['score_physical_count'] == n and value['score_seconds'] >= 0, 'Missing actual outer inference workload')
    for component, data in (('residual', final), ('prior', {key[len('prior_B_'):]: item for key, item in final.items() if key.startswith('prior_B_')})):
        work = value[component]
        if not data:
            check(work == {}, 'B invented prior inference workload'); continue
        tau, gamma = _scalar(data, 'tau'), _scalar(data, 'gamma'); width, c = len(data['original_train_b']), data['alpha'].shape[1]
        calls = 0 if gamma is None else n * (1 + int(tau != 0 and bool(np.any(data['U']))))
        expected = dict(raw_distance_evaluation_count=calls, raw_distance_pair_count=calls * width,
            reference_distance_evaluation_count=calls, reference_distance_pair_count=calls * len(data['old_indices']),
            kernel_evaluation_count=n * int(gamma is not None), kernel_pair_count=n * width * int(gamma is not None),
            adapter_physical_evaluation_count=n * int(gamma is not None and tau != 0),
            dictionary_physical_evaluation_count=n * int(gamma is not None and tau != 0), intercept_addition_count=n * c)
        for key, expected_value in expected.items(): check(work.get(key, 0) == expected_value, 'Actual single-record score work mismatch: ' + component + '/' + key)
        check(work['score_seconds'] >= 0, 'Negative measured inference duration')
    for key in set(value['residual']) | set(value['prior']):
        if key != 'score_seconds': check(value[key] == value['residual'].get(key, 0) + value['prior'].get(key, 0), 'Outer prior/residual work total mismatch: ' + key)


def _verify_baseline(stage, entry, labels, registry, resolver):
    data = resolver(stage['final_state_ref']); ids = stage['training_physical_ids']; n, c = len(ids), len(registry)
    check(data['alpha'].shape == (n, c) and stage['optimizer_steps'] == 0
          and stage['all_states_estimated_from_trainfold_only'] is True and stage['source_validation'] is None, 'R0 physical/state boundary mismatch')
    y = np.asarray([registry.index(labels[pid]) for pid in ids]); d = _distances(data['original_train_b'], data['original_train_a'])
    s0 = float(np.sum(d[np.triu_indices(n, 1)]) / n)
    tau = None if c == 1 else float(np.median(np.min(np.where(y[:, None] != y[None, :], d, np.inf), axis=1)))
    close(_scalar(data, 'tau'), tau, 'R0 original bandwidth mismatch'); close(stage['interaction_centered_trace'], s0, 'R0 original trace mismatch')
    fact = int(c > 1 and s0 > 0); gamma = _scalar(data, 'gamma')
    if fact:
        rm1 = _radial(d, tau, 1.0, True); expected_gamma = s0 / float(-2 * np.sum(rm1[np.triu_indices(n, 1)]) / n)
        close(gamma, expected_gamma, 'R0 gamma mismatch'); K, _ = affine_analysis._center(rm1, rm1, np.full(n, 1 / n), gamma)
        # R0 has no free intercept.  Authenticate its actual reference-difference
        # deployment terms from its complete training geometry; no affine gauge
        # equivalence can justify changing the baseline centering measure.
        reference, reference_self = rm1[0], float(rm1[0, 0])
        diff = (rm1 - rm1[:, :1]) - reference[None, :] + reference_self
        mean = np.mean(diff, axis=0); grand = float(np.mean(mean))
        _array_close(data['reference_kernel'], reference, 'R0 reference kernel mismatch')
        close(_scalar(data, 'reference_self'), reference_self, 'R0 reference self mismatch')
        _array_close(data['center_mean'], mean, 'R0 complete-train center mean mismatch')
        close(_scalar(data, 'center_grand'), grand, 'R0 complete-train center grand mismatch')
        _array_close((K + np.eye(n)) @ data['alpha'], np.eye(c)[y] - 1 / c, 'Independent R0 ridge equation mismatch')
    else:
        check(gamma is None, 'Zero R0 invented scale'); _array_close(data['alpha'], np.zeros((n, c)), 'Zero R0 invented coefficients')
        _array_close(data['reference_kernel'], np.zeros(n), 'Zero R0 reference kernel mismatch')
        _array_close(data['center_mean'], np.zeros(n), 'Zero R0 center mean mismatch')
        check(_scalar(data, 'reference_self') == _scalar(data, 'center_grand') == 0., 'Zero R0 centering scalar mismatch')
    check(stage['factorization_calls'] == fact and stage['effective_degrees_of_freedom_extra_triangular_solves'] == 2 * fact,
          'R0 actual factor/EDF work mismatch')
    return fact


def _check_events(entry, by_prep, by_stage, binding):
    seen = {name: dict(initial=0, final=0, gradients=0, trials=0, steps=0) for name in by_stage}; prepared = set()
    for event in entry['training_events']:
        check(event['schema'] == SCHEMA and event['method'] == METHOD and event['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION'
              and event['source_validation'] is None and event['scope'] == entry['scope'] and event['fold'] == entry['fold']
              and event['outer_trial'] == entry['trial'] and all(event.get(key) == value for key, value in binding.items()), 'Training event source/path binding mismatch')
        name, kind = event['state'], event['event']
        if kind == 'CONDITIONAL_JOINT_PREPARED':
            pn = name.removesuffix('_prepare'); check(pn in by_prep and pn not in prepared, 'Duplicate/unknown preparation event')
            check(event['prepared_state_ref'] == by_prep[pn]['prepared_state_ref'] and event['prior_folds'] == by_prep[pn]['prior_folds'], 'Preparation event source/ref mismatch')
            prepared.add(pn); continue
        check(name in by_stage, 'Unknown training stage'); stage = by_stage[name]; counts = seen[name]
        mapping = dict(CONDITIONAL_JOINT_GRADIENT='gradients', CONDITIONAL_JOINT_TRIAL='trials', CONDITIONAL_JOINT_STEP='steps')
        if kind in mapping:
            key = mapping[kind]; index = counts[key]; check(index < len(stage[key]), 'Extra training event')
            check(all(event.get(field) == value for field, value in stage[key][index].items()), 'Full event/solver trace mismatch'); counts[key] += 1
        elif kind == 'CONDITIONAL_JOINT_INITIAL':
            check(counts['initial'] == 0 and event['state_ref'] == stage['initialization_state_ref']
                  and event.get('objective') == stage.get('initial_objective'), 'Initial event/cache mismatch'); counts['initial'] += 1
        elif kind == 'CONDITIONAL_JOINT_FINAL':
            check(counts['final'] == 0 and event['mode'] == stage['mode'] and event['state_ref'] == stage['final_state_ref'], 'FINAL real envelope mismatch')
            actual = event['audit']
            check(actual['final_state_ref'] == stage['final_state_ref'] and actual['preparation'] == stage['preparation']
                  and actual.get('final_objective') == stage.get('final_objective') and actual['stop_reason'] == stage['stop_reason'], 'Final audit/state mismatch')
            for key in STAGE_COUNTERS: check(actual[key] == stage[key], 'Final real work mismatch: ' + key)
            counts['final'] += 1
        else: raise ValueError('Unknown Conditional training event')
    check(prepared == set(by_prep) and all(v['initial'] == v['final'] == 1 and all(v[key] == len(by_stage[name][key])
          for key in ('gradients', 'trials', 'steps')) for name, v in seen.items()), 'Incomplete full solver event coverage')


def verify_record(record, split, old, resolver, expected_binding=None):
    finite_tree(record); binding = record['inheritance_binding']; labels = {pid: split['registered_classes'][int(y)] for pid, y in zip(split['support_ids'], split['support_labels'])}
    classes, old, k = sorted(split['registered_classes']), sorted(old), split['k']
    check(record['schema'] == SCHEMA and record['method'] == METHOD and record['scope'] == SCOPE
          and record['query_rows_used'] == record['source_rows_used'] == 0 and binding.get('split_id') == split['split_id'], 'Parent method/access/split mismatch')
    if expected_binding is not None: check(all(binding.get(key) == value for key, value in expected_binding.items()), 'Parent escaped current run/row')
    check(all(record[key] == value for key, value in split_identity(split, old).items()), 'Parent declared identity mismatch')
    groups = {cls: sorted(pid for pid in labels if labels[pid] == cls) for cls in classes}
    check(len(labels) == len(split['support_ids']) and all(len(values) == k for values in groups.values())
          and record['classes'] == classes and record['old_classes'] == old and record['support_count'] == len(labels), 'Physical K/classes/IDs mismatch')
    expected = []
    if k == 1:
        check(record['fold_count'] == 0 and record['folds'] == [] and record['physical_fold_assignment'] == []
              and record['oof'] is record['oneshot_proxy'] is None and record['full_support'] is not None
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'True K1 independent held metric invented')
        expected.append((record['full_support'], set(labels), set(), 'support_full_k1', None))
    else:
        nfolds = min(k, 3); assignments = {pid: j % nfolds for values in groups.values() for j, pid in enumerate(values)}
        check(record['fold_count'] == len(record['folds']) == nfolds and record['full_support'] is None
              and {item['physical_id']: (item['class_id'], item['fold']) for item in record['physical_fold_assignment']}
              == {pid: (labels[pid], assignments[pid]) for pid in labels}, 'OOF physical assignment mismatch')
        for j, entry in enumerate(record['folds']):
            held = {pid for pid in labels if assignments[pid] == j}; expected.append((entry, set(labels) - held, held, 'support_oof', j))
        proxy = record['oneshot_proxy']; check(proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
            and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent', 'Complete proxy anchor coverage mismatch')
        for j, entry in enumerate(proxy['trials']):
            train = {values[j] for values in groups.values()}; expected.append((entry, train, set(labels) - train, 'support_oneshot_proxy', j))
    counts = dict.fromkeys(COUNTERS[4:], 0); logs, events, training = [], [], []; peak = 0
    for entry, train, held, scope, index in expected:
        coords = dict(binding, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        check(entry['scope'] == scope and entry['parent_k'] == k and entry['train_k'] == len(train) // len(classes)
              and entry['held_k'] == len(held) // len(classes) and entry['fold'] == (index if scope == 'support_oof' else None)
              and entry['trial'] == (index if scope == 'support_oneshot_proxy' else None), 'Outer physical path coordinates mismatch')
        btrain, bheld = {pid for pid in train if labels[pid] in old}, {pid for pid in held if labels[pid] in old}
        for key, ids in (('b_training_ids', btrain), ('c_training_ids', train), ('b_ids', bheld), ('c_ids', held)):
            check(entry[key] == sorted(ids), 'Actual sorted physical path mismatch: ' + key)
        check(entry['held_labels'] == {pid: labels[pid] for pid in held} and entry['b_classes'] == old and entry['c_classes'] == classes
              and set(entry['paths']) == set(PATHS), 'Paired physical class/held mismatch')
        reuse = classes == old
        check(entry['c_reuses_b0'] is entry['c_reuses_b_candidates'] is reuse
              and [s['state'] for s in entry['stages']] == (['B0'] if reuse else ['B0', 'C0'])
              and [p['state'] for p in entry['preparations']] == (['B'] if reuse else ['B', 'C'])
              and [s['state'] for s in entry['candidate_stages']] == (['B_CONDITIONAL'] if reuse else ['B_CONDITIONAL', 'C_CONDITIONAL_seq']), 'new0 exact-reuse stage structure mismatch')
        def refs(value):
            if isinstance(value, dict):
                if {'path', 'arrays', 'array_summaries'} <= set(value):
                    namespace = json.loads(value['namespace']); check(all(namespace.get(key) == item for key, item in coords.items()), 'Numeric archive escaped physical path')
                    check(namespace['state'] in ('OUTER_SUPPORT_HELD', 'B0', 'C0', 'B_prepare', 'C_prepare', 'B_CONDITIONAL', 'C_CONDITIONAL_seq'), 'Unknown archive state namespace')
                else:
                    for item in value.values(): refs(item)
            elif isinstance(value, list):
                for item in value: refs(item)
        refs(entry)
        for item in entry['stages'] + entry['preparations'] + entry['candidate_stages']:
            check(all(item[key] == value for key, value in coords.items()), 'Stage current-run/path binding mismatch')
        for item, keys in [(p, PREPARATION_COUNTERS) for p in entry['preparations']] + [(s, STAGE_COUNTERS) for s in entry['candidate_stages']]:
            check(all(type(item[key]) is int and item[key] >= 0 for key in keys), 'Invalid actual core counter type/value')
        raw = resolver(entry['outer_features_state_ref']) if held else None
        if held: check(set(raw) == set(BRANCHES) and all(len(value) == len(held) for value in raw.values())
                       and entry['prediction_status'] == 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN', 'Fixed held feature/score evidence mismatch')
        else: check(entry['outer_features_state_ref'] is None and entry['prediction_status'] == 'NO_HELD_PREDICTIONS', 'True K1 held evidence invented')
        for stage in entry['stages']:
            is_b = stage['state'] == 'B0'; registry = old if is_b else classes; tids = entry['b_training_ids' if is_b else 'c_training_ids']
            check(stage['training_physical_ids'] == tids, 'R0 physical train binding mismatch')
            fact = _verify_baseline(stage, entry, labels, registry, resolver)
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += fact
            counts['baseline_head_triangular_solve_count'] += 2 * fact; counts['baseline_effective_df_triangular_solve_count'] += 2 * fact
            logs.append(dict(event='BASE_FIT', **stage))
            if held:
                subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if is_b else list(range(len(held)))
                scores = head_score(resolver(stage['final_state_ref']), {key: value[subset] for key, value in raw.items()})
                _array_close(entry['paths']['R0']['b_scores' if is_b else 'c_scores'], scores, 'Independent R0 outer fixed score mismatch')
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
        by_prep = {p['state']: p for p in entry['preparations']}; by_stage = {s['state']: s for s in entry['candidate_stages']}
        for prep, stage in zip(entry['preparations'], entry['candidate_stages']):
            check(stage['preparation_ref'] == prep['state'], 'Preparation/stage ordering mismatch')
            verify_preparation(prep, entry, labels, old, resolver); final = verify_candidate(stage, prep, entry, by_stage['B_CONDITIONAL'], resolver)
            logs.extend((dict(event='CONDITIONAL_PREPARATION', **prep), dict(event='CANDIDATE_FIT', **stage)))
            for key in PREPARATION_COUNTERS: counts[key] += prep[key]
            for key in STAGE_COUNTERS: counts[key] += stage[key]
            counts['trained_conditional_stage_count'] += int(stage['optimizer_steps'] > 0)
            if held:
                is_b = stage['mode'] == 'B'; subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if is_b else list(range(len(held)))
                scores = head_score(final, {key: value[subset] for key, value in raw.items()})
                _array_close(entry['paths']['R_CONDITIONAL_seq']['b_scores' if is_b else 'c_scores'], scores, 'Independent actual-B plus residual outer score mismatch')
                verify_score_workload(stage['score_workload'], final, len(subset))
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
            training.append(dict(split_id=record['split_id'], scope=scope, fold=entry['fold'], trial=entry['trial'], k=k,
                new_count=record['new_count'], state=stage['state'], train_k=entry['train_k'], latent_rank=prep['latent_rank'],
                trainable_parameter_count=stage['trainable_parameter_count'], trained_parameter_count=stage['trained_parameter_count'],
                optimizer_steps=stage['optimizer_steps'], stop_reason=stage['stop_reason'],
                initial_objective=stage.get('initial_objective'), final_objective=stage.get('final_objective'),
                gradients=stage['gradients'], trials=stage['trials'], steps=stage['steps'],
                initialization_state_ref=stage['initialization_state_ref'], final_state_ref=stage['final_state_ref'],
                evidence_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION'))
        # Only now join the independently fixed scores to legal support-held truth.
        if held:
            evidence = {name: dict(**{key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')},
                b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
            fresh = assess_paths(evidence, old)
            for name in PATHS: check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]), 'Fixed-score post-truth diagnostics mismatch')
        else:
            for path in entry['paths'].values(): check(path['b_scores'] == path['c_scores'] == [] and path['diagnostic'] is path['held_comparisons'] is None
                and path['metrics'] == dict.fromkeys(METRICS), 'True K1 metrics fabricated')
        _check_events(entry, by_prep, by_stage, binding); events.extend(entry['training_events']); counts['sequence_paths'] += 1
        if reuse:
            for name in PATHS: check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'new0 changed actual B scores')
        final_stage = by_stage['B_CONDITIONAL' if reuse else 'C_CONDITIONAL_seq']
        check(entry['deployment_C_state_bytes'] == {'R_CONDITIONAL_seq': final_stage['resident_numeric_state_bytes']}
              and entry['minimum_deployment_numeric_state_bytes'] == {'R_CONDITIONAL_seq': final_stage['deployment_numeric_state_bytes']}, 'Resident/minimum deployment byte scope mismatch')
        peak = max(peak, final_stage['resident_numeric_state_bytes'])
    counts['candidate_preparation_count'] = counts['ajlr_preparation_count']; counts['candidate_stage_count'] = counts['ajlr_stage_count']
    counts['baseline_triangular_solve_count'] = counts['baseline_head_triangular_solve_count'] + counts['baseline_effective_df_triangular_solve_count']
    counts['head_fit_count'] = sum(counts[key] for key in ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'prior_head_fit_count'))
    counts['factorization_count'] = sum(counts[key] for key in ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'prior_factorization_count'))
    check(all(type(record[key]) is int and record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == peak, 'Parent actual work aggregate mismatch')
    if k > 1:
        check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'), 'OOF physical pooling mismatch')
        check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy parent-first H/gap/decline aggregation mismatch')
    resolver.verify_tree(record)
    return logs, events, dict(oof=None if k == 1 else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if k == 1 else record['oneshot_proxy']['parent_mean_metrics']), training


def completion_check(done, spec):
    """Require exactly the declared dynamic rows/physical matrix before traces."""
    count = len(spec['rows']); exact = budget_for_spec(spec)['total']['exact']
    check(done['status'] == STATUS and done['model_rows'] == done['completed_rows'] == count
          and done['run_id'] == spec['run_id'], 'Declared row matrix incomplete')
    for key, value in exact.items(): check(done[key] == value, 'Incomplete declared physical coverage: ' + key)


def verify_stage_stream(stream, logs, split_id):
    for value in logs:
        check(next(stream, None) == dict(compact_event(value), schema=SCHEMA, method=METHOD, split_id=split_id), 'Actual fit stage stream mismatch')


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.) + record['fit_seconds']
    for row in logs:
        group = row['event'].lower()
        for key, value in row.items():
            if key.endswith('_seconds') and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group + '_' + key + '_sum'; resources[name] = resources.get(name, 0.) + value
            if key.endswith('_bytes') and type(value) is int:
                name = group + '_maximum_' + key; resources[name] = max(resources.get(name, 0), value)
        if row['event'] == 'CANDIDATE_FIT':
            forwards = ([] if row.get('initial_objective') is None else [row['initial_objective']]) + [v['objective'] for v in row['trials']]
            backwards = [v['objective'] for v in row['gradients']]
            for key, values in (('objective_forward_seconds_sum', forwards), ('objective_backward_seconds_sum', backwards)):
                resources[key] = resources.get(key, 0.) + sum(value['objective_seconds'] for value in values)
            resources['final_head_forward_seconds_sum'] = resources.get('final_head_forward_seconds_sum', 0.) + row['final_fit']['forward_seconds']
            for component in ('residual', 'prior'):
                for key, value in (row.get('score_workload') or {}).get(component, {}).items():
                    if type(value) in (int, float):
                        name = 'outer_' + component + '_' + key + '_sum'; resources[name] = resources.get(name, 0) + value


def _add(groups, key, metrics):
    group = groups.setdefault(key, {metric: [] for metric in METRICS})
    for metric, value in metrics.items(): group[metric].append(value)


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows(csv_record({key: row.get(key) for key in fields}) for row in rows)


def _resolved_argv(spec, row):
    co = spec['probe']['cohorts'][row['cohort']]
    return [str(PurePosixPath(spec['code']['cwd']) / 'tools/evaluate_d92_conditional_joint_probe.py'),
        '--support-features', row['support_features'], '--capsule', co['capsule'], '--output', row['output_root'] + '/probe',
        '--config', co['evaluation_config'], '--expected-capsule-id', co['capsule_id'],
        '--expected-checkpoint-sha256', row['expected_checkpoint_sha256'], '--expected-model-seed', str(row['seeds']['model']),
        '--run-id', spec['run_id'], '--row-id', row['row_id']]


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec; validate_spec(spec)
    root, out = Path(run_root or spec['execution']['remote_run_root']), Path(output)
    if out.exists(): raise FileExistsError(out)
    launch, done, state = [read(root / name) for name in ('startup.json', 'complete.json', 'state.json')]
    completion_check(done, spec)
    check(launch['spec'] == spec and launch['commit'] == done['commit'] and set(state) == {row['row_id'] for row in spec['rows']}
          and all(value['status'] == STATUS for value in state.values()), 'Complete run/source commit/row state mismatch')
    check(all(value['query_access'] is False and value['source_sample_access'] is False for value in (launch, done)), 'Forbidden run access')
    expected_supervisor = [str(PurePosixPath(spec['code']['cwd']) / 'tools/run_d92_conditional_joint_probe.py'),
        '--spec', str(PurePosixPath(spec['code']['cwd']) / spec['spec_path']), '--commit', done['commit']]
    check(launch['argv'] == expected_supervisor, 'Supervisor resolved argv mismatch')
    lanes = []
    # Complete bindings/inventories precede every fixed-score/held trace read.
    for row in spec['rows']:
        co = spec['probe']['cohorts'][row['cohort']]; lane = root / row['row_id'] / 'probe'
        marker = verify_marker(lane / 'probe_complete.json', spec, row); startup = read(lane / 'startup.json'); check_bind(startup, row, co)
        expected = _resolved_argv(spec, row); process = read(root / row['row_id'] / 'probe.log.process.json')
        check(startup['argv'] == expected and startup['python'] == spec['code']['environment']
              and process['argv'] == [startup['python'], '-u', *expected] and process['pid'] == startup['pid']
              and process['cwd'] == spec['code']['cwd'] and process['kind'] == 'probe', 'Evaluator process/resolved argv/core path mismatch')
        check(startup['run_id'] == marker['run_id'] == spec['run_id'] and startup['row_id'] == marker['row_id'] == row['row_id']
              and startup['schema'] == marker['schema'] == SCHEMA and startup['method'] == marker['method'] == METHOD
              and startup['scope'] == marker['scope'] == SCOPE, 'Row schema/actual run binding mismatch')
        expected_config = dict(algorithm=PROBE_CONFIG, producer_matrix=co['matrix'], selection=co['selection'])
        check(startup['config'] == expected_config and startup['episodes'] == spec['probe']['budget']['per_row'][row['row_id']]['exact']['episodes']
              and startup['producer_episodes'] == co['expected_split_count'], 'Dynamic selected axes/config coverage mismatch')
        for value in (startup, marker):
            check(all(value['channel'][key] == expected_value for key, expected_value in CHANNEL.items()) and value['scenarios'] == SCENARIOS
                  and value['query_rows_used'] == value['source_rows_used'] == 0 and value['truth_read'] is False, 'Channel/access mismatch')
            check(value['payload_audit']['new_source_payload_bytes'] == value['payload_audit']['new_ground_statistics_bytes'] == 0, 'Additional forbidden source payload')
        check(all(startup[key] is False for key in ('query_iq_access', 'checkpoint_loaded', 'encoder_updated'))
              and startup['actual_A'] is None and startup['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION', 'Invented A/forbidden fit access')
        cache_root = Path(startup['support_features']); check(cache_root == Path(row['support_features']), 'Declared cache reference changed')
        feature, plan, provenance = [read(cache_root / name) for name in ('features_complete.json', 'support_splits.json', 'checkpoint_provenance.json')]
        check_bind(feature, row, co)
        for value in (feature, plan): check(value['capsule_id'] == co['capsule_id'] and value['checkpoint_sha256'] == row['expected_checkpoint_sha256'], 'Producer cache/capsule binding mismatch')
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == co['expected_split_count']
              and provenance == startup['provenance'] and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH'
              and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance'] == [], 'Source-only contract/provenance mismatch')
        old = feature['classes']; check(len(old) == 6, 'Actual six-old registry required')
        chosen = selected_tasks([(split, None, None) for split in plan['splits']], co['selection'], old)
        verify_artifact_inventory(lane, marker); resolver = StateResolver(lane)
        lanes.append((row, lane, marker, {split['split_id']: split for split, _, _ in chosen}, old, startup, resolver))
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; training = []; archives = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    resource_strata = {name: {} for name in ('by_k_new_count', 'by_receiver_scene', 'by_model_cohort', 'by_row')}
    old_by_k = {}
    for row, lane, marker, expected, old, startup, resolver in lanes:
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stages = iter(jsonlines(lane / 'fit_stages.jsonl'))
        full_events = iter(jsonlines(lane / 'training_events.jsonl')); small_events = iter(jsonlines(lane / 'training_events_compact.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane / 'fit_trace.jsonl'), jsonlines(lane / 'compact.jsonl')):
            check(record is not None and small is not None and record['split_id'] in expected and record['split_id'] not in seen, 'Duplicate/unexpected/missing parent trace')
            sid = record['split_id']; seen.add(sid); split = expected[sid]
            labels = dict(zip(split['support_ids'], split['support_labels']))
            old_ids = tuple(sorted(pid for pid, y in labels.items() if split['registered_classes'][int(y)] in old))
            old_key = (row['row_id'], split['receiver'], split['scenario'], split['k'], split['support_seed'])
            check(old_by_k.setdefault(old_key, old_ids) == old_ids, 'Old physical support changed across new-count rows')
            logs, events, measurements, trace = verify_record(record, split, old, resolver, dict(run_id=spec['run_id'], row_id=row['row_id'], split_id=sid))
            check(small == compact_record(record), 'Compact parent evidence mismatch'); verify_stage_stream(stages, logs, sid)
            for event in events:
                value = dict(event, split_id=sid)
                check(next(full_events, None) == value and next(small_events, None) == compact_event(value), 'Full/compact training event inventory mismatch')
            training.extend(dict(row_id=row['row_id'], model_seed=row['seeds']['model'], cohort=row['cohort'], receiver=record['receiver'], scenario=record['scenario'], **value) for value in trace)
            accumulate_resources(resources, record, logs)
            keys = dict(by_k_new_count=(record['k'], record['new_count']), by_receiver_scene=(row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']),
                        by_model_cohort=(row['seeds']['model'], row['cohort'], record['k'], record['new_count']), by_row=(row['row_id'],))
            for name, key in keys.items():
                cell = resource_strata[name].setdefault(key, {}); accumulate_resources(cell, record, logs); cell['parents'] = cell.get('parents', 0) + 1
                for counter in COUNTERS[4:]: cell[counter] = cell.get(counter, 0) + record[counter]
            counts['episodes'] += 1; counts['k1_episodes'] += int(record['k'] == 1); counts['oof_episodes'] += int(record['k'] > 1)
            counts['proxy_anchor_count'] += small['proxy_anchor_count']
            for counter in COUNTERS[4:]: counts[counter] += record[counter]
            for diagnostic, paths in measurements.items():
                if paths is None: paths = {path: dict.fromkeys(METRICS) for path in PATHS}
                population = 'old_only' if record['new_count'] == 0 else 'new_present'
                for path, values in paths.items():
                    _add(strata['overall'], (diagnostic, path, population), values)
                    _add(strata['by_k_new_count'], (diagnostic, path, record['k'], record['new_count']), values)
                    _add(strata['by_receiver_scene'], (diagnostic, path, row['cohort'], record['receiver'], record['scenario'], record['k'], record['new_count']), values)
                    _add(strata['by_model_cohort'], (diagnostic, path, row['seeds']['model'], row['cohort'], record['k'], record['new_count']), values)
        check(seen == set(expected) and next(stages, None) is next(full_events, None) is next(small_events, None) is None, 'Missing declared parents or extra events')
        resolver.finalize(); manifest = resolver.manifest
        check(marker['state_archive_file_count'] == manifest['file_count'] and marker['state_archive_file_bytes'] == manifest['total_file_bytes']
              and marker['state_archive_numeric_bytes'] == manifest['numeric_array_bytes'], 'Marker actual archive byte mismatch')
        check(all(counts[key] == marker[key] == state[row['row_id']][key] for key in COUNTERS), 'Row actual count mismatch')
        for key in COUNTERS: coverage[key] += counts[key]
        resolver.clear_cache(); archives.append(dict(row_id=row['row_id'], root=str(lane), manifest=str(lane / 'state_manifest.json'),
            file_count=manifest['file_count'], file_bytes=manifest['total_file_bytes'], numeric_array_bytes=manifest['numeric_array_bytes'],
            archive_seconds=manifest['archive_seconds'], by_phase=manifest['by_phase'], cache=resolver.cache_statistics(), independent_analysis_work=dict(resolver.analysis_work)))
    check(all(coverage[key] == done[key] for key in COUNTERS), 'Run actual count mismatch')
    budget = budget_for_spec(spec)['total']
    check(all(coverage[key] == value for key, value in budget['exact'].items()) and all(coverage[key] <= value for key, value in budget['maximum'].items()), 'Dynamic structural budget exceeded')
    check(coverage['candidate_stage_count'] == len(training) and done['finished'] >= launch['started'], 'Complete training-stage/wall evidence mismatch')
    resources.update(actual_counters=dict(coverage), structural_budget=budget, run_wall_seconds=done['finished'] - launch['started'],
        lane_wall_seconds_sum=sum(marker['wall_seconds'] for _, _, marker, _, _, _, _ in lanes),
        maximum_lane_peak_rss_bytes=max((marker['peak_process_rss_bytes'] for _, _, marker, _, _, _, _ in lanes if marker['peak_process_rss_bytes'] is not None), default=None),
        peak_gpu_memory_bytes=None, deployment_package_bytes=None, incremental_transmission_bytes=None,
        additional_ground_data_payload_bytes=0, additional_ground_statistics_bytes=0, separate_spectral_seconds=None,
        independent_analysis_work={key: sum(item['independent_analysis_work'][key] for item in archives) for key in archives[0]['independent_analysis_work']},
        triangular_rhs_count_scope='Per actual system: columns=r; elements=n*r; dense=n*n*r; proxy not measured FLOPs',
        resident_state_scope='Core resident numeric buffers include actual B; archive bytes, minimum deployment and process RSS are distinct. Alias layouts cannot be reconstructed from NPZ copies.',
        positive_kernel_evidence='Independently rebuilt raw Gaussian/equivalence geometry plus actual saved projection/residual factor residuals; residual-only certificate positive_kernel_checked remains false',
        unmeasured_reason='CPU only; no deployment transport/energy measured; three candidate spectral diagnostics included in head timing but not independently timed',
        hardware=[dict(row_id=row['row_id'], hardware=startup['hardware'], blas_environment=startup['blas_environment']) for row, _, _, _, _, startup, _ in lanes])
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'cohort', 'receiver', 'scenario', 'k', 'new_count'), by_model_cohort=('diagnostic', 'path', 'model_seed', 'cohort', 'k', 'new_count'))
    tables = {name: _statistics(groups, dimensions[name]) for name, groups in strata.items()}
    rdimensions = dict(by_k_new_count=('k', 'new_count'), by_receiver_scene=('cohort', 'receiver', 'scenario', 'k', 'new_count'),
                       by_model_cohort=('model_seed', 'cohort', 'k', 'new_count'), by_row=('row_id',))
    resource_tables = {name: [dict(zip(rdimensions[name], key), **value) for key, value in sorted(groups.items())] for name, groups in resource_strata.items()}
    summary = dict(status=SUMMARY_STATUS, summary_schema=SUMMARY_SCHEMA, schema=SCHEMA, method=METHOD, scope=SCOPE,
        run_id=spec['run_id'], release_commit=done['commit'], model_rows=len(spec['rows']), coverage=coverage, resources=resources,
        resource_statistics=resource_tables, statistics=tables, algorithm=PROBE_CONFIG, channel=CHANNEL, old_class_count=6,
        actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        query_rows_used=0, source_rows_used=0, training_stage_count=len(training), state_archives=archives,
        state_archive_file_count=sum(value['file_count'] for value in archives), state_archive_file_bytes=sum(value['file_bytes'] for value in archives),
        raw_training_sources=[dict(row_id=row['row_id'], fit_trace=str(lane / 'fit_trace.jsonl'), full_training_events=str(lane / 'training_events.jsonl'),
            compact_training_events=str(lane / 'training_events_compact.jsonl')) for row, lane, _, _, _, _, _ in lanes],
        interpretation=['Support-only OOF and proxy; no query accessed. Legal held truth is joined after independently fixed scores.',
            'B/C optimize pooled class RMS CE only; no proximal +Z gradient. Ball .5 and 4x12 trials use actual projected delta; final is the last accepted state.',
            'All old physical support and every registered output column are constrained. Actual B mapping, fold, classes and physical IDs remain frozen; new0 reuses B.',
            'True K1 fits full closed heads and has N/A independent held metrics. Proxy is a separate within-parent diagnostic.',
            'H, gap and decline are computed within each parent before parent averaging. Proxy anchors are averaged within parent first. Repeated old support across new counts is not independent replication.',
            'A and B-A remain N/A until a legal actual ground packet is bound; R0/B0 is not A.',
            'Projection/residual and both adjoints have distinct RHS costs; candidate positive C performs three actual spectral diagnostics. Rejected trials and priors are included.',
            'Independent KKT adjoint general solves add analysis work and are reported separately. Residual-only certificates do not claim fresh PSD spectra.',
            'No accuracy threshold, promotion, training-peak selection or performance feedback changes method state.'])
    out.mkdir(parents=True, exist_ok=False); write_json(out / 'summary.json', summary)
    for name, rows in tables.items(): _write_csv(out / (name + '.csv'), rows)
    for name, rows in resource_tables.items(): _write_csv(out / ('resources_' + name + '.csv'), rows)
    with (out / 'training_objectives.jsonl').open('x', encoding='utf-8') as stream:
        for row in training: stream.write(json.dumps(row, allow_nan=False) + '\n')
    _write_csv(out / 'training_objectives.csv', [compact_event(row) for row in training])
    cells = {}
    for row in tables['by_k_new_count']: cells.setdefault(tuple(row[key] for key in dimensions['by_k_new_count']), {})[row['metric']] = row['mean']
    fields = ('B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
    lines = ['# ConditionalJoint 支持集独立分析', '', 'A 与 B−A 为 N/A。准确率按百分数、差值按百分点。真实 K1 的独立 held 为 N/A；proxy 单列。', '',
        '| 诊断 | 路径 | K | 新类数 | B 旧 | C 旧 | C 新 | H | 注册下降 | 新旧差 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, values in sorted(cells.items()): lines.append('| ' + ' | '.join([str(value) for value in key] + ['N/A' if values[field] is None else f'{100 * values[field]:.3f}' for field in fields]) + ' |')
    lines += ['', *['- ' + value for value in summary['interpretation']], '']; (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))


if __name__ == '__main__': main()
