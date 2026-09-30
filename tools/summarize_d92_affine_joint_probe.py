"""Independently certify AFFINE_JOINT arrays, workload and all paired support parents."""
import argparse
from collections import OrderedDict
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from evaluate_d92_affine_joint_probe import (
    CHANNEL, SCENARIOS, SCOPE, STATUS, SCHEMA, METHOD, PATHS, METRICS, BRANCHES, PROBE_CONFIG,
    COUNTERS, EXACT_COUNTS, MAX_COUNTS, PREPARATION_COUNTERS, STAGE_COUNTERS,
    check, read, scalars, csv_record, split_identity, selected_tasks, compact_event,
    assess_paths, pooled_assess, parent_mean, json_native, compact_record, validate_spec, verify_marker,
)
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import _distances
from summarize_d92_registration_diagnostic import _statistics, write_json
from summarize_d92_branch_support_probe import jsonlines, check_bind, finite_tree, close
from d92_affine_analysis_math import center_kernel_vjp

SUMMARY_STATUS = 'COMPLETE_AFFINE_JOINT_PROBE_VERIFIED'



class StateResolver:
    """Check archive content and cache only immutable numeric arrays, never math results."""
    def __init__(self, root, cache_budget_bytes=64*1024*1024, max_entries=64):
        self.root = Path(root).resolve()
        self.manifest = read(self.root/'state_manifest.json')
        check(self.manifest['schema'] == 'd92_affine_joint_state_archive_v1' and
            self.manifest['method'] == METHOD and self.manifest['status'] == 'COMPLETE' and
            self.manifest['prediction_formula'] == 'actual_B_prior_plus_centered_residual_plus_analytic_intercept',
            'Incomplete state archive')
        files = self.manifest['files']
        self.refs = {ref['path']: ref for ref in files}
        check(len(files) == len(self.refs) == self.manifest['file_count'], 'Duplicate/missing state archive inventory')
        check(self.manifest['total_file_bytes'] == sum(ref['file_bytes'] for ref in files), 'Archive byte inventory mismatch')
        self.used, self.verified, self.cache = set(), set(), OrderedDict()
        check(type(cache_budget_bytes) is int and cache_budget_bytes >= 0 and
            type(max_entries) is int and max_entries >= 0, 'Invalid archive cache budget')
        self.cache_budget_bytes, self.max_entries = cache_budget_bytes, max_entries
        self.cache_numeric_bytes = 0; self.cache_sizes = {}; self.file_mtimes = {}
        self.cache_counts = dict(load_count=0, hit_count=0, eviction_count=0,
            loaded_numeric_bytes=0, evicted_numeric_bytes=0, oversized_load_count=0,
            uncached_load_count=0, clear_count=0, cleared_entry_count=0,
            cleared_numeric_bytes=0, peak_numeric_bytes=0, peak_entries=0)

    def _file_state(self, path, ref):
        info = path.stat()
        check(info.st_size == ref['file_bytes'], 'Archived state file size mismatch')
        first = self.file_mtimes.setdefault(ref['path'], info.st_mtime_ns)
        check(info.st_mtime_ns == first, 'Archived state file mtime changed after first read')

    def _reserve(self, predicted_bytes):
        # Reserve against manifest numeric nbytes before materializing any array.
        # Oversized files empty the resident cache, are fully read/checked, and
        # are returned without residency. No numeric or math check is skipped.
        while self.cache and (self.cache_numeric_bytes+predicted_bytes > self.cache_budget_bytes
                or len(self.cache) >= self.max_entries):
            name, _ = self.cache.popitem(last=False); size = self.cache_sizes.pop(name)
            self.cache_numeric_bytes -= size
            self.cache_counts['eviction_count'] += 1
            self.cache_counts['evicted_numeric_bytes'] += size

    def clear_cache(self):
        """Release a completed lane's arrays while retaining reference/inventory state."""
        self.cache_counts['clear_count'] += 1
        self.cache_counts['cleared_entry_count'] += len(self.cache)
        self.cache_counts['cleared_numeric_bytes'] += self.cache_numeric_bytes
        self.cache.clear(); self.cache_sizes.clear(); self.cache_numeric_bytes = 0

    def cache_statistics(self):
        return dict(self.cache_counts, numeric_byte_budget=self.cache_budget_bytes,
            entry_cap=self.max_entries, resident_numeric_bytes=self.cache_numeric_bytes,
            resident_entry_count=len(self.cache))

    def __call__(self, ref):
        check(isinstance(ref, dict) and ref.get('path') in self.refs and ref == self.refs[ref['path']],
            'State reference missing from exact archive manifest')
        relative = Path(ref['path'])
        check(not relative.is_absolute() and relative.parts and relative.parts[0] == 'state_arrays', 'State path escaped archive')
        path = (self.root/relative).resolve()
        check(path.is_relative_to(self.root/'state_arrays') and path.suffix == '.npz', 'State path escaped archive')
        self.used.add(ref['path'])
        self._file_state(path, ref)
        if ref['path'] in self.cache:
            self.cache_counts['hit_count'] += 1
            self.cache.move_to_end(ref['path']); return self.cache[ref['path']]
        predicted = sum(meta['nbytes'] for meta in ref['arrays'].values())
        self._reserve(predicted)
        self.cache_counts['load_count'] += 1
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
        self._file_state(path, ref)
        actual = sum(value.nbytes for value in arrays.values())
        self.cache_counts['loaded_numeric_bytes'] += actual
        self.verified.add(ref['path'])
        if self.cache_budget_bytes and actual > self.cache_budget_bytes: self.cache_counts['oversized_load_count'] += 1
        if self.cache_budget_bytes and self.max_entries and actual <= self.cache_budget_bytes:
            self.cache[ref['path']] = arrays; self.cache_sizes[ref['path']] = actual
            self.cache_numeric_bytes += actual
            self.cache_counts['peak_numeric_bytes'] = max(self.cache_counts['peak_numeric_bytes'], self.cache_numeric_bytes)
            self.cache_counts['peak_entries'] = max(self.cache_counts['peak_entries'], len(self.cache))
        else: self.cache_counts['uncached_load_count'] += 1
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
    check(value['schema'] == 'd92_affine_joint_artifacts_v1' and value['method'] == METHOD and value['status'] == STATUS
        and marker['artifact_manifest'] == 'artifact_manifest.json', 'Missing complete artifact inventory')
    listed = {row['path']: row['file_bytes'] for row in value['files']}
    actual = {path.relative_to(root).as_posix(): path.stat().st_size for path in root.rglob('*')
              if path.is_file() and path.name != 'artifact_manifest.json'}
    check(len(listed) == len(value['files']) and listed == actual, 'Artifact file inventory/size mismatch')


def coordinate_state(ref, resolver, rank):
    arrays = resolver(ref)
    Z, U = arrays["Z"], arrays["U"]
    check(Z.shape == (736, rank) and U.shape == (736, 8), "AFFINE_JOINT coordinate state shape mismatch")
    return Z, U


def dct_initial():
    k = np.arange(736, dtype=np.float64)
    return np.asarray([np.full(736, 1/math.sqrt(736))]+[
        math.sqrt(2/736)*np.cos(math.pi*j*(k+.5)/736) for j in range(1, 8)])


def verify_coordinate_map(Z, U, anchor, W):
    expected = anchor.copy() if not np.any(Z) else anchor+Z@W.T
    check(np.array_equal(U, expected), "AFFINE_JOINT original-U coordinate reconstruction mismatch")


def verify_latent_coordinates(prep, resolver):
    arrays = resolver(prep["prepared_state_ref"])
    H, W = arrays["H"], arrays["W"]
    n, rank = prep["train_physical_count"], prep["latent_rank"]
    check(H.shape == (n, 8) and W.shape == (8, rank) and 0 <= rank <= 8,
          "AFFINE_JOINT dictionary/coordinate shape mismatch")
    eta = 128*sys.float_info.epsilon*max(n, 8)
    close(prep["rank_energy_threshold"], eta, "AFFINE_JOINT frozen numerical rank threshold mismatch")
    check(prep["dictionary_physical_evaluation_count"] == n, "AFFINE_JOINT dictionary physical workload mismatch")
    if not prep['rank_estimated']:
        check(rank == prep['latent_svd_count'] == 0 and prep['singular_values'] is None
            and prep['whitening_residual'] is prep['whitening_tolerance'] is None
            and np.array_equal(arrays['singular_values'], np.zeros(8)) and prep['no_information'],
            'AFFINE_JOINT invented skipped coordinate decomposition')
        close(prep['dictionary_rms'], float(np.linalg.norm(H))/math.sqrt(n), 'AFFINE_JOINT dictionary RMS mismatch')
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
          "AFFINE_JOINT latent numerical rank/SVD count mismatch")
    check(np.allclose(values, prep["singular_values"], rtol=eta, atol=0.),
          "AFFINE_JOINT full dictionary singular spectrum mismatch")
    check(np.array_equal(arrays['singular_values'], np.asarray(prep['singular_values'])),
          'AFFINE_JOINT archived/reported singular spectrum mismatch')
    if rank:
        expected_W = (vt[retained].T/spectrum[retained])/scale
        check(np.allclose(W@W.T, expected_W@expected_W.T, rtol=tolerance,
            atol=tolerance*float(np.linalg.norm(expected_W@expected_W.T))),
            'AFFINE_JOINT retained SVD coordinate subspace mismatch')
    white = (H@W)/math.sqrt(n)
    error = float(np.linalg.norm(white.T@white-np.eye(rank), ord=2)) if rank else 0.
    check(error <= tolerance, "AFFINE_JOINT dictionary whitening residual exceeded")
    close(prep["whitening_tolerance"], tolerance, "AFFINE_JOINT whitening tolerance mismatch")
    close(prep["whitening_residual"], error, "AFFINE_JOINT whitening residual mismatch")
    close(prep["dictionary_rms"], float(np.linalg.norm(H))/math.sqrt(n), "AFFINE_JOINT dictionary RMS mismatch")
    check(prep["dictionary_physical_evaluation_count"] == n, "AFFINE_JOINT optimizer coordinate physical binding mismatch")
    return H, W


def _array_close(actual, expected, message, scale=1.):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message+' shape/finite')
    tol = 128*np.finfo(np.float64).eps*max((1, *a.shape))*max(1., float(scale), float(np.linalg.norm(b)))
    check(float(np.linalg.norm(a-b)) <= tol, message)


def _scalar(data, key):
    value = np.asarray(data[key]); check(value.size <= 1, 'Scalar archive shape mismatch: '+key)
    return None if value.size == 0 else float(value.reshape(-1)[0])


def adapted_blocks(raw, U):
    """Independent sample-wise frozen DCT/tangent map; no fitting is invoked."""
    from scipy.special import erf
    b, a = interaction._blocks(**raw, allow_empty=True)
    original = np.concatenate((b, a), axis=1); unit = np.zeros_like(original)
    slices = (slice(0, 160), slice(160, 256), slice(256, 416), slice(416, 576), slice(576, 736))
    rho = np.zeros((len(original), 5))
    for j, sl in enumerate(slices):
        for i, row in enumerate(original):
            rho[i, j] = float(np.linalg.norm(row[sl]))
            if rho[i, j]: unit[i, sl] = row[sl]/rho[i, j]
    out = original.copy(); V0 = dct_initial()
    if not np.any(U): return b, a
    for i in range(len(out)):
        z = np.sum(V0*(unit[i]/math.sqrt(5.))[None, :], axis=1)
        h = .5*z*(1+erf(z/math.sqrt(2.))); v = np.sum(U*h[None, :], axis=1)
        for j, sl in enumerate(slices):
            if not rho[i, j]: continue
            w = unit[i, sl]; t = v[sl]-w*float(np.sum(w*v[sl]))
            delta = .25*t/float(np.hypot(.25, np.linalg.norm(t)))
            value = w+delta; out[i, sl] = rho[i, j]*value/np.linalg.norm(value)
    return out[:, :256], out[:, 256:]


def _head_parts(data, prefix=''):
    return {key[len(prefix):]: value for key, value in data.items() if key.startswith(prefix)} if prefix else data


def _raw_kernel(distance, tau, gamma):
    if gamma is None: return np.zeros_like(distance), np.zeros_like(distance)
    if tau == 0:
        raw = np.asarray(distance == 0, dtype=np.float64); return raw, raw-1.
    check(tau is not None and tau > 0 and gamma > 0, 'Undefined fixed kernel scale')
    return np.exp(-distance/tau), np.expm1(-distance/tau)


def _center(raw, cross, q, gamma):
    if gamma is None: return np.zeros_like(raw), np.zeros_like(cross)
    left, right, grand = raw@q, q@raw, float(q@raw@q)
    K = gamma*(raw-left[:, None]-right[None, :]+grand)
    L = gamma*(cross-(cross@q)[:, None]-right[None, :]+grand)
    return K, L


def head_score(data, raw, *, prefix=''):
    """Evaluate the saved classification function one physical sample at a time."""
    part = _head_parts(data, prefix)
    b, a = interaction._blocks(**raw, allow_empty=True)
    U = part.get('U', np.zeros((736, 8)))
    ub, ua = adapted_blocks(raw, U)
    tb, ta = part['original_train_b'], part['original_train_a']
    ab, aa = part.get('adapted_train_b', tb), part.get('adapted_train_a', ta)
    tau, gamma = _scalar(part, 'tau'), _scalar(part, 'gamma')
    alpha = part['alpha']; scores = np.zeros((len(b), alpha.shape[1]))
    for i in range(len(b)):
        d0 = _distances(b[i:i+1], a[i:i+1], tb, ta)
        d = d0 if tau == 0 else .5*(d0+_distances(ub[i:i+1], ua[i:i+1], ab, aa))
        _, cross = _raw_kernel(d, tau, gamma)
        if 'q' in part:
            _, Kcross = _center(part['raw_train_minus_one'], cross, part['q'], gamma)
        elif gamma is None:
            Kcross = np.zeros_like(cross)
        else:
            row = cross[0]; diff = (row-row[0])-part['reference_kernel']+_scalar(part, 'reference_self')
            Kcross = gamma*(diff-float(diff.mean())-part['center_mean']+_scalar(part, 'center_grand'))[None, :]
        scores[i] = (Kcross@alpha)[0]+part.get('intercept', np.zeros(alpha.shape[1]))
    if not prefix and any(key.startswith('prior_B_') for key in data):
        prior = head_score(data, raw, prefix='prior_B_')
        indices = np.asarray(data['prior_old_class_indices'], dtype=int) if 'prior_old_class_indices' in data else None
        check(indices is not None and len(indices) == prior.shape[1], 'Missing prior old class-column map')
        scores[:, indices] += prior
    return scores


def verify_head(ref, audit, resolver, prior=None):
    """Verify the full affine saddle equations from saved support-only arrays."""
    data = resolver(ref); y = np.asarray(data['train_labels'], dtype=int); yh = np.asarray(data['held_labels'], dtype=int)
    ids = audit['training_physical_ids']; held = audit['held_physical_ids']; classes = audit['classes']; n, c = len(ids), len(classes)
    check(y.shape == (n,) and yh.shape == (len(held),) and all(0 <= v < c for v in y), 'Head label/physical binding mismatch')
    oldref = audit['old_reference_physical_ids']; check(oldref and set(oldref) <= set(ids), 'Old reference escaped head train')
    q = np.asarray([1/len(oldref) if pid in oldref else 0. for pid in ids])
    _array_close(data['q'], q, 'Fixed old physical reference measure mismatch')
    d0 = _distances(data['original_train_b'], data['original_train_a'])
    dh0 = _distances(data['original_held_b'], data['original_held_a'], data['original_train_b'], data['original_train_a'])
    du = _distances(data['adapted_train_b'], data['adapted_train_a'])
    dhu = _distances(data['adapted_held_b'], data['adapted_held_a'], data['adapted_train_b'], data['adapted_train_a'])
    tau, gamma = _scalar(data, 'tau'), _scalar(data, 'gamma')
    distance, cross_distance = (d0, dh0) if tau == 0 else (.5*(d0+du), .5*(dh0+dhu))
    _array_close(data['distance'], distance, 'Actual joint train distance mismatch')
    _array_close(data['cross_distance'], cross_distance, 'Actual joint cross distance mismatch')
    oldpos = [ids.index(pid) for pid in oldref]; od = d0[np.ix_(oldpos, oldpos)]; oy = y[oldpos]
    s0 = float(np.sum(od[np.triu_indices(len(oldpos), 1)])/len(oldpos))
    check(_scalar(data, 's0') == s0, 'Original old-reference trace changed')
    expected_tau = None if len(set(oy)) == 1 else float(np.median(np.min(np.where(oy[:, None] != oy[None, :], od, np.inf), axis=1)))
    close(tau, expected_tau, 'Fixed original old-reference bandwidth mismatch')
    if tau is None or s0 == 0: check(gamma is None, 'Invented zero-information old scale')
    else:
        om1 = np.asarray(od == 0, dtype=float)-1. if tau == 0 else np.expm1(-od/tau)
        trace = float(-2*np.sum(om1[np.triu_indices(len(oldpos), 1)])/len(oldpos))
        close(gamma, s0/trace, 'Fixed original old-reference gamma mismatch')
    raw, rm1 = _raw_kernel(distance, tau, gamma); cross, cm1 = _raw_kernel(cross_distance, tau, gamma)
    for key, expected in (('raw_train', raw), ('raw_cross', cross), ('raw_train_minus_one', rm1), ('raw_cross_minus_one', cm1)):
        _array_close(data[key], expected, 'Raw fixed kernel mismatch: '+key)
    K, L = _center(rm1, cm1, q, gamma)
    _array_close(data['K'], K, 'Reference-centered K mismatch'); _array_close(data['L'], L, 'Reference-centered L mismatch')
    _array_close(K, K.T, 'Kernel symmetry mismatch'); _array_close(q@K, np.zeros(n), 'Old-reference kernel mean mismatch')
    Y = np.eye(c)[y]-1/c; _array_close(data['Y'], Y, 'Target contains intercept/recentering')
    M, Mh = data['M_train'], data['M_held']
    check(M.shape == (n, c) and Mh.shape == (len(held), c), 'Function prior shape mismatch')
    if prior is None:
        _array_close(M, np.zeros_like(M), 'B prior must be zero'); _array_close(Mh, np.zeros_like(Mh), 'B held prior must be zero')
    else:
        p, pclasses = prior
        check(set(pclasses) <= set(classes), 'Prior classes escaped registered output')
        _array_close(p['original_train_b'], data['original_train_b'][oldpos], 'Prior old training geometry changed')
        _array_close(p['original_train_a'], data['original_train_a'][oldpos], 'Prior old training auxiliary changed')
        def prior_at(ob, oa):
            pb, pa = p['original_train_b'], p['original_train_a']; pab, paa = p['adapted_train_b'], p['adapted_train_a']
            U = p['U']; features = {key: p[key] for key in BRANCHES} if all(key in p for key in BRANCHES) else None
            # Saved head cross geometry gives current-point mapping directly;
            # the adapter is independently checked at stage/outer-score boundary.
            blocks = np.concatenate((ob, oa), axis=1)
            pseudo = dict(z_id=blocks[:, :160], fft=blocks[:, 160:256], t_emb=blocks[:, 256:416], f_emb=blocks[:, 416:576], pa_local=blocks[:, 576:])
            ab, aa = adapted_blocks(pseudo, U)
            pd = .5*(_distances(ob, oa, pb, pa)+_distances(ab, aa, pab, paa))
            ptau, pgamma = _scalar(p, 'tau'), _scalar(p, 'gamma')
            if ptau == 0: pd = _distances(ob, oa, pb, pa)
            _, pcross = _raw_kernel(pd, ptau, pgamma)
            _, pl = _center(p['raw_train_minus_one'], pcross, p['q'], pgamma)
            answer = np.zeros((len(ob), c)); answer[:, [classes.index(v) for v in pclasses]] = pl@p['alpha']+p['intercept']
            return answer
        _array_close(M, prior_at(data['original_train_b'], data['original_train_a']), 'C train prior is not actual B function')
        _array_close(Mh, prior_at(data['original_held_b'], data['original_held_a']), 'C held prior is not actual B function')
    E = Y-M; _array_close(data['E'], E, 'Residual target was recentered')
    alpha, intercept, z = data['alpha'], data['intercept'], data['schur_z']
    check(alpha.shape == E.shape and intercept.shape == (c,) and z.shape == (n,)
        and data['schur_s'].shape == (), 'Affine coefficient/intercept/Schur shape mismatch')
    matrix = K+np.eye(n); s = _scalar(data, 'schur_s'); trace = float(np.trace(K))
    _array_close(data['combined_rhs'], np.column_stack((E, np.ones(n))), 'Combined C+1 head RHS mismatch')
    _array_close(matrix@z, np.ones(n), 'Schur z does not solve A z = e')
    close(s, float(z.sum()), 'Schur denominator mismatch')
    tolerance = 128*np.finfo(float).eps*max(n, c); schur_tol = tolerance*max(1., float(n), abs(s))
    check(s > 0 and n/(1+trace)-schur_tol <= s <= n+schur_tol, 'Schur positive spectral bound exceeded')
    F = alpha+z[:, None]*intercept
    _array_close(matrix@F, E, 'Schur base solution does not solve A F = E')
    _array_close(intercept, F.sum(axis=0)/s, 'Analytic intercept differs from Schur solution')
    numerator = float(np.linalg.norm(matrix@alpha+intercept[None, :]-E))
    denominator = (1+trace)*float(np.linalg.norm(alpha))+math.sqrt(n)*float(np.linalg.norm(intercept))+float(np.linalg.norm(E))
    residual = numerator/denominator if denominator else 0.; tolerance = 128*np.finfo(float).eps*max(n, c)
    check(residual <= tolerance, 'Canonical actual-trace normal equation residual exceeded')
    if data['chol'].size: _array_close(data['chol']@data['chol'].T, matrix, 'Saved Cholesky does not factor K+I')
    if gamma is None:
        check(data['chol'].shape == (0, 0), 'Zero kernel retained an invented factor')
        _array_close(z, np.ones(n), 'Zero kernel Schur z mismatch')
        _array_close(intercept, E.mean(axis=0), 'Zero kernel intercept is not mean residual target')
        _array_close(alpha, E-intercept, 'Zero kernel affine coefficients mismatch')
    else: check(data['chol'].shape == (n, n), 'Nonzero kernel missing factor')
    train_scores, scores = M+K@alpha+intercept, Mh+L@alpha+intercept
    _array_close(data['train_scores'], train_scores, 'Closed head train function mismatch')
    _array_close(data['scores'], scores, 'Closed head held function mismatch')
    _array_close(alpha.sum(axis=1), np.zeros(n), 'Canonical coefficients class-sum mismatch')
    _array_close(alpha.sum(axis=0), np.zeros(c), 'Affine coefficients sample-sum mismatch', scale=math.sqrt(n)*np.linalg.norm(alpha))
    _array_close(intercept.sum(), np.asarray(0.), 'Analytic intercept class-sum mismatch')
    _array_close(train_scores.sum(axis=1), np.zeros(n), 'Train score class-sum mismatch')
    _array_close(scores.sum(axis=1), np.zeros(len(held)), 'Held score class-sum mismatch')
    close(_scalar(data, 'actual_trace'), float(np.trace(K)), 'Actual kernel trace mismatch')
    sample_norm = float(np.linalg.norm(alpha.sum(axis=0))); sample_denom = math.sqrt(n)*float(np.linalg.norm(alpha))
    expected_audit = dict(normal_equation_absolute_residual=numerator, sample_sum_alpha_norm=sample_norm,
        sample_sum_alpha_residual=sample_norm/sample_denom if sample_denom else 0., schur_s=s,
        schur_s_lower_bound=n/(1+trace), schur_s_upper_bound=float(n), schur_s_tolerance=schur_tol,
        intercept_norm=float(np.linalg.norm(intercept)), analytic_intercept_parameter_count=c,
        analytic_intercept_contrast_count=c-1, condition_bound=1+trace,
        head_training_loss_data=.5*float(np.sum((train_scores-Y)**2)),
        head_training_loss_ridge=.5*float(np.sum(alpha*(K@alpha))))
    expected_audit['head_training_loss_total'] = expected_audit['head_training_loss_data']+expected_audit['head_training_loss_ridge']
    for key, expected in expected_audit.items():
        if key in audit: close(audit[key], expected, 'Affine head scalar mismatch: '+key)
    for key, expected in (('expected_old_reference_mean', q@M+intercept), ('fitted_old_reference_mean', q@train_scores),
                          ('residual_sample_mean', E.mean(axis=0))):
        if key in audit: _array_close(audit[key], expected, 'Affine head mean mismatch: '+key)
    _array_close(q@train_scores, q@M+intercept, 'Full old-reference mean omitted prior or intercept')
    for key, expected in (('normal_equation_residual', residual), ('actual_kernel_trace', float(np.trace(K)))):
        if key in audit: close(audit[key], expected, 'Head numerical audit mismatch: '+key)
    identity = not np.any(data.get('U', np.zeros((736, 8)))) or tau == 0
    work = dict(head_fit_count=1, factorization_count=int(gamma is not None),
        head_triangular_solve_count=2*int(gamma is not None), ajlr_forward_evaluation_count=1,
        raw_distance_evaluation_count=0 if identity else 1+int(len(held) > 0),
        raw_distance_pair_count=0 if identity else n*(n-1)//2+len(held)*n,
        reference_distance_evaluation_count=0 if identity else 1+int(len(held) > 0),
        reference_distance_pair_count=0 if identity else len(oldref)*(len(oldref)-1)//2+len(oldref)*(n-len(oldref))+len(held)*len(oldref),
        kernel_evaluation_count=(1+int(len(held) > 0))*int(gamma is not None),
        kernel_pair_count=(n*n+len(held)*n)*int(gamma is not None), adapter_physical_evaluation_count=n+len(held))
    work.update(head_triangular_rhs_count=2*(c+1)*int(gamma is not None),
        head_triangular_rhs_element_count=2*n*(c+1)*int(gamma is not None),
        head_triangular_dense_work_unit_count=2*n*n*(c+1)*int(gamma is not None),
        intercept_fit_count=1, intercept_addition_count=(n+len(held))*c)
    for key, expected in work.items():
        if key in audit: check(type(audit[key]) is int and audit[key] == expected, 'Actual head forward work mismatch: '+key)
    return data


def verify_preparation(prep, entry, labels, old, resolver):
    b = prep['state'] == 'B'; ids = entry['b_training_ids' if b else 'c_training_ids']
    classes = entry['b_classes' if b else 'c_classes']; k = entry['train_k']
    check(prep['training_physical_ids'] == ids and prep['classes'] == classes and prep['old_classes'] == old
        and prep['train_physical_count'] == len(ids) and prep['class_count'] == len(classes)
        and prep['ajlr_preparation_count'] == 1 and prep['inherited_state'] is (not b)
        and prep['inherited_adapter_from'] == (None if b else 'B_AFFINE')
        and prep['source_inputs'] is prep['query_fit'] is False, 'AFFINE_JOINT preparation binding mismatch')
    verify_latent_coordinates(prep, resolver)
    folds = prep['inner_folds']; nfolds = 0 if k == 1 else min(k, 3)
    check(len(folds) == nfolds and len(prep['prior_folds']) == (0 if b else nfolds), 'AFFINE_JOINT inner/prior fold coverage mismatch')
    groups = {cls: sorted(pid for pid in ids if labels[pid] == cls) for cls in classes}
    assignment = {pid: index % nfolds for values in groups.values() for index, pid in enumerate(values)} if nfolds else {}
    seen = set(); factors = 0; prior_arrays = []
    for j, fold in enumerate(folds):
        held = {pid for pid in ids if assignment[pid] == j}; train = set(ids)-held
        check(fold['inner_fold'] == j and set(fold['training_physical_ids']) == train
            and len(fold['training_physical_ids']) == len(train) and set(fold['held_physical_ids']) == held
            and len(fold['held_physical_ids']) == len(held) and not seen.intersection(held)
            and not (train | held).intersection(entry['c_ids']), 'AFFINE_JOINT inner/outer isolation mismatch')
        check(fold['all_head_statistics_from_old_inner_train_only'] is True
            and fold['old_reference_physical_ids'] == [pid for pid in fold['training_physical_ids'] if labels[pid] in old],
            'AFFINE_JOINT old-reference physical binding mismatch')
        seen.update(held)
        if b: check(fold['prior_source'] == 'ZERO', 'Invented B function prior')
        else:
            prior = prep['prior_folds'][j]
            check(prior['training_physical_ids'] == [pid for pid in fold['training_physical_ids'] if labels[pid] in old]
                and prior['held_physical_ids'] == [pid for pid in fold['held_physical_ids'] if labels[pid] in old]
                and prior['classes'] == old and prior['old_reference_physical_ids'] == prior['training_physical_ids']
                and fold['prior_source'] == 'FROZEN_ACTUAL_B_U_OLD_INNER_TRAIN_HEAD'
                and fold['prior_ref'] == prior['head_state_ref'], 'Actual B prior inner-fold binding mismatch')
            arrays = verify_head(prior['head_state_ref'], prior, resolver)
            verify_head(prior['head_state_ref'], prior['final_fit'], resolver)
            prior_arrays.append(arrays)
            check(np.array_equal(arrays['U'], resolver(prep['prepared_state_ref'])['anchor_U']), 'Prior adapter differs from actual B')
            factors += int(arrays['chol'].size > 0)
    check(not folds or seen == set(ids), 'AFFINE_JOINT inner held coverage mismatch')
    check(prep['prior_head_fit_count'] == (0 if b else nfolds) and prep['prior_factorization_count'] == factors
        and prep['prior_triangular_solve_count'] == 2*factors
        and prep['prior_score_evaluation_count'] == (0 if b else 2*nfolds+1)
        and prep['prior_score_physical_count'] == (0 if b else (nfolds+1)*len(ids)), 'Actual prior preparation work mismatch')
    for suffix, exponent in (('rhs_count', 0), ('rhs_element_count', 1), ('dense_work_unit_count', 2)):
        expected = sum(2*(len(a['alpha'])**exponent)*(a['alpha'].shape[1]+1) for a in prior_arrays if a['chol'].size)
        check(prep['prior_triangular_'+suffix] == expected, 'Prior combined C+1 RHS work mismatch: '+suffix)
    check(prep['prior_intercept_fit_count'] == len(prior_arrays)
        and prep['prior_intercept_addition_count'] == sum((len(a['train_labels'])+len(a['held_labels']))*len(old) for a in prior_arrays)
            +(0 if b else (nfolds+1)*len(ids)*len(old)), 'Prior analytic intercept fitting/addition work mismatch')
    expected_reason = 'PHYSICAL_K1' if k == 1 else 'ZERO_DICTIONARY_RANK' if prep['latent_rank'] == 0 else \
        'NO_OLD_KERNEL_INFORMATION' if prep['final_problem']['trace_scale'] is None else \
        'ALL_INNER_GEOMETRY_DEGENERATE' if not any(f['trace_scale'] is not None and f['bandwidth_tau'] is not None and f['bandwidth_tau'] > 0 for f in folds) else None
    check(prep['no_information_reason'] == expected_reason and prep['no_information'] is (expected_reason is not None), 'AFFINE_JOINT no-update reason mismatch')
    check(prep['preparation_seconds'] >= 0 and prep['prepared_distance_evaluation_count'] ==
        (3*(nfolds+1) if b else 5*nfolds+2), 'Actual original/preparation distance work mismatch')


def _distance_reverse(b, a, tb, ta, adj):
    """Independent VJP for ||b-tb||²+||a-ta||²+||b⊗a-tb⊗ta||²."""
    gb, ga, gtb, gta = np.zeros_like(b), np.zeros_like(a), np.zeros_like(tb), np.zeros_like(ta)
    for i in range(len(b)):
        w = adj[i, :, None]; aa = float(a[i]@a[i]); bb = float(b[i]@b[i])
        adot, bdot = ta@a[i], tb@b[i]
        leftb = 2*((1+aa)*b[i]-(1+adot[:, None])*tb)
        lefta = 2*((1+bb)*a[i]-(1+bdot[:, None])*ta)
        rightb = 2*((1+np.sum(ta*ta, axis=1)[:, None])*tb-(1+adot[:, None])*b[i])
        righta = 2*((1+np.sum(tb*tb, axis=1)[:, None])*ta-(1+bdot[:, None])*a[i])
        gb[i] = np.sum(w*leftb, axis=0); ga[i] = np.sum(w*lefta, axis=0)
        gtb += w*rightb; gta += w*righta
    return gb, ga, gtb, gta


def _adapter_reverse(ob, oa, U, g):
    from scipy.special import erf
    original = np.concatenate((ob, oa), axis=1); result = np.zeros_like(U)
    slices = (slice(0, 160), slice(160, 256), slice(256, 416), slice(416, 576), slice(576, 736))
    for row, incoming in zip(original, g):
        unit = np.zeros(736); norms = []
        for sl in slices:
            rho = float(np.linalg.norm(row[sl])); norms.append(rho)
            if rho: unit[sl] = row[sl]/rho
        z = np.sum(dct_initial()*(unit/math.sqrt(5.))[None, :], axis=1)
        h = .5*z*(1+erf(z/math.sqrt(2.))); v = np.sum(U*h[None, :], axis=1); gv = np.zeros(736)
        for sl, rho in zip(slices, norms):
            if not rho: continue
            w = unit[sl]; t = v[sl]-w*float(w@v[sl]); den = float(np.hypot(.25, np.linalg.norm(t)))
            s = w+.25*t/den; sn = float(np.linalg.norm(s)); y = s/sn
            gd = rho/sn*(incoming[sl]-y*float(y@incoming[sl]))
            gt = .25/den*(gd-t*float(t@gd)/(den*den)); gv[sl] = gt-w*float(w@gt)
        result += gv[:, None]*h[None, :]
    return result


def verify_companion(data, aggregate, j, G, fold, enabled=True):
    """Saddle adjoint, complete centering VJP and original-U gradient."""
    n, c = data['alpha'].shape; tau, gamma = _scalar(data, 'tau'), _scalar(data, 'gamma')
    active = enabled and gamma is not None and tau is not None and tau > 0 and len(G) > 0
    expected_calls = 2*int(active)
    check(fold['derivative_triangular_solve_count'] == expected_calls and fold['ce_adjoint_solve_count'] == int(active),
        'Affine adjoint branch/solve work mismatch')
    for suffix, amount in (('rhs_count', expected_calls*c), ('rhs_element_count', expected_calls*n*c),
                          ('dense_work_unit_count', expected_calls*n*n*c)):
        check(fold.get('derivative_triangular_'+suffix, 0) == amount, 'Affine adjoint C RHS work mismatch: '+suffix)
    names = ('adjoint_T', 'adjoint_eta', 'adjoint_g_b', 'adjoint_rhs')
    keys = ['fold_'+str(j)+'_'+name for name in names]
    if not active:
        check(not any(key in aggregate for key in keys), 'Invented zero-kernel/zero-bandwidth companion')
        return np.zeros((736, 8))
    T, eta, g_b, rhs = [aggregate[key] for key in keys]
    check(T.shape == rhs.shape == (n, c) and eta.shape == g_b.shape == (c,), 'Affine companion shapes mismatch')
    _array_close(g_b, G.sum(axis=0), 'Intercept gradient omitted sumRowsG')
    _array_close(rhs, data['L'].T@G, 'Adjoint C-column RHS mismatch')
    matrix = data['K']+np.eye(n)
    _array_close(matrix@T+eta[None, :], rhs, 'Affine saddle adjoint first equation mismatch')
    _array_close(T.sum(axis=0), g_b, 'Affine saddle adjoint sample sum is not g_b')
    V = T+data['schur_z'][:, None]*eta
    _array_close(eta, (V.sum(axis=0)-g_b)/_scalar(data, 'schur_s'), 'Affine companion Schur correction mismatch')
    close(fold['adjoint_g_b_norm'], float(np.linalg.norm(g_b)), 'Adjoint intercept gradient norm mismatch')
    close(fold['adjoint_sample_sum_residual'], float(np.linalg.norm(T.sum(axis=0)-g_b)), 'Adjoint sample-sum audit mismatch')
    barL = G@data['alpha'].T; rawK = -(T@data['alpha'].T); barK = .5*(rawK+rawK.T)
    barR, barQ = center_kernel_vjp(barK, barL, data['q'], gamma)
    _array_close(barR, gamma*barK, 'Complete centering VJP did not cancel under affine saddle constraints')
    _array_close(barQ, gamma*barL, 'Cross centering VJP omitted sample-zero alpha constraint')
    dd = -.5*barR*data['raw_train']/tau; dc = -.5*barQ*data['raw_cross']/tau
    b, a, hb, ha = [data[key] for key in ('adapted_train_b', 'adapted_train_a', 'adapted_held_b', 'adapted_held_a')]
    gb, ga, rb, ra = _distance_reverse(b, a, b, a, dd)
    hgb, hga, tgb, tga = _distance_reverse(hb, ha, b, a, dc)
    return _adapter_reverse(data['original_train_b'], data['original_train_a'], data['U'], np.concatenate((gb+rb+tgb, ga+ra+tga), axis=1)) + \
        _adapter_reverse(data['original_held_b'], data['original_held_a'], data['U'], np.concatenate((hgb, hga), axis=1))


def verify_objective(value, Z, U, prep, resolver, aggregate_ref=None):
    folds = value['inner_folds']; classes = prep['classes']; c = len(classes)
    check(len(folds) == len(prep['inner_folds']) == len(value['head_refs']) and value['temperature'] == 1., 'AFFINE_JOINT objective fold/temperature mismatch')
    sums = np.zeros(c); counts = np.zeros(c, dtype=np.int64); scores = []; labels = []; heads = []
    for j, (fold, physical, ref) in enumerate(zip(folds, prep['inner_folds'], value['head_refs'])):
        check(fold['head_ref'] == ref and all(fold[key] == physical[key] for key in
            ('classes', 'training_physical_ids', 'held_physical_ids', 'old_reference_physical_ids', 'bandwidth_tau', 'trace_scale')),
            'AFFINE_JOINT objective physical/reference binding mismatch')
        prior = None if prep['state'] == 'B' else (resolver(prep['prior_folds'][j]['head_state_ref']), prep['old_classes'])
        data = verify_head(ref, fold, resolver, prior)
        heads.append(data)
        check(np.array_equal(data['U'], U), 'Objective head U differs from accepted/trial state')
        f, y = data['scores'], np.asarray(data['held_labels'], dtype=int)
        top = f.max(axis=1); ce = np.log(np.exp(f-top[:, None]).sum(axis=1))+top-f[np.arange(len(f)), y]
        fs = np.bincount(y, weights=ce, minlength=c); fc = np.bincount(y, minlength=c)
        _array_close(fold['held_ce_sums'], fs, 'AFFINE_JOINT per-fold CE sums mismatch')
        check(fold['held_ce_counts'] == fc.tolist() and fold['held_correct_count'] == int(np.sum(np.argmax(f, axis=1) == y)), 'AFFINE_JOINT physical CE counts/accuracy mismatch')
        sums += fs; counts += fc; scores.append(f); labels.append(y)
    check(np.all(counts == prep['train_k']), 'CE must pool each physical inner-held row once')
    means = sums/counts; risk = float(np.linalg.norm(means))/math.sqrt(c); prox = .5*float(np.sum(Z*Z))
    for key, expected in (('class_ce_sums', sums), ('class_ce_counts', counts), ('class_ce_means', means)):
        _array_close(value[key], expected, 'AFFINE_JOINT pooled class CE mismatch: '+key)
    for key, expected in (('loss_ce', risk), ('loss_task', risk), ('RMSCE', risk), ('loss_proximal', prox), ('loss_total', risk+prox)):
        close(value[key], expected, 'AFFINE_JOINT risk/proximal mismatch: '+key)
    check(value['inner_head_fit_count'] == (0 if value['forward_cache_reused'] else len(folds))
        and value['inner_factorization_count'] == (0 if value['forward_cache_reused'] else sum(f['factorization_count'] for f in folds))
        and value['inner_objective_evaluation_count'] == int(not value['forward_cache_reused']), 'AFFINE_JOINT cached forward charged incorrectly')
    check(value['derivative_triangular_solve_count'] == sum(f['derivative_triangular_solve_count'] for f in folds)
        and value['ce_adjoint_solve_count'] == sum(f['ce_adjoint_solve_count'] for f in folds), 'AFFINE_JOINT CE adjoint actual cost mismatch')
    if aggregate_ref is not None:
        aggregate = resolver(aggregate_ref)
        _array_close(aggregate['scores'], np.concatenate(scores), 'Aggregate held score coordinate mismatch')
        check(np.array_equal(aggregate['labels'], np.concatenate(labels)), 'Aggregate held label binding mismatch')
        for key, expected in (('class_ce_sums', sums), ('class_ce_counts', counts), ('class_ce_means', means), ('RMSCE', np.asarray(risk)), ('prox', np.asarray(prox))):
            _array_close(aggregate[key], expected, 'Aggregate risk arrays mismatch: '+key)
        if 'g_Z' in aggregate:
            gU = np.zeros((736, 8))
            for j, (data, fold, f, y) in enumerate(zip(heads, folds, scores, labels)):
                ex = np.exp(f-f.max(axis=1)[:, None]); G = ex/ex.sum(axis=1)[:, None]; G[np.arange(len(y)), y] -= 1
                if risk: G *= (means[y]/(c*risk*counts[y]))[:, None]
                else: G.fill(0.)
                gU += verify_companion(data, aggregate, j, G, fold, enabled=risk > 0)
            W = resolver(prep['prepared_state_ref'])['W']
            _array_close(aggregate['g_Z'], gU@W+Z, 'Complete CE/prox gradient in Z coordinates mismatch', scale=np.linalg.norm(gU)*np.linalg.norm(W)+np.linalg.norm(Z))
    for key in ('head_triangular_rhs_count', 'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
                'intercept_fit_count', 'intercept_addition_count'):
        expected = 0 if value['forward_cache_reused'] else sum(f[key] for f in folds)
        check(value[key] == expected, 'Cached affine forward work mismatch: '+key)
    for suffix in ('rhs_count', 'rhs_element_count', 'dense_work_unit_count'):
        key = 'derivative_triangular_'+suffix
        check(value[key] == sum(f.get(key, 0) for f in folds), 'Affine objective derivative RHS aggregate mismatch')


def verify_candidate(stage, prep, entry, b_stage, resolver):
    rank = prep['latent_rank']; coordinates = resolver(prep['prepared_state_ref']); H, W = coordinates['H'], coordinates['W']
    anchor = np.zeros((736, 8)) if stage['mode'] == 'B' else resolver(b_stage['final_state_ref'])['U']
    check(stage['status'] == 'AFFINE_JOINT_STAGE_COMPLETE' and stage['config'] == PROBE_CONFIG
        and stage['preparation']['prepared_state_ref'] == prep['prepared_state_ref']
        and stage['preparation']['inner_folds'] == prep['inner_folds']
        and stage['preparation']['prior_folds'] == prep['prior_folds']
        and stage['training_physical_ids'] == prep['training_physical_ids']
        and stage['source_validation'] is None and stage['no_information'] is prep['no_information'], 'AFFINE_JOINT stage preparation/config mismatch')
    def state(ref):
        data = resolver(ref); z, u = coordinate_state(ref, resolver, rank)
        check(np.array_equal(data['W'], W) and np.array_equal(data['anchor_U'], anchor)
            and np.array_equal(data['singular_values'], coordinates['singular_values']), 'AFFINE_JOINT coordinate/anchor binding mismatch')
        verify_coordinate_map(z, u, anchor, W)
        return data, z, u
    initial, Z, U = state(stage['initialization_state_ref'])
    check(np.array_equal(Z, np.zeros((736, rank))) and np.array_equal(U, anchor), 'AFFINE_JOINT exact B-to-C initialization mismatch')
    steps, gradients, trials = stage['steps'], stage['gradients'], stage['trials']
    check(stage['optimizer_steps'] == len(steps) <= 4 and stage['optimizer_iterations'] == len(gradients) <= 4
        and stage['backward_evaluation_count'] == len(gradients) and stage['trial_count'] == stage['trial_attempt_count'] == len(trials) <= 48
        and stage['accepted_trial_count'] == len(steps) and stage['rejected_trial_count'] == len(trials)-len(steps), 'AFFINE_JOINT actual solver counts mismatch')
    objectives = []; current = None; path_length = 0.; used = 0; accepted_steps = []
    if prep['no_information']:
        check(not steps and not gradients and not trials and stage['initial_objective'] is stage['final_objective'] is None
            and stage['stop_reason'] == prep['no_information_reason'], 'Invented no-information adaptation')
    else:
        verify_objective(stage['initial_objective'], Z, U, prep, resolver, stage['initialization_state_ref'])
        objectives.append(stage['initial_objective']); current = stage['initial_objective']['loss_total']
    for iteration, event in enumerate(gradients, 1):
        data, gz, gu = state(event['state_ref']); g = data['g_Z']; direction = data['d_Z']
        check(event['iteration'] == iteration and np.array_equal(gz, Z) and np.array_equal(gu, U) and g.shape == direction.shape == Z.shape,
            'AFFINE_JOINT gradient cache/coordinate binding mismatch')
        norm = float(np.linalg.norm(g)); expected = -g/norm if norm else np.zeros_like(g)
        _array_close(direction, expected, 'AFFINE_JOINT normalized negative total gradient mismatch')
        close(event['gradient_norm'], norm, 'AFFINE_JOINT full gradient norm mismatch'); close(event['direction_norm'], float(np.linalg.norm(direction)), 'AFFINE_JOINT direction norm mismatch')
        verify_objective(event['objective'], Z, U, prep, resolver, event['state_ref'])
        check(event['objective']['forward_cache_reused'] is True and event['objective']['backward_evaluation_count'] == 1, 'Gradient refitted accepted heads')
        close(event['objective']['loss_total'], current, 'Gradient cache changed loss')
        group = [trial for trial in trials if trial['iteration'] == iteration]
        check(len(group) <= 12 and (norm > 0 or not group), 'AFFINE_JOINT trial after zero gradient')
        winner = None
        for j, trial in enumerate(group, 1):
            check(winner is None and trial['trial'] == j and trial['step_size'] == .125*.5**(j-1)
                and trial['gradient_state_ref'] == event['state_ref'], 'AFFINE_JOINT fixed first-acceptable trial order mismatch')
            td, tz, tu = state(trial['state_ref']); delta = tz-Z
            check(np.array_equal(tz, Z+trial['step_size']*direction) and np.array_equal(td['delta_Z'], delta), 'AFFINE_JOINT actual trial displacement mismatch')
            verify_objective(trial['objective'], tz, tu, prep, resolver, trial['state_ref']); after = trial['objective']['loss_total']
            dot = float(np.sum(g*delta)); tol = 128*float(np.finfo(float).eps)*max(1., abs(current), abs(after), abs(current+1e-4*dot))
            armijo = bool(after <= current+1e-4*dot+tol); nonincrease = bool(after <= current+tol)
            check(trial['armijo_pass'] is armijo and trial['objective_nonincrease_pass'] is nonincrease
                and trial['accepted'] is (armijo and nonincrease), 'AFFINE_JOINT exact two-condition acceptance mismatch')
            for key, expected in (('comparison_tolerance', tol), ('gradient_dot_delta', dot), ('update_norm', float(np.linalg.norm(delta))), ('loss_before', current), ('loss_after', after)):
                close(trial[key], expected, 'AFFINE_JOINT trial scalar mismatch: '+key)
            used += 1; objectives.append(trial['objective'])
            if trial['accepted']: winner = trial
        if winner is not None:
            step = steps[len(accepted_steps)]
            check(step['iteration'] == iteration and step['trial'] == winner['trial'] and step['state_ref'] == winner['state_ref']
                and step['objective'] == winner['objective'], 'AFFINE_JOINT accepted trial/step binding mismatch')
            accepted_steps.append(step); path_length += winner['update_norm']; _, Z, U = state(winner['state_ref']); current = winner['loss_after']
        else: check(iteration == len(gradients), 'AFFINE_JOINT continued after exhausted/zero gradient iteration')
    final, fz, fu = state(stage['final_state_ref'])
    check(used == len(trials) and accepted_steps == steps and np.array_equal(fz, Z) and np.array_equal(fu, U), 'AFFINE_JOINT final state is not last accepted state')
    check(path_length <= .5+128*np.finfo(float).eps*max(1, Z.size), 'AFFINE_JOINT normalized coordinate path exceeded')
    if not prep['no_information']:
        verify_objective(stage['final_objective'], Z, U, prep, resolver, stage['final_objective_state_ref'])
        check(stage['final_objective']['forward_cache_reused'] is True and stage['final_objective']['backward_evaluation_count'] == 0, 'AFFINE_JOINT final cache recomputed training work')
        close(stage['final_objective']['loss_total'], current, 'AFFINE_JOINT final risk differs from last accepted state')
        reason = stage['stop_reason']; check(reason in ('MAX_ITERATIONS', 'ZERO_GRADIENT', 'TRIAL_BUDGET_EXHAUSTED'), 'Unknown AFFINE_JOINT bounded stop')
        if reason == 'MAX_ITERATIONS': check(len(steps) == len(gradients) == 4, 'Premature AFFINE_JOINT max-iteration stop')
        if reason == 'ZERO_GRADIENT': check(gradients[-1]['gradient_norm'] == 0, 'False AFFINE_JOINT zero-gradient stop')
        if reason == 'TRIAL_BUDGET_EXHAUSTED': check(len([t for t in trials if t['iteration'] == len(gradients)]) == 12 and not trials[-1]['accepted'], 'False AFFINE_JOINT budget exhausted stop')
    for key in ('inner_objective_evaluation_count', 'inner_head_fit_count', 'inner_factorization_count'):
        check(stage[key] == sum(value[key] for value in objectives), 'AFFINE_JOINT actual objective workload mismatch: '+key)
    for key in ('derivative_triangular_solve_count', 'ce_adjoint_solve_count'):
        check(stage[key] == sum(g['objective'][key] for g in gradients), 'AFFINE_JOINT actual cached CE adjoint work mismatch')
    for suffix in ('rhs_count', 'rhs_element_count', 'dense_work_unit_count'):
        key = 'derivative_triangular_'+suffix
        check(stage[key] == sum(g['objective'][key] for g in gradients), 'Stage adjoint C RHS count mismatch: '+suffix)
    for key in ('head_triangular_rhs_count', 'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
                'intercept_fit_count', 'intercept_addition_count'):
        check(stage[key] == sum(o[key] for o in objectives)+stage['final_fit'][key], 'Stage complete/rejected-trial affine work mismatch: '+key)
    check(stage['head_triangular_solve_count'] == 2*(stage['inner_factorization_count']+stage['final_factorization_count'])
        and stage['ajlr_forward_evaluation_count'] == stage['inner_head_fit_count']+stage['final_head_fit_count'], 'AFFINE_JOINT solve/forward actual accounting mismatch')
    p = None
    if stage['mode'] == 'C_seq':
        bdata = resolver(b_stage['final_state_ref'])
        for key, value in bdata.items(): check(np.array_equal(final['prior_B_'+key], value), 'Final C prior is not exact current B state: '+key)
        p = (bdata, prep['old_classes'])
    verify_head(stage['final_state_ref'], stage['final_fit'], resolver, p)
    raw = {key: final[key] for key in BRANCHES}; ob, oa = interaction._blocks(**raw)
    ub, ua = adapted_blocks(raw, U)
    for key, expected in (('original_train_b', ob), ('original_train_a', oa), ('adapted_train_b', ub), ('adapted_train_a', ua), ('V0', dct_initial())):
        _array_close(final[key], expected, 'AFFINE_JOINT final geometry/dictionary mismatch: '+key)
    original = np.concatenate((ob, oa), axis=1); units = np.zeros_like(original)
    for sl in (slice(0, 160), slice(160, 256), slice(256, 416), slice(416, 576), slice(576, 736)):
        for i, row in enumerate(original):
            norm = np.linalg.norm(row[sl])
            if norm: units[i, sl] = row[sl]/norm
    from scipy.special import erf
    H_expected = []
    for row in units:
        z = np.sum(dct_initial()*(row/math.sqrt(5.))[None, :], axis=1); H_expected.append(.5*z*(1+erf(z/math.sqrt(2.))))
    _array_close(H, np.asarray(H_expected), 'Prepared dictionary is not bound to current support')
    if stage['mode'] == 'C_seq':
        oldpos2 = [j for j, pid in enumerate(prep['training_physical_ids']) if pid in b_stage['training_physical_ids']]
        for key in BRANCHES: check(np.array_equal(final[key][oldpos2], bdata[key]), 'C old physical support differs from actual B: '+key)
    displacement = np.stack([np.sum((U-anchor)*h[None, :], axis=1) for h in H]); actual = float(np.sum(displacement*displacement)/len(H))
    close(stage['pre_tangent_displacement_mean_squared'], actual, 'AFFINE_JOINT measured functional displacement mismatch')
    close(stage['loss_proximal'] if 'loss_proximal' in stage else .5*np.sum(Z*Z), .5*float(np.sum(Z*Z)), 'AFFINE_JOINT proximal state mismatch')
    close(stage['coordinate_norm'], float(np.linalg.norm(Z)), 'AFFINE_JOINT coordinate norm mismatch')
    close(stage['u_update_norm'], float(np.linalg.norm(U-anchor)), 'AFFINE_JOINT original U increment mismatch')
    check(stage['coordinate_parameter_count'] == 736*rank and stage['maximum_trainable_parameter_count'] == 5888
        and stage['trainable_parameter_count'] == (0 if prep['no_information'] else 736*rank)
        and stage['trained_parameter_count'] == (736*rank if steps else 0), 'AFFINE_JOINT active/trained parameter accounting mismatch')
    check(stage['final_head_fit_count'] == 1 and stage['final_factorization_count'] == int(final['chol'].size > 0)
        and stage['persistent_state_bytes'] > 0 and stage['adapter_state_bytes'] == 94208
        and stage['fit_seconds'] >= 0 and stage['score_seconds'] >= 0, 'AFFINE_JOINT final closed head/state work mismatch')
    n, c = final['alpha'].shape
    check(stage['analytic_intercept_parameter_count'] == c and stage['analytic_intercept_contrast_count'] == c-1
        and stage['analytic_coefficient_parameter_count'] == n*c and stage['analytic_head_parameter_count'] == (n+1)*c,
        'Analytic head parameters conflated with optimizer coordinates')
    # The core copies each retained immutable numeric field. Count these actual
    # shapes independently; only global V0 is shared between returned C and B.
    resident = 8*(23552+744*rank+3708*n+8+(5+int(final['chol'].size > 0))*n*n+6*n*c+c)
    deploy = 8*(11776+2208*n+4*n+n*c+c)
    if stage['mode'] == 'C_seq':
        resident += b_stage['persistent_state_bytes']-47104
        deploy += b_stage['deployment_numeric_state_bytes']-47104
    check(stage['persistent_state_bytes'] == resident and stage['deployment_numeric_state_bytes'] == deploy,
        'Actual persistent/deployment numeric buffer shapes mismatch')
    check(stage['training_coordinate_state_bytes'] == 8*(744*rank+8*n+8+5888), 'Actual training-coordinate buffers mismatch')
    check(stage['head_state_bytes'] == 8*(736*n+n*c+c+n*(1 if _scalar(final, 'gamma') is None else 2)),
        'Actual affine head buffers omitted intercept or double-counted shared views')


def verify_score_workload(value, final, n):
    check(value['score_physical_count'] == n and value['score_seconds'] >= 0 and value['feature_geometry_seconds'] >= 0
        and value['reference_distance_scope'] == 'subset_of_raw_distance_work_not_additive', 'Outer inference workload scope mismatch')
    def expected(data):
        gamma, tau = _scalar(data, 'gamma'), _scalar(data, 'tau'); m = len(data['q']); r = int(np.count_nonzero(data['q']))
        dc = n*(1+int(tau != 0 and np.any(data['U']))) if gamma is not None else 0
        result = dict(raw_distance_evaluation_count=dc, raw_distance_pair_count=dc*m,
            reference_distance_evaluation_count=dc, reference_distance_pair_count=dc*r,
            kernel_evaluation_count=n if gamma is not None else 0, kernel_pair_count=n*m if gamma is not None else 0,
            adapter_physical_evaluation_count=n if gamma is not None and tau != 0 else 0,
            dictionary_physical_evaluation_count=n if gamma is not None and tau != 0 else 0,
            intercept_addition_count=n*data['alpha'].shape[1])
        return result
    residual = expected(final); prior = expected(_head_parts(final, 'prior_B_')) if any(k.startswith('prior_B_') for k in final) else dict.fromkeys(residual, 0)
    for component, costs in (('residual', residual), ('prior', prior)):
        for key, amount in costs.items(): check(value[component].get(key, 0) == amount, 'Outer '+component+' executed work mismatch: '+key)
    for key in residual: check(value.get(key, 0) == residual[key]+prior[key], 'Outer component total mismatch: '+key)


def verify_record(record, split, old, resolver=None, expected_binding=None):
    finite_tree(record); check(resolver is not None, 'Complete AFFINE_JOINT state archive resolver required')
    binding = record['inheritance_binding']
    check(record['schema'] == SCHEMA and record['method'] == METHOD and
        all(isinstance(binding.get(key), str) and binding[key] for key in ('run_id', 'row_id'))
        and binding.get('split_id') == split['split_id'], 'Parent method/run/row/split binding mismatch')
    if expected_binding is not None:
        check(all(binding.get(key) == val for key, val in expected_binding.items()), 'Parent escaped current run/row')
    check(all(record[key] == value for key, value in split_identity(split, old).items()), 'Parent identity mismatch')
    classes, old = sorted(split['registered_classes']), sorted(old)
    labels = {pid: split['registered_classes'][y] for pid, y in zip(split['support_ids'], split['support_labels'])}
    k = split['k']; n = len(labels); groups = {cls: sorted(pid for pid in labels if labels[pid] == cls) for cls in classes}
    check(all(len(v) == k for v in groups.values()) and record['classes'] == classes and record['old_classes'] == old
        and record['support_count'] == n and record['old_class_count'] == len(old) and record['new_class_count'] == len(classes)-len(old), 'Parent physical K/registry mismatch')
    check(record['scope'] == SCOPE and record['query_rows_used'] == record['source_rows_used'] == 0, 'Forbidden parent access')
    expected = []
    if k == 1:
        check(record['fold_count'] == 0 and record['folds'] == [] and record['physical_fold_assignment'] == []
            and record['oof'] is record['oneshot_proxy'] is None and record['full_support'] is not None
            and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'K1 independent holdout fabricated')
        expected.append((record['full_support'], set(labels), set(), 'support_full_k1', None))
    else:
        folds = min(k, 3); assignments = {pid: j % folds for ids in groups.values() for j, pid in enumerate(ids)}
        physical = record['physical_fold_assignment']
        check(len(physical) == n and {v['physical_id']: (v['class_id'], v['fold']) for v in physical}
            == {pid: (labels[pid], assignments[pid]) for pid in labels}, 'Physical fold assignment mismatch')
        check(record['full_support'] is None and record['fold_count'] == len(record['folds']) == folds
            and record['heldout_unavailable_reason'] is None, 'OOF parent coverage mismatch')
        for fold, entry in enumerate(record['folds']):
            held = {pid for pid in labels if assignments[pid] == fold}; expected.append((entry, set(labels)-held, held, 'support_oof', fold))
        proxy = record['oneshot_proxy']
        check(proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
            and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent', 'All-anchor proxy coverage mismatch')
        for trial, entry in enumerate(proxy['trials']):
            train = {v[trial] for v in groups.values()}; expected.append((entry, train, set(labels)-train, 'support_oneshot_proxy', trial))
    counts = dict.fromkeys(COUNTERS[4:], 0); logs = []; events = []; training = []; peak = 0
    for entry, train, held, scope, index in expected:
        coords = dict(binding, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        def check_refs(value):
            if isinstance(value, dict):
                if {'path', 'arrays', 'array_summaries'} <= set(value):
                    namespace = json.loads(value['namespace'])
                    check(all(namespace.get(key) == val for key, val in coords.items()), 'Archive escaped current run/row/physical path')
                    check(namespace['state'] in ('OUTER_SUPPORT_HELD', 'B0', 'C0', 'B_prepare', 'C_prepare', 'B_AFFINE', 'C_AFFINE_seq'),
                        'Unknown archive stage namespace')
                else:
                    for child in value.values(): check_refs(child)
            elif isinstance(value, list):
                for child in value: check_refs(child)
        check_refs(entry)
        for item in entry['preparations']+entry['candidate_stages']+entry['stages']:
            check(all(item.get(key) == val for key, val in coords.items()), 'Actual B/C stage run/row/path mismatch')
        for item, keys in [(p, PREPARATION_COUNTERS) for p in entry['preparations']]+[(s, STAGE_COUNTERS) for s in entry['candidate_stages']]:
            check(all(type(item.get(key)) is int and item[key] >= 0 for key in keys), 'Invalid stage actual-work counter type')
        btrain = {pid for pid in train if labels[pid] in old}; bheld = {pid for pid in held if labels[pid] in old}
        for key, ids in (('b_training_ids', btrain), ('c_training_ids', train), ('b_ids', bheld), ('c_ids', held)):
            check(len(entry[key]) == len(ids) and set(entry[key]) == ids, 'Physical path mismatch: '+key)
        check(not train.intersection(held) and entry['held_labels'] == {pid: labels[pid] for pid in held}
            and entry['b_classes'] == old and entry['c_classes'] == classes and set(entry['paths']) == set(PATHS), 'Path paired class/held mismatch')
        check(entry['scope'] == scope and entry['parent_k'] == k and entry['fold'] == (index if scope == 'support_oof' else None)
            and entry['trial'] == (index if scope == 'support_oneshot_proxy' else None)
            and entry['train_k'] == len(train)//len(classes) and entry['held_k'] == len(held)//len(classes), 'Outer path coordinates mismatch')
        reuse = classes == old
        check(entry['c_reuses_b0'] is entry['c_reuses_b_candidates'] is reuse
            and [p['state'] for p in entry['preparations']] == (['B'] if reuse else ['B', 'C'])
            and [s['state'] for s in entry['candidate_stages']] == (['B_AFFINE'] if reuse else ['B_AFFINE', 'C_AFFINE_seq'])
            and [s['state'] for s in entry['stages']] == (['B0'] if reuse else ['B0', 'C0']), 'N0/shared B stage structure mismatch')
        raw = resolver(entry['outer_features_state_ref']) if held else None
        if held:
            check(set(raw) == set(BRANCHES) and all(len(v) == len(entry['c_ids']) for v in raw.values()), 'Outer support-held raw feature binding mismatch')
            evidence = {name: dict(**{key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')},
                b_scores=entry['paths'][name]['b_scores'], c_scores=entry['paths'][name]['c_scores']) for name in PATHS}
            fresh = assess_paths(evidence, old)
            for name in PATHS:
                check(entry['paths'][name] == dict(b_scores=evidence[name]['b_scores'], c_scores=evidence[name]['c_scores'], **fresh[name]), 'Fixed-score paired diagnostic mismatch')
        else:
            check(entry['outer_features_state_ref'] is None, 'K1 fabricated held features')
            for value in entry['paths'].values():
                check(value['b_scores'] == value['c_scores'] == [] and value['diagnostic'] is value['held_comparisons'] is None
                    and value['metrics'] == dict.fromkeys(METRICS), 'K1 independent scores/metrics fabricated')
        for stage in entry['stages']:
            b = stage['state'] == 'B0'; tids = entry['b_training_ids' if b else 'c_training_ids']; registry = old if b else classes
            check(stage['training_physical_ids'] == tids and stage['train_physical_count'] == len(tids)
                and stage['class_count'] == len(registry) and stage['optimizer_steps'] == 0
                and stage['all_states_estimated_from_trainfold_only'] is True and stage['source_validation'] is None,
                'R0 state boundary mismatch')
            fact = int(len(registry) > 1 and stage['interaction_centered_trace'] > 0)
            check(stage['factorization_calls'] == fact and stage['effective_degrees_of_freedom_extra_triangular_solves'] == 2*fact
                and 0 <= stage['normal_equation_residual'] <= stage['numerical_tolerance']
                and 0 <= stage['trace_relative_error'] <= stage['numerical_tolerance'], 'R0 numerical/work certification mismatch')
            counts['baseline_head_fit_count'] += 1; counts['baseline_factorization_count'] += fact
            counts['baseline_head_triangular_solve_count'] += 2*fact; counts['baseline_effective_df_triangular_solve_count'] += 2*fact
            logs.append(dict(event='BASE_FIT', **stage))
            if held:
                subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if b else list(range(len(entry['c_ids'])))
                selected = {key: v[subset] for key, v in raw.items()}; scores = head_score(resolver(stage['final_state_ref']), selected)
                _array_close(entry['paths']['R0']['b_scores' if b else 'c_scores'], scores, 'R0 saved outer score function mismatch')
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
        by_prep = {p['state']: p for p in entry['preparations']}; by_stage = {s['state']: s for s in entry['candidate_stages']}
        # The evaluator emits each preparation immediately before its fitted stage.
        for prep, stage in zip(entry['preparations'], entry['candidate_stages']):
            check(stage['preparation_ref'] == prep['state'], 'Candidate/preparation stage order mismatch')
            verify_preparation(prep, entry, labels, old, resolver); logs.append(dict(event='AFFINE_PREPARATION', **prep))
            for key in PREPARATION_COUNTERS: counts[key] += prep[key]
            verify_candidate(stage, prep, entry, by_stage['B_AFFINE'], resolver)
            logs.append(dict(event='CANDIDATE_FIT', **stage)); counts['trained_ajlr_stage_count'] += int(stage['optimizer_steps'] > 0)
            for key in STAGE_COUNTERS: counts[key] += stage[key]
            if held:
                b = stage['mode'] == 'B'; subset = [entry['c_ids'].index(pid) for pid in entry['b_ids']] if b else list(range(len(entry['c_ids'])))
                scores = head_score(resolver(stage['final_state_ref']), {key: v[subset] for key, v in raw.items()})
                _array_close(entry['paths']['R_AFFINE_seq']['b_scores' if b else 'c_scores'], scores, 'AFFINE_JOINT saved outer prior+residual function mismatch')
                verify_score_workload(stage['score_workload'], resolver(stage['final_state_ref']), len(subset))
                counts['final_score_evaluation_count'] += 1; counts['final_score_physical_count'] += len(subset)
            training.append(dict(split_id=record['split_id'], scope=scope, fold=entry['fold'], trial=entry['trial'],
                k=k, new_count=record['new_count'], state=stage['state'], train_k=entry['train_k'],
                latent_rank=prep['latent_rank'], trainable_parameter_count=stage['trainable_parameter_count'],
                trained_parameter_count=stage['trained_parameter_count'], optimizer_steps=stage['optimizer_steps'],
                stop_reason=stage['stop_reason'], initial_objective=None if stage['initial_objective'] is None else compact_event(stage['initial_objective']),
                final_objective=None if stage['final_objective'] is None else compact_event(stage['final_objective']),
                steps=[compact_event(v) for v in stage['steps']], gradients=[compact_event(v) for v in stage['gradients']], trials=[compact_event(v) for v in stage['trials']],
                initialization_state_ref=stage['initialization_state_ref'], final_state_ref=stage['final_state_ref'],
                evidence_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION', full_vectors_source='Original state_arrays; independently verified coordinates and heads'))
        per_stage = {key: dict(initial=0, final=0, gradients=0, trials=0, steps=0) for key in by_stage}; prepared_seen = set()
        for event in entry['training_events']:
            check(event['objective_scope'] == 'INNER_SUPPORT_TRAINING_NOT_VALIDATION' and event['source_validation'] is None
                and event['scope'] == scope and event['fold'] == entry['fold'] and event['outer_trial'] == entry['trial'], 'Training event coordinates/access mismatch')
            check(event['schema'] == SCHEMA and event['method'] == METHOD and all(event.get(key) == binding[key] for key in binding),
                'Training event current-run binding mismatch')
            name, kind = event['state'], event['event']
            if kind == 'AFFINE_JOINT_PREPARED':
                pn = name.removesuffix('_prepare'); check(pn in by_prep and pn not in prepared_seen, 'Duplicate/unknown preparation event')
                check(event['prepared_state_ref'] == by_prep[pn]['prepared_state_ref'] and event['prior_folds'] == by_prep[pn]['prior_folds'], 'Preparation event array/prior mismatch')
                prepared_seen.add(pn); continue
            check(name in by_stage, 'Unknown training event stage'); stage = by_stage[name]; seen = per_stage[name]
            mapping = dict(AFFINE_JOINT_GRADIENT='gradients', AFFINE_JOINT_TRIAL='trials', AFFINE_JOINT_STEP='steps')
            if kind in mapping:
                key = mapping[kind]; index2 = seen[key]; check(index2 < len(stage[key]), 'Training event exceeds solver trace')
                check(all(event.get(field) == value for field, value in stage[key][index2].items()), 'Event/full solver trace mismatch')
                seen[key] += 1
            elif kind == 'AFFINE_JOINT_INITIAL':
                check(seen['initial'] == 0 and event['state_ref'] == stage['initialization_state_ref']
                    and event.get('objective') == stage['initial_objective'], 'Initial event/cache mismatch'); seen['initial'] += 1
            elif kind == 'AFFINE_JOINT_FINAL':
                check(seen['final'] == 0 and event['final_state_ref'] == stage['final_state_ref'] and event['final_objective'] == stage['final_objective']
                    and event['stop_reason'] == stage['stop_reason'], 'Final event/cache mismatch')
                for key in STAGE_COUNTERS: check(event[key] == stage[key], 'Final event actual work mismatch: '+key)
                seen['final'] += 1
            else: raise ValueError('Unknown AFFINE_JOINT training event kind')
        check(prepared_seen == set(by_prep) and all(v['initial'] == v['final'] == 1 and all(v[key] == len(by_stage[name][key])
            for key in ('gradients', 'trials', 'steps')) for name, v in per_stage.items()), 'Complete solver event coverage mismatch')
        events.extend(entry['training_events']); counts['sequence_paths'] += 1
        if reuse:
            for name in PATHS: check(entry['paths'][name]['b_scores'] == entry['paths'][name]['c_scores'], 'N0 must reuse B scores exactly')
        final_stage = by_stage['B_AFFINE' if reuse else 'C_AFFINE_seq']; check(entry['deployment_C_state_bytes'] == dict(R_AFFINE_seq=final_stage['persistent_state_bytes']), 'Deployable C state bytes mismatch')
        peak = max(peak, final_stage['persistent_state_bytes'])
    counts['baseline_triangular_solve_count'] = counts['baseline_head_triangular_solve_count']+counts['baseline_effective_df_triangular_solve_count']
    counts['head_fit_count'] = sum(counts[key] for key in ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'prior_head_fit_count'))
    counts['factorization_count'] = sum(counts[key] for key in ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'prior_factorization_count'))
    check(all(type(record[key]) is int and record[key] == value for key, value in counts.items()) and record['persistent_state_bytes'] == peak, 'Actual parent aggregate workload mismatch')
    if k > 1:
        check(record['oof'] == dict(paths=pooled_assess(record['folds'], labels, classes, old), aggregation='one_record_per_physical_held_id'), 'OOF physical score pooling mismatch')
        check(record['oneshot_proxy']['parent_mean_metrics'] == parent_mean(record['oneshot_proxy']['trials']), 'Proxy parent-first aggregation mismatch')
    resolver.verify_tree(record)
    return logs, events, dict(oof=None if k == 1 else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if k == 1 else record['oneshot_proxy']['parent_mean_metrics']), training


def accumulate_resources(resources, record, logs):
    resources['parent_wall_seconds_sum'] = resources.get('parent_wall_seconds_sum', 0.)+record['fit_seconds']
    for row in logs:
        group = row['event'].lower()
        for key, value in row.items():
            if key.endswith('_seconds') and value is not None:
                check(value >= 0, 'Negative measured duration'); name = group+'_'+key+'_sum'
                resources[name] = resources.get(name, 0.)+value
            if key.endswith('_bytes') and type(value) is int:
                name = group+'_maximum_'+key; resources[name] = max(resources.get(name, 0), value)
        if row['event'] == 'CANDIDATE_FIT':
            forwards = ([] if row['initial_objective'] is None else [row['initial_objective']])+[v['objective'] for v in row['trials']]
            backwards = [v['objective'] for v in row['gradients']]
            for key, values in (('objective_forward_seconds_sum', forwards), ('objective_backward_seconds_sum', backwards)):
                resources[key] = resources.get(key, 0.)+sum(v['objective_seconds'] for v in values)
            resources['final_head_forward_seconds_sum'] = resources.get('final_head_forward_seconds_sum', 0.)+row['final_fit']['forward_seconds']
            for key, value in (row.get('score_workload') or {}).items():
                if type(value) in (int, float): resources['outer_candidate_'+key+'_sum'] = resources.get('outer_candidate_'+key+'_sum', 0)+value
            for component in ('residual', 'prior'):
                for key, value in (row.get('score_workload') or {}).get(component, {}).items():
                    if type(value) in (int, float):
                        name = 'outer_'+component+'_'+key+'_sum'; resources[name] = resources.get(name, 0)+value


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
        check(startup['run_id'] == marker['run_id'] == spec['run_id'] and startup['row_id'] == marker['row_id'] == row['row_id']
            and startup['schema'] == marker['schema'] == SCHEMA and startup['method'] == marker['method'] == METHOD,
            'Startup/marker method or current-run binding mismatch')
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
    # Every lane inventory and COMPLETE numeric manifest must pass before the
    # first outer-held feature or fixed-score trace is opened.
    resolvers = {}
    for row, lane, marker, _, _, _ in lanes:
        verify_artifact_inventory(lane, marker); resolvers[row['row_id']] = StateResolver(lane)
    coverage = dict.fromkeys(COUNTERS, 0); resources = {}; training = []; archive_sources = []
    strata = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_cohort')}
    resource_strata = {name: {} for name in ('by_k_new_count', 'by_receiver_scene', 'by_model_cohort', 'by_row')}
    for row, lane, marker, expected, old, startup in lanes:
        resolver = resolvers[row['row_id']]
        seen = set(); counts = dict.fromkeys(COUNTERS, 0); stages = iter(jsonlines(lane/'fit_stages.jsonl'))
        event_stream = iter(jsonlines(lane/'training_events.jsonl')); small_events = iter(jsonlines(lane/'training_events_compact.jsonl'))
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact length mismatch'); sid = record['split_id']
            check(sid in expected and sid not in seen, 'Unexpected/duplicate parent'); seen.add(sid)
            logs, events, measurements, train = verify_record(record, expected[sid], old, resolver,
                dict(run_id=spec['run_id'], row_id=row['row_id'], split_id=sid)); accumulate_resources(resources, record, logs)
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
        resolver.clear_cache()
        archive_sources.append(dict(row_id=row['row_id'], root=str(lane), manifest=str(lane/'state_manifest.json'),
            file_count=manifest['file_count'], file_bytes=manifest['total_file_bytes'], numeric_array_bytes=manifest['numeric_array_bytes'],
            archive_seconds=manifest['archive_seconds'], by_phase=manifest['by_phase'], cache=resolver.cache_statistics()))
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
        fitted_state_scope='Persistent state includes actual retained training coordinates, Schur/combined RHS and head caches; deployment_numeric_state_bytes separately excludes those training buffers and includes analytic intercept and actual B prior. Archives and frozen Phase1 are separate.',
        triangular_rhs_count_scope='Per actual triangular call: r RHS columns, n*r elements, n*n*r dense work units; dense work is a proxy, not measured FLOPs',
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
         schema=SCHEMA, method=METHOD, algorithm=PROBE_CONFIG, channel=CHANNEL, statistics=tables,
        old_class_count=6, actual_A=None, adaptation_gain_B_minus_A=None, automatic_promotion=False, performance_gate=None,
        query_rows_used=0, source_rows_used=0, training_stage_count=len(training),
        raw_training_sources=[dict(row_id=row['row_id'], fit_trace=str(lane/'fit_trace.jsonl'),
            full_training_events=str(lane/'training_events.jsonl'), compact_training_events=str(lane/'training_events_compact.jsonl'))
            for row, lane, _, _, _, _ in lanes],
        state_archives=archive_sources, state_archive_file_count=sum(v['file_count'] for v in archive_sources),
        state_archive_file_bytes=sum(v['file_bytes'] for v in archive_sources),
        interpretation=[
            'R0 is original BranchLocalRidge. R_AFFINE_seq is the only fixed candidate. C inherits the actual current-row B function and original U_B; no state crosses a parent, fold or model. The previous two-candidate run and this single-candidate run have different workloads.',
            'Each inner prior B head uses only old inner-train records under frozen actual U_B. Final C embeds the actual final B head. Fixed old physical reference measure, tau and gamma are inherited; its reference embeddings move with current U_C.',
            'The analytic affine residual head uses A=I+K, F=A^-1(Y-M), z=A^-1 e, s=e^T z, b=e^T F/s, alpha=F-z b. The intercept is unpenalized, e^T alpha=0, and full old-reference mean is q^T M+b. Targets are not sample-recentered. All registered classes compete in one argmax.',
            'Class means of CE pool all physical inner-held losses across folds before RMS reduction. The objective adds 0.5||Z||^2. Those labels supervise the adapter and do not constitute independent validation.',
            'Four updates and twelve trials per update are fixed. The normalized negative total gradient uses the first trial satisfying both Armijo and objective nonincrease. Rejected trials do not replace the last accepted cache; budget exhaustion is not convergence.',
            'True K1 executes legal B/C closed heads without adapter supervision or held metrics. Proxy trainK1 executes the same head semantics; it is a within-parent diagnostic and cannot replace genuine K1 holdout.',
            'A is unavailable; B minus B0 is a support-classifier increment, not B minus ground A.',
            'OOF pools each physical held row once. Proxy averages all anchors within parent then weights parents equally.',
            'Old-only reuse and repeated old support across new-count rows are not independent observations.',
            'Coordinate parameter count, gradient-trainable count and actually updated count are distinct. U plus materialized fixed V0 is 94208 bytes; C also retains actual B and both fitted heads. Parameter count is not training FLOPs or a starborne measurement.',
            'Baseline primal and EDF solves, prior preparation/score, trial/final heads, affine CE adjoints and outer prior/residual inference are recorded separately. Every head uses C+1 RHS columns and every adjoint C; prior uses old class count plus one. Rejected trials are fully charged. Dense triangular work units are not measured FLOPs. Reference pair counts are subsets of raw work.',
            'Original NPZ arrays retain coordinates, gradients, directions, kernels, q, priors, labels, residual targets, intercepts, Schur buffers and scores. The summary independently verifies affine equations, complete intercept companion, centering VJP, CE/prox gradient, current-run actual B, every RHS counter and full event/file inventory. It creates no missing final-full-C-Z0 head.',
            'Registration-only final full C at Z0 cannot be separated from later adapter changes when that full head was not archived. Inner Z0 heads concern their own supervised inner-held sets; they do not identify that final outer-held decomposition.',
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
    lines = ['# AFFINE_JOINT-LocalRidge support pilot', '',
        '完整160 parent、4 row、R0和R_AFFINE_seq两路径均已核验；旧类固定6个。A与B−A为N/A。真实K1完整闭式头已执行，独立持出指标为N/A。', '',
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
        f"实际有效训练阶段{coverage['trained_ajlr_stage_count']}个、函数坐标更新{coverage['optimizer_steps']}次、内层head拟合{coverage['inner_head_fit_count']}次、新增最终head拟合{coverage['final_head_fit_count']}次。",
        f"实测运行墙钟{resources['run_wall_seconds']:.3f} s；完整适配器和分类头状态、线程与分项工作量见summary.json。", '']
    lines += ['- '+line for line in summary['interpretation']]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8'); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--run-root'); parser.add_argument('--output', required=True)
    result = summarize(**vars(parser.parse_args())); print(json.dumps(dict(status=result['status'], coverage=result['coverage'])))

if __name__ == '__main__': main()
