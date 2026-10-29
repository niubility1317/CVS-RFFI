"""L-only nuisance metrics and truth-blind source G10 predictions.

The nested input schema is reused from star_relative_source, with this run's
identity/output. No old run/root/state is consumed. F0 and DURA stay frozen;
V chooses only global thresholds, after all L aggregates are immutable.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import source_open_readout as geometry_core
from . import source_practical_readout as risk
from . import star_nuisance_core as core
from . import star_relative_source as previous

RUN_ID = '20261029-phase3-star-nuisance-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_nuisance_source_v1'
STATUS = 'STAR_NUISANCE_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS = 'STAR_NUISANCE_SOURCE_THRESHOLDS_FROZEN'
PATHS = ('P0', 'P1')
VARIANTS = ('control', 'rx', 'rxday')
METHODS = ('MD', 'RMD')
CANDIDATES = tuple(v+'::'+m+'::'+p for v in VARIANTS for m in METHODS for p in PATHS)
PRIMARY = 'rxday::RMD::P1'
COUNTS = dict(L_s=6300, V=27000)
NOVEL_COUNT = 3150
RECEIVERS, SCENES = previous.RECEIVERS, previous.SCENES
FIELDS = {'run_id', 'code_commit', 'source_config', 'novel_public_root', 'novel_capsule_id',
          'novel_split_id', 'output_root', 'batch_size'}
KNOWN_FIELDS = {'ids', 'y', 'rx_i', 'day_i', 'scenario', 'scores', 'closed', 'baseline_closed'}
NOVEL_FIELDS = {'ids', 'rx_i', 'day_i', 'scores', 'closed'}

# Reuse the already-verified raw source binding, fixed G10 and full F0/G paths.
source_metadata = previous.source_metadata
_contract = previous._contract
_backend = previous._backend
_smoke = previous._smoke
_observations = previous._observations
_branches = previous._branches
_snapshot = previous._snapshot
_check_models = previous._check_models
_encode = previous._encode
_save_geometry = previous._save_geometry
_npz = previous._npz
_write = previous._write
_read = previous._read
_load_raw = previous._load_raw


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != FIELDS or config.get('run_id') != RUN_ID
            or not isinstance(config.get('code_commit'), str) or re.fullmatch(r'[0-9a-f]{40}', config['code_commit']) is None
            or type(config.get('batch_size')) is not int or config['batch_size'] != 128
            or any(not isinstance(config[k], str) or not config[k].strip()
                   for k in FIELDS - {'source_config', 'batch_size'})
            or not isinstance(config.get('source_config'), dict)):
        raise ValueError('Only the fixed eight-field source nuisance configuration is accepted')
    source = config['source_config']
    if (source.get('run_id') != RUN_ID or source.get('code_commit') != config['code_commit']
            or source.get('batch_size') != config['batch_size']
            or source.get('output_root') != config['output_root']):
        raise ValueError('Nested source inputs must use this run, code, batch and new output root')
    # Schema reuse only: no old run is executed and no old output is opened.
    previous.validate_config(dict(source, run_id=previous.RUN_ID))
    return config


def _load_novel(public_root, capsule_id, split_id):
    from .star_source_novel import load_public
    return load_public(public_root, capsule_id, split_id)


def _calibration_inputs(ids, y, closed, baseline, scores, rx, scenario):
    ids, y, closed, baseline, scores, rx, scenario = map(np.asarray,
        (ids, y, closed, baseline, scores, rx, scenario))
    n = len(ids) if ids.ndim else 0
    if (not n or ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != n
            or np.any(np.char.strip(ids) == '') or scores.shape != (n, len(CANDIDATES))
            or scores.dtype.kind != 'f' or not np.isfinite(scores).all()
            or any(a.shape != (n,) or a.dtype.kind not in 'iu' for a in (y, rx))
            or closed.shape != (n, 2) or closed.dtype.kind not in 'iu'
            or baseline.shape != (n, 2) or baseline.dtype.kind not in 'iu'
            or np.any((y < 0) | (y >= 6)) or np.any((closed < 0) | (closed >= 6))
            or not np.array_equal(baseline, closed) or not np.isin(rx, RECEIVERS).all()
            or scenario.shape != (n,) or scenario.dtype.kind != 'U' or not np.isin(scenario, SCENES).all()):
        raise ValueError('Aligned known-only six-class values and same-path original F0 baselines required')
    return ids, y, closed, baseline, scores, rx, scenario


def calibrate_known(ids, y, closed, baseline, scores, rx, scenario):
    """Maximal scalar per fixed candidate, intact ties, 42 known-only risks."""
    ids, y, closed, baseline, scores, rx, scenario = _calibration_inputs(ids, y, closed, baseline, scores, rx, scenario)
    masks = list(previous._masks(rx, scenario))
    coverage = {name: set(y[mask].tolist()) == set(range(6)) for name, mask in masks}
    methods, thresholds = {}, []
    for j, name in enumerate(CANDIDATES):
        path_index = PATHS.index(name.rsplit('::', 1)[1])
        new, old, values = closed[:, path_index], baseline[:, path_index], scores[:, j]
        local = {key: risk._local_maximum(y[mask], new[mask], old[mask], values[mask])
                 for key, mask in masks if coverage[key]}
        feasible = all(coverage.values()) and all(value[1] for value in local.values())
        threshold = min(value[0] for value in local.values()) if feasible else float(values.min())
        groups = {}
        for key, mask in masks:
            if coverage[key]:
                groups[key] = dict(risk._risk_group(y[mask], new[mask], old[mask], values[mask], threshold), certified=True)
            else:
                groups[key] = dict(certified=False, constraints_passed=False, reason='missing_known_class',
                    known_count=int(mask.sum()), class_counts=np.bincount(y[mask], minlength=6).tolist())
        levels = np.unique(values); index = int(np.searchsorted(levels, threshold))
        with np.errstate(over='ignore'):
            following = float(levels[index+1]) if index+1 < len(levels) else float(np.nextafter(levels[-1], np.inf))
        next_passed = feasible and np.isfinite(following) and all(
            risk._risk_group(y[mask], new[mask], old[mask], values[mask], following)['constraints_passed'] for _, mask in masks)
        methods[name] = dict(threshold=float(threshold), constraints_passed=all(v['constraints_passed'] for v in groups.values()),
            no_rejection_feasible=feasible, all_groups_class_complete=all(coverage.values()), zero_rejection_fallback=not feasible,
            next_candidate_threshold=following if np.isfinite(following) else None, next_candidate_passed=bool(next_passed),
            local_maximum_thresholds={k: v[0] for k, v in local.items()}, groups=groups)
        thresholds.append(threshold)
    return np.asarray(thresholds, np.float64), dict(status=CALIBRATION_STATUS, schema=SCHEMA, run_id=RUN_ID,
        candidate_names=list(CANDIDATES), group_count=42, known_count=len(ids), methods=methods,
        constraints_passed=all(v['constraints_passed'] for v in methods.values()), source_only=True, target_reads=0,
        source_V_labels_used=True, source_V_metric_fit=False, source_V_sigma_fit=False, unknown_inputs_used=False,
        limits=dict(known_macro_frr=.05, original_closed_minus_new_open=.02),
        baseline='same-input same-path original F0 complete head',
        acceptance_rule='score >= maximal legal global scalar; ties intact; missing class is uncertifiable')


def _score_paths(backend, received, states, device, batch_size):
    values = np.empty((len(received), len(CANDIDATES)), np.float64)
    closed = np.empty((len(received), len(PATHS)), np.int64)
    for pi, path in enumerate(PATHS):
        state = states[path]
        for start in range(0, len(received), batch_size):
            end = min(start+batch_size, len(received))
            branches, logits = _encode(backend, received[start:end], path, device)
            if logits.shape != (end-start, 6) or not np.isfinite(logits).all():
                raise ValueError('Finite full original six-class logits must preserve row order')
            closed[start:end, pi] = logits.argmax(axis=1)
            for variant in VARIANTS:
                score = core.metric_scores(branches, state['geometry'], state['metrics'][variant])
                if score.shape != (end-start, 2) or not np.isfinite(score).all():
                    raise ValueError('Two finite source-frozen MD/RMD scores required')
                for mi, method in enumerate(METHODS):
                    values[start:end, CANDIDATES.index(variant+'::'+method+'::'+path)] = score[:, mi]
            print('STAR_NUISANCE_PREDICT_PROGRESS', path, end, len(received), flush=True)
    return values, closed


def _check_states(states):
    for state in states.values():
        if not previous._same(dict(geometry=state['geometry'], metrics=state['metrics']), state['before']):
            raise ValueError('Source L geometry/metric/normalization changed after fitting')


def run(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('A new immutable source nuisance output root is required')
    source = config['source_config']; started = time.perf_counter()
    contract, metadata = source_metadata(source)
    from .star_channel import validate_mixture, validate_episode
    mixture = _read(source['mixture_config_path']); validate_mixture(mixture); validate_episode(source['episode'])
    backend = _backend(source, device, with_dura=True); snapshots = _snapshot(backend)
    smoke = _smoke(backend, source, device); _check_models(snapshots)
    if smoke.get('status') != 'PASS':
        raise ValueError('Actual frozen F0/G synthetic-star checkpoint smoke must pass before source IQ')
    output.mkdir(parents=True, exist_ok=False); _write(output/'smoke.json', smoke)
    raw = _load_raw(source['raw_mainsig'])
    received, _ = _observations(raw, metadata['L_s'], contract, source, mixture)
    states = {}
    for path in PATHS:
        branches, _ = _branches(backend, received, path, device, config['batch_size'])
        y = metadata['L_s']['y']; geometry = geometry_core.fit_geometry(branches, y)
        metrics = {v: core.fit_metric(branches, y, metadata['L_s']['rx_i'], metadata['L_s']['day_i'], geometry, v)
                   for v in VARIANTS}
        states[path] = dict(geometry=geometry, metrics=metrics, before=copy.deepcopy(dict(geometry=geometry, metrics=metrics)))
        row = output/'source'/path; row.mkdir(parents=True)
        _save_geometry(row/'geometry.npz', geometry)
        for variant, metric in metrics.items():
            (row/variant).mkdir(); _npz(row/variant/'metric.npz', core.to_arrays(metric))
        _check_models(snapshots)
        print('STAR_NUISANCE_L_FROZEN', path, len(y), flush=True)
    del received, branches
    # V can choose thresholds only. Every metric/projection/scale is already saved.
    received, scenarios = _observations(raw, metadata['V'], contract, source, mixture)
    del raw
    known_scores, known_closed = _score_paths(backend, received, states, device, config['batch_size'])
    del received
    _check_states(states); _check_models(snapshots)
    meta = metadata['V']; baseline = known_closed.copy()
    thresholds, calibration = calibrate_known(meta['ids'], meta['y'], known_closed, baseline,
        known_scores, meta['rx_i'], scenarios)
    calibration_root = output/'source/calibration'; calibration_root.mkdir(parents=True)
    with (calibration_root/'thresholds.npy').open('xb') as stream:
        np.save(stream, thresholds, allow_pickle=False)
    _npz(output/'known_predictions.npz', dict(ids=meta['ids'], y=meta['y'], rx_i=meta['rx_i'], day_i=meta['day_i'],
        scenario=scenarios, scores=known_scores, closed=known_closed, baseline_closed=baseline))
    _write(calibration_root/'complete.json', dict(calibration, code_commit=config['code_commit']))
    print(CALIBRATION_STATUS, flush=True)
    # This is the first access to the source-heldout novel public pool.
    manifest, ids, received, novel_meta = _load_novel(config['novel_public_root'], config['novel_capsule_id'], config['novel_split_id'])
    try:
        if (len(ids) != NOVEL_COUNT or ids.dtype.kind != 'U' or len(set(ids.tolist())) != len(ids)
                or set(ids.tolist()) & set(meta['ids'].tolist()) or set(ids.tolist()) & set(metadata['L_s']['ids'].tolist())):
            raise ValueError('Source novel public IDs/count must be unique and separate from original L/V')
        novel_scores, novel_closed = _score_paths(backend, received, states, device, config['batch_size'])
        _check_states(states); _check_models(snapshots)
        _npz(output/'novel_predictions.npz', dict(ids=ids, rx_i=novel_meta['rx_i'], day_i=novel_meta['day_i'],
            scores=novel_scores, closed=novel_closed))
    finally:
        if isinstance(received, np.memmap): received._mmap.close()
    total = sum(COUNTS.values()) + NOVEL_COUNT
    result = dict(status=STATUS, schema=SCHEMA, run_id=RUN_ID, code_commit=config['code_commit'], config=config,
        candidate_names=list(CANDIDATES), variant_names=list(VARIANTS), method_names=list(METHODS), path_names=list(PATHS),
        primary_candidate=PRIMARY, source_only=True, target_reads=0, target_labels_read=0,
        source_U_cache_reads=0, source_U_records_consumed=0, source_OE_records_consumed=0, source_novel_truth_reads=0,
        new_neural_updates=0, new_encoder_updates=0, new_identity_head_updates=0, new_G_updates=0, optimizer_updates=0,
        known_count=COUNTS['V'], novel_count=NOVEL_COUNT, L_physical_count=COUNTS['L_s'],
        new_star_observations=sum(COUNTS.values()), prebuilt_novel_star_observations=NOVEL_COUNT,
        F0_forward_rows=2*total, G_forward_rows=total, synthetic_F0_rows=4, synthetic_G_rows=2,
        reader_identity=backend.get('reader_identity'), dura_identity=backend.get('dura_identity'),
        novel_handle={k: config['novel_'+k] for k in ('capsule_id', 'split_id')},
        novel_public_schema=manifest.get('schema'), same_received_for_P0_P1=True, clean_model_inputs=0,
        model_parameters_buffers_unchanged=True, L_aggregates_unchanged_during_V=True,
        all_thresholds_frozen_before_novel=True, source_L_labels_used_for_statistics=True, source_V_labels_used_for_thresholds=True,
        source_V_metric_fit=False, source_V_sigma_fit=False, source_bank_deployed=False, source_IQ_saved=False,
        baseline='same-input same-path original F0 complete head',
        calibration_ref='source/calibration/complete.json', thresholds_ref='source/calibration/thresholds.npy',
        known_predictions_ref='known_predictions.npz', novel_predictions_ref='novel_predictions.npz',
        smoke=smoke, seconds=time.perf_counter()-started, final_phase3_complete=False,
        scope='source-only falsification on repeated heldout novel TX; not target confirmation or registration')
    _write(output/'complete.json', result)
    print(STATUS, flush=True)
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_NUISANCE_SOURCE_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
                          argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('run'); command.add_argument('--config', required=True); command.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    done = run(_read(args.config), args.device)
    print(json.dumps(dict(status=done['status'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
