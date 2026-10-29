"""Frozen F0 source-only G10 statistics and known-only risk calibration.

Only original L/V metadata select records from ManySig. The raw PKL is loaded
as a whole, but only those records enter observe/model/fit. Normalized source
IQ caches, U_s, OE, E51, source novel, and target inputs are never opened.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np

from . import source_open_readout as geometry_core
from . import source_practical_readout as risk
from . import star_relative_core as core
from . import unlabeled_registration_core as registration
from .branch_export import DIMENSIONS
from .data import MAIN_RX_NAMES

RUN_ID = '20261029-phase3-star-relative-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_relative_source_v1'
STATUS = 'STAR_RELATIVE_SOURCE_FROZEN'
CALIBRATION_STATUS = 'STAR_RELATIVE_SOURCE_THRESHOLDS_FROZEN'
PATHS = ('P0', 'P1')
COUNTS = dict(L_s=6300, V=27000)
RECEIVERS = (1, 3, 4, 6, 8)
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban', 'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
EPISODE = dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001)
READER_FIELDS = {'base_release', 'frozen_sources', 'source_output', 'model_seed'}
FIELDS = {'run_id', 'code_commit', 'reader_config', 'source_cache', 'source_contract', 'raw_mainsig',
          'mixture_config_path', 'episode', 'dura_checkpoint', 'dura_release', 'output_root', 'batch_size'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _reader_config(config):
    reader = config.get('reader_config')
    if (not isinstance(reader, dict) or set(reader) != READER_FIELDS or reader.get('model_seed') != 2026092701
            or type(reader['model_seed']) is not int
            or any(not isinstance(reader[k], str) or not reader[k] for k in ('base_release', 'source_output'))
            or not isinstance(reader['frozen_sources'], dict) or set(reader['frozen_sources']) != {'F', 'AF', 'BASE', 'FOLD', 'G'}
            or any(not isinstance(v, str) or not v for v in reader['frozen_sources'].values())):
        raise ValueError('Only the original frozen F0 reader configuration is allowed')
    return reader


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != FIELDS or config.get('run_id') != RUN_ID
            or not isinstance(config.get('code_commit'), str) or re.fullmatch(r'[0-9a-f]{40}', config['code_commit']) is None
            or type(config.get('batch_size')) is not int or config['batch_size'] != 128
            or any(not isinstance(config[k], str) or not config[k] for k in FIELDS - {'reader_config', 'episode', 'batch_size'})
            or not isinstance(config.get('episode'), dict)):
        raise ValueError('Only the fixed source-only twelve-field configuration is accepted')
    episode = {k: v for k, v in config['episode'].items() if k != 'condition'}
    if (episode != EPISODE or set(config['episode']) - set(EPISODE) not in (set(), {'condition'})
            or config['episode'].get('condition', 'MIX') != 'MIX'
            or any(type(episode[k]) is not int for k in ('scene_seed', 'channel_seed'))):
        raise ValueError('The fixed G10 MIX01 source realization is required')
    _reader_config(config)
    return config


def _contract(config):
    contract = _read(config['source_contract'])
    if (not isinstance(contract, dict) or contract.get('classes') != CLASSES
            or contract.get('source_rxs') != list(RECEIVERS) or contract.get('source_days') != [1, 2, 3]
            or contract.get('counts') != dict(L_s=COUNTS['L_s'], U_s=56700, V=COUNTS['V'])
            or contract.get('num_classes') != 6 or contract.get('split_seed') != 392005
            or type(contract.get('split_seed')) is not int
            # Original F0 keeps preprocessing in cache metadata; later external
            # contracts also inline it. Neither schema requires inventing keys.
            or contract.get('equalized', 1) != 1
            or contract.get('out_len', 256) != 256 or contract.get('normalize', True) is not True
            or not isinstance(contract.get('role_ids'), dict)):
        raise ValueError('The original F0 source contract/known roster must be retained')
    return contract


def source_metadata(config):
    """Read L/V metadata only; never SourceDataset or *_x.npy/U_s metadata."""
    contract = _contract(config)
    roles, seen = {}, set()
    fields = {'y', 'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain', 'ids'}
    for role, n in COUNTS.items():
        with np.load(Path(config['source_cache']) / (role + '_metadata.npz'), allow_pickle=False) as stored:
            if set(stored.files) != fields:
                raise ValueError('Original visible L/V metadata fields differ')
            meta = {k: stored[k].copy() for k in fields}
        ids = meta['ids']
        if (ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != n
                or any(not str(v) for v in ids) or ids.tolist() != contract['role_ids'].get(role)
                or seen & set(ids.tolist())):
            raise ValueError('Original source L/V physical order/count or disjointness differs')
        for k in fields - {'ids'}:
            if meta[k].shape != (n,) or meta[k].dtype.kind not in 'iu':
                raise ValueError('Aligned integer source metadata required: ' + k)
        if (set(meta['y'].tolist()) != set(range(6)) or not np.isin(meta['rx_i'], RECEIVERS).all()
                or not np.isin(meta['day_i'], [1, 2, 3]).all() or not np.all(meta['eq_i'] == 1)
                or np.any(meta['sig_i'] < 0) or np.any(meta['base_index'] < 0)
                or not np.array_equal(meta['domain'], np.searchsorted(RECEIVERS, meta['rx_i']) * 3 + meta['day_i'] - 1)):
            raise ValueError('Source class/RX/day/representation/packet metadata differs')
        seen.update(ids.tolist()); roles[role] = meta
    return contract, roles


def raw_packet(value):
    """Layout only: no crop, padding, FP32/RMS conversion, or quality selection."""
    value = np.asarray(value)
    if value.shape == (256,) and value.dtype.kind == 'c':
        result = np.stack((value.real, value.imag))
    elif value.dtype.kind == 'f' and value.shape == (256, 2):
        result = value.T
    elif value.dtype.kind == 'f' and value.shape == (2, 256):
        result = value
    else:
        raise ValueError('Original raw source packet must retain exactly 256 IQ samples')
    if not np.isfinite(result).all():
        raise ValueError('An originally selected source packet is nonfinite; do not reselect')
    return np.array(result, copy=True)


def raw_records(payload, metadata, contract):
    """Yield exact original packets, mapping model y through named TX classes."""
    if not isinstance(payload, dict) or not {'data', 'tx_list', 'rx_list', 'capture_date_list', 'equalized_list'}.issubset(payload):
        raise ValueError('Original ManySig raw layout metadata is absent')
    txs, rxs = list(map(str, payload['tx_list'])), list(map(str, payload['rx_list']))
    days, eqs = list(map(str, payload['capture_date_list'])), list(payload['equalized_list'])
    if (len(set(txs)) != len(txs) or set(txs) != set(CLASSES) or rxs != list(MAIN_RX_NAMES)
            or days != ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23']
            or len(eqs) <= 1 or eqs[1] != 1):
        raise ValueError('Original ManySig named class/RX/date/eq namespace differs')
    for i, sid in enumerate(metadata['ids']):
        y, rx, day, eq, sig = (int(metadata[k][i]) for k in ('y', 'rx_i', 'day_i', 'eq_i', 'sig_i'))
        if y not in range(6) or rx not in RECEIVERS or day not in (1, 2, 3) or eq != 1 or sig < 0:
            raise ValueError('Only original source L/V records may be indexed')
        tx = txs.index(contract['classes'][y])
        try:
            record = payload['data'][tx][rx][day][eq][sig]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError('An originally mapped raw source record is absent') from exc
        yield raw_packet(record), rxs[rx], days[day], str(sid)


def _load_raw(path):
    # Original PKL IO is whole-file, not per-record random access. No U_s role
    # cache is opened; only the exact L/V metadata are indexed after this load.
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def _observe(raw, **kwargs):
    from .star_channel import observe
    return observe(raw, **kwargs)


def _backend(config, device, with_dura=True):
    """Load only F0/G and contract metadata; shared by source and target paths."""
    import torch
    from . import runner
    reader_config = _reader_config(config)
    contract = _contract(config)
    reader = runner._load_model(reader_config, contract, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader_identity = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100,
        model_seed=reader_config['model_seed'], source_output=reader_config['source_output'],
        source_contract=config['source_contract'], base_release=reader_config['base_release'])
    dura, dura_identity = None, None
    if with_dura:
        release = Path(config['dura_release']); modules = release / 'experiments'
        if not (modules / 'pamf_r65_four_route/runtime.py').is_file() or not (modules / 'pamf_stage2_inference/engine.py').is_file():
            raise ValueError('The fixed complete DURA release is missing')
        package = importlib.import_module('experiments')
        package.__path__ = [str(modules)] + [str(p) for p in package.__path__ if Path(p) != modules]
        sys.path[:0] = [str(release), str(release / 'code')]
        from experiments.pamf_r65_four_route.runtime import load_checkpoint
        from experiments.pamf_stage2_inference.engine import InferenceEngine
        model, payload = load_checkpoint(config['dura_checkpoint'], device)
        if ((payload.get('arm'), payload.get('epoch'), payload.get('updates'), payload.get('code_commit')) !=
                ('stage2', 20, 4940, 'b2a2ce937a64654e17e84c2aa817b9da4257a48f')
                or sum(p.numel() for p in model.parameters()) != 266951):
            raise ValueError('Only the fixed original DURA v1 R69 may be loaded')
        model.eval().requires_grad_(False)
        dura = (model, InferenceEngine(model, reuse=True, graph=True, native=False))
        dura_identity = dict(deployment_id='dura_v1', checkpoint=config['dura_checkpoint'], release_root=config['dura_release'],
            arm='stage2', epoch=20, updates=4940, training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f',
            parameters=266951, reuse=True, graph=True, native=False, precision='FP32', TF32=False)
    return dict(reader=reader, dura=dura, reader_identity=reader_identity, dura_identity=dura_identity)


def _snapshot(backend):
    from .development_transfer import _snapshot_model
    models = [backend['reader']] + ([backend['dura'][0]] if backend['dura'] is not None else [])
    return [(model, _snapshot_model(model)) for model in models]


def _check_models(snapshots):
    from .development_transfer import _check_model
    for model, before in snapshots:
        _check_model(model, before)


def _encode(backend, values, path, device):
    import torch
    from .source_open_export import _tensor, extract_branches
    if path not in PATHS:
        raise ValueError('Only fixed P0/P1 computation paths are allowed')
    with torch.no_grad():
        x = _tensor(values, device)
        if path == 'P1':
            if backend['dura'] is None:
                raise ValueError('P1 requires frozen DURA')
            x = backend['dura'][1](x)[0].detach().clone()
        if x.dtype != torch.float32 or tuple(x.shape) != (len(values), 2, 256) or not bool(torch.isfinite(x).all()):
            raise ValueError('One finite FP32 star observation must reach F0')
        branches, logits = extract_branches(backend['reader'], x)
        arrays = {name: np.asarray(value.detach().cpu().tolist(), np.float32) for name, value in branches.items()}
        result = np.asarray(logits.detach().cpu().tolist(), np.float32)
    return arrays, result


def _smoke(backend, config, device):
    mixture = _read(config['mixture_config_path'])
    n = np.arange(256)
    raw = [np.stack((np.cos(.07*n), np.sin(.07*n))), np.stack((np.cos(.11*n), np.sin(.11*n)))]
    episode = {k: v for k, v in config['episode'].items() if k != 'condition'}
    received = np.stack([_observe(x, physical_id='star-relative-synthetic-'+str(i), receiver='1-19', day='2021_03_08',
        episode=episode, mixture=mixture)[0] for i, x in enumerate(raw)])
    for path in PATHS:
        branches, logits = _encode(backend, received, path, device)
        if set(branches) != set(DIMENSIONS) or logits.shape != (2, 6):
            raise ValueError('Actual F0/G synthetic checkpoint smoke differs')
    result = dict(status='PASS', real_checkpoint=True, synthetic_physical_rows=2, synthetic_F0_rows=4,
        synthetic_G_rows=2, paths=list(PATHS), source_IQ_reads=0, source_V_reads=0, target_reads=0, optimizer_updates=0)
    print('STAR_RELATIVE_SOURCE_CHECKPOINT_SMOKE_PASS', 2, flush=True)
    return result


def _observations(payload, meta, contract, config, mixture):
    episode = {k: v for k, v in config['episode'].items() if k != 'condition'}
    values = np.empty((len(meta['ids']), 2, 256), np.float32); scenes = []
    for i, (raw, receiver, day, sid) in enumerate(raw_records(payload, meta, contract)):
        received, private = _observe(raw, physical_id=sid, receiver=receiver, day=day, episode=episode, mixture=mixture)
        if received.shape != (2, 256) or received.dtype != np.float32 or not np.isfinite(received).all() or private['scenario'] not in SCENES:
            raise ValueError('G10 must return one finite star-only source observation')
        values[i] = received; scenes.append(private['scenario'])
    return values, np.asarray(scenes)


def _branches(backend, received, path, device, batch_size):
    rows = {key: [] for key in DIMENSIONS}; logits = []
    for start in range(0, len(received), batch_size):
        b, l = _encode(backend, received[start:start+batch_size], path, device)
        for key in DIMENSIONS:
            rows[key].append(b[key])
        logits.append(l)
    return {k: np.concatenate(v) for k, v in rows.items()}, np.concatenate(logits)


def _same(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    return left == right


def _statistics(branches, scores, meta, geometry):
    fused = np.asarray(branches['fused160'], np.float64)
    unit = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
    density = registration.source_density(registration.embedding(branches, geometry), unit, meta['y'], meta['rx_i'], meta['day_i'])
    return dict(eps=np.asarray(density['eps'], np.float64), radius=np.asarray(density['radius'], np.float64),
        sigma=np.maximum(np.std(scores, axis=0, ddof=0), 1e-6).astype(np.float64),
        anchor_units=np.asarray(density['anchor_units'], np.float32),
        anchor_scores=np.asarray([scores[index].mean(axis=0) for index in density['group_indices']], np.float64),
        group_keys=np.asarray(density['group_keys'], np.int64))


def _masks(rx, scene):
    yield 'overall', np.ones(len(rx), bool)
    for r in RECEIVERS:
        yield 'rx/'+str(r), rx == r
    for s in SCENES:
        yield 'scene/'+s, scene == s
    for r in RECEIVERS:
        for s in SCENES:
            yield 'rxscene/'+str(r)+'/'+s, (rx == r) & (scene == s)


def calibrate_known(ids, y, closed, scores, rx, scenario):
    """One scalar/method; exact Fraction risks and all 42 predeclared groups."""
    ids, y, closed, scores, rx, scenario = map(np.asarray, (ids, y, closed, scores, rx, scenario))
    n = len(ids) if ids.ndim else 0
    if (not n or ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != n or np.any(np.char.strip(ids) == '')
            or scores.shape != (n, 7) or scores.dtype.kind != 'f' or not np.isfinite(scores).all()
            or any(x.shape != (n,) or x.dtype.kind not in 'iu' for x in (y, closed, rx))
            or np.any((y < 0) | (y >= 6)) or np.any((closed < 0) | (closed >= 6))
            or not np.isin(rx, RECEIVERS).all() or scenario.shape != (n,) or scenario.dtype.kind != 'U'
            or not np.isin(scenario, SCENES).all()):
        raise ValueError('Only aligned six-known-class source V calibration inputs are accepted')
    masks = list(_masks(rx, scenario)); methods = {}; thresholds = []
    for j, method in enumerate(core.METHODS):
        values = scores[:, j]
        coverage = {name: set(y[mask].tolist()) == set(range(6)) for name, mask in masks}
        local = {name: risk._local_maximum(y[mask], closed[mask], closed[mask], values[mask])
                 for name, mask in masks if coverage[name]}
        feasible = all(coverage.values()) and all(v[1] for v in local.values())
        threshold = min(v[0] for v in local.values()) if feasible else float(values.min())
        groups = {}
        for name, mask in masks:
            if coverage[name]:
                groups[name] = risk._risk_group(y[mask], closed[mask], closed[mask], values[mask], threshold)
                groups[name]['certified'] = True
            else:
                groups[name] = dict(certified=False, constraints_passed=False, reason='missing_known_class',
                    known_count=int(mask.sum()), class_counts=np.bincount(y[mask], minlength=6).tolist())
        levels = np.unique(values); i = int(np.searchsorted(levels, threshold))
        next_level = float(levels[i+1]) if i+1 < len(levels) else float(np.nextafter(levels[-1], np.inf))
        next_pass = False if not feasible else all(risk._risk_group(y[mask], closed[mask], closed[mask], values[mask], next_level)['constraints_passed'] for _, mask in masks)
        methods[method] = dict(threshold=threshold, no_rejection_feasible=feasible,
            constraints_passed=all(g['constraints_passed'] for g in groups.values()), all_groups_class_complete=all(coverage.values()),
            next_candidate_threshold=next_level if np.isfinite(next_level) else None, next_candidate_passed=next_pass,
            zero_rejection_fallback=not feasible, groups=groups, local_maximum_thresholds={k: v[0] for k, v in local.items()})
        thresholds.append(threshold)
    return np.asarray(thresholds, np.float64), dict(schema=SCHEMA, status=CALIBRATION_STATUS,
        method_names=list(core.METHODS), known_count=n, group_count=42, methods=methods,
        constraints_passed=all(v['constraints_passed'] for v in methods.values()),
        class_count=6, receivers=list(RECEIVERS), scenario_order=list(SCENES), source_only=True,
        source_V_labels_used=True, source_V_metric_fit=False, source_V_sigma_fit=False,
        original_F0_closed_equals_baseline=True, unknown_inputs_used=False, target_reads=0,
        limits=dict(known_macro_frr=.05, original_closed_minus_new_open=.02),
        group_scope='overall + 5 RX + 6 actual scenes + 30 RX-by-scene; missing class is uncertifiable',
        acceptance_rule='score >= maximal legal global scalar with intact ties; uncertifiable uses min score and reports failure')


def _npz(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def _save_geometry(path, geometry):
    arrays = dict(schema=np.asarray(geometry['schema']), class_count=np.asarray(6, np.int64))
    for name, branch in geometry['branches'].items():
        arrays.update({name+'__'+k: v for k, v in branch.items()})
    _npz(path, arrays)


def _load_geometry(path):
    names = ('mean', 'scale', 'prototypes', 'whitener')
    expected = {'schema', 'class_count'} | {n+'__'+k for n in DIMENSIONS for k in names}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != expected:
            raise ValueError('Frozen geometry aggregate fields differ')
        result = dict(schema=z['schema'].item(), class_count=int(z['class_count'].item()),
            branches={n: {k: z[n+'__'+k].copy() for k in names} for n in DIMENSIONS})
    geometry_core._geometry(result)
    return result


def _save_metric(path, metric):
    core._metric(metric)
    arrays = dict(schema=np.asarray(metric['schema']), score_mean=metric['score_mean'], score_scale=metric['score_scale'])
    for name, branch in metric['branches'].items():
        arrays.update({name+'__'+k: v for k, v in branch.items()})
    _npz(path, arrays)


def _load_metric(path):
    names = ('class_means', 'background_mean', 'within_whitener', 'background_whitener')
    expected = {'schema', 'score_mean', 'score_scale'} | {n+'__'+k for n in core.BRANCHES for k in names}
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != expected:
            raise ValueError('Frozen relative metric aggregate fields differ')
        result = dict(schema=z['schema'].item(), score_mean=z['score_mean'].copy(), score_scale=z['score_scale'].copy(),
            branches={n: {k: z[n+'__'+k].copy() for k in names} for n in core.BRANCHES})
    core._metric(result)
    return result


def run(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('New source output root cannot overwrite existing state')
    started = time.perf_counter()
    contract, metadata = source_metadata(config)
    from .star_channel import validate_mixture, validate_episode
    mixture = _read(config['mixture_config_path']); validate_mixture(mixture); validate_episode(config['episode'])
    backend = _backend(config, device, with_dura=True); snapshots = _snapshot(backend)
    smoke = _smoke(backend, config, device); _check_models(snapshots)
    if smoke.get('status') != 'PASS':
        raise ValueError('Actual synthetic-star checkpoint smoke must pass before source IQ')
    raw = _load_raw(config['raw_mainsig'])
    received, _ = _observations(raw, metadata['L_s'], contract, config, mixture)
    states = {}
    for path in PATHS:
        branches, logits = _branches(backend, received, path, device, config['batch_size'])
        geometry = geometry_core.fit_geometry(branches, metadata['L_s']['y'])
        metric = core.fit_metric(branches, metadata['L_s']['y'], geometry)
        scores = core.all_scores(branches, logits, geometry, metric)
        statistics = _statistics(branches, scores, metadata['L_s'], geometry)
        states[path] = dict(geometry=geometry, metric=metric, statistics=statistics,
                            before=copy.deepcopy(dict(geometry=geometry, metric=metric, statistics=statistics)))
        _check_models(snapshots)
        print('STAR_RELATIVE_L_FIT_FROZEN', path, len(received), flush=True)
    del received, branches, logits, scores
    # All geometry, MD/RMD normalization, density, anchors, and sigma are now L-frozen.
    received, scenes = _observations(raw, metadata['V'], contract, config, mixture)
    del raw
    validation = {}
    for path in PATHS:
        state = states[path]; score_rows, closed_rows = [], []
        for start in range(0, len(received), config['batch_size']):
            branches, logits = _encode(backend, received[start:start+config['batch_size']], path, device)
            score_rows.append(core.all_scores(branches, logits, state['geometry'], state['metric']))
            closed_rows.append(logits.argmax(axis=1).astype(np.int64))
        values, closed = np.concatenate(score_rows), np.concatenate(closed_rows)
        thresholds, report = calibrate_known(metadata['V']['ids'], metadata['V']['y'], closed, values, metadata['V']['rx_i'], scenes)
        if not _same({k: state[k] for k in ('geometry', 'metric', 'statistics')}, state['before']):
            raise ValueError('L-frozen geometry/metric/statistics changed while consuming V')
        _check_models(snapshots)
        validation[path] = dict(scores=values, closed=closed, baseline_closed=closed.copy(), ids=metadata['V']['ids'],
            y=metadata['V']['y'], rx_i=metadata['V']['rx_i'], day_i=metadata['V']['day_i'], scenario=scenes)
        state.update(thresholds=thresholds, calibration=report)
    del received
    output.mkdir(parents=True, exist_ok=False)
    for path, state in states.items():
        row = output / 'by_path' / path; (row / 'calibration').mkdir(parents=True)
        _save_geometry(row / 'geometry.npz', state['geometry']); _save_metric(row / 'metric.npz', state['metric'])
        with (row / 'thresholds.npy').open('xb') as stream:
            np.save(stream, state['thresholds'], allow_pickle=False)
        _npz(row / 'statistics.npz', state['statistics'])
        _write(row / 'calibration/complete.json', dict(state['calibration'], path=path, run_id=RUN_ID, code_commit=config['code_commit']))
        vroot = output / 'source_V' / path; vroot.mkdir(parents=True)
        _npz(vroot / 'data.npz', validation[path])
    total = sum(COUNTS.values())
    result = dict(status=STATUS, schema=SCHEMA, run_id=RUN_ID, code_commit=config['code_commit'], config=config,
        path_names=list(PATHS), method_names=list(core.METHODS), primary_method=core.METHODS[6],
        reader_identity=backend['reader_identity'], dura_identity=backend['dura_identity'],
        identities={k: backend[k] for k in ('reader_identity', 'dura_identity')},
        by_path={path: dict(root='by_path/'+path, calibration_ref='by_path/'+path+'/calibration/complete.json',
            calibration_constraints_passed=states[path]['calibration']['constraints_passed']) for path in PATHS},
        source_only=True, L_physical_count=COUNTS['L_s'], V_physical_count=COUNTS['V'], source_physical_count=total,
        F0_source_forward_rows=2*total, G_source_forward_rows=total, new_star_observations=total,
        additional_independent_physical_records=0, same_received_for_P0_P1=True, clean_model_inputs=0,
        source_L_labels_used_for_statistics=True, source_V_labels_used_for_thresholds=True,
        sigma_source='source_L_all_scores.std(ddof=0), floor1e-6', geometry_metric_normalization_density_anchors_sigma_frozen_before_V=True,
        source_U_cache_reads=0, source_U_records_consumed=0, source_OE_records_consumed=0, source_novel_records_consumed=0,
        target_reads=0, target_labels_read=0, new_neural_updates=0, new_encoder_updates=0, new_identity_head_updates=0,
        new_G_updates=0, optimizer_updates=0, source_bank_deployed=False, source_IQ_saved=False,
        source_L_branches_lifetime='memory only until source fitting; not deployed or saved',
        source_V_ref='source_V/{path}/data.npz; known calibration diagnostics only',
        raw_IO_scope='whole ManySig PKL unpickled; only exact L/V metadata-indexed records consumed; no U_s cache opened',
        physical_index='metadata y -> named TX lookup; rx_i/day_i/eq_i/sig_i; base_index never used as packet index',
        overlay_physical_id='original contract role_id; role is not added to channel seed', episode=config['episode'],
        mixture_configuration=mixture, model_parameters_buffers_unchanged=True, L_aggregates_unchanged_during_V=True,
        smoke=smoke, seconds=time.perf_counter()-started, final_phase3_complete=False)
    _write(output / 'complete.json', result)
    print(STATUS, flush=True)
    return result


def load_source(root, path):
    """Read aggregate files/markers only; never source rows, IQ, or V arrays."""
    root = Path(root)
    if path not in PATHS:
        raise ValueError('Frozen source path must be P0 or P1')
    marker = _read(root / 'complete.json')
    if (marker.get('status') != STATUS or marker.get('schema') != SCHEMA or marker.get('run_id') != RUN_ID
            or marker.get('path_names') != list(PATHS) or marker.get('method_names') != list(core.METHODS)
            or marker.get('primary_method') != core.METHODS[6] or marker.get('source_only') is not True
            or marker.get('target_reads') != 0 or marker.get('new_neural_updates') != 0
            or marker.get('model_parameters_buffers_unchanged') is not True
            or marker.get('L_aggregates_unchanged_during_V') is not True
            or marker.get('geometry_metric_normalization_density_anchors_sigma_frozen_before_V') is not True):
        raise ValueError('The complete frozen star-only source marker is required')
    row = root / 'by_path' / path
    calibration = _read(row / 'calibration/complete.json')
    if (calibration.get('status') != CALIBRATION_STATUS or calibration.get('schema') != SCHEMA
            or calibration.get('run_id') != marker['run_id'] or calibration.get('code_commit') != marker['code_commit']
            or calibration.get('path') != path or calibration.get('method_names') != list(core.METHODS)
            or calibration.get('group_count') != 42 or calibration.get('source_V_metric_fit') is not False
            or calibration.get('source_V_sigma_fit') is not False or calibration.get('target_reads') != 0):
        raise ValueError('The matching independent known-only threshold marker is required')
    geometry, metric = _load_geometry(row / 'geometry.npz'), _load_metric(row / 'metric.npz')
    thresholds = np.load(row / 'thresholds.npy', allow_pickle=False)
    if thresholds.dtype != np.float64 or thresholds.shape != (7,) or not np.isfinite(thresholds).all():
        raise ValueError('Seven frozen source thresholds are required')
    shapes = dict(eps=(), radius=(), sigma=(7,), anchor_units=(90, 160), anchor_scores=(90, 7), group_keys=(90, 3))
    with np.load(row / 'statistics.npz', allow_pickle=False) as z:
        if set(z.files) != set(shapes):
            raise ValueError('Only source aggregate statistics are deployable')
        statistics = {k: z[k].copy() for k in z.files}
    for k, shape in shapes.items():
        value = statistics[k]
        dtype = np.int64 if k == 'group_keys' else np.float32 if k == 'anchor_units' else np.float64
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('Frozen source statistic shape/dtype differs: ' + k)
    expected_keys = np.asarray([(label, rx, day) for label in range(6) for rx in RECEIVERS for day in (1, 2, 3)], np.int64)
    if (statistics['eps'] <= 0 or statistics['radius'] <= 0 or np.any(statistics['sigma'] < 1e-6)
            or not np.array_equal(statistics['group_keys'], expected_keys)
            or not np.allclose(np.linalg.norm(statistics['anchor_units'], axis=1), 1., atol=2e-6)):
        raise ValueError('Positive source scales and unit group anchors are required')
    return dict(marker=marker, geometry=geometry, metric=metric, thresholds=thresholds, statistics=statistics)


def main(argv=None):
    print(json.dumps(dict(event='STAR_RELATIVE_SOURCE_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
                          argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run_parser = sub.add_parser('run'); run_parser.add_argument('--config', required=True); run_parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = run(_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
