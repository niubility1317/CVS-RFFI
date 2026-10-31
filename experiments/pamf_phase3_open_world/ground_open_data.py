"""Source clean -> fixed DURA, with disjoint calibration and blinded queries.

The original clean roles remain unchanged. No raw dataset, target, private
novel truth, or G10 teacher is used to construct the clean training roles.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch

from .ground_open_source import SourceDataset
from . import fresh_world_run as fresh
from . import star_relative_source as source
from . import source_diverse_export as diverse
from . import source_open_export as auxiliary


RUN_ID = '20261101-phase3-ground-open-s2026092701-r03'
SCHEMA = 'pamf_phase3_ground_open_data_v1'
STATUS = 'GROUND_OPEN_DATA_COMPLETE'
ROLE_STATUS = 'GROUND_OPEN_ROLE_COMPLETE'
QUERY_STATUS = 'GROUND_OPEN_QUERY_COMPLETE'
REFERENCE_CODE = 'f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
CACHE_RUN = '20261029-phase3-star-representation-s2026092701-r01'
SPLIT_SEED = 2026110101
CAPSULE_ID = RUN_ID+'-source-clean-v1'
SPLIT_ID = 'sourceV-half-y-rx-day-s2026110101'
RECEIVERS = (1, 3, 4, 6, 8)
COUNTS = dict(L=6300, Us=56700, OE=6134, V=27000, novel=3150)
ROLE_COUNTS = dict(L=6300, Us=56700, OE=6134, CAL=13500)
CELL_SIZE = 300
ROLES = ('L', 'Us', 'OE', 'CAL')
CONDITIONS = ('source_clean_DURA', 'source_star_DURA')
FIELDS = {'run_id', 'code_commit', 'reference_config', 'source_config', 'diverse_data_root',
          'oe_data_root', 'split_seed', 'output_root', 'batch_size'}
META_FIELDS = {'ids', 'rx_i', 'day_i', 'domain'}
_read, _write = source._read, source._write


def validate_config(c):
    if (not isinstance(c, dict) or set(c) != FIELDS or c.get('run_id') != RUN_ID
            or re.fullmatch('[0-9a-f]{40}', str(c.get('code_commit'))) is None
            or type(c.get('split_seed')) is not int or c['split_seed'] != SPLIT_SEED
            or type(c.get('batch_size')) is not int or c['batch_size'] != 128
            or any(not isinstance(c.get(k), str) or not c[k].strip()
                   for k in ('diverse_data_root', 'oe_data_root', 'output_root'))):
        raise ValueError('Exact nine-field source-clean configuration/seed/batch128 required')
    r = fresh.validate_config(c['reference_config'])
    if r['code_commit'] != REFERENCE_CODE:
        raise ValueError('Only the fixed fresh f4b5 reference is permitted')
    s = c['source_config']
    if not isinstance(s, dict) or s.get('run_id') != CACHE_RUN:
        raise ValueError('The original source-representation source configuration is required')
    # Reuse the twelve-field constructor/input schema, not its old run outputs.
    source.validate_config(dict(s, run_id=source.RUN_ID))
    out = Path(c['output_root']).resolve()
    roots = [r['output_root'], c['diverse_data_root'], c['oe_data_root'],
             s['source_cache'], s['output_root'], s['reader_config']['source_output']]
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in roots):
        raise ValueError('New immutable ground data cannot overlap any parent/source inputs')
    return c


def _reference(c):
    r = c['reference_config']; m = _read(Path(r['output_root'])/'complete.json')
    expected = dict(schema=fresh.SCHEMA, status=fresh.STATUS, run_id=fresh.RUN_ID,
                    code_commit=REFERENCE_CODE, config=r, all_three_DURA_conditions_complete=True,
                    all_conditions_through_DURA=True, source_models_frozen_before_V=True,
                    all_thresholds_frozen_before_novel_and_Q=True, classification_unchanged=True,
                    private_truth_reads=0, target_U_reads=0)
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError('Complete unchanged fixed fresh reference required; no target inputs are opened')
    s = c['source_config']; reader = s['reader_config']
    original = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100,
                    model_seed=reader['model_seed'], source_output=reader['source_output'],
                    source_contract=s['source_contract'], base_release=reader['base_release'])
    if m.get('parent_F0_identity') != original:
        raise ValueError('Clean source roles and reference must share the original F0 ancestor')
    return m


def _contract(c):
    d = _read(c['source_config']['source_contract'])
    if (d.get('classes') != source.CLASSES or d.get('source_rxs') != list(RECEIVERS)
            or d.get('source_days') != [1, 2, 3] or d.get('split_seed') != 392005
            or d.get('counts') != dict(L_s=COUNTS['L'], U_s=COUNTS['Us'], V=COUNTS['V'])
            or d.get('equalized', 1) != 1 or d.get('out_len', 256) != 256
            or d.get('normalize', True) is not True or not isinstance(d.get('role_ids'), dict)):
        raise ValueError('Original eq1/once-RMS source392005 roles required')
    return d


def _metadata(ids, meta, n, labelled=False):
    fields = META_FIELDS | ({'y'} if labelled else set())
    if (not isinstance(ids, np.ndarray) or ids.shape != (n,) or ids.dtype.kind != 'U'
            or len(set(ids.tolist())) != n or np.any(np.char.strip(ids) == '') or set(meta) != fields
            or not np.array_equal(meta['ids'], ids)):
        raise ValueError('Exact role-blind/public metadata and unique aligned IDs required')
    for k in fields-{'ids'}:
        if meta[k].shape != (n,) or meta[k].dtype != np.int64:
            raise ValueError('Public metadata must retain aligned integer fields: '+k)
    rx, day = meta['rx_i'], meta['day_i']
    if (not np.isin(rx, RECEIVERS).all() or not np.isin(day, (1, 2, 3)).all()
            or not np.array_equal(meta['domain'], np.searchsorted(RECEIVERS, rx)*3+day-1)
            or (labelled and not np.isin(meta['y'], range(6)).all())):
        raise ValueError('Only the original source RX/day and visible known labels are permitted')


def _split_validation(ids, y, rx, day, seed):
    n = 6*len(RECEIVERS)*3*CELL_SIZE
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != n
            or any(a.shape != (n,) or a.dtype != np.int64 for a in (y, rx, day))
            or CELL_SIZE % 2 or not np.isin(y, range(6)).all()
            or not np.isin(rx, RECEIVERS).all() or not np.isin(day, (1, 2, 3)).all()):
        raise ValueError('Complete ninety-cell source V with an even fixed cell size required')
    rng = np.random.default_rng(seed); cal, test = [], []
    for cls in range(6):
        for receiver in RECEIVERS:
            for date in (1, 2, 3):
                rows = np.flatnonzero((y == cls) & (rx == receiver) & (day == date))
                if len(rows) != CELL_SIZE:
                    raise ValueError('Every known TX/RX/day V cell must retain exactly its original count')
                rows = rows[np.argsort(ids[rows], kind='stable')]
                rows = rows[rng.permutation(CELL_SIZE)]
                cal.extend(rows[:CELL_SIZE//2]); test.extend(rows[CELL_SIZE//2:])
    return np.sort(np.asarray(cal, np.int64)), np.sort(np.asarray(test, np.int64))


def _clean_auxiliary(c, role):
    if role == 'OE':
        handle = diverse._diverse_handle(c['diverse_data_root'])
        iq, meta = diverse._public(c['diverse_data_root'])
    elif role == 'novel':
        handle = auxiliary._oe_handles(c['oe_data_root'])['novel_validation']
        iq, meta = auxiliary._oe_public(c['oe_data_root'], 'novel_validation')
    else:
        raise ValueError('Only existing clean OE/novel public data may be opened')
    return iq, meta, handle


def _array(x, n, shape=(2, 256)):
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.float32 or tuple(x.shape) != (n,)+shape or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite exact FP32 source tensors required')
        x = np.asarray(x.detach().cpu().tolist(), np.float32)
    else:
        x = np.asarray(x)
    if x.dtype != np.float32 or x.shape != (n,)+shape or not np.isfinite(x).all():
        raise ValueError('Finite exact FP32 source IQ/logits required')
    return x


def _compute(b, values, device, logits):
    x = _array(values, len(values))
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        restored = b['dura'][1](torch.tensor(x.tolist(), dtype=torch.float32, device=device))[0].detach().clone()
        result = _array(restored, len(x))
        teacher = b['reader'].classify_features(b['reader'].features(restored)) if logits else None
    return result, None if teacher is None else _array(teacher, len(x), (6,))


def _smoke(b, device):
    n = np.arange(256)
    clean = np.stack([np.stack((np.cos(w*n), np.sin(w*n))) for w in (.07, .11)]).astype(np.float32)
    restored, logits = _compute(b, clean, device, True)
    if restored.shape != (2, 2, 256) or logits.shape != (2, 6):
        raise ValueError('Actual original F0/DURA synthetic clean smoke failed')
    print('GROUND_OPEN_ACTUAL_F0_DURA_CLEAN_SMOKE_PASS', flush=True)
    return dict(status='PASS', actual_F0_DURA_checkpoints=True, synthetic_G_forward_rows=2,
                synthetic_F_forward_rows=2, synthetic_input='clean_P0_to_DURA', real_source_reads=0)


def _restore_all(b, values, c, device, teacher, name):
    n = len(values); iq = np.empty((n, 2, 256), np.float32)
    logits = np.empty((n, 6), np.float32) if teacher else None
    for start in range(0, n, c['batch_size']):
        end = min(start+c['batch_size'], n)
        iq[start:end], current = _compute(b, values[start:end], device, teacher)
        if teacher: logits[start:end] = current
        if start == 0 or end == n or start % 4096 == 0:
            print('GROUND_OPEN_DATA_PROGRESS', name, end, n, flush=True)
    return iq, logits


def _save(path, value):
    with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)


def _npz(path, values):
    with path.open('xb') as stream: np.savez(stream, **values)


def _identity(pm, b):
    return dict(parent_F0_identity=pm['parent_F0_identity'], dura_identity=pm['dura_identity'],
                loaded_dura_identity=b['dura_identity'])


def _export(folder, name, iq, ids, metadata, c, identities, teacher=None, baseline=None):
    labelled = name in ('L', 'CAL'); _metadata(ids, metadata, len(iq), labelled)
    _array(iq, len(ids)); folder.mkdir(parents=True, exist_ok=False)
    _save(folder/'iq.npy', iq); _save(folder/'ids.npy', ids); _npz(folder/'metadata.npz', metadata)
    files = {'iq.npy', 'ids.npy', 'metadata.npz', 'manifest.json'}
    if teacher is not None:
        _save(folder/'teacher_logits.npy', _array(teacher, len(ids), (6,))); files.add('teacher_logits.npy')
    if baseline is not None:
        _save(folder/'baseline_logits.npy', _array(baseline, len(ids), (6,))); files.add('baseline_logits.npy')
    m = dict(schema=SCHEMA, status=ROLE_STATUS if name in ROLES else QUERY_STATUS, run_id=RUN_ID,
             code_commit=c['code_commit'], role=name if name in ROLES else 'source_query',
             count=len(ids), iq_shape=list(iq.shape), precision='FP32', input_path='G(clean)' if name != CONDITIONS[1] else 'G(G10_received)',
             phase3_data_status='VALIDATED_ONCE', capsule_id=CAPSULE_ID, split_id=SPLIT_ID,
             source_only=True, metadata_fields=sorted(metadata), files=sorted(files), **identities)
    if name in CONDITIONS: m['condition'] = name
    _write(folder/'manifest.json', m)
    return m


def build(c, device='cuda:0'):
    validate_config(c); out = Path(c['output_root'])
    if out.exists(): raise FileExistsError('Ground data outputs are immutable; preserve existing output')
    pm = _reference(c); b = fresh._backend(c['reference_config'], device)
    if (b['reader_identity'] != pm['parent_F0_identity'] or b['dura_identity'] != pm['loaded_dura_identity']
            or b['dura_identity'].get('checkpoint') != c['source_config']['dura_checkpoint']
            or b['dura_identity'].get('release_root') != c['source_config']['dura_release']):
        raise ValueError('Original source F0 and fixed DURA input ancestors differ')
    models = [b['reader'], b['dura'][0]] + ([b['model']] if 'model' in b else [])
    if any(p.requires_grad or p.dtype != torch.float32 for model in models for p in model.parameters()) or any(m.training for model in models for m in model.modules()):
        raise ValueError('All original models must remain frozen FP32 eval')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    snapshots = source._snapshot(b)
    if 'model' in b: snapshots += source._snapshot(dict(reader=b['model'], dura=None))
    smoke = _smoke(b, device); source._check_models(snapshots)
    if smoke.get('status') != 'PASS': raise ValueError('Actual clean checkpoint smoke must pass first')
    out.mkdir(parents=True, exist_ok=False); _write(out/'smoke.json', smoke)
    contract = _contract(c); identities = _identity(pm, b); roles = {}; handles = {}; seen = set()
    counters = dict(G=0, F=0)
    def claim(ids):
        if seen & set(ids.tolist()): raise ValueError('Original source physical roles must remain disjoint')
        seen.update(ids.tolist())
    for role in ('L', 'Us', 'OE'):
        if role == 'OE':
            values, meta, handles[role] = _clean_auxiliary(c, role)
        else:
            dataset = SourceDataset(c['source_config']['source_cache'], 'L_s' if role == 'L' else 'U_s', contract)
            values = dataset.x; meta = dataset.metadata
        ids = meta['ids']; public = {k:meta[k] for k in META_FIELDS | ({'y'} if role == 'L' else set())}
        _metadata(ids, public, COUNTS[role], role == 'L'); claim(ids)
        try:
            restored, teacher = _restore_all(b, values, c, device, role == 'L', role)
            roles[role] = _export(out/'roles'/role, role, restored, ids, public, c, identities, teacher)
            counters['G'] += len(ids); counters['F'] += len(ids) if role == 'L' else 0
        finally:
            close(dict(iq=values))
        del values, restored, teacher
        if role != 'OE': del dataset
    dataset = SourceDataset(c['source_config']['source_cache'], 'V', contract)
    meta = dataset.metadata; ids = meta['ids']
    public = {k:meta[k] for k in META_FIELDS | {'y'}}
    _metadata(ids, public, COUNTS['V'], True); claim(ids)
    cal, test = _split_validation(ids, meta['y'], meta['rx_i'], meta['day_i'], c['split_seed'])
    v_iq, v_logits = _restore_all(b, dataset.x, c, device, True, 'V')
    counters['G'] += len(ids); counters['F'] += len(ids)
    cal_meta = {k:v[cal] for k, v in public.items()}
    roles['CAL'] = _export(out/'roles/CAL', 'CAL', v_iq[cal], ids[cal], cal_meta, c, identities, v_logits[cal])
    old_v = fresh._source_rows(c['reference_config']['output_root'], 'source_DURA', 'known')
    star_v = fresh._iq_role(c['reference_config'], 'V', pm)
    try:
        if (not np.array_equal(old_v['ids'], ids) or not np.array_equal(star_v['ids'], ids)
                or any(not np.array_equal(old_v[k], meta[k]) for k in ('y', 'rx_i', 'day_i'))
                or any(not np.array_equal(star_v['metadata'][k], meta[k]) for k in ('y', 'rx_i', 'day_i', 'domain'))):
            raise ValueError('Original clean/star V roles must retain identical source labels and ID order')
        star_known = _array(star_v['restored'][test], len(test)).copy()
    finally: close(star_v)
    (out/'private').mkdir()
    _npz(out/'private/known_test_truth.npz', dict(ids=ids[test], y=meta['y'][test], rx_i=meta['rx_i'][test],
         day_i=meta['day_i'][test], star_scenario=old_v['scenario'][test]))
    query_meta = {k:meta[k][test] for k in META_FIELDS}
    clean_known, clean_logits = v_iq[test], v_logits[test]
    old_known_logits = _array(old_v['default_logits'][test], len(test), (6,))
    del dataset, v_iq, v_logits, star_v, old_v
    nv_values, nv_meta, handles['novel'] = _clean_auxiliary(c, 'novel')
    nv_ids = nv_meta['ids']; nv_public = {k:nv_meta[k] for k in META_FIELDS}
    _metadata(nv_ids, nv_public, COUNTS['novel']); claim(nv_ids)
    try: nv_iq, nv_logits = _restore_all(b, nv_values, c, device, True, 'novel')
    finally: close(dict(iq=nv_values))
    counters['G'] += len(nv_ids); counters['F'] += len(nv_ids)
    old_nv = fresh._source_rows(c['reference_config']['output_root'], 'source_DURA', 'novel')
    star_nv = fresh._iq_role(c['reference_config'], 'novel', pm)
    try:
        if (not np.array_equal(old_nv['ids'], nv_ids) or not np.array_equal(star_nv['ids'], nv_ids)
                or any(not np.array_equal(star_nv['metadata'][k], nv_public[k]) for k in META_FIELDS-{'ids'})):
            raise ValueError('Clean/star heldout source novel IDs and public domains differ')
        star_novel = _array(star_nv['restored'], len(nv_ids)).copy()
    finally: close(star_nv)
    q_meta = {k:np.concatenate((query_meta[k], nv_public[k])) for k in META_FIELDS}
    permutation = np.random.default_rng(c['split_seed']+1).permutation(len(test)+len(nv_ids))
    q_meta = {k:v[permutation] for k, v in q_meta.items()}; q_ids = q_meta['ids']
    queries = {}
    for name, iq, logits in (
            (CONDITIONS[0], np.concatenate((clean_known,nv_iq)), np.concatenate((clean_logits,nv_logits))),
            (CONDITIONS[1], np.concatenate((star_known,star_novel)), np.concatenate((old_known_logits,_array(old_nv['default_logits'],len(nv_ids),(6,)))))):
        queries[name] = _export(out/'queries'/name, name, iq[permutation], q_ids, q_meta, c, identities, baseline=logits[permutation])
    source._check_models(snapshots)
    expected = dict(G=sum(COUNTS.values()), F=COUNTS['L']+COUNTS['V']+COUNTS['novel'])
    if counters != expected: raise ValueError('Actual clean DURA/F0 compute exposure differs')
    counts = {k:(COUNTS['V']//2 if k == 'CAL' else COUNTS[k]) for k in ROLES}
    counts.update(V=COUNTS['V'], known_TEST=len(test), novel=COUNTS['novel'], source_Q=len(q_ids))
    done = dict(schema=SCHEMA, status=STATUS, run_id=RUN_ID, code_commit=c['code_commit'], config=c,
        phase3_data_status='VALIDATED_ONCE', capsule_id=CAPSULE_ID, split_id=SPLIT_ID,
        source_only=True, counts=counts, roles=roles, queries=queries, **identities,
        source_config=c['source_config'], original_source_contract_handle={k:contract[k] for k in ('classes','source_rxs','source_days','split_seed','counts')},
        clean_auxiliary_handles=handles, input_processing='existing_eq1_FP32_once_unit_RMS_no_new_channel',
        role_input_path='G(clean)', source_star_input_path='original_fixed_G10_P1', second_RMS_applied=False,
        source_roles_disjoint=True, V_calibration_test_disjoint=True, paired_query_ID_order_equal=True,
        known_TEST_labels_public=False, query_known_unknown_roles_public=False,
        known_TEST_truth_ref='private/known_test_truth.npz', original_models_unchanged=True,
        model_parameters_buffers_unchanged=True, teacher_input_path='G(clean)',
        real_F_forward_rows=counters['F'], G_forward_rows=counters['G'], real_signal_forward_rows=0,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'], synthetic_G_forward_rows=smoke['synthetic_G_forward_rows'],
        source_clean_IQ_reads=sum(COUNTS.values()), source_star_cached_IQ_reads=len(test)+COUNTS['novel'],
        channel_observations_generated=0, optimizer_updates=0, new_independent_physical_records=0,
        target_U_reads=0, target_Q_reads=0, target_reads=0, target_truth_reads=0, formal_target_reads=0,
        private_truth_reads=0, Us_labels_read=0, OE_TX_labels_read=0, novel_TX_labels_read=0,
        final_phase3_complete=False)
    _write(out/'complete.json', done)
    return done


def _load(root, name, query):
    if name not in (CONDITIONS if query else ROLES): raise ValueError('Only declared source roles/queries may be loaded')
    folder = Path(root)/('queries' if query else 'roles')/name
    m = _read(folder/'manifest.json'); n = COUNTS['V']//2+COUNTS['novel'] if query else COUNTS['V']//2 if name == 'CAL' else COUNTS[name]
    labelled = name in ('L', 'CAL'); fields = META_FIELDS | ({'y'} if labelled else set())
    files = {'manifest.json', 'ids.npy', 'iq.npy', 'metadata.npz'}
    if labelled: files.add('teacher_logits.npy')
    if query: files.add('baseline_logits.npy')
    expected = dict(schema=SCHEMA, status=QUERY_STATUS if query else ROLE_STATUS, run_id=RUN_ID,
        role='source_query' if query else name, count=n, iq_shape=[n,2,256], precision='FP32',
        input_path='G(G10_received)' if name == CONDITIONS[1] else 'G(clean)',
        phase3_data_status='VALIDATED_ONCE', capsule_id=CAPSULE_ID, split_id=SPLIT_ID, source_only=True,
        metadata_fields=sorted(fields), files=sorted(files))
    if (any(m.get(k) != v for k,v in expected.items()) or (query and m.get('condition') != name)
            or re.fullmatch('[0-9a-f]{40}', str(m.get('code_commit'))) is None
            or any(not isinstance(m.get(k),dict) or not m[k] for k in ('parent_F0_identity','dura_identity','loaded_dura_identity'))
            or {p.name for p in folder.iterdir()} != files):
        raise ValueError('Complete source-clean role manifest/file contract differs')
    ids = np.load(folder/'ids.npy', allow_pickle=False)
    with np.load(folder/'metadata.npz', allow_pickle=False) as a:
        if set(a.files) != fields: raise ValueError('Labels/role/private fields must not leak into public metadata')
        meta = {k:a[k].copy() for k in a.files}
    _metadata(ids, meta, n, labelled); result = dict(manifest=m, ids=ids, metadata=meta)
    try:
        for key, shape in [('iq',(n,2,256))] + ([('teacher_logits',(n,6))] if labelled else []) + ([('baseline_logits',(n,6))] if query else []):
            value = np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False); result[key] = value
            if not isinstance(value,np.memmap) or value.shape != shape or value.dtype != np.float32:
                raise ValueError('Exact readonly FP32 source arrays required')
    except BaseException:
        close(result); raise
    return result


def load_role(root, role): return _load(root, role, False)


def load_query(root, condition): return _load(root, condition, True)


def close(bundle):
    values = bundle.values() if isinstance(bundle, dict) else (bundle,)
    for value in values:
        if isinstance(value, np.memmap) and not value._mmap.closed: value._mmap.close()


def main(argv=None):
    print(json.dumps(dict(event='GROUND_OPEN_DATA_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a = p.parse_args(argv);done = build(_read(a.config), a.device);print(done['status'],flush=True)


if __name__ == '__main__': main()
