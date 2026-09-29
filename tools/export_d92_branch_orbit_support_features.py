"""Four-phase native branch features from registered support IQ only.

Every received physical observation is read once; each view is forwarded
separately through the unchanged Phase1 model. No query IQ or fitted state.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from export_d92_branch_support_features import (
    SupportIQ, support_plan, FrozenBranches, load_native, historical_fft96,
    BRANCHES, check, read, write, strings, peak_process_rss,
)

CACHE_SCHEMA = 'd92_branch_orbit_support_features_v1'
CACHE_NAME = 'support_branch_orbit_features.npz'
FEATURE_CONTRACT = dict(input_shape=[2,256], view='four_full_length_quarter_phase_rotations',
    view_count=4, phase_degrees=[0,90,180,270], identity_feature_key='feat_joint',
    branch_keys=list(BRANCHES), branch_dim=160, fft_dim=96,
    fft='historical_spectral_logmag_sketch', fft_input='original_received_observation',
    fft_norm_floor=1e-8, cache_dtype='float32',
    normalization='raw_native_aux_no_additional_branch_normalization',
    native_forward_scope='one_physical_one_view_per_call')
MATRIX_KEYS = ('receivers','scenarios','ks','new_counts','support_seeds')


def make_received_views(iq):
    """Pure mathematical views; original observation is exactly view zero."""
    iq = np.asarray(iq)
    check(iq.dtype == np.float32 and iq.ndim == 3 and iq.shape[1:] == (2,256)
          and np.isfinite(iq).all(), 'Expected finite float32 support IQ')
    views = np.empty((len(iq),4,2,256), dtype=np.float32)
    views[:,0] = iq
    views[:,1,0], views[:,1,1] = -iq[:,1], iq[:,0]
    views[:,2] = -iq
    views[:,3,0], views[:,3,1] = iq[:,1], -iq[:,0]
    return views


def transform_support(iq, infer):
    views = make_received_views(iq)
    arrays = {key: np.empty((len(iq),4,160), dtype=np.float32) for key in BRANCHES}
    for physical in range(len(iq)):
        for view in range(4):
            values = infer(views[physical:physical+1, view])
            check(set(values) == set(BRANCHES), 'Unexpected native branch members')
            for key in BRANCHES:
                value = values[key]
                check(value.dtype == np.float32 and value.shape == (1,160)
                      and np.isfinite(value).all(), 'Invalid native branch: '+key)
                arrays[key][physical,view] = value[0]
    return arrays


def load_support(*, support_features, capsule, expected_capsule_id, expected_checkpoint_sha256, expected_model_seed, config):
    check(set(config) == {'algorithm', 'matrix'} and isinstance(config['algorithm'], dict), 'Frozen probe configuration mismatch')
    matrix = config['matrix']
    check(set(matrix) == set(MATRIX_KEYS), 'Explicit complete matrix required')
    for key in MATRIX_KEYS:
        values = matrix[key]
        check(isinstance(values, list) and bool(values) and len(set(values)) == len(values), 'Invalid matrix axis: '+key)
        check(all(isinstance(v, str) and v for v in values) if key in ('receivers', 'scenarios') else
              all(type(v) is int and v >= (1 if key == 'ks' else 0) for v in values), 'Invalid matrix values: '+key)
    expected_cells = set(itertools.product(*(matrix[k] for k in MATRIX_KEYS)))
    manifest = read(Path(capsule) / 'manifest.json')
    check(manifest.get('protocol_schema') == 'p2_min_v1' and manifest.get('phase2_data_status') == 'VALIDATED_ONCE'
          and manifest.get('capsule_id') == expected_capsule_id and manifest.get('split_count') == len(expected_cells), 'Capsule manifest/matrix mismatch')
    root = Path(support_features)
    marker, startup, provenance, plan = [read(root / name) for name in
        ('features_complete.json', 'startup.json', 'checkpoint_provenance.json', 'support_splits.json')]
    for record in (marker, startup, provenance, plan):
        check(record.get('checkpoint_sha256') == expected_checkpoint_sha256, 'Checkpoint binding mismatch')
    for record in (marker, startup, plan):
        check(record.get('capsule_id') == expected_capsule_id and record.get('schema') == CACHE_SCHEMA, 'Support cache capsule/schema mismatch')
    old = marker.get('classes')
    check(strings(old) and provenance.get('classes') == old and startup.get('provenance') == provenance,
          'Checkpoint class/provenance mismatch')
    check(all(v.get('model_seed') == expected_model_seed for v in (marker, startup, provenance))
          and provenance.get('verdict') == 'MATCHED_SOURCE_ONLY_SCRATCH' and provenance.get('source_role_comparison') == 'EXACT_MATCH'
          and provenance.get('checkpoint_epoch') == 200 and provenance.get('checkpoint_inheritance') == []
          and provenance.get('target_access_before_freeze') is False, 'Checkpoint source-only provenance mismatch')
    check(marker.get('status') == 'BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE' and marker.get('feature_contract') == FEATURE_CONTRACT
          and startup.get('feature_contract') == FEATURE_CONTRACT and marker.get('dtype') == 'float32'
          and marker.get('view_count_per_observation') == 4, 'Support feature formula mismatch')
    for record in (marker, startup):
        check(all(record.get(k) is False for k in ('query_iq_access', 'source_data_access', 'truth_read', 'adapted_state_inherited'))
              and record.get('query_rows_read') == 0 and record.get('native_eval') is True
              and record.get('new_source_payload_bytes') == 0 and record.get('new_ground_statistics_bytes') == 0,
              'Forbidden feature producer access/state')
    check(marker.get('query_used_for_fitting') is False and startup.get('query_fit_access') is False
          and marker.get('encoder_updated') is False and marker.get('native_parameters_unchanged') is True
          and marker.get('native_buffers_unchanged') is True, 'Frozen encoder mismatch')
    check(type(marker.get('model_file_bytes')) is int and marker['model_file_bytes'] > 0
          and provenance.get('model_file_bytes') == startup.get('model_file_bytes') == marker['model_file_bytes'], 'Model package bytes mismatch')
    path = root / CACHE_NAME
    with np.load(path, allow_pickle=False) as cache:
        check(set(cache.files) == set(BRANCHES) | {'fft'} | {'ids', 'indices', 'labels', 'checkpoint_sha256', 'capsule_id', 'feature_contract_json'}, 'Unexpected support cache members')
        check(cache['checkpoint_sha256'].item() == expected_checkpoint_sha256 and cache['capsule_id'].item() == expected_capsule_id
              and json.loads(cache['feature_contract_json'].item()) == FEATURE_CONTRACT, 'NPZ binding mismatch')
        ids, indices, labels = cache['ids'], cache['indices'], cache['labels']
        arrays = {k: cache[k] for k in (*BRANCHES, 'fft')}
    check(ids.ndim == 1 and ids.dtype.kind in 'SU' and strings(ids.astype(str).tolist())
          and ids.astype(str).tolist() == sorted(ids.astype(str).tolist()), 'Invalid physical ID registry')
    count = len(ids)
    check(indices.shape == (count,) and indices.dtype == np.int64 and len(set(indices.tolist())) == count and np.all(indices >= 0)
          and labels.shape == (count,) and labels.dtype.kind in 'SU' and all(labels.astype(str)), 'Invalid physical indices/labels')
    for key, value in arrays.items():
        check(value.dtype == np.float32 and value.shape == ((count,96) if key == 'fft' else (count,4,160)) and np.isfinite(value).all(), 'Invalid feature array: '+key)
    check(marker.get('count') == count and marker.get('feature_array_bytes') == sum(v.nbytes for v in arrays.values())
          and marker.get('index_bytes') == indices.nbytes and marker.get('registry_array_bytes') == ids.nbytes + labels.nbytes
          and marker.get('feature_file_bytes') == path.stat().st_size and marker.get('shapes') == {k: list(v.shape) for k, v in arrays.items()}
          and marker.get('support_iq_rows_read') == marker.get('support_physical_observation_count') == count
          and marker.get('native_physical_forward_count') == marker.get('native_view_forward_count') == marker.get('identity_reference_checks') == marker.get('native_batch_calls') == 4*count
          and marker.get('native_batch_size') == startup.get('native_batch_size') == 1
          and marker.get('smoke_forward_count') == marker.get('smoke_batch_calls') == 4
          and marker.get('native_total_physical_forward_count') == marker.get('native_total_batch_calls') == 4*count+4
          and marker.get('synthetic_smoke', {}).get('status') == 'PASS'
          and marker['synthetic_smoke'].get('query_rows_read') == 0
          and marker['synthetic_smoke'].get('frozen_state_unchanged') is True
          and marker.get('identity_check_additional_encoder_forwards') == 0, 'Support cache counts/bytes mismatch')
    check(set(plan) == {'schema', 'capsule_id', 'checkpoint_sha256', 'splits'} and isinstance(plan['splits'], list), 'Unexpected support plan')
    splits = plan['splits']; check(len(splits) == len(expected_cells) == marker.get('split_count'), 'Partial support matrix')
    lookup = {str(v): i for i, v in enumerate(ids)}; seen_ids = set(); seen_cells = set(); used = set(); tasks = []
    allowed = {'split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes', 'support_indices', 'support_ids', 'support_labels'}
    for split in splits:
        check(set(split) == allowed and isinstance(split['split_id'], str) and split['split_id'] and split['split_id'] not in seen_ids, 'Unexpected/duplicate support split')
        seen_ids.add(split['split_id']); classes = split['registered_classes']; k = split['k']
        check(strings(classes) and classes[:len(old)] == old and type(k) is int and k > 0, 'Registered classes/K mismatch')
        cell = (split['receiver'], split['scenario'], k, len(classes)-len(old), split['support_seed'])
        check(cell in expected_cells and cell not in seen_cells, 'Unexpected/duplicate matrix cell'); seen_cells.add(cell)
        support_ids = split['support_ids']; support_indices = split['support_indices']; y = split['support_labels']
        check(strings(support_ids) and len(support_ids) == len(classes)*k and isinstance(y, list) and len(y) == len(support_ids)
              and isinstance(support_indices, list) and len(support_indices) == len(support_ids)
              and all(type(v) is int for v in y + support_indices) and set(y) == set(range(len(classes)))
              and all(y.count(i) == k for i in range(len(classes))) and all(v in lookup for v in support_ids), 'Invalid task physical support')
        positions = np.asarray([lookup[v] for v in support_ids], dtype=np.int64)
        check(indices[positions].tolist() == support_indices and labels[positions].astype(str).tolist() == [classes[v] for v in y], 'Physical index/ID/class mismatch')
        used.update(support_ids); tasks.append((split, positions, np.asarray(y, dtype=np.int64)))
    check(seen_cells == expected_cells and used == set(lookup), 'Incomplete matrix or extra cached physical rows')
    arrays = {k: np.frombuffer(v.tobytes(), dtype=np.float32).reshape(v.shape) for k, v in arrays.items()}
    return arrays, tasks, old, marker, startup, provenance



def export(*, checkpoint_root, native_code, source_contract, source_receivers, seed, capsule, output,
           expected_checkpoint_sha256, expected_capsule_id, device='cpu', batch_size=32):
    out, capsule = Path(output), Path(capsule)
    if out.exists(): raise FileExistsError(out)
    check(type(batch_size) is int and batch_size > 0, 'Positive batch size required')
    check(isinstance(expected_checkpoint_sha256, str) and len(expected_checkpoint_sha256) == 64
          and all(c in '0123456789abcdef' for c in expected_checkpoint_sha256), 'Invalid expected checkpoint SHA')
    started = time.perf_counter(); reader = SupportIQ(capsule/'received.npz'); owns_output = False
    try:
        manifest, splits, indices, labels = support_plan(capsule, reader.ids, expected_capsule_id, source_receivers)
        reader.allow(indices)
        timing = dict(metadata_seconds=time.perf_counter()-started)
        stage = time.perf_counter()
        infer, provenance = load_native(checkpoint_root=checkpoint_root, native_code=native_code, source_contract=source_contract,
            seed=seed, expected_checkpoint_sha256=expected_checkpoint_sha256, device=device)
        timing['checkpoint_load_seconds'] = time.perf_counter()-stage
        old = provenance['classes']
        check(all(split['registered_classes'][:len(old)] == old for split in splits), 'Frozen old class registry mismatch')
        out.mkdir(parents=True, exist_ok=False)
        owns_output = True
        startup = dict(schema=CACHE_SCHEMA, argv=sys.argv, python=sys.executable, pid=os.getpid(),
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed,
            feature_contract=FEATURE_CONTRACT, device=device, batch_size=batch_size, native_batch_size=1, source_receivers=source_receivers,
            provenance=provenance, query_iq_access=False, query_rows_read=0, query_fit_access=False, truth_read=False,
            source_data_access=False, adapted_state_inherited=False, native_eval=True,
            new_source_payload_bytes=0, new_ground_statistics_bytes=0, model_already_deployed=None,
            model_incremental_transfer_bytes=None, model_deployment_unknown_reason='Deployment status unknown',
            model_file_bytes=provenance['model_file_bytes'], model_file_bytes_scope='Full training checkpoint package, not a minimal inference package')
        write(out/'startup.json', startup); print(json.dumps(dict(event='STARTUP', **startup)), flush=True)
        stage = time.perf_counter()
        synthetic = np.random.Generator(np.random.PCG64(0)).standard_normal((1,2,256)).astype(np.float32)
        smoke_values = transform_support(synthetic, infer)
        check(all(v.shape == (1,4,160) and np.isfinite(v).all() for v in smoke_values.values()), 'Synthetic orbit smoke failed')
        check(historical_fft96(synthetic).shape == (1,96), 'Synthetic FFT smoke failed')
        infer.verify_frozen()
        timing['synthetic_smoke_seconds'] = time.perf_counter()-stage
        smoke = dict(status='PASS', input='synthetic_PCG64_seed0', physical_observation_count=1,
            view_forward_count=4, native_batch_calls=4, query_rows_read=0, frozen_state_unchanged=True,
            seconds=timing['synthetic_smoke_seconds'])
        print(json.dumps(dict(event='NATIVE_SYNTHETIC_ORBIT_SMOKE', **smoke)), flush=True)
        counts_before_support = (infer.physical_forward_count, infer.batch_calls, infer.identity_reference_checks)
        arrays = {key: np.empty((len(indices), 4, 160), dtype=np.float32) for key in BRANCHES}
        arrays['fft'] = np.empty((len(indices), 96), dtype=np.float32)
        timing.update(support_read_seconds=0., native_forward_seconds=0., fft_seconds=0., frozen_verify_seconds=0.)
        for offset in range(0, len(indices), batch_size):
            stop = min(offset+batch_size, len(indices)); stage = time.perf_counter()
            iq = reader.take(indices[offset:stop]); timing['support_read_seconds'] += time.perf_counter()-stage
            stage = time.perf_counter(); branches = transform_support(iq, infer); timing['native_forward_seconds'] += time.perf_counter()-stage
            for key in BRANCHES: arrays[key][offset:stop] = branches[key]
            stage = time.perf_counter(); arrays['fft'][offset:stop] = historical_fft96(iq); timing['fft_seconds'] += time.perf_counter()-stage
            stage = time.perf_counter(); infer.verify_frozen(); timing['frozen_verify_seconds'] += time.perf_counter()-stage
            print(json.dumps(dict(event='SUPPORT_FEATURE_PROGRESS', completed=stop, total=len(indices),
                native_batch_calls=infer.batch_calls, elapsed_seconds=time.perf_counter()-started)), flush=True)
        stage = time.perf_counter()
        indices_array = np.asarray(indices, dtype=np.int64); ids = reader.ids[indices]; class_ids = np.asarray(labels)
        np.savez(out/CACHE_NAME, **arrays, ids=ids, labels=class_ids, indices=indices_array,
            checkpoint_sha256=np.asarray(expected_checkpoint_sha256), capsule_id=np.asarray(expected_capsule_id),
            feature_contract_json=np.asarray(json.dumps(FEATURE_CONTRACT, sort_keys=True)))
        write(out/'support_splits.json', dict(schema=CACHE_SCHEMA, capsule_id=expected_capsule_id,
            checkpoint_sha256=expected_checkpoint_sha256, splits=splits))
        write(out/'checkpoint_provenance.json', provenance)
        timing['write_seconds'] = time.perf_counter()-stage; timing['total_seconds'] = time.perf_counter()-started
        metadata_bytes = {name: (out/name).stat().st_size for name in ('startup.json', 'support_splits.json', 'checkpoint_provenance.json')}
        marker = dict(status='BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE', schema=CACHE_SCHEMA, feature_contract=FEATURE_CONTRACT,
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed, classes=old,
            count=len(indices), split_count=len(splits), dtype='float32', shapes={k: list(v.shape) for k, v in arrays.items()},
            feature_array_bytes=sum(v.nbytes for v in arrays.values()), index_bytes=indices_array.nbytes,
            registry_array_bytes=ids.nbytes+class_ids.nbytes, feature_file_bytes=(out/CACHE_NAME).stat().st_size,
            metadata_file_bytes=metadata_bytes,
            artifact_file_bytes_excluding_completion_marker=(out/CACHE_NAME).stat().st_size+sum(metadata_bytes.values()),
            model_file_bytes=provenance['model_file_bytes'], view_count_per_observation=4,
            native_physical_forward_count=infer.physical_forward_count-counts_before_support[0],
            native_view_forward_count=infer.physical_forward_count-counts_before_support[0],
            native_batch_calls=infer.batch_calls-counts_before_support[1],
            identity_reference_checks=infer.identity_reference_checks-counts_before_support[2],
            identity_check_additional_encoder_forwards=0, native_batch_size=1,
            support_physical_observation_count=len(indices), smoke_forward_count=4, smoke_batch_calls=4,
            synthetic_smoke=smoke, native_total_physical_forward_count=infer.physical_forward_count,
            native_total_batch_calls=infer.batch_calls,
            support_iq_rows_read=reader.rows_read, iq_access='ZIP_STORED_NPY_readonly_mmap_support_indices_only',
            capsule_revalidated=False, received_file_rehashed=False, native_eval=True, native_parameters_unchanged=True,
            native_buffers_unchanged=True, encoder_updated=False, active_branches=infer.attributes,
            query_iq_access=False, query_rows_read=0, query_used_for_fitting=False, source_data_access=False, truth_read=False,
            adapted_state_inherited=False, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
            timing=timing, peak_process_rss_bytes=peak_process_rss(),
            peak_process_rss_reason='Linux process high-water RSS; null when unavailable on this platform',
            byte_scope='Support feature/index/registry arrays and cache file; received-side storage, not source transmission')
        write(out/'features_complete.json', marker); print(json.dumps(marker, allow_nan=False), flush=True)
        return marker
    except Exception as exc:
        if owns_output and not (out/'features_complete.json').exists() and not (out/'technical_failure.json').exists():
            write(out/'technical_failure.json', dict(status='TECHNICAL_FAILURE', error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint-root', 'native-code', 'source-contract', 'source-receivers', 'capsule', 'output',
                'expected-checkpoint-sha256', 'expected-capsule-id'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = vars(parser.parse_args()); args['source_receivers'] = json.loads(args['source_receivers']); export(**args)


if __name__ == '__main__': main()
