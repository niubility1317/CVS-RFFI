"""Export all received observations through one frozen, original-view forward.

Distinct from the support-only probe cache: this label-free cache permits query
inference but never fitting. Every native call contains exactly one observation.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from export_d92_branch_support_features import (BRANCHES, FEATURE_CONTRACT, SupportIQ, check,
    read, write, load_native, historical_fft96, peak_process_rss)
from export_d92_mv_kme_features import validate_capsule, validate_origin

ROOT = Path(__file__).resolve().parents[1]
CACHE_SCHEMA = 'd92_branch_received_features_v1'
CACHE_NAME = 'received_branch_features.npz'


def local_core():
    name = '_d92_branch_ridge_local'
    if name not in sys.modules:
        package = types.ModuleType(name); package.__path__ = [str(ROOT/'code/cvsrffi')]
        sys.modules[name] = package
    return importlib.import_module(name+'.d92_branch_ridge')


def export(*, row_root, source, contract, native_code, seed, capsule, output, config,
           expected_capsule_id, expected_checkpoint_sha256, device='cpu', batch_size=128):
    out, capsule = Path(output), Path(capsule)
    if out.exists(): raise FileExistsError(out)
    check(set(config) == {'algorithm'} and config['algorithm'] == local_core().FROZEN_CONFIG, 'Frozen BranchRidge config mismatch')
    check(type(batch_size) is int and batch_size > 0, 'Positive IO batch size required')
    started = time.perf_counter(); manifest = validate_capsule(capsule, expected_capsule_id)
    previous, origin = validate_origin(row_root, capsule, expected_checkpoint_sha256)
    check(previous['seed'] == seed, 'Model seed mismatch')
    # Reuse only the mmap storage implementation. This separately registered
    # label-free exporter explicitly authorizes all received rows for forward.
    reader = SupportIQ(capsule/'received.npz'); owns_output = False
    try:
        check(manifest['received_count'] == len(reader.ids) and manifest['signal_shape'] == [2, 256], 'Capsule shape/count mismatch')
        reader.allow(list(range(len(reader.ids))))
        timing = dict(metadata_seconds=time.perf_counter()-started); stage = time.perf_counter()
        infer, provenance = load_native(checkpoint_root=source, native_code=native_code, source_contract=contract,
            seed=seed, expected_checkpoint_sha256=expected_checkpoint_sha256, device=device)
        check(provenance['classes'] == origin['classes'], 'Source class mapping mismatch')
        timing['checkpoint_load_seconds'] = time.perf_counter()-stage
        out.mkdir(parents=True, exist_ok=False); owns_output = True
        startup = dict(schema=CACHE_SCHEMA, config=config, feature_contract=FEATURE_CONTRACT, argv=sys.argv,
            python=sys.executable, pid=os.getpid(), capsule_id=expected_capsule_id,
            checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed, device=device,
            provenance=provenance, query_fit_access=False, truth_read=False, source_data_access=False,
            adapted_state_inherited=False, encoder_updated=False, native_eval=True, native_batch_size=1,
            io_batch_size=batch_size, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
            model_already_deployed=None, model_incremental_transfer_bytes=None,
            model_deployment_unknown_reason='Deployment state unknown', model_file_bytes=provenance['model_file_bytes'],
            model_file_bytes_scope='Full training checkpoint package, not a minimal inference package')
        write(out/'startup.json', startup); print(json.dumps(dict(event='STARTUP', **startup)), flush=True)
        stage = time.perf_counter()
        synthetic = np.random.Generator(np.random.PCG64(0)).standard_normal((1, 2, 256)).astype(np.float32)
        synthetic_branches = infer(synthetic)
        synthetic_features = local_core()._features(**synthetic_branches, fft=historical_fft96(synthetic))
        check(synthetic_features.shape == (1, 736) and np.isfinite(synthetic_features).all(), 'Synthetic branch feature smoke failed')
        infer.verify_frozen()
        timing['synthetic_smoke_seconds'] = time.perf_counter()-stage
        smoke = dict(status='PASS', input='synthetic_PCG64_seed0', physical_forward_count=1,
            native_batch_calls=1, feature_shape=[1, 736], query_rows_read=0,
            seconds=timing['synthetic_smoke_seconds'], frozen_state_unchanged=True)
        print(json.dumps(dict(event='NATIVE_SYNTHETIC_BRANCH_SMOKE', **smoke)), flush=True)
        counts_before_received = (infer.physical_forward_count, infer.batch_calls, infer.identity_reference_checks)
        arrays = {key: np.empty((len(reader.ids), 160), dtype=np.float32) for key in BRANCHES}
        arrays['fft'] = np.empty((len(reader.ids), 96), dtype=np.float32)
        timing.update(received_read_seconds=0., native_forward_seconds=0., fft_seconds=0., frozen_verify_seconds=0.)
        for start in range(0, len(reader.ids), batch_size):
            stop = min(start+batch_size, len(reader.ids)); stage = time.perf_counter()
            iq = reader.take(list(range(start, stop))); timing['received_read_seconds'] += time.perf_counter()-stage
            for position in range(len(iq)):
                one = iq[position:position+1]; stage = time.perf_counter(); values = infer(one)
                timing['native_forward_seconds'] += time.perf_counter()-stage
                for key in BRANCHES: arrays[key][start+position] = values[key][0]
                stage = time.perf_counter(); arrays['fft'][start+position] = historical_fft96(one)[0]
                timing['fft_seconds'] += time.perf_counter()-stage
            stage = time.perf_counter(); infer.verify_frozen(); timing['frozen_verify_seconds'] += time.perf_counter()-stage
            print(json.dumps(dict(event='FEATURE_PROGRESS', completed=stop, total=len(reader.ids),
                native_batch_calls=infer.batch_calls, elapsed_seconds=time.perf_counter()-started)), flush=True)
        stage = time.perf_counter()
        np.savez(out/CACHE_NAME, **arrays, ids=reader.ids, checkpoint_sha256=np.asarray(expected_checkpoint_sha256),
            capsule_id=np.asarray(expected_capsule_id), feature_contract_json=np.asarray(json.dumps(FEATURE_CONTRACT, sort_keys=True)))
        write(out/'checkpoint_provenance.json', provenance)
        timing['write_seconds'] = time.perf_counter()-stage; timing['total_seconds'] = time.perf_counter()-started
        marker = dict(status='BRANCH_FEATURES_COMPLETE', schema=CACHE_SCHEMA, feature_contract=FEATURE_CONTRACT,
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, model_seed=seed,
            classes=provenance['classes'], count=len(reader.ids), dtype='float32', shapes={k:list(v.shape) for k,v in arrays.items()},
            feature_array_bytes=sum(v.nbytes for v in arrays.values()), registry_array_bytes=reader.ids.nbytes,
            feature_file_bytes=(out/CACHE_NAME).stat().st_size, model_file_bytes=provenance['model_file_bytes'],
            native_physical_forward_count=infer.physical_forward_count-counts_before_received[0],
            native_batch_calls=infer.batch_calls-counts_before_received[1],
            identity_reference_checks=infer.identity_reference_checks-counts_before_received[2],
            smoke_forward_count=1, smoke_batch_calls=1, synthetic_smoke=smoke,
            native_total_physical_forward_count=infer.physical_forward_count,
            native_total_batch_calls=infer.batch_calls, support_cache_reuse_count=0,
            identity_check_additional_encoder_forwards=0,
            native_batch_size=1, view_count_per_observation=1, query_used_for_fitting=False,
            source_data_access=False, truth_read=False, adapted_state_inherited=False, encoder_updated=False,
            native_eval=True, native_parameters_unchanged=True, native_buffers_unchanged=True,
            active_branches=infer.attributes, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
            capsule_revalidated=False, received_file_rehashed=False, timing=timing,
            peak_process_rss_bytes=peak_process_rss(), peak_process_rss_reason='Linux process high-water RSS; null when unavailable')
        write(out/'features_complete.json', marker); print(json.dumps(marker, allow_nan=False), flush=True)
        return marker
    except Exception as exc:
        if owns_output and not (out/'features_complete.json').exists():
            write(out/'technical_failure.json', dict(status='TECHNICAL_FAILURE', error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        reader.close()


def load_features(*, branch_features, capsule, row_root, expected_capsule_id, expected_checkpoint_sha256, config):
    """Pure immutable feature binding; never load IQ values, weights or fit state."""
    root = Path(branch_features); manifest = validate_capsule(capsule, expected_capsule_id)
    marker, startup, provenance = [read(root/name) for name in ('features_complete.json', 'startup.json', 'checkpoint_provenance.json')]
    previous, origin = validate_origin(row_root, capsule, expected_checkpoint_sha256)
    check(set(config) == {'algorithm'} and config['algorithm'] == local_core().FROZEN_CONFIG and startup['config'] == config,
          'Frozen BranchRidge feature config mismatch')
    for record in (marker, startup, provenance):
        check(record.get('checkpoint_sha256') == expected_checkpoint_sha256 and record.get('model_seed') == previous['seed'], 'Feature model binding mismatch')
    check(marker.get('status') == 'BRANCH_FEATURES_COMPLETE' and startup.get('provenance') == provenance
          and marker.get('classes') == provenance.get('classes') == origin['classes'], 'Feature class/provenance mismatch')
    check(provenance.get('verdict') == 'MATCHED_SOURCE_ONLY_SCRATCH' and provenance.get('source_role_comparison') == 'EXACT_MATCH'
          and provenance.get('checkpoint_epoch') == 200 and provenance.get('checkpoint_inheritance') == []
          and provenance.get('target_access_before_freeze') is False, 'Invalid source-only checkpoint provenance')
    for record in (marker, startup):
        check(record.get('schema') == CACHE_SCHEMA and record.get('capsule_id') == expected_capsule_id
              and record.get('feature_contract') == FEATURE_CONTRACT and record.get('native_batch_size') == 1
              and record.get('native_eval') is True and all(record.get(k) is False for k in
              ('truth_read', 'source_data_access', 'adapted_state_inherited', 'encoder_updated')), 'Full received feature schema/access mismatch')
    check(marker.get('query_used_for_fitting') is False and startup.get('query_fit_access') is False
          and marker.get('native_parameters_unchanged') is marker.get('native_buffers_unchanged') is True
          and marker.get('view_count_per_observation') == 1 and marker.get('dtype') == 'float32'
          and marker.get('new_source_payload_bytes') == marker.get('new_ground_statistics_bytes') == 0, 'Frozen feature boundary mismatch')
    with np.load(root/CACHE_NAME, allow_pickle=False) as cache:
        check(set(cache.files) == {*BRANCHES, 'fft', 'ids', 'checkpoint_sha256', 'capsule_id', 'feature_contract_json'}, 'Unexpected full cache members')
        check(cache['checkpoint_sha256'].item() == expected_checkpoint_sha256 and cache['capsule_id'].item() == expected_capsule_id
              and json.loads(cache['feature_contract_json'].item()) == FEATURE_CONTRACT, 'Feature NPZ binding mismatch')
        arrays = {key:cache[key] for key in (*BRANCHES, 'fft')}; ids = cache['ids']
    with np.load(Path(capsule)/'received.npz', allow_pickle=False) as cache:
        physical_ids = cache['ids']  # Only identifier member; never IQ.
    check(ids.ndim == 1 and ids.dtype.kind in 'SU' and len(set(ids.tolist())) == len(ids)
          and np.array_equal(ids, physical_ids) and len(ids) == manifest['received_count'], 'Full cache physical ID mismatch')
    check(all(v.dtype == np.float32 and v.shape == (len(ids), 96 if k == 'fft' else 160) and np.isfinite(v).all()
              for k,v in arrays.items()), 'Invalid branch feature shape/value')
    check(marker.get('count') == marker.get('native_physical_forward_count') == marker.get('native_batch_calls')
          == marker.get('identity_reference_checks') == len(ids) and marker.get('identity_check_additional_encoder_forwards') == 0
          and marker.get('shapes') == {k:list(v.shape) for k,v in arrays.items()}
          and marker.get('feature_array_bytes') == sum(v.nbytes for v in arrays.values())
          and marker.get('registry_array_bytes') == ids.nbytes and marker.get('feature_file_bytes') == (root/CACHE_NAME).stat().st_size
          and type(marker.get('model_file_bytes')) is int and marker['model_file_bytes'] > 0
          and marker['model_file_bytes'] == provenance.get('model_file_bytes') == startup.get('model_file_bytes'), 'Feature byte/count mismatch')
    check(marker.get('smoke_forward_count') == marker.get('smoke_batch_calls') == 1
          and marker.get('native_total_physical_forward_count') == marker.get('native_total_batch_calls') == len(ids)+1
          and marker.get('support_cache_reuse_count') == 0 and marker.get('synthetic_smoke', {}).get('status') == 'PASS'
          and marker['synthetic_smoke'].get('query_rows_read') == 0
          and marker['synthetic_smoke'].get('frozen_state_unchanged') is True, 'Synthetic smoke/count mismatch')
    arrays = {k:np.frombuffer(v.tobytes(), dtype=np.float32).reshape(v.shape) for k,v in arrays.items()}
    ids.setflags(write=False)
    return arrays, ids, marker, previous, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('row-root', 'source', 'contract', 'native-code', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128, help='IO grouping only; every native call remains singleton')
    args = vars(parser.parse_args()); args['config'] = read(args['config']); export(**args)


if __name__ == '__main__': main()
