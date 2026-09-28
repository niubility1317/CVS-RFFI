"""Evaluate frozen branch information on registered support only; no query API."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import numpy as np
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG, probe_branch_support
from export_d92_branch_support_features import CACHE_SCHEMA, CACHE_NAME, FEATURE_CONTRACT

SCOPE = 'SUPPORT_OOF_DIAGNOSTIC_NOT_QUERY_EVALUATION'
MATRIX_KEYS = ('receivers', 'scenarios', 'ks', 'new_counts', 'support_seeds')
BRANCHES = ('z_id', 't_emb', 'f_emb', 'pa_local', 'fft')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def strings(values):
    return isinstance(values, list) and bool(values) and all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values)


def scalars(value):
    """Keep nested scalar diagnostics, excluding physical records and arrays."""
    if not isinstance(value, dict):
        return value
    return {k: scalars(v) for k, v in value.items() if isinstance(v, dict) or v is None or isinstance(v, (str, int, float, bool))}


def csv_record(record):
    return {k: 'N/A' if v is None else json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in record.items()}


def peak_rss():
    if sys.platform.startswith('linux'):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return None


def load_support(*, support_features, capsule, expected_capsule_id, expected_checkpoint_sha256, expected_model_seed, config):
    check(set(config) == {'algorithm', 'matrix'} and config['algorithm'] == FROZEN_CONFIG, 'Frozen probe configuration mismatch')
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
    check(marker.get('status') == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and marker.get('feature_contract') == FEATURE_CONTRACT
          and startup.get('feature_contract') == FEATURE_CONTRACT and marker.get('dtype') == 'float32'
          and marker.get('view_count_per_observation') == 1, 'Support feature formula mismatch')
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
        check(set(cache.files) == set(BRANCHES) | {'ids', 'indices', 'labels', 'checkpoint_sha256', 'capsule_id', 'feature_contract_json'}, 'Unexpected support cache members')
        check(cache['checkpoint_sha256'].item() == expected_checkpoint_sha256 and cache['capsule_id'].item() == expected_capsule_id
              and json.loads(cache['feature_contract_json'].item()) == FEATURE_CONTRACT, 'NPZ binding mismatch')
        ids, indices, labels = cache['ids'], cache['indices'], cache['labels']
        arrays = {k: cache[k] for k in BRANCHES}
    check(ids.ndim == 1 and ids.dtype.kind in 'SU' and strings(ids.astype(str).tolist())
          and ids.astype(str).tolist() == sorted(ids.astype(str).tolist()), 'Invalid physical ID registry')
    count = len(ids)
    check(indices.shape == (count,) and indices.dtype == np.int64 and len(set(indices.tolist())) == count and np.all(indices >= 0)
          and labels.shape == (count,) and labels.dtype.kind in 'SU' and all(labels.astype(str)), 'Invalid physical indices/labels')
    for key, value in arrays.items():
        check(value.dtype == np.float32 and value.shape == (count, 96 if key == 'fft' else 160) and np.isfinite(value).all(), 'Invalid feature array: '+key)
    check(marker.get('count') == count and marker.get('feature_array_bytes') == sum(v.nbytes for v in arrays.values())
          and marker.get('index_bytes') == indices.nbytes and marker.get('registry_array_bytes') == ids.nbytes + labels.nbytes
          and marker.get('feature_file_bytes') == path.stat().st_size and marker.get('shapes') == {k: list(v.shape) for k, v in arrays.items()}
          and marker.get('native_physical_forward_count') == marker.get('support_iq_rows_read') == marker.get('identity_reference_checks') == count
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


def evaluate(*, support_features, capsule, output, config, expected_capsule_id, expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    arrays, tasks, old, producer, extraction, provenance = load_support(support_features=support_features, capsule=capsule,
        expected_capsule_id=expected_capsule_id, expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_seed=expected_model_seed, config=config)
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(model_file_bytes=producer['model_file_bytes'], model_file_bytes_scope='Full training checkpoint package; not a minimal inference package',
        model_already_deployed=None, model_incremental_transfer_bytes=None, model_deployment_unknown_reason='Deployment state unknown',
        new_source_payload_bytes=0, new_ground_statistics_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], feature_extraction_timing=producer['timing'],
        native_physical_forward_count=producer['native_physical_forward_count'], native_batch_calls=producer['native_batch_calls'],
        metadata_file_bytes=producer.get('metadata_file_bytes'),
        artifact_file_bytes_excluding_completion_marker=producer.get('artifact_file_bytes_excluding_completion_marker'))
    startup = dict(scope=SCOPE, argv=sys.argv, python=sys.executable, pid=os.getpid(), config=config, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=expected_model_seed, provenance=provenance,
        support_features=str(support_features), capsule_manifest=str(Path(capsule)/'manifest.json'), episodes=len(tasks),
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False, adapted_state_inherited=False,
        cross_row_adapted_state_reuse=False, checkpoint_loaded=False, encoder_updated=False, payload_audit=payload,
        source_validation=None, source_validation_reason='No source samples or source feature banks accessed',
        learning_rate=None, epoch=None, optimizer_steps=0, unavailable_reason='Analytical support probes; no optimizer or deployment head',
        blas_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    write(out/'startup.json', startup); print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    fields = ['split_id','receiver','scenario','k','new_count','support_seed','completed','total','scope','classes','support_count',
        'fold_count','numerical','oof','reconstruction','paired','fit_seconds','factorization_count','optimizer_steps','persistent_state_bytes',
        'heldout_unavailable_reason','fit_call_seconds','log_write_seconds','total_seconds','peak_process_rss_bytes','query_rows_used','source_rows_used']
    total_factorizations = 0; k1_count = 0
    with (out/'fit_trace.jsonl').open('x',encoding='utf-8') as trace, (out/'compact.jsonl').open('x',encoding='utf-8') as compact, \
         (out/'compact.csv').open('x',encoding='utf-8',newline='') as csvfile, (out/'fit_stages.jsonl').open('x',encoding='utf-8') as stages:
        writer = csv.DictWriter(csvfile,fieldnames=fields); writer.writeheader()
        stage_writer = None
        with (out/'fit_stages.csv').open('x',encoding='utf-8',newline='') as stage_csv:
            for number, (split, positions, labels) in enumerate(tasks, 1):
                started = time.perf_counter()
                audit = probe_branch_support(**{k: v[positions] for k,v in arrays.items()},support_labels=labels,
                    support_ids=split['support_ids'],classes=split['registered_classes'],old_classes=old)
                elapsed = time.perf_counter()-started
                k = split['k']; expected_folds = 0 if k == 1 else min(k,3)
                check(audit['k'] == k and audit['support_count'] == len(positions) and set(audit['classes']) == set(split['registered_classes'])
                      and audit['fold_count'] == expected_folds and len(audit['folds']) == expected_folds
                      and audit['factorization_count'] == expected_folds*6 and audit['optimizer_steps'] == 0
                      and audit['persistent_state_bytes'] == 0, 'Core support audit mismatch')
                if k == 1:
                    check(all(audit[key] is None for key in ('oof','reconstruction','paired')), 'K1 must not have held diagnostics')
                    k1_count += 1
                total_factorizations += audit['factorization_count']
                coords = {key:split[key] for key in ('split_id','receiver','scenario','k','support_seed')}
                coords['new_count'] = len(split['registered_classes'])-len(old)
                record = dict(audit, **{key:value for key,value in coords.items() if key != 'k'}, scope=SCOPE,
                    registered_classes=split['registered_classes'],fit_call_seconds=elapsed,query_rows_used=0,source_rows_used=0)
                write_started = time.perf_counter()
                trace.write(json.dumps(record,allow_nan=False)+'\n'); trace.flush()
                for fold in audit['folds']:
                    for stage in fold['stages']:
                        row = dict(split_id=split['split_id'],fold=fold['fold'],**scalars(stage))
                        if stage_writer is None:
                            stage_writer = csv.DictWriter(stage_csv, fieldnames=list(row)); stage_writer.writeheader()
                        stages.write(json.dumps(row,allow_nan=False)+'\n'); stage_writer.writerow(csv_record(row))
                        print(json.dumps(dict(event='ANALYTICAL_SUPPORT_FIT',**row),allow_nan=False),flush=True)
                stages.flush(); stage_csv.flush()
                small = dict(coords,completed=number,total=len(tasks),scope=SCOPE,classes=len(split['registered_classes']),
                    **{key:scalars(audit[key]) for key in ('support_count','fold_count','numerical','oof','reconstruction','paired',
                       'fit_seconds','factorization_count','optimizer_steps','persistent_state_bytes','heldout_unavailable_reason')},fit_call_seconds=elapsed,
                    log_write_seconds=time.perf_counter()-write_started,total_seconds=time.perf_counter()-started,
                    peak_process_rss_bytes=peak_rss(),query_rows_used=0,source_rows_used=0)
                compact.write(json.dumps(small,allow_nan=False)+'\n'); compact.flush(); writer.writerow(csv_record(small)); csvfile.flush()
                print(json.dumps(dict(event='SUPPORT_EPISODE_COMPLETE',**small),allow_nan=False),flush=True)
    marker = dict(status='SUPPORT_PROBE_COMPLETE',scope=SCOPE,capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256,model_seed=expected_model_seed,episodes=len(tasks),k1_episodes=k1_count,
        oof_episodes=len(tasks)-k1_count,algorithm=FROZEN_CONFIG,matrix=config['matrix'],query_rows_used=0,source_rows_used=0,
        truth_read=False,factorization_count=total_factorizations,optimizer_steps=0,persistent_state_bytes=0,payload_audit=payload,
        peak_process_rss_bytes=peak_rss(),peak_process_rss_reason='Linux process high-water RSS; null on unsupported platform')
    write(out/'probe_complete.json',marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--expected-model-seed',type=int,required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); evaluate(**args)


if __name__ == '__main__': main()
