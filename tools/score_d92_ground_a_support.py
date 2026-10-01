"""Freeze label-free ground-A scores, then pair legal support-held A/B/C.

This supplement never edits an existing method summary or runs adaptation.
The prediction entry accepts neither truth nor a training trace.  Truth and
actual same-run B/C scores are read only after all A predictions are persisted.
"""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
from cvsrffi.d92_ground_classifier_a import GroundFeatureContract, GroundHeadMetadata
from export_d92_ground_classifier_a_packet import load_packet
from export_d92_branch_support_features import CACHE_NAME, CACHE_SCHEMA, FEATURE_CONTRACT

SCHEMA = 'd92_ground_a_support_pairing_v1'
STATUS = 'GROUND_A_SUPPORT_PAIRING_COMPLETE'
SCOPE = 'SUPPORT_ONLY_GROUND_A_PAIRED_OLD_HELD_NOT_QUERY_EVALUATION'
PREDICTION_SCHEMA = 'd92_ground_a_support_predictions_v1'
PREDICTION_STATUS = 'GROUND_A_SUPPORT_PREDICTIONS_FIXED'
PREDICTION_FILE = 'ground_a_predictions.npz'
METHODS = {
    'd92_affine_joint_local_ridge_v1': ('D92-AffineJointLocalRidge-v1', 'AFFINE_JOINT_PROBE_COMPLETE',
        'SUPPORT_ONLY_AFFINE_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION', 'R_AFFINE_seq', 'B_AFFINE', 'C_AFFINE_seq'),
    'd92_conditional_joint_local_ridge_v1': ('D92-ConditionalJointLocalRidge-v1', 'CONDITIONAL_JOINT_PROBE_COMPLETE',
        'SUPPORT_ONLY_CONDITIONAL_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION', 'R_CONDITIONAL_seq', 'B_CONDITIONAL', 'C_CONDITIONAL_seq'),
    'd92_margin_joint_local_ridge_v1': ('D92-MarginJointLocalRidge-v1', 'MARGIN_JOINT_PROBE_COMPLETE',
        'SUPPORT_ONLY_MARGIN_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION', 'R_MARGIN_seq', 'B_MARGIN', 'C_MARGIN_seq'),
}
METRICS = ('A_old_accuracy', 'B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy', 'C_h',
           'adaptation_gain_B_minus_A', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
BINDING_KEYS = ('run_id', 'row_id', 'checkpoint_sha256', 'capsule_id', 'model_seed')


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def _strings(values, *, empty=False):
    return (isinstance(values, (list, tuple)) and (empty or bool(values))
            and all(isinstance(value, str) and value for value in values) and len(set(values)) == len(values))


def _binding(run_id, row_id, digest, capsule_id, model_seed):
    require(all(isinstance(value, str) and value for value in (run_id, row_id, capsule_id)), 'Explicit run/row/capsule binding required')
    GroundFeatureContract(digest, 'z_id', 'feat_joint', 'raw', 'float32', 160)
    require(type(model_seed) is int and model_seed >= 0, 'Explicit model seed required')
    return dict(run_id=run_id, row_id=row_id, checkpoint_sha256=digest, capsule_id=capsule_id, model_seed=model_seed)


def _separate_output(output, *inputs):
    resolved = Path(output).resolve()
    require(all(not resolved.is_relative_to(Path(root).resolve()) for root in inputs),
            'Supplement output must be outside the original packet/cache/trace directories')


def _selected_support_ids(root, binding, selection):
    """Resolve an explicit row selection from manifest metadata, without truth."""
    require(isinstance(selection, dict) and set(selection) <= {'receiver_scenes', 'support_seed', 'ks', 'new_counts', 'splits'}
            and isinstance(selection.get('splits'), list) and bool(selection['splits']), 'Explicit current-row support selection required')
    plan = read(root / 'support_splits.json')
    require(plan.get('schema') == CACHE_SCHEMA and plan.get('checkpoint_sha256') == binding['checkpoint_sha256']
            and plan.get('capsule_id') == binding['capsule_id'] and isinstance(plan.get('splits'), list), 'Selected support manifest binding mismatch')
    lookup = {split['split_id']: split for split in plan['splits']}
    require(len(lookup) == len(plan['splits']), 'Ambiguous support manifest split IDs')
    selected, seen = set(), set()
    for identity in selection['splits']:
        require(set(identity) == {'split_id', 'receiver', 'scenario', 'k', 'new_count', 'support_seed', 'registered_classes'}, 'Selected split metadata schema mismatch')
        sid = identity['split_id']; require(sid in lookup and sid not in seen, 'Selected support split missing/duplicated'); seen.add(sid)
        split = lookup[sid]
        actual = {key: split[key] for key in ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes')}
        actual['new_count'] = len(split['registered_classes']) - 6
        require(identity == actual and type(split['k']) is int and split['k'] > 0
                and _strings(split['registered_classes']) and _strings(split['support_ids'])
                and len(split['support_ids']) == split['k'] * len(split['registered_classes']), 'Selected physical support identity mismatch')
        # support_labels/indices are intentionally not indexed here.  The row
        # metadata establishes authorized IDs, not a class/role filter for A.
        selected.update(split['support_ids'])
    require(bool(selected), 'Empty current-row support selection')
    return sorted(selected)


def load_prediction_inputs(*, support_features, head, binding, selection):
    """Read selected raw z_id and IDs; manifest selection uses metadata only."""
    root = Path(support_features).resolve()
    marker, startup, provenance = [read(root / name) for name in
        ('features_complete.json', 'startup.json', 'checkpoint_provenance.json')]
    require(head.metadata.checkpoint_sha256 == binding['checkpoint_sha256'], 'Packet/checkpoint binding mismatch')
    for value in (marker, startup, provenance):
        require(value.get('checkpoint_sha256') == binding['checkpoint_sha256'] and value.get('model_seed') == binding['model_seed'],
                'Cache checkpoint/model seed binding mismatch')
    for value in (marker, startup):
        require(value.get('schema') == CACHE_SCHEMA and value.get('capsule_id') == binding['capsule_id'], 'Cache schema/capsule binding mismatch')
        require(value.get('feature_contract') == FEATURE_CONTRACT, 'Raw feat_joint cache contract mismatch')
        require(all(value.get(key) is False for key in ('query_iq_access', 'source_data_access', 'truth_read', 'adapted_state_inherited'))
                and value.get('query_rows_read') == 0 and value.get('native_eval') is True
                and value.get('new_source_payload_bytes') == value.get('new_ground_statistics_bytes') == 0,
                'Forbidden cache producer access/state')
    classes = marker.get('classes')
    require(_strings(classes) and len(classes) == 6 and set(classes) == set(head.classes)
            and provenance.get('classes') == classes and startup.get('provenance') == provenance, 'Ground class set/provenance mismatch')
    require(provenance.get('verdict') == 'MATCHED_SOURCE_ONLY_SCRATCH' and provenance.get('source_role_comparison') == 'EXACT_MATCH'
            and provenance.get('checkpoint_epoch') == 200 and provenance.get('checkpoint_inheritance') == []
            and provenance.get('target_access_before_freeze') is False, 'Source-only cache provenance required')
    require(marker.get('status') == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and marker.get('dtype') == 'float32'
            and marker.get('view_count_per_observation') == 1 and marker.get('query_used_for_fitting') is False
            and startup.get('query_fit_access') is False and marker.get('encoder_updated') is False
            and marker.get('native_parameters_unchanged') is True and marker.get('native_buffers_unchanged') is True,
            'Incomplete or modified raw support feature cache')
    path = root / CACHE_NAME
    require(marker.get('feature_file_bytes') == path.stat().st_size, 'Cache actual file byte mismatch')
    with np.load(path, allow_pickle=False) as cache:
        require(set(cache.files) == {'z_id', 'fft', 't_emb', 'f_emb', 'pa_local', 'ids', 'indices', 'labels',
                'checkpoint_sha256', 'capsule_id', 'feature_contract_json'}, 'Unexpected support cache members')
        require(cache['checkpoint_sha256'].item() == binding['checkpoint_sha256'] and cache['capsule_id'].item() == binding['capsule_id']
                and json.loads(cache['feature_contract_json'].item()) == FEATURE_CONTRACT, 'Raw cache numeric binding mismatch')
        z_id, ids = cache['z_id'], cache['ids']
    require(ids.ndim == 1 and ids.dtype.kind in 'SU', 'Invalid feature physical ID representation')
    physical_ids = ids.astype(str).tolist()
    require(_strings(physical_ids) and physical_ids == sorted(physical_ids), 'Invalid feature physical ID registry')
    require(z_id.dtype == np.float32 and z_id.shape == (len(ids), 160), 'Ground A requires raw float32 z_id[N,160]')
    require(marker.get('count') == len(ids) and marker.get('shapes', {}).get('z_id') == list(z_id.shape), 'Raw z_id count/shape mismatch')
    require(head.metadata.feature_contract == GroundFeatureContract(binding['checkpoint_sha256'], 'z_id', 'feat_joint', 'raw', 'float32', 160),
            'Packet raw feature contract mismatch')
    selected_ids = _selected_support_ids(root, binding, selection)
    lookup = {pid: i for i, pid in enumerate(physical_ids)}
    require(set(selected_ids) <= set(lookup), 'Selected physical support ID missing from raw cache')
    selected_z = z_id[[lookup[pid] for pid in selected_ids]]
    require(np.isfinite(selected_z).all(), 'Nonfinite selected raw z_id')
    return dict(z_id=np.frombuffer(selected_z.tobytes(), dtype=np.float32).reshape(selected_z.shape), physical_ids=selected_ids,
                support_features=str(root), cache_file=str(path), cache_file_bytes=path.stat().st_size,
                cache_marker=marker, provenance=provenance)


def predict_support(*, packet, support_features, output, run_id, row_id, selection,
                    expected_checkpoint_sha256, expected_capsule_id, expected_model_seed):
    """Persist every A score/prediction before any support truth/trace is read."""
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    _separate_output(out, packet, support_features)
    binding = _binding(run_id, row_id, expected_checkpoint_sha256, expected_capsule_id, expected_model_seed)
    started = time.perf_counter(); tick = time.perf_counter(); head = load_packet(packet)
    packet_load_seconds = time.perf_counter() - tick
    packet_root = Path(packet).resolve(); packet_marker = read(packet_root / 'complete.json')
    packet_metadata = read(packet_root / 'metadata.json')
    packet_seed = packet_metadata['existing_source_only_provenance'].get('model_seed')
    require(packet_seed is None or packet_seed == expected_model_seed, 'Packet/cache model seed mismatch')
    tick = time.perf_counter(); inputs = load_prediction_inputs(support_features=support_features, head=head, binding=binding, selection=selection)
    cache_read_seconds = time.perf_counter() - tick
    out.mkdir(parents=True, exist_ok=False)
    try:
        scores = np.empty((len(inputs['physical_ids']), 6), dtype=np.float32)
        bridge_seconds = score_seconds = 0.
        for i, row in enumerate(inputs['z_id']):
            # The local Torch/NumPy ABI need not support shared arrays.  Scalar
            # list conversion preserves each float32 value, and is measured.
            tick = time.perf_counter(); tensor = torch.tensor([row.tolist()], dtype=torch.float32, device='cpu')
            bridge_seconds += time.perf_counter() - tick
            tick = time.perf_counter(); values = head.score(z_id=tensor, feature_contract=head.metadata.feature_contract)
            score_seconds += time.perf_counter() - tick
            require(values.dtype == torch.float32 and tuple(values.shape) == (1, 6), 'Native Ground A score dtype/shape mismatch')
            scores[i] = np.asarray(values.cpu().tolist()[0], dtype=np.float32)
        require(np.isfinite(scores).all(), 'Nonfinite fixed A scores')
        columns = scores.argmax(axis=1); predictions = [head.classes[int(i)] for i in columns]
        with (out / PREDICTION_FILE).open('xb') as stream:
            np.savez(stream, scores=scores, physical_ids=np.asarray(inputs['physical_ids']),
                     ordered_classes=np.asarray(head.classes), prediction_columns=columns.astype(np.int64), predictions=np.asarray(predictions))
        result = dict(schema=PREDICTION_SCHEMA, status=PREDICTION_STATUS, scope=SCOPE, binding=binding,
            ordered_classes=list(head.classes), physical_record_count=len(scores), prediction_file=PREDICTION_FILE,
            selection=selection, selected_support_ids=inputs['physical_ids'], selected_split_count=len(selection['splits']),
            prediction_file_bytes=(out / PREDICTION_FILE).stat().st_size,
            feature_contract=asdict(head.metadata.feature_contract), head_metadata=asdict(head.metadata),
            support_features=inputs['support_features'], support_cache_file=inputs['cache_file'], support_cache_file_bytes=inputs['cache_file_bytes'],
            source_only_provenance=inputs['provenance'], packet=str(packet_root),
            packet_schema=packet_marker['schema'], packet_total_local_file_bytes=packet_marker['packet_total_file_bytes'],
            prediction_policy='each_physical_record_all_six_native_ground_columns_first_native_column_ties',
            prediction_fixed_before_support_truth_join=True, truth_rows_read=0, query_rows_used=0, source_rows_used=0,
            calibration_performed=False, training_performed=False, ground_column_order_preserved=True,
            resources=dict(packet_load_seconds=packet_load_seconds, raw_cache_read_seconds=cache_read_seconds,
                float32_scalar_list_bridge_seconds=bridge_seconds, native_single_record_score_seconds=score_seconds,
                wall_seconds=time.perf_counter() - started, score_call_count=len(scores), scored_physical_count=len(scores),
                packet_total_local_file_bytes=packet_marker['packet_total_file_bytes'],
                head_weight_file_bytes=(packet_root / 'head_weight.float32.bin').stat().st_size,
                raw_z_id_numeric_bytes=inputs['z_id'].nbytes, score_numeric_bytes=scores.nbytes,
                incremental_transmission_bytes=None, deployment_peak_memory_bytes=None, gpu_peak_memory_bytes=None,
                energy=None, unmeasured_reason='Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured',
                hardware=dict(device='cpu', machine=platform.machine(), platform=platform.platform(), torch_version=str(torch.__version__)),
                byte_scope='Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory'))
        write(out / 'prediction_complete.json', result)
        # Read back the persisted evidence before returning a fixed state.
        load_fixed_predictions(out)
        return result
    except Exception as exc:
        write(out / 'prediction_failed.json', dict(schema=PREDICTION_SCHEMA, status='GROUND_A_SUPPORT_PREDICTION_FAILED',
              error_type=type(exc).__name__, error=str(exc)))
        raise


def load_fixed_predictions(predictions):
    root = Path(predictions).resolve()
    require(not (root / 'prediction_failed.json').exists(), 'Failed A predictions cannot be joined to truth')
    marker = read(root / 'prediction_complete.json')
    require(marker['schema'] == PREDICTION_SCHEMA and marker['status'] == PREDICTION_STATUS and marker['scope'] == SCOPE
            and marker['prediction_fixed_before_support_truth_join'] is True and marker['truth_rows_read'] == 0
            and marker['query_rows_used'] == marker['source_rows_used'] == 0, 'A predictions are not independently fixed')
    _binding(**dict(run_id=marker['binding']['run_id'], row_id=marker['binding']['row_id'], digest=marker['binding']['checkpoint_sha256'],
                    capsule_id=marker['binding']['capsule_id'], model_seed=marker['binding']['model_seed']))
    require(marker['prediction_file'] == PREDICTION_FILE and (root / PREDICTION_FILE).stat().st_size == marker['prediction_file_bytes'], 'Prediction file byte mismatch')
    with np.load(root / PREDICTION_FILE, allow_pickle=False) as saved:
        require(set(saved.files) == {'scores', 'physical_ids', 'ordered_classes', 'prediction_columns', 'predictions'}, 'Unexpected prediction members')
        values = {key: saved[key] for key in saved.files}
    scores, ids, classes, columns, names = [values[key] for key in ('scores', 'physical_ids', 'ordered_classes', 'prediction_columns', 'predictions')]
    require(ids.ndim == classes.ndim == names.ndim == 1 and all(value.dtype.kind in 'SU' for value in (ids, classes, names)), 'Invalid fixed prediction registry')
    ids, classes, names = ids.astype(str).tolist(), classes.astype(str).tolist(), names.astype(str).tolist()
    require(_strings(ids) and ids == sorted(ids) and ids == marker['selected_support_ids']
            and _strings(classes) and len(classes) == 6 and classes == marker['ordered_classes'], 'Fixed native class/selected physical order mismatch')
    require(scores.dtype == np.float32 and scores.shape == (len(ids), 6) and np.isfinite(scores).all()
            and columns.dtype == np.int64 and columns.shape == (len(ids),) and len(names) == len(ids)
            and marker['physical_record_count'] == len(ids), 'Fixed score shape/dtype/count mismatch')
    require(np.array_equal(columns, scores.argmax(axis=1)) and names == [classes[int(i)] for i in columns], 'Fixed prediction differs from saved full scores')
    require(marker['head_metadata']['ordered_classes'] == classes
            and marker['head_metadata']['checkpoint_sha256'] == marker['binding']['checkpoint_sha256']
            and marker['head_metadata']['feature_contract'] == marker['feature_contract'], 'Fixed packet/feature binding mismatch')
    metadata = dict(marker['head_metadata']); metadata['feature_contract'] = GroundFeatureContract(**metadata['feature_contract'])
    GroundHeadMetadata(**metadata)
    require(marker['training_performed'] is marker['calibration_performed'] is False
            and marker['ground_column_order_preserved'] is True
            and marker['prediction_policy'] == 'each_physical_record_all_six_native_ground_columns_first_native_column_ties',
            'Fixed Ground A inference policy mismatch')
    return dict(marker=marker, physical_ids=ids, ordered_classes=classes, scores=scores, predictions=dict(zip(ids, names)))


def _load_truth(fixed, support_splits):
    """This is the first point at which target support truth is accessed."""
    marker = fixed['marker']; root = Path(marker['support_features'])
    require(Path(support_splits).resolve() == (root / 'support_splits.json').resolve(), 'Truth plan differs from the fixed raw cache binding')
    plan = read(support_splits); binding = marker['binding']
    require(set(plan) == {'schema', 'capsule_id', 'checkpoint_sha256', 'splits'} and plan['schema'] == CACHE_SCHEMA
            and plan['capsule_id'] == binding['capsule_id'] and plan['checkpoint_sha256'] == binding['checkpoint_sha256'], 'Support truth plan binding mismatch')
    require(Path(marker['support_cache_file']).resolve() == (root / CACHE_NAME).resolve()
            and (root / CACHE_NAME).stat().st_size == marker['support_cache_file_bytes'], 'Fixed cache reference changed')
    with np.load(root / CACHE_NAME, allow_pickle=False) as cache:
        ids, indices, labels = cache['ids'], cache['indices'], cache['labels']
        require(cache['checkpoint_sha256'].item() == binding['checkpoint_sha256'] and cache['capsule_id'].item() == binding['capsule_id'], 'Truth cache binding mismatch')
    require(ids.ndim == labels.ndim == 1 and ids.dtype.kind in 'SU' and labels.dtype.kind in 'SU'
            and len(labels) == len(ids) and len(set(ids.astype(str).tolist())) == len(ids)
            and indices.dtype == np.int64 and indices.shape == (len(ids),) and len(set(indices.tolist())) == len(ids)
            and np.all(indices >= 0), 'Truth physical registry differs from fixed predictions')
    lookup = {pid: i for i, pid in enumerate(ids.astype(str).tolist())}
    require(set(fixed['physical_ids']) <= set(lookup), 'Selected fixed physical ID missing from truth registry')
    truths = {pid: str(labels[lookup[pid]]) for pid in fixed['physical_ids']}
    index_by_id = {pid: int(indices[lookup[pid]]) for pid in fixed['physical_ids']}
    require(all(isinstance(value, str) and value for value in truths.values()), 'Invalid support truth class ID')
    selected = {item['split_id']: item for item in marker['selection']['splits']}
    require(len(selected) == marker['selected_split_count'], 'Fixed selected split count mismatch')
    splits = {}; used = set()
    for split in plan['splits']:
        if split['split_id'] not in selected: continue
        require(set(split) == {'split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes', 'support_indices', 'support_ids', 'support_labels'}, 'Support split schema mismatch')
        sid, classes, pids, y, k = [split[key] for key in ('split_id', 'registered_classes', 'support_ids', 'support_labels', 'k')]
        identity = {key: split[key] for key in ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes')}
        identity['new_count'] = len(classes) - 6
        require(selected[sid] == identity, 'Fixed selected split identity changed before truth join')
        require(isinstance(sid, str) and sid and sid not in splits and _strings(classes) and _strings(pids)
                and set(fixed['ordered_classes']) <= set(classes) and type(k) is int and k > 0
                and type(split['support_seed']) is int and split['support_seed'] >= 0, 'Support split identity/ground registry mismatch')
        require(isinstance(y, list) and len(y) == len(pids) == k * len(classes) and all(type(value) is int and 0 <= value < len(classes) for value in y)
                and all(y.count(i) == k for i in range(len(classes))) and isinstance(split['support_indices'], list)
                and len(split['support_indices']) == len(pids), 'Support split physical K/truth mismatch')
        require(all(pid in truths for pid in pids) and [truths[pid] for pid in pids] == [classes[i] for i in y]
                and [index_by_id[pid] for pid in pids] == split['support_indices'], 'Missing or mismatched physical support truth')
        splits[sid] = split; used.update(pids)
    require(set(splits) == set(selected) and used == set(fixed['physical_ids']), 'Extra/missing selected physical rows in support plan')
    return splits, truths


def _trace_context(fixed, fit_trace):
    path = Path(fit_trace).resolve(); require(path.name == 'fit_trace.jsonl', 'Explicit original fit_trace.jsonl required')
    startup, complete = [read(path.parent / name) for name in ('startup.json', 'probe_complete.json')]
    binding = fixed['marker']['binding']; schema = startup.get('schema')
    require(schema in METHODS, 'Unsupported actual support method schema')
    method, status, scope, candidate, bstate, cstate = METHODS[schema]
    for value in (startup, complete):
        require(all(value.get(key) == binding[key] for key in BINDING_KEYS), 'Training trace escaped current run/row/checkpoint/capsule/seed')
        require(value.get('schema') == schema and value.get('method') == method and value.get('scope') == scope
                and value.get('query_rows_used') == value.get('source_rows_used') == 0 and value.get('truth_read') is False, 'Support trace method/access boundary mismatch')
    require(complete.get('status') == status and complete.get('episodes') == startup.get('episodes'), 'Incomplete original support trace')
    require(Path(startup['support_features']).resolve() == Path(fixed['marker']['support_features']).resolve(), 'Actual B used a different raw feature cache')
    require(startup['provenance'] == fixed['marker']['source_only_provenance'], 'Actual B provenance differs from fixed A cache')
    selected = startup['config']['selection']['splits']
    require(isinstance(selected, list) and bool(selected) and complete['selection'] == startup['config']['selection']
            and len(selected) == startup['episodes'], 'Actual selected parent coverage mismatch')
    require(startup['config']['selection'] == fixed['marker']['selection'], 'Ground A prediction selection differs from the actual current row')
    expected = {item['split_id']: item for item in selected}
    require(len(expected) == len(selected), 'Duplicate selected support split')
    return dict(path=path, binding=binding, schema=schema, method=method, scope=scope, candidate=candidate,
                bstate=bstate, cstate=cstate, expected=expected, complete=complete)


def _namespace_refs(value, coords):
    if isinstance(value, dict):
        if {'path', 'arrays', 'array_summaries', 'namespace'} <= set(value):
            namespace = json.loads(value['namespace'])
            require(all(namespace.get(key) == item for key, item in coords.items()), 'Actual adapted B/C state belongs to another run/path')
        else:
            for item in value.values(): _namespace_refs(item, coords)
    elif isinstance(value, (tuple, list)):
        for item in value: _namespace_refs(item, coords)


def _scores(value, n, c):
    array = np.asarray(value, dtype=np.float64)
    if n == 0: require(value == [], 'K1 invented held scores'); return np.empty((0, c))
    require(array.shape == (n, c) and np.isfinite(array).all(), 'Actual B/C score shape/finite mismatch')
    return array


def _path_predictions(entry, train, held, classes, old, truths, context, parent_k):
    require(entry['parent_k'] == parent_k and entry['train_k'] == len(train) // len(classes)
            and entry['held_k'] == len(held) // len(classes), 'Actual path physical K mismatch')
    bt = sorted(pid for pid in train if truths[pid] in old); bh = sorted(pid for pid in held if truths[pid] in old)
    require(entry['b_training_ids'] == bt and entry['c_training_ids'] == sorted(train) and entry['b_ids'] == bh and entry['c_ids'] == sorted(held),
            'Actual B/C old-held physical pairing mismatch')
    require(entry['b_classes'] == old and entry['c_classes'] == classes and entry['held_labels'] == {pid: truths[pid] for pid in held}, 'Actual trace ground/held truth mapping mismatch')
    expected_status = 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN' if held else 'NO_HELD_PREDICTIONS'
    if context['schema'] in ('d92_conditional_joint_local_ridge_v1','d92_margin_joint_local_ridge_v1') or 'prediction_status' in entry:
        require(entry.get('prediction_status') == expected_status, 'Actual B/C predictions were not fixed')
    reuse = classes == old
    require(entry['c_reuses_b_candidates'] is reuse and [stage['state'] for stage in entry['candidate_stages']]
            == ([context['bstate']] if reuse else [context['bstate'], context['cstate']]), 'Missing actual adapted B/C stages; R0/B0 cannot supply A/B')
    # The existing executable passes only run/row/split into method context.
    # Checkpoint/capsule/seed are bound by the actual lane startup/completion,
    # not invented as missing fields in the historical stage/namespace ABI.
    coords = dict(run_id=context['binding']['run_id'], row_id=context['binding']['row_id'], split_id=context['split_id'],
                  **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
    outer = entry['outer_features_state_ref']
    if held:
        require(isinstance(outer, dict) and {'path', 'arrays', 'array_summaries', 'namespace'} <= set(outer), 'Missing fixed outer-held feature archive')
        widths = dict(z_id=160, fft=96, t_emb=160, f_emb=160, pa_local=160)
        require(set(outer['arrays']) == set(outer['array_summaries']) == set(widths)
                and all(outer['arrays'][key]['shape'] == [len(held), width] and outer['arrays'][key]['dtype'] == 'float64'
                        for key, width in widths.items()), 'Fixed outer-held feature archive shape/dtype mismatch')
        require(json.loads(outer['namespace']).get('state') == 'OUTER_SUPPORT_HELD', 'Outer-held archive state mismatch')
        _namespace_refs(outer, coords)
    else:
        require(outer is None, 'K1 invented outer-held archive')
    # Affine v1 predates prediction_status. Its frozen ENTRY first archives the
    # exact outer features, scores the final same-path states, then constructs
    # assess_paths evidence. Require those real references and full scores,
    # alongside the completed lane and physical pairing; do not add a field to
    # historical traces or treat an absent Conditional status as this contract.
    for stage in entry['candidate_stages']:
        is_b = stage['state'] == context['bstate']
        require(all(stage.get(key) == item for key, item in coords.items()) and stage['mode'] == ('B' if is_b else 'C_seq')
                and stage['training_physical_ids'] == (bt if is_b else sorted(train)) and stage['final_state_ref'] is not None,
                'Actual B/C stage inheritance/path binding mismatch')
        _namespace_refs(stage, coords)
    if context['schema'] == 'd92_margin_joint_local_ridge_v1':
        preparations=entry['preparations']
        require([prep['state'] for prep in preparations]==(['B'] if reuse else ['B','C']),
                'Missing actual Margin B/C preparation')
        for prep,stage in zip(preparations,entry['candidate_stages']):
            is_b=prep['state']=='B'
            require(all(prep.get(key)==item for key,item in coords.items())
                and prep['training_physical_ids']==(bt if is_b else sorted(train))
                and prep['inherited_adapter_from']==(None if is_b else context['bstate'])
                and prep['inherited_state'] is (not is_b)
                and stage['preparation_ref']==prep['state']
                and stage['inherited_from_B'] is (not is_b), 'Margin actual B-to-C inheritance mismatch')
            _namespace_refs(prep,coords)
        if not reuse:
            b_ref=entry['candidate_stages'][0]['final_state_ref']
            c_stage=entry['candidate_stages'][1]
            require(preparations[1]['final_problem']['prior_source']=='FROZEN_CURRENT_ACTUAL_B'
                and preparations[1]['final_problem']['prior_ref']==b_ref
                and c_stage['preparation']['final_problem']['prior_ref']==b_ref,
                'Margin final prior is not the same-path actual B state')
    scores = entry['paths'].get(context['candidate'])
    require(scores is not None, 'Actual adapted path missing; no R0/B0 fallback')
    b = _scores(scores['b_scores'], len(bh), len(old)); c = _scores(scores['c_scores'], len(held), len(classes))
    if reuse: require(np.array_equal(b, c), 'new0 changed actual B predictions')
    return dict(b={pid: old[int(index)] for pid, index in zip(bh, b.argmax(axis=1))},
                c={pid: classes[int(index)] for pid, index in zip(sorted(held), c.argmax(axis=1))},
                old_held_ids=bh, held_ids=sorted(held), scope=entry['scope'], fold=entry['fold'], trial=entry['trial'],
                fixed_score_contract='margin_explicit_status_and_same_path_actual_B_prior' if context['schema'] == 'd92_margin_joint_local_ridge_v1'
                    else 'conditional_explicit_status_plus_archived_final_states' if context['schema'] == 'd92_conditional_joint_local_ridge_v1'
                    else 'affine_v1_archived_outer_features_and_final_states_before_assess_paths')


def _metrics(examples, all_held, old, truths):
    if not examples: return dict.fromkeys(METRICS)
    a = sum(row['A_prediction'] == row['truth'] for row in examples) / len(examples)
    b = sum(row['B_prediction'] == row['truth'] for row in examples) / len(examples)
    co = sum(row['C_prediction'] == row['truth'] for row in examples) / len(examples)
    new = [(pid, name) for pid, name in all_held.items() if truths[pid] not in old]
    cn = None if not new else sum(name == truths[pid] for pid, name in new) / len(new)
    h = None if cn is None else (0. if cn + co == 0 else 2 * cn * co / (cn + co))
    return dict(A_old_accuracy=a, B_old_accuracy=b, C_old_accuracy=co, C_new_accuracy=cn, C_h=h,
                adaptation_gain_B_minus_A=b - a, total_old_accuracy_drop=b - co,
                C_abs_new_old_gap=None if cn is None else abs(cn - co))


def _parent(record, split, truths, fixed, context):
    old, classes, k = sorted(fixed['ordered_classes']), sorted(split['registered_classes']), split['k']
    identity = {key: split[key] for key in ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes')}
    identity['new_count'] = len(classes) - 6
    require(all(record.get(key) == value for key, value in identity.items()) and context['expected'][split['split_id']] == identity,
            'Actual parent split identity differs from selected legal support')
    require(record['schema'] == context['schema'] and record['method'] == context['method'] and record['scope'] == context['scope']
            and record['classes'] == classes and record['old_classes'] == old and record['support_count'] == len(split['support_ids'])
            and record['old_class_count'] == 6 and record['new_class_count'] == len(classes) - 6
            and record['query_rows_used'] == record['source_rows_used'] == 0, 'Actual parent schema/classes/access mismatch')
    path_binding = dict(run_id=context['binding']['run_id'], row_id=context['binding']['row_id'], split_id=split['split_id'])
    require(all(record['inheritance_binding'].get(key) == value for key, value in path_binding.items()),
            'Actual adapted B reused from another run/row/split')
    context = dict(context, split_id=split['split_id']); pids = set(split['support_ids'])
    groups = {name: sorted(pid for pid in pids if truths[pid] == name) for name in classes}
    paths = []
    if k == 1:
        require(record['fold_count'] == 0 and record['folds'] == [] and record['oneshot_proxy'] is None
                and record['oof'] is None and record['full_support'] is not None
                and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'K1 independent held result invented')
        entry = record['full_support']; require(entry['scope'] == 'support_full_k1' and entry['fold'] is entry['trial'] is None, 'K1 full-support path mismatch')
        _path_predictions(entry, pids, set(), classes, old, truths, context, k)
        return dict(**identity, path=context['candidate'], oof=dict(metrics=dict.fromkeys(METRICS), old_held_physical_count=0,
                    unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT'), proxy=None, paired_paths=[])
    nf = min(k, 3); assignment = {pid: i % nf for group in groups.values() for i, pid in enumerate(group)}
    require(record['fold_count'] == len(record['folds']) == nf and record['full_support'] is None
            and len(record['physical_fold_assignment']) == len(pids)
            and {row['physical_id']: (row['class_id'], row['fold']) for row in record['physical_fold_assignment']}
            == {pid: (truths[pid], fold) for pid, fold in assignment.items()}, 'OOF physical assignment incomplete')
    pooled_c, examples = {}, []
    for j, entry in enumerate(record['folds']):
        held = {pid for pid in pids if assignment[pid] == j}
        require(entry['scope'] == 'support_oof' and entry['fold'] == j and entry['trial'] is None, 'OOF path coordinate mismatch')
        value = _path_predictions(entry, pids - held, held, classes, old, truths, context, k)
        value['examples'] = [dict(physical_id=pid, truth=truths[pid], A_prediction=fixed['predictions'][pid],
            B_prediction=value['b'][pid], C_prediction=value['c'][pid]) for pid in value['old_held_ids']]
        require(not set(pooled_c) & set(value['c']), 'Repeated physical held ID in OOF')
        pooled_c.update(value['c']); examples.extend(value['examples']); paths.append(value)
    old_ids = sorted(pid for pid in pids if truths[pid] in old)
    require(sorted(row['physical_id'] for row in examples) == old_ids and set(pooled_c) == pids, 'Missing A/B/C paired OOF physical IDs')
    oof = dict(metrics=_metrics(examples, pooled_c, old, truths), old_held_physical_count=len(examples), old_held_physical_ids=old_ids)
    proxy = record['oneshot_proxy']
    require(proxy is not None and proxy['trial_count'] == len(proxy['trials']) == k and proxy['proxy_train_k'] == 1
            and proxy['aggregation'] == 'all_anchors_mean_within_parent_then_equal_parent', 'Proxy anchor coverage incomplete')
    trial_metrics = []
    for j, entry in enumerate(proxy['trials']):
        train = {group[j] for group in groups.values()}; held = pids - train
        require(entry['scope'] == 'support_oneshot_proxy' and entry['fold'] is None and entry['trial'] == j, 'Proxy path coordinate mismatch')
        value = _path_predictions(entry, train, held, classes, old, truths, context, k)
        value['examples'] = [dict(physical_id=pid, truth=truths[pid], A_prediction=fixed['predictions'][pid],
            B_prediction=value['b'][pid], C_prediction=value['c'][pid]) for pid in value['old_held_ids']]
        trial_metrics.append(_metrics(value['examples'], value['c'], old, truths)); paths.append(value)
    proxy_metrics = {key: None if any(row[key] is None for row in trial_metrics) else sum(row[key] for row in trial_metrics) / k for key in METRICS}
    return dict(**identity, path=context['candidate'], oof=oof,
                proxy=dict(metrics=proxy_metrics, trial_count=k, aggregation='all_anchors_mean_within_parent_then_equal_parent'), paired_paths=paths)


def _statistics(parents, binding):
    tables = {name: {} for name in ('overall', 'by_k_new_count', 'by_receiver_scene', 'by_model_row')}
    for parent in parents:
        for diagnostic in ('oof', 'proxy'):
            value = parent[diagnostic]; metrics = dict.fromkeys(METRICS) if value is None else value['metrics']
            keys = dict(overall=(diagnostic, parent['path'], 'old_only' if parent['new_count'] == 0 else 'new_present'),
                by_k_new_count=(diagnostic, parent['path'], parent['k'], parent['new_count']),
                by_receiver_scene=(diagnostic, parent['path'], parent['receiver'], parent['scenario'], parent['k'], parent['new_count']),
                by_model_row=(diagnostic, parent['path'], binding['model_seed'], binding['row_id'], parent['k'], parent['new_count']))
            for name, key in keys.items():
                cell = tables[name].setdefault(key, {metric: [] for metric in METRICS})
                for metric in METRICS: cell[metric].append(metrics[metric])
    dimensions = dict(overall=('diagnostic', 'path', 'population'), by_k_new_count=('diagnostic', 'path', 'k', 'new_count'),
        by_receiver_scene=('diagnostic', 'path', 'receiver', 'scenario', 'k', 'new_count'),
        by_model_row=('diagnostic', 'path', 'model_seed', 'row_id', 'k', 'new_count'))
    result = {}
    for name, groups in tables.items():
        rows = []
        for key, metrics in sorted(groups.items()):
            for metric, values in metrics.items():
                present = [v for v in values if v is not None]
                rows.append(dict(zip(dimensions[name], key), metric=metric, parent_count=len(values), measured_parent_count=len(present),
                    null_parent_count=len(values) - len(present), mean=None if not present else sum(present) / len(present),
                    minimum=None if not present else min(present), maximum=None if not present else max(present)))
        result[name] = rows
    return result


def score_fixed_support(*, predictions, fit_trace, support_splits):
    """Independent truth join and pairing; no model, packet, fit or query call."""
    fixed = load_fixed_predictions(predictions)
    splits, truths = _load_truth(fixed, support_splits)
    context = _trace_context(fixed, fit_trace); require(set(context['expected']) <= set(splits), 'Selected parent missing from legal support truth')
    seen, parents = set(), []
    with context['path'].open(encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'Empty original trace record')
            record = json.loads(line); sid = record['split_id']
            require(sid in context['expected'] and sid not in seen, 'Duplicate/unexpected original support parent')
            seen.add(sid); parents.append(_parent(record, splits[sid], truths, fixed, context))
    require(seen == set(context['expected']) and len(parents) == context['complete']['episodes'], 'Partial original support trace cannot supply A/B pairing')
    return dict(schema=SCHEMA, status=STATUS, scope=SCOPE, binding=fixed['marker']['binding'], run_id=fixed['marker']['binding']['run_id'],
        row_id=fixed['marker']['binding']['row_id'], method=context['method'], method_schema=context['schema'], adapted_path=context['candidate'],
        prediction_archive=str(Path(predictions).resolve()), fit_trace=str(context['path']), support_splits=str(Path(support_splits).resolve()),
        ordered_ground_classes=fixed['ordered_classes'], ground_prediction_record_count=len(fixed['physical_ids']), parent_count=len(parents),
        k1_parent_count=sum(parent['k'] == 1 for parent in parents), actual_A_scope='MATCHED_OLD_SUPPORT_HELD_ONLY_NOT_QUERY_OR_DEPLOYMENT_ACCURACY',
        parents=parents, statistics=_statistics(parents, fixed['marker']['binding']), resources=fixed['marker']['resources'],
        query_rows_used=0, source_rows_used=0, original_summary_modified=False, original_trace_modified=False,
        training_performed=False, calibration_performed=False, model_called_during_truth_join=False,
        interpretation=['All six native ground columns compete for every explicitly selected current-row support record before support truth is joined.',
            'A and actual adapted B/C use the same row/split/physical old-held records; no R0/B0 or other-run adapted B is substituted.',
            'True K1 has no independent held A or B-A. One-shot proxy is separate and averaged within parent first.',
            'OOF is pooled once per held physical ID within parent; H/gap/decline are computed within parent before parent averaging.',
            'This new supplement leaves previous method summaries and reports unchanged and cannot feed adaptation or query selection.'])


def _csv(path, rows):
    if not rows: return
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows: writer.writerow({key: 'N/A' if value is None else value for key, value in row.items()})


def score_support(*, packet, support_features, fit_trace, output, run_id, row_id, selection=None,
                  expected_checkpoint_sha256, expected_capsule_id, expected_model_seed):
    """Create an exclusive supplement, preserving fixed scores on join failure."""
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    _separate_output(out, packet, support_features, Path(fit_trace).parent)
    binding = _binding(run_id, row_id, expected_checkpoint_sha256, expected_capsule_id, expected_model_seed)
    # Only startup selection metadata is read before prediction.  The actual
    # fit trace and every support truth value remain behind the fixed-score step.
    startup = read(Path(fit_trace).parent / 'startup.json')
    require(all(startup.get(key) == value for key, value in binding.items()), 'Prediction selection belongs to another run/row')
    require(startup.get('schema') in METHODS and startup.get('method') == METHODS[startup['schema']][0]
            and Path(startup['support_features']).resolve() == Path(support_features).resolve(), 'Prediction selection/raw cache differs from the actual row')
    actual_selection = startup['config']['selection']
    require(selection is None or selection == actual_selection, 'Explicit selection differs from actual current-row metadata')
    predict_support(packet=packet, support_features=support_features, output=out, run_id=run_id, row_id=row_id,
        selection=actual_selection,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_capsule_id=expected_capsule_id, expected_model_seed=expected_model_seed)
    try:
        result = score_fixed_support(predictions=out, fit_trace=fit_trace, support_splits=Path(support_features) / 'support_splits.json')
        write(out / 'pairing.json', result)
        for name, rows in result['statistics'].items(): _csv(out / (name + '.csv'), rows)
        with (out / 'paired_parents.jsonl').open('x', encoding='utf-8') as stream:
            for parent in result['parents']: stream.write(json.dumps(parent, ensure_ascii=False, allow_nan=False) + '\n')
        lines = ['# Ground A 支持集配对补充', '', '所有数值仅为同 row、同物理旧类 support-held 诊断；真实 K1 为 N/A。准确率用百分数，差值用百分点。', '',
            '| 诊断 | K | 新类数 | A 旧 | B 旧 | B−A | C 旧 | C 新 | H | 旧类下降 | 新旧差 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        cells = {}
        for row in result['statistics']['by_k_new_count']:
            cells.setdefault((row['diagnostic'], row['k'], row['new_count']), {})[row['metric']] = row['mean']
        fields = ('A_old_accuracy', 'B_old_accuracy', 'adaptation_gain_B_minus_A', 'C_old_accuracy', 'C_new_accuracy', 'C_h', 'total_old_accuracy_drop', 'C_abs_new_old_gap')
        for key, metrics in sorted(cells.items()):
            lines.append('| ' + ' | '.join([str(value) for value in key] + ['N/A' if metrics[field] is None else f'{100 * metrics[field]:.3f}' for field in fields]) + ' |')
        (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        write(out / 'complete.json', dict(schema=SCHEMA, status=STATUS, binding=result['binding'], parent_count=result['parent_count'],
            query_rows_used=0, original_summary_modified=False, prediction_status=PREDICTION_STATUS))
        return result
    except Exception as exc:
        write(out / 'scoring_failed.json', dict(schema=SCHEMA, status='GROUND_A_SUPPORT_PAIRING_FAILED',
            error_type=type(exc).__name__, error=str(exc), fixed_predictions_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'support-features', 'fit-trace', 'output', 'run-id', 'row-id', 'expected-checkpoint-sha256', 'expected-capsule-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    result = score_support(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'], parent_count=result['parent_count'], query_rows_used=0), allow_nan=False))


if __name__ == '__main__': main()
