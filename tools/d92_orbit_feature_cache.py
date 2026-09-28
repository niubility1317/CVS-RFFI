"""Read-only binding of frozen received-orbit caches; no adaptation/model loading.

The producer used full-length IQ256 phases (0, pi/2, pi, 3pi/2), raw
native identity160 and historical FFT96 (floor 1e-8), stored as float32.
OSC applies its own unit normalization (floor 1e-12) after this boundary.
"""
import json
from pathlib import Path

import numpy as np
from export_d92_mv_kme_features import read, write, validate_capsule, validate_origin, peak_process_rss

ROOT = Path(__file__).resolve().parents[1]
CACHE_SCHEMA = 'd92_bnna_received_views_v1'
CACHE_NAME = 'received_bnna_features.npz'
PRODUCER_CONFIG = ROOT / 'configs/d92_bnna_frozen_20260929.json'
FEATURE_CONTRACT = dict(input_shape=[2, 256], phases_quarter_turns=[0, 1, 2, 3],
    identity_dim=160, fft_dim=96, fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8)
FORWARD_SCOPE = 'One observation, four fixed full-length phase views per call; no cross-query statistics'


def _strings(value):
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value) and len(set(value)) == len(value)


def load_features(*, orbit_features, capsule, row_root, expected_capsule_id,
                  expected_checkpoint_sha256, algorithm):
    """Read cache, extraction metadata and capsule IDs only, never IQ or fit state."""
    producer = read(PRODUCER_CONFIG)
    if (set(producer) != {'algorithm'} or producer['algorithm'].get('method') != 'D92-BNNA-v1'
            or any(producer['algorithm'].get(k) != v or algorithm.get(k) != v for k, v in FEATURE_CONTRACT.items())):
        raise ValueError('Frozen orbit producer/consumer feature formula mismatch')
    root = Path(orbit_features)
    marker = read(root / 'features_complete.json')
    provenance = read(root / 'checkpoint_provenance.json')
    extraction = read(root / 'startup.json')
    previous, origin = validate_origin(row_root, capsule, expected_checkpoint_sha256)
    if (not _strings(origin.get('classes'))
            or origin.get('checkpoint_sha256') != expected_checkpoint_sha256
            or marker.get('status') != 'BNNA_FEATURES_COMPLETE' or marker.get('schema') != CACHE_SCHEMA
            or marker.get('capsule_id') != expected_capsule_id or marker.get('checkpoint_sha256') != expected_checkpoint_sha256
            or marker.get('algorithm') != producer['algorithm'] or marker.get('classes') != origin['classes']
            or marker.get('model_seed') != previous['seed'] or marker.get('dtype') != 'float32'
            or any(marker.get(key) is not False for key in ('query_used_for_fitting', 'source_data_access', 'truth_read', 'encoder_updated'))
            or marker.get('native_eval') is not True or marker.get('native_buffers_unchanged') is not True
            or marker.get('view_count_per_observation') != 4 or marker.get('native_forward_scope') != FORWARD_SCOPE
            or marker.get('new_ground_statistics_bytes') != 0
            or provenance.get('checkpoint_sha256') != expected_checkpoint_sha256 or provenance.get('classes') != origin['classes']
            or provenance.get('model_seed') != previous['seed'] or provenance.get('checkpoint_inheritance') != []
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('verdict') != 'MATCHED_SOURCE_ONLY_SCRATCH'
            or provenance.get('source_role_comparison') != 'EXACT_MATCH' or provenance.get('checkpoint_epoch') != 200
            or type(marker.get('model_file_bytes')) is not int or marker['model_file_bytes'] <= 0
            or provenance.get('model_file_bytes') != marker['model_file_bytes']
            or extraction.get('config') != producer or extraction.get('provenance') != provenance
            or extraction.get('capsule_id') != expected_capsule_id or extraction.get('checkpoint_sha256') != expected_checkpoint_sha256
            or any(extraction.get(key) is not False for key in ('query_fit_access', 'source_data_access', 'truth_read'))):
        raise ValueError('Frozen orbit feature provenance mismatch')
    feature_path = root / CACHE_NAME
    with np.load(feature_path, allow_pickle=False) as data:
        if set(data.files) != {'identity_views', 'fft', 'ids', 'checkpoint_sha256', 'capsule_id', 'algorithm_json'}:
            raise ValueError('Unexpected orbit cache members')
        if (data['checkpoint_sha256'].item() != expected_checkpoint_sha256 or data['capsule_id'].item() != expected_capsule_id
                or json.loads(data['algorithm_json'].item()) != producer['algorithm']):
            raise ValueError('Orbit cache binding mismatch')
        identity, fft, ids = data['identity_views'], data['fft'], data['ids']
    # Lazy NPZ member access: IQ is never materialized or inspected.
    with np.load(Path(capsule) / 'received.npz', allow_pickle=False) as data:
        physical_ids = data['ids']
    if (ids.ndim != 1 or ids.dtype.kind not in 'SU' or physical_ids.dtype.kind not in 'SU'
            or not len(ids) or len(set(ids)) != len(ids) or not np.array_equal(ids, physical_ids)
            or any(not value for value in ids.astype(str))
            or identity.shape != (len(ids), 4, 160) or fft.shape != (len(ids), 96)
            or identity.dtype != np.float32 or fft.dtype != np.float32
            or not np.isfinite(identity).all() or not np.isfinite(fft).all()
            or marker.get('count') != len(ids) or marker.get('identity_views_shape') != list(identity.shape)
            or marker.get('fft_shape') != list(fft.shape) or marker.get('identity_views_bytes') != identity.nbytes
            or marker.get('fft_bytes') != fft.nbytes or marker.get('feature_array_bytes') != identity.nbytes + fft.nbytes
            or marker.get('feature_file_bytes') != feature_path.stat().st_size):
        raise ValueError('Orbit cache physical ID/shape/byte mismatch')
    identity = np.frombuffer(identity.tobytes(), dtype=np.float32).reshape(identity.shape)
    fft = np.frombuffer(fft.tobytes(), dtype=np.float32).reshape(fft.shape)
    ids = ids.astype(str); ids.setflags(write=False)
    return identity, fft, ids, marker, previous


def validate_split(split, manifest, ids, old_classes):
    """Pure physical split boundary, independent of previous method predictors."""
    classes = split['registered_classes']; support = split['support_indices']; query = split['query_indices']
    labels = np.asarray(split['support_labels'])
    if (split.get('protocol_schema') != 'p2_min_v1' or split.get('phase2_data_status') != 'VALIDATED_ONCE'
            or split.get('capsule_id') != manifest['capsule_id'] or not _strings(classes)
            or classes[:len(old_classes)] != old_classes or type(split.get('k')) is not int or split['k'] < 1
            or not support or not query or any(type(i) is not int for i in support + query)
            or len(set(support)) != len(support) or len(set(query)) != len(query) or set(support) & set(query)
            or min(support + query) < 0 or max(support + query) >= len(ids)
            or labels.shape != (len(support),) or labels.dtype.kind not in 'iu'
            or set(labels.tolist()) != set(range(len(classes)))
            or not np.all(np.bincount(labels, minlength=len(classes)) == split['k'])):
        raise ValueError('Split identity, physical support/query or registration mismatch')
    if {'query_labels', 'query_truth', 'query_roles', 'query_class_counts'} & set(split):
        raise ValueError('Forbidden query information in split')
    return np.asarray(support), np.asarray(query), labels
