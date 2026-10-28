"""Convert fixed historical observations outside the truth-blind predictor.

Only this builder reads the legacy truth ledger.  ``load_public`` accepts the
three public files and has no truth or source-data input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PUBLIC_FIELDS = frozenset({
    'schema', 'data_status', 'capsule_id', 'split_id', 'count', 'iq_shape',
    'registered_classes', 'processing_route', 'mode', 'equalization_enabled',
    'fs_hz', 'single_observation',
})
PUBLIC_FILES = frozenset({'manifest.json', 'ids.npy', 'received.npy'})
MAIN_RX_NAMES = (
    '1-1', '1-19', '14-7', '18-2', '19-2', '2-1',
    '2-19', '20-1', '3-19', '7-14', '7-7', '8-8',
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:24]


def _classes(values: Sequence[str], name: str, count: int | None = None) -> list[str]:
    result = list(values)
    if (not result or any(not isinstance(v, str) or not v.strip() for v in result)
            or len(set(result)) != len(result)
            or (count is not None and len(result) != count)):
        raise ValueError(f'Invalid {name} class list')
    return result


def _receivers(values: Sequence[str | int]) -> list[str]:
    result = []
    for value in values:
        text = str(value)
        if text.isdecimal():
            index = int(text)
            if index >= len(MAIN_RX_NAMES):
                raise ValueError('Receiver index is outside the Main split')
            text = MAIN_RX_NAMES[index]
        if text not in MAIN_RX_NAMES:
            raise ValueError(f'Unknown receiver {value!r}')
        result.append(text)
    if not result or len(set(result)) != len(result):
        raise ValueError('Receiver list must be nonempty and unique')
    return result


def _finite(iq: np.ndarray) -> None:
    for start in range(0, len(iq), 512):
        if not np.isfinite(iq[start:start + 512]).all():
            raise ValueError('Received IQ must be finite')


def _validate_arrays(ids: np.ndarray, iq: np.ndarray, count: int) -> None:
    if ids.ndim != 1 or len(ids) != count or ids.dtype.kind != 'U':
        raise ValueError('Opaque IDs must be a count-matched Unicode vector')
    if any(not str(value) for value in ids) or len(set(ids.tolist())) != count:
        raise ValueError('Opaque IDs must be nonempty and have no duplicate ID')
    if iq.dtype != np.dtype('float32') or iq.shape != (count, 2, 256):
        raise ValueError('Received IQ dtype/shape/count must be float32[N,2,256]')
    _finite(iq)


def load_public(root: str | Path) -> tuple[dict[str, Any], np.ndarray, np.memmap]:
    """Read only the public manifest, opaque IDs and read-only received memmap."""
    root = Path(root)
    if set(p.name for p in root.iterdir()) != PUBLIC_FILES:
        raise ValueError('Public input directory must contain only the three public files')
    manifest = _read_json(root / 'manifest.json')
    if not isinstance(manifest, dict) or set(manifest) != PUBLIC_FIELDS:
        raise ValueError('Public manifest fields do not match the whitelist')
    count = manifest['count']
    if type(count) is not int or count <= 0:
        raise ValueError('Public manifest count must be a positive integer')
    if (manifest['schema'] != 'phase3_min_v1'
            or manifest['data_status'] != 'VALIDATED_ONCE'
            or manifest['processing_route'] != 'residual'
            or manifest['mode'] != 'post_sync'
            or manifest['equalization_enabled'] is not False
            or manifest['single_observation'] is not True
            or manifest['fs_hz'] != 25000000
            or manifest['iq_shape'] != [count, 2, 256]):
        raise ValueError('Public manifest schema, shape or observation contract differs')
    _classes(manifest['registered_classes'], 'registered', count=6)
    if any(not isinstance(manifest[k], str) or not manifest[k] for k in ('capsule_id', 'split_id')):
        raise ValueError('Public manifest requires capsule_id and split_id')
    ids = np.load(root / 'ids.npy', allow_pickle=False)
    iq = np.load(root / 'received.npy', mmap_mode='r', allow_pickle=False)
    if not isinstance(iq, np.memmap):
        raise ValueError('Received input must be an NPY memory map')
    _validate_arrays(ids, iq, count)
    return manifest, ids, iq


def _legacy_manifest(manifest: dict[str, Any]) -> None:
    channel = manifest.get('channel', {})
    checks = manifest.get('checks', {})
    if (manifest.get('protocol_schema') != 'p2_min_v1'
            or manifest.get('phase2_data_status') != 'VALIDATED_ONCE'
            or not manifest.get('capsule_id')
            or manifest.get('signal_shape') != [2, 256]
            or channel.get('route') != 'residual'
            or channel.get('mode') != 'post_sync'
            or channel.get('equalization_enabled') is not False
            or channel.get('fs_hz') != 25000000
            or any(checks.get(k) is not True for k in (
                'unique_physical_observation', 'scenes_physically_disjoint',
                'support_query_physically_disjoint', 'query_labels_absent',
                'source_paths_absent', 'finite_received'))):
        raise ValueError('Legacy capsule lacks the matching validated observation contract')


def _canonical(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    index = row.get('signal_index')
    if type(index) is not int or index < 0:
        raise ValueError('Invalid canonical physical signal index')
    for key in ('transmitter', 'receiver', 'day'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f'Canonical physical record has no {key}')
    # The validated legacy capsule has a single equalized=1 representation.
    return row['transmitter'], row['receiver'], row['day'], 1, index


def build_from_practical_capsule(
    source_root: str | Path,
    output_root: str | Path,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
    receivers: Sequence[str | int],
    evaluation_seed: int = 2026102801,
    include_discovery: bool = False,
) -> dict[str, Any]:
    """Copy existing query IQ without synthesis, adaptation or label exposure.

    The legacy capsule proves its own support/query disjointness, but does not
    certify that its support records never appeared in another historical
    query.  Requesting discovery therefore returns a refusal alongside the
    usable evaluation.  No historical query becomes training data.
    """
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f'Output root already exists: {output}')
    known = _classes(known_classes, 'known', count=6)
    unknown = _classes(unknown_classes, 'unknown')
    if set(known) & set(unknown):
        raise ValueError('Known and unknown class sets overlap')
    receiver_names = _receivers(receivers)
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise ValueError('Evaluation seed must be a nonnegative integer')
    source = Path(source_root)
    if (source / 'capsule').is_dir():
        source = source / 'capsule'
    manifest = _read_json(source / 'manifest.json')
    _legacy_manifest(manifest)
    source_report = _read_json(source.parent / 'builder_report.json')
    config = source_report.get('config', {})
    if config.get('old_classes') != known:
        raise ValueError('Known class order differs from the historical source roster')
    source_receivers = config.get('source_receivers')
    if (not isinstance(source_receivers, list) or not source_receivers
            or set(source_receivers) & set(receiver_names)):
        raise ValueError('Source and target receiver boundary is invalid or unverified')
    ledger = _read_json(source.parent / 'score_only' / 'truth.json')
    if not isinstance(ledger, dict):
        raise ValueError('Legacy builder truth ledger must be ID-keyed')
    with np.load(source / 'received.npz', allow_pickle=False) as stored:
        if set(stored.files) != {'iq', 'ids'}:
            raise ValueError('Legacy received NPZ must contain only IQ and IDs')
        source_ids, source_iq = stored['ids'], stored['iq']
    count = manifest.get('received_count')
    if type(count) is not int or count <= 0:
        raise ValueError('Legacy received count is invalid')
    _validate_arrays(source_ids, source_iq, count)
    if len(ledger) != count or set(ledger) != set(source_ids.tolist()):
        raise ValueError('Legacy ledger and physical observation IDs differ')
    canonical_seen = set()
    positions = set()
    selected = []
    available = set()
    for source_id, row in ledger.items():
        index = row.get('index')
        if (type(index) is not int or not 0 <= index < count
                or index in positions or str(source_ids[index]) != source_id
                or row.get('physical_id') != source_id):
            raise ValueError('Legacy physical ID/index mapping differs')
        positions.add(index)
        canonical = _canonical(row)
        if canonical in canonical_seen:
            raise ValueError('Canonical physical record is reused across role or scene')
        canonical_seen.add(canonical)
        if (row.get('pool_role') not in ('query', 'support_pool')
                or row.get('scene') not in manifest['scenarios']):
            raise ValueError('Legacy physical role or scene is invalid')
        if row['receiver'] in receiver_names:
            available.add(row['transmitter'])
            if row['transmitter'] in known + unknown and row['pool_role'] == 'query':
                selected.append((index, source_id, row, canonical))
    if not set(known + unknown) <= available:
        raise ValueError('Requested known or unknown class is absent from the selected receivers')
    coverage = {(r['receiver'], r['transmitter'], r['scene']) for _, _, r, _ in selected}
    expected = {(rx, tx, scene) for rx in receiver_names for tx in known + unknown
                for scene in manifest['scenarios']}
    if coverage != expected:
        raise ValueError('Selected query class/receiver/scenario coverage is incomplete')
    order = np.random.default_rng(evaluation_seed).permutation(len(selected))
    selected = [selected[int(i)] for i in order]
    capsule_id = 'phase3-' + _identity(dict(source=manifest['capsule_id'],
        known=known, unknown=unknown, receivers=receiver_names, seed=evaluation_seed,
        query_ids=[sid for _, sid, _, _ in selected]))
    opaque = np.asarray(['p3-' + _identity([manifest['capsule_id'], evaluation_seed, sid])
                         for _, sid, _, _ in selected])
    public_manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE',
        capsule_id=capsule_id, split_id=_identity([capsule_id, 'evaluation']),
        count=len(selected), iq_shape=[len(selected), 2, 256],
        registered_classes=known, processing_route='residual', mode='post_sync',
        equalization_enabled=False, fs_hz=25000000, single_observation=True)
    public = output / 'evaluation' / 'public'
    truth_root = output / 'evaluation' / 'truth'
    output.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True)
    truth_root.mkdir()
    np.save(public / 'received.npy', source_iq[[i for i, _, _, _ in selected]])
    np.save(public / 'ids.npy', opaque)
    _write_json(public / 'manifest.json', public_manifest)
    labels = {name: i for i, name in enumerate(known + unknown)}
    truth_path = truth_root / 'truth.npz'
    np.savez(truth_path, ids=opaque,
        y=np.asarray([labels[r['transmitter']] for _, _, r, _ in selected], dtype=np.int64),
        receiver=np.asarray([MAIN_RX_NAMES.index(r['receiver']) for _, _, r, _ in selected],
                            dtype=np.int64),
        scenario=np.asarray([r['scene'] for _, _, r, _ in selected]))
    discovery_status = ('REFUSED_HISTORICAL_QUERY_PROVENANCE_UNVERIFIED'
                        if include_discovery else 'NOT_REQUESTED')
    report_path = output / 'builder_report.json'
    _write_json(report_path, dict(source_root=str(source),
        source_capsule_id=manifest['capsule_id'], historical_exposed=True, non_blind=True,
        observation_action='COPY_EXISTING_RECEIVED_ONLY', channel_generated=False,
        source_target_receivers_disjoint=True, canonical_physical_unique=True,
        legacy_support_query_canonical_disjoint=True, historical_query_fit=False,
        evaluation_seed=evaluation_seed, known_classes=known, unknown_classes=unknown,
        selected_receivers=receiver_names, discovery_status=discovery_status,
        discovery_reason=('Legacy validation does not establish support exclusion from '
            'all other historical queries; no discovery input was released.'
            if include_discovery else None),
        evaluation_mapping=[dict(opaque_id=str(oid), source_physical_id=sid,
            source_role=r['pool_role'], canonical_record=list(canonical))
            for oid, (_, sid, r, canonical) in zip(opaque, selected)]))
    load_public(public)
    return dict(evaluation=dict(public_root=str(public), truth_path=str(truth_path),
                               manifest=public_manifest),
                discovery=None, discovery_status=discovery_status,
                builder_report=str(report_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--known-classes', nargs=6, required=True)
    parser.add_argument('--unknown-classes', nargs='+', required=True)
    parser.add_argument('--receivers', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=2026102801)
    parser.add_argument('--include-discovery', action='store_true')
    args = parser.parse_args()
    result = build_from_practical_capsule(args.source_root, args.output_root,
        args.known_classes, args.unknown_classes, args.receivers,
        evaluation_seed=args.seed, include_discovery=args.include_discovery)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
