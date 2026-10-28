"""Export only validated source21 OE with the original frozen six-class F."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import runner
from . import source_open_export as source
from . import source_open_run as parent
from .source_domain_oe_readout import domains_from_metadata


KEYS = {'run_id', 'row_id', 'code_commit', 'parent_run_root', 'diverse_data_root', 'output_root'}
PARENT_RUN = '20261029-phase3-source-open-readout-s2026092701-r01'
PARENT_ROW = 'source-open-readout-s2026092701'
COUNT = 6134
CAPSULE_ID = 'source-oe-diverse-9dd73fa533b19a64fd4db8d9'
SPLIT_ID = '5e0f620d5c7c72512e9e9edd'
FEATURE_REF = 'exposure_train/public/features.npz'


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the six declared source21 export inputs are accepted')
    if any(not isinstance(config[key], str) or not config[key].strip() for key in KEYS):
        raise ValueError('Nonempty source21 export strings required')
    if len(config['code_commit']) != 40:
        raise ValueError('Actual execution Git commit required')
    return config


def _parent(config):
    marker = parent._export_marker(Path(config['parent_run_root']) / 'export')
    previous = marker['config']
    source.validate_config(previous)
    if (previous['run_id'] != PARENT_RUN or previous['row_id'] != PARENT_ROW
            or previous['model_seed'] != 2026092701):
        raise ValueError('Only the fixed original full-six-class source-open parent is allowed')
    # Preserve the old export/code identity. New execution code is recorded at
    # top level, never substituted into this frozen-cache identity.
    identity = parent._identity(marker)
    base = {key: previous[key] for key in runner.COMMON | runner.SOURCE}
    base['output_root'] = config['output_root']
    return base, identity


def _zero(marker, key):
    if type(marker.get(key)) is not int or marker[key] != 0:
        raise ValueError('Source-only model-free data boundary differs: ' + key)


def _diverse_handle(root):
    marker = runner._read(Path(root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_OE_DIVERSE_DATA_VALIDATED'
            or marker.get('data_status') != 'VALIDATED_ONCE' or marker.get('source_only') is not True
            or type(marker.get('count')) is not int or marker['count'] != COUNT
            or marker.get('counts') != {'exposure_train': COUNT} or marker.get('exposure_train_count') != COUNT):
        raise ValueError('The validated new source21 training-only capsule is required')
    for key in ('target_reads', 'original_L_s_reads', 'original_U_s_reads', 'original_V_reads',
                'novel_validation_IQ_reads', 'model_reads', 'model_predictions', 'optimizer_updates',
                'channel_observations_generated'):
        _zero(marker, key)
    domains = marker.get('source_domains', {})
    if (domains.get('receiver_names') != source.RX_NAMES or domains.get('receiver_indices') != source.MAIN_RX
            or domains.get('dates') != source.DATES or domains.get('day_indices') != [1, 2, 3]
            or domains.get('domain_count') != 15):
        raise ValueError('Original source RX/day domains required')
    handle = marker.get('exposure_train')
    if (not isinstance(handle, dict) or type(handle.get('count')) is not int or handle['count'] != COUNT
            or handle.get('capsule_id') != CAPSULE_ID or handle.get('split_id') != SPLIT_ID
            or handle.get('data_status') != 'VALIDATED_ONCE' or handle.get('iq_dtype') != 'float32'
            or handle.get('iq_shape') != [COUNT, 2, 256]):
        raise ValueError('Source21 minimal capsule/split/count handle differs')
    return {key: handle[key] for key in ('capsule_id', 'split_id', 'count')}


def _public(root):
    # Absolute paths in the old builder marker are provenance only. All actual
    # reads are joined to the explicitly supplied, possibly relocated root.
    public = Path(root) / 'exposure_train/public'
    if {path.name for path in public.iterdir()} != {'iq.npy', 'ids.npy', 'metadata.npz'}:
        raise ValueError('Source21 public whitelist differs')
    ids = np.load(public / 'ids.npy', allow_pickle=False)
    iq = np.load(public / 'iq.npy', allow_pickle=False, mmap_mode='r')
    try:
        with np.load(public / 'metadata.npz', allow_pickle=False) as stored:
            if set(stored.files) != {'rx_i', 'day_i', 'domain'}:
                raise ValueError('Only source RX/day/domain metadata may be public')
            meta = {key: stored[key].copy() for key in stored.files}
        if (ids.shape != (COUNT,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == '')
                or len(set(ids.tolist())) != COUNT or iq.shape != (COUNT, 2, 256) or iq.dtype != np.float32):
            raise ValueError('Source21 unique IDs or FP32 IQ shape differs')
        expected_domains = domains_from_metadata(meta['rx_i'], meta['day_i'])
        domains = meta['domain']
        if (domains.shape != (COUNT,) or domains.dtype.kind not in 'iu'
                or not np.array_equal(domains, expected_domains)
                or set(np.unique(domains).tolist()) != set(range(15))):
            raise ValueError('Source21 Main RX/day domain mapping/coverage differs')
        meta['ids'] = ids
        return iq, meta
    except BaseException:
        if isinstance(iq, np.memmap):
            iq._mmap.close()
        raise


def export(config, device='cuda:0'):
    validate_config(config)
    if Path(config['output_root']).exists():
        raise FileExistsError('A new immutable source21 export root is required')
    base, identity = _parent(config)
    contract, output = runner._prepare('calibrate', base)
    if (contract['counts']['L_s'] != source.COUNTS['source_L'] or contract['counts']['V'] != source.COUNTS['source_V']
            or contract.get('source_rxs') != source.MAIN_RX or contract.get('source_days') != [1, 2, 3]):
        raise ValueError('Original full source contract required for real L smoke')
    reader = runner._load_model(base, contract, device)
    source._validate_model(reader)
    labeled = runner._source_dataset(base, 'L_s', contract)
    lmeta = source._source_meta(labeled, 'L_s', contract)
    output.mkdir(parents=True, exist_ok=False)
    branches, logits = source.extract_branches(reader, source._tensor(labeled.x[:2], device))
    runner._write(output / 'smoke.json', dict(status='PASS', actual_checkpoint=True, source_samples=2,
        source_role='original_L_s', validation_reads=0, OE_reads=0, original_U_s_reads=0,
        target_reads=0, optimizer_updates=0, model_classes=6, parameters=232993,
        branch_dimensions=source.DIMENSIONS, logit_shape=list(logits.shape)))
    print('SOURCE_DIVERSE_REAL_F_SMOKE_PASS', config['row_id'], 2, flush=True)
    del labeled
    handle = _diverse_handle(config['diverse_data_root'])
    values, meta = _public(config['diverse_data_root'])
    try:
        if set(meta['ids'].tolist()) & set(lmeta['ids'].tolist()):
            raise ValueError('Source L and source21 opaque IDs overlap')
        export_config = dict(base, row_id=config['row_id'])
        source._export_role(reader, values, meta, 'exposure_train', export_config, device, output)
    finally:
        if isinstance(values, np.memmap):
            values._mmap.close()
    source._validate_model(reader)
    result = dict(status='SOURCE_DIVERSE_OE_FEATURES_EXPORTED', run_id=config['run_id'], row_id=config['row_id'],
        code_commit=config['code_commit'], config=config, count=COUNT, feature_ref=FEATURE_REF,
        parent_export_identity=identity, source_OE_handle=handle, source_only=True,
        target_reads=0, original_U_s_reads=0, OE_truth_reads=0, optimizer_updates=0,
        encoder_updates=0, identity_head_updates=0, head_changed=False, model_classes=6, parameters=232993,
        source_L_smoke_samples=2, source_L_loaded_for_smoke=True,
        validation_IQ_reads=0, old_OE_train_reads=0, novel_validation_reads=0,
        model_forward_samples=COUNT + 2, model_forward_roles=dict(original_L_smoke=2, diverse_exposure_train=COUNT),
        branch_names=list(source.BRANCHES), dimensions=dict(source.DIMENSIONS), source_OE_TX_roster_read=False)
    runner._write(output / 'complete.json', result)
    print('SOURCE_DIVERSE_OE_FEATURES_EXPORTED', config['row_id'], COUNT, flush=True)
    return result


run = export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = export(runner._read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], count=result['count'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
