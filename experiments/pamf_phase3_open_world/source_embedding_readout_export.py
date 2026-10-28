"""Frozen E51 eight-branch export for matched source membership readouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import runner, source_open_export as source, source_open_run as parent
from . import source_open_embedding_train as trainer, source_embedding_eval as embedding


KEYS = {'run_id', 'row_id', 'code_commit', 'train_config', 'checkpoint', 'parent_run_root', 'oe_data_root', 'output_root'}
COUNTS = dict(source.COUNTS)
EMBEDDING_RUN = '20261029-phase3-source-embedding-increment-s2026092701-r02'
EMBEDDING_COMMIT = '28d53ea9b38c19f642c59e31f178a3f65f40894d'
ARMS = ('supervised_only', 'uniform_oe')


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != KEYS
            or any(not isinstance(config[key], str) or not config[key].strip() for key in KEYS)
            or len(config['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in config['code_commit'])):
        raise ValueError('Only the eight declared frozen E51 source export inputs are accepted')
    return config


def validate_encoder_identity(identity, arm):
    if (not isinstance(identity, dict) or arm not in ARMS or identity.get('arm') != arm
            or identity.get('run_id') != EMBEDDING_RUN or identity.get('row_id') != arm + '-s2026092701'
            or identity.get('code_commit') != EMBEDDING_COMMIT or not isinstance(identity.get('checkpoint'), str)):
        raise ValueError('The two fixed, completed r02 encoder states are required')
    path = Path(identity['checkpoint'])
    if (path.name != 'final.pt' or path.parent.name != 'train' or path.parent.parent.name != arm
            or path.parent.parent.parent.name != 'rows' or path.parent.parent.parent.parent.name != EMBEDDING_RUN):
        raise ValueError('Encoder checkpoint must belong to the matching fixed r02 row')
    return identity


def _encoder(config, device):
    train_config = runner._read(config['train_config'])
    checked = {key: train_config[key] for key in ('run_id', 'row_id', 'code_commit')}
    checked.update(train_config=config['train_config'], checkpoint=config['checkpoint'])
    train_config, marker = embedding._training(checked)
    identity = {key: train_config[key] for key in ('run_id', 'row_id', 'arm', 'code_commit')}
    identity.update(checkpoint=config['checkpoint'], updates=222, cumulative_updates=11322)
    validate_encoder_identity(identity, train_config['arm'])
    if config['row_id'] != identity['row_id']:
        raise ValueError('New readout row and frozen encoder arm differ')
    wrapper = trainer.load_trained_model(train_config, config['checkpoint'], device)
    reader = wrapper.id_backbone.encoder
    source._validate_model(reader)
    return reader, train_config, identity


def _parent(config, train_config):
    _, identity = embedding._parent(config, train_config)
    return identity


def export(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists(): raise FileExistsError('A new immutable E51 feature export root is required')
    reader, train_config, encoder = _encoder(config, device)
    original = _parent(config, train_config)
    contract = trainer._contract(train_config)
    labeled = runner._source_dataset(train_config, 'L_s', contract)
    lmeta = source._source_meta(labeled, 'L_s', contract)
    output.mkdir(parents=True, exist_ok=False)
    _, logits = source.extract_branches(reader, source._tensor(labeled.x[:2], device))
    runner._write(output / 'smoke.json', dict(status='PASS', actual_checkpoint=True, source_samples=2,
        encoder_state_identity=encoder, model_classes=6, parameters=232993, logits_shape=list(logits.shape),
        validation_reads=0, OE_reads=0, target_reads=0, optimizer_updates=0))
    print('SOURCE_EMBEDDING_BRANCH_SMOKE_PASS', config['row_id'], 2, flush=True)
    export_config = dict(batch_size=128, row_id=config['row_id'])
    source._export_role(reader, labeled.x, lmeta, 'source_L', export_config, device, output)
    seen = set(lmeta['ids'].tolist()); del labeled
    validation = runner._source_dataset(train_config, 'V', contract)
    meta = source._source_meta(validation, 'V', contract)
    if seen & set(meta['ids'].tolist()): raise ValueError('Source L/V physical roles overlap')
    seen.update(meta['ids'].tolist())
    source._export_role(reader, validation.x, meta, 'source_V', export_config, device, output)
    del validation
    handles = source._oe_handles(config['oe_data_root'])
    if handles != original['source_OE_handles']:
        raise ValueError('The unchanged original source4/novel3 capsules are required')
    for role in ('exposure_train', 'novel_validation'):
        iq, meta = source._oe_public(config['oe_data_root'], role)
        try:
            if seen & set(meta['ids'].tolist()): raise ValueError('Source/OE physical roles overlap')
            seen.update(meta['ids'].tolist())
            source._export_role(reader, iq, meta, role, export_config, device, output)
        finally:
            if isinstance(iq, np.memmap): iq._mmap.close()
    source._validate_model(reader)
    result = dict(status='SOURCE_EMBEDDING_BRANCH_FEATURES_EXPORTED', config=config, arm=train_config['arm'],
        source_only=True, target_reads=0, original_U_s_reads=0, OE_truth_reads=0, optimizer_updates=0,
        new_encoder_updates=0, new_identity_head_updates=0, head_changed=False, default_head_replaced=False,
        historical_encoder_updates=222, historical_identity_head_updates=222, model_classes=6, parameters=232993,
        encoder_state_identity=encoder, parent_export_identity=original, source_OE_handles=handles,
        counts=COUNTS.copy(), feature_refs=source.FEATURE_REFS.copy(), branch_names=list(source.BRANCHES),
        dimensions=dict(source.DIMENSIONS), model_forward_samples=sum(COUNTS.values()) + 2,
        source_L_smoke_samples=2, source_OE_TX_roster_read=False, source_validation_updates=0,
        readout_normalization='raw new-encoder branches; no feature statistics fitted by exporter')
    runner._write(output / 'complete.json', result)
    return result


run = export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv); result = export(runner._read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], arm=result['arm']))); return 0


if __name__ == '__main__': raise SystemExit(main())
