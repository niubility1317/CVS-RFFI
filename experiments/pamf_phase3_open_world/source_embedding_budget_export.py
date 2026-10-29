"""Frozen E60 eight-branch export for the independent 2220-step candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import runner, source_open_export as source, source_open_run as parent
from . import source_open_embedding_train as trainer, source_embedding_readout_export as previous


RUN_ID = '20261029-phase3-source-embedding-budget-s2026092701-r01'
KEYS = set(previous.KEYS)
COUNTS = dict(source.COUNTS)
ARMS = ('supervised_only', 'uniform_oe')
validate_config = previous.validate_config
_parent = previous._parent


def validate_encoder_identity(identity, arm):
    if (not isinstance(identity, dict) or arm not in ARMS or identity.get('arm') != arm
            or identity.get('run_id') != RUN_ID or identity.get('row_id') != arm + '-s2026092701'
            or not isinstance(identity.get('code_commit'), str) or len(identity['code_commit']) != 40
            or any(c not in '0123456789abcdef' for c in identity['code_commit'])
            or not isinstance(identity.get('checkpoint'), str)
            or type(identity.get('updates')) is not int or identity['updates'] != 2220
            or type(identity.get('cumulative_updates')) is not int or identity['cumulative_updates'] != 13320):
        raise ValueError('The own fixed E50+2220 source encoder identity is required')
    path = Path(identity['checkpoint'])
    if (path.name != 'final.pt' or path.parent.name != 'train' or path.parent.parent.name != arm
            or path.parent.parent.parent.name != 'rows' or path.parent.parent.parent.parent.name != RUN_ID):
        raise ValueError('Budget checkpoint must belong to its own matching arm')
    return identity


def _encoder(config, device):
    train_config = runner._read(config['train_config']); trainer.validate_config(train_config)
    arm = train_config['arm']
    identity = {key: train_config[key] for key in ('run_id', 'row_id', 'arm', 'code_commit')}
    identity.update(checkpoint=config['checkpoint'], updates=train_config['steps'], cumulative_updates=11100 + train_config['steps'])
    validate_encoder_identity(identity, arm)
    if (any(config[key] != train_config[key] for key in ('run_id', 'row_id', 'code_commit'))
            or Path(config['checkpoint']).resolve() != (Path(train_config['output_root']) / 'final.pt').resolve()):
        raise ValueError('Export and own budget row/checkpoint identity differ')
    marker = runner._read(Path(train_config['output_root']) / 'complete.json')
    if (marker.get('status') != 'SOURCE_OPEN_EMBEDDING_FROZEN' or marker.get('schema') != trainer.SCHEMA
            or marker.get('config') != train_config or marker.get('arm') != arm or marker.get('updates') != 2220
            or marker.get('parent_updates') != 11100 or marker.get('cumulative_updates') != 13320
            or marker.get('increment_epoch') != 60 or marker.get('optimizer_reset') is not True
            or marker.get('source_only') is not True or marker.get('default_head_joint') is not True
            or marker.get('identity_head_replaced') is not False or marker.get('model_classes') != 6
            or marker.get('parameters') != 232993 or any(marker.get(key) != 0 for key in
                ('U_reads', 'validation_reads', 'novel_validation_reads', 'target_reads', 'OE_truth_reads'))):
        raise ValueError('Complete own E60 source-only cold-start budget required')
    wrapper = trainer.load_trained_model(train_config, config['checkpoint'], device)
    reader = wrapper.id_backbone.encoder; source._validate_model(reader)
    return reader, train_config, identity


def export(config, device='cuda:0'):
    validate_config(config); output = Path(config['output_root'])
    if output.exists(): raise FileExistsError('A new immutable E60 branch export root is required')
    reader, train_config, encoder = _encoder(config, device)
    original = _parent(config, train_config); contract = trainer._contract(train_config)
    labeled = runner._source_dataset(train_config, 'L_s', contract)
    lmeta = source._source_meta(labeled, 'L_s', contract)
    output.mkdir(parents=True, exist_ok=False)
    _, logits = source.extract_branches(reader, source._tensor(labeled.x[:2], device))
    runner._write(output / 'smoke.json', dict(status='PASS', actual_checkpoint=True, source_samples=2,
        encoder_state_identity=encoder, model_classes=6, parameters=232993, logits_shape=list(logits.shape),
        validation_reads=0, OE_reads=0, target_reads=0, optimizer_updates=0))
    print('SOURCE_BUDGET_BRANCH_SMOKE_PASS', config['row_id'], 2, flush=True)
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
    if handles != original['source_OE_handles']: raise ValueError('Original source4/novel3 capsules must be retained')
    for role in ('exposure_train', 'novel_validation'):
        iq, meta = source._oe_public(config['oe_data_root'], role)
        try:
            if seen & set(meta['ids'].tolist()): raise ValueError('Source/OE physical roles overlap')
            seen.update(meta['ids'].tolist())
            source._export_role(reader, iq, meta, role, export_config, device, output)
        finally:
            if isinstance(iq, np.memmap): iq._mmap.close()
    source._validate_model(reader)
    result = dict(status='SOURCE_BUDGET_BRANCH_FEATURES_EXPORTED', config=config, arm=train_config['arm'],
        source_only=True, target_reads=0, original_U_s_reads=0, OE_truth_reads=0, optimizer_updates=0,
        new_encoder_updates=0, new_identity_head_updates=0, head_changed=False, default_head_replaced=False,
        historical_encoder_updates=2220, historical_identity_head_updates=2220, model_classes=6, parameters=232993,
        encoder_state_identity=encoder, parent_export_identity=original, source_OE_handles=handles,
        counts=COUNTS.copy(), feature_refs=source.FEATURE_REFS.copy(), branch_names=list(source.BRANCHES),
        dimensions=dict(source.DIMENSIONS), model_forward_samples=sum(COUNTS.values()) + 2,
        source_L_smoke_samples=2, source_OE_TX_roster_read=False, source_validation_updates=0,
        readout_normalization='raw new-E60 branches; exporter fits no statistics')
    runner._write(output / 'complete.json', result); return result


run = export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv); result = export(runner._read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], arm=result['arm']))); return 0


if __name__ == '__main__': raise SystemExit(main())
