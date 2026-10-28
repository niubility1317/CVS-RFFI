"""Private source-proxy selection and frozen full-feature export for Phase3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import branch_export, source_fold
from .data import _identity


KEYS = branch_export.KEYS | {'branch_export_root'}
FROZEN_KEYS = {'train_features__fused160', 'train_y', 'train_ids',
    'validation_features__fused160', 'validation_logits', 'validation_y',
    'validation_ids', 'validation_receivers'}
_read, _write = branch_export._read, branch_export._write


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the declared source stream export inputs are accepted')
    if not isinstance(config['branch_export_root'], str) or not config['branch_export_root']:
        raise ValueError('The own frozen branch export root is required')
    branch_export.validate_config({key: config[key] for key in branch_export.KEYS})
    return config


def _load_labeled(config, contract):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    import torch
    with np.load(Path(config['source_cache']) / 'L_s_metadata.npz', allow_pickle=False) as stored:
        metadata = {name: stored[name].copy() for name in stored.files}
    source_fold._metadata(metadata, 'L_s', contract)
    labeled = SourceDataset(config['source_cache'], 'L_s', contract)
    if (not torch.is_tensor(labeled.x) or tuple(labeled.x.shape) != (6300, 2, 256)
            or labeled.x.dtype != torch.float32 or not bool(torch.isfinite(labeled.x).all())):
        raise ValueError('The original private L_s loader must retain finite FP32[6300,2,256]')
    if not isinstance(labeled.metadata, dict) or set(labeled.metadata) != set(metadata):
        raise ValueError('Loaded source L metadata differs')
    if any(not np.array_equal(labeled.metadata[key], value) for key, value in metadata.items()):
        raise ValueError('Loaded source L physical ordering differs')
    return metadata, labeled.x


def _select_l(config, contract, metadata, values):
    import torch
    source_fold._spec_and_contract(contract, config['fold'])
    metadata = source_fold._metadata(metadata, 'L_s', contract)
    if (not torch.is_tensor(values) or tuple(values.shape) != (6300, 2, 256)
            or values.dtype != torch.float32):
        raise ValueError('Original private L_s tensor shape differs')
    train = np.flatnonzero(np.isin(metadata['y'], [2, 3, 4, 5]) & (metadata['rx_i'] != 8))
    discovery = np.flatnonzero((metadata['rx_i'] == 8) & np.isin(metadata['y'], [0, 2, 3, 4, 5]))
    hidden = np.flatnonzero((metadata['rx_i'] == 8) & (metadata['y'] == 1))
    if (len(train) != 3360 or len(discovery) != 1050 or len(hidden) != 210
            or set(train.tolist()) & set(discovery.tolist())):
        raise ValueError('Private source training/discovery allocation differs')
    return metadata, train, discovery, hidden


def _load_frozen(config, parent):
    root = Path(config['branch_export_root'])
    marker = _read(root / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_BRANCHES_EXPORTED'
            or marker.get('source_only') is not True or marker.get('target_reads') != 0
            or marker.get('optimizer_updates') != 0 or marker.get('head_changed') is not False
            or marker.get('parent_epochs') != 20 or marker.get('parent_updates') != 540
            or marker.get('train_count') != 3360 or marker.get('validation_count') != 27000
            or marker.get('heldout_source_rx') != 8 or marker.get('features_ref') != 'source_branches.npz'
            or marker.get('source_manifest') != parent['manifest']):
        raise ValueError('The own complete source branch export is required')
    previous = marker.get('config')
    branch_export.validate_config(previous)
    for key in ('source_model_root', 'source_cache', 'source_contract', 'code_runtime', 'fold', 'model_seed'):
        if previous[key] != config[key]:
            raise ValueError('Frozen branch source identity differs: ' + key)
    if Path(previous['output_root']).resolve() != root.resolve():
        raise ValueError('Frozen branch output root identity differs')
    # Read exactly the eight needed arrays; other historical branches are untouched.
    with np.load(root / 'source_branches.npz', allow_pickle=False) as stored:
        if not FROZEN_KEYS.issubset(stored.files):
            raise ValueError('The required frozen full-feature fields are absent')
        return {name: stored[name].copy() for name in FROZEN_KEYS}


def _validate_frozen(config, contract, lmeta, train_indices, arrays, parent):
    with np.load(Path(config['source_cache']) / 'V_metadata.npz', allow_pickle=False) as stored:
        vmeta = {name: stored[name].copy() for name in stored.files}
    selected = source_fold.select_fold(lmeta, vmeta, contract, config['fold'])
    if selected['manifest'] != parent['manifest'] or not np.array_equal(selected['train_indices'], train_indices):
        raise ValueError('The current source fold differs from the frozen own parent')
    for key in ('train_y', 'train_ids', 'validation_y', 'validation_ids', 'validation_receivers'):
        if not np.array_equal(arrays[key], selected[key]):
            raise ValueError('Frozen full-feature source association differs: ' + key)
    for key, shape in (('train_features__fused160', (3360, 160)),
                       ('validation_features__fused160', (27000, 160)), ('validation_logits', (27000, 4))):
        value = arrays[key]
        if value.dtype != np.dtype('float32') or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Finite frozen full-feature FP32 shape differs: ' + key)
    for key in ('train_ids', 'validation_ids'):
        if arrays[key].dtype.kind != 'U':
            raise ValueError('Frozen canonical source IDs must remain Unicode')
    for key in ('train_y', 'validation_y', 'validation_receivers'):
        if arrays[key].dtype != np.dtype('int64'):
            raise ValueError('Frozen source labels/RX must remain int64')
    calibration = (arrays['validation_y'] < 4) & (arrays['validation_receivers'] != 8)
    future = arrays['validation_receivers'] == 8
    if int(calibration.sum()) != 14400 or int(future.sum()) != 5400 or np.any(calibration & future):
        raise ValueError('Seen-source calibration/future held-out allocations differ')
    return selected, vmeta, calibration, future


def _forward(model, values):
    import torch
    if (model.training or any(parameter.requires_grad for parameter in model.parameters())
            or values.dtype != torch.float32 or tuple(values.shape[1:]) != (2, 256)
            or not len(values) or not bool(torch.isfinite(values).all())):
        raise ValueError('Frozen own model and finite FP32 source I/Q required')
    with torch.no_grad():
        features = model.features(values)
        logits = model.classify_features(features)
    for value, shape in ((features, (len(values), 160)), (logits, (len(values), 4))):
        if value.dtype != torch.float32 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Complete own fused features/default-head logits differ')
    return (np.asarray(features.detach().cpu().tolist(), dtype=np.float32),
            np.asarray(logits.detach().cpu().tolist(), dtype=np.float32))


def _index(values, indices):
    import torch
    return values.index_select(0, torch.tensor(indices.tolist(), dtype=torch.long, device=values.device))


def _opaque(canonical):
    ids = np.asarray(['p3s-' + _identity(['source_stream_proxy_physical', str(value)]) for value in canonical])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Opaque source stream ID collision')
    return ids


def _write_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)


def _write_role(output, role, features, logits, canonical, labels):
    ids = _opaque(canonical)
    order = np.argsort(ids, kind='stable')
    ids, features, logits, canonical, labels = ids[order], features[order], logits[order], canonical[order], labels[order]
    public, truth = output / role / 'public', output / role / 'truth'
    public.mkdir(parents=True)
    truth.mkdir()
    _write_npz(public / 'features.npz', ids=ids, features=features, logits=logits)
    _write_npz(truth / 'truth.npz', ids=ids, y=labels.astype(np.int64))
    mapping = [dict(opaque_id=str(opaque), canonical_source_id=str(original), model_label=int(label))
               for opaque, original, label in zip(ids, canonical, labels)]
    return mapping


def export(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Source stream export cannot be overwritten: ' + str(output))
    base = {key: config[key] for key in branch_export.KEYS}
    parent = branch_export._parent(base)
    model = branch_export._load_model(base, parent, device)
    contract = _read(config['source_contract'])
    source_fold._spec_and_contract(contract, config['fold'])
    lmeta, labeled = _load_labeled(config, contract)
    lmeta, train_indices, u_indices, hidden_indices = _select_l(config, contract, lmeta, labeled)
    output.mkdir(parents=True, exist_ok=False)
    smoke_features, smoke_logits = _forward(model, _index(labeled, train_indices[:2]).to(device=device))
    _write(output / 'smoke.json', dict(status='PASS', source_samples=2, source_role='selected_L_s_train',
        feature_shape=list(smoke_features.shape), logit_shape=list(smoke_logits.shape), validation_reads=0,
        target_reads=0, optimizer_updates=0, own_checkpoint=True))
    print('PHASE3_SOURCE_STREAM_SMOKE_PASS', config['row_id'], flush=True)
    arrays = _load_frozen(config, parent)
    selected, vmeta, calibration, future = _validate_frozen(config, contract, lmeta, train_indices, arrays, parent)
    canonical_roles = dict(train=lmeta['ids'][train_indices], calibration=arrays['validation_ids'][calibration],
        discovery=lmeta['ids'][u_indices], evaluation=arrays['validation_ids'][future])
    ids = [str(value) for values in canonical_roles.values() for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError('Source L/calibration/U/future physical roles must be mutually disjoint')
    feature_parts, logit_parts = [], []
    for start in range(0, len(u_indices), config['batch_size']):
        index = u_indices[start:start + config['batch_size']]
        features, logits = _forward(model, _index(labeled, index).to(device=device))
        feature_parts.append(features); logit_parts.append(logits)
        print('PHASE3_SOURCE_STREAM_U_EXPORT', config['row_id'], start + len(index), len(u_indices), flush=True)
    remap = np.asarray([4, 5, 0, 1, 2, 3], dtype=np.int64)
    _write_npz(output / 'source_inputs.npz', train_features=arrays['train_features__fused160'],
        train_y=arrays['train_y'], calibration_features=arrays['validation_features__fused160'][calibration])
    mapping = dict(discovery=_write_role(output, 'discovery', np.concatenate(feature_parts),
        np.concatenate(logit_parts), canonical_roles['discovery'], remap[lmeta['y'][u_indices]]),
        evaluation=_write_role(output, 'evaluation', arrays['validation_features__fused160'][future],
            arrays['validation_logits'][future], canonical_roles['evaluation'], arrays['validation_y'][future]))
    public_ids = [row['opaque_id'] for values in mapping.values() for row in values]
    if len(set(public_ids)) != len(public_ids):
        raise ValueError('U/future opaque IDs must remain disjoint')
    _write(output / 'fold_manifest.json', dict(source_manifest=selected['manifest'],
        canonical_train_ids=canonical_roles['train'].tolist(), canonical_calibration_ids=canonical_roles['calibration'].tolist(),
        hidden_rx8_label1_L_ids=lmeta['ids'][hidden_indices].tolist(), opaque_to_canonical=mapping))
    _write(output / 'builder_report.json', dict(config=config, source_only=True,
        full_L_loaded_for_private_selection_only=True, original_L_count=6300,
        original_F0_L_hidden_label_proxy=True, original_U_s_opened=False,
        hidden_rx8_label1_L_count=210, hidden_rx8_label1_model_forwards=0,
        model_forward_samples=1052, model_forward_roles={'selected_train_smoke': 2, 'discovery': 1050},
        validation_IQ_loaded=False, validation_model_forwards=0, frozen_branch_arrays_read=sorted(FROZEN_KEYS),
        calibration_role='known V from source RX1/3/4/6 only', future_role='readonly V RX8 only',
        source_label_order=[2, 3, 4, 5, 0, 1], target_reads=0, optimizer_updates=0,
        new_blind_target_evidence=False,
        cross_split_historical_exposure='RX8 was a target receiver in other historical splits; source proxy only',
        historical_F1_F2_query_artifacts_read=False, physical_roles_disjoint=True,
        private_truth_not_provided_to_algorithm=True))
    result = dict(status='SOURCE_STREAM_INPUTS_EXPORTED', source_only=True, target_reads=0,
        optimizer_updates=0, head_changed=False, train_count=3360, calibration_count=14400,
        discovery_count=1050, future_count=5400, heldout_source_rx=8, parent_epochs=20, parent_updates=540,
        config=config, source_manifest=selected['manifest'], source_inputs_ref='source_inputs.npz',
        discovery_features_ref='discovery/public/features.npz', future_features_ref='evaluation/public/features.npz',
        discovery_truth_ref='discovery/truth/truth.npz', future_truth_ref='evaluation/truth/truth.npz',
        full_L_loaded_for_private_selection_only=True, original_U_s_opened=False,
        new_blind_target_evidence=False, physical_roles_disjoint=True)
    _write(output / 'complete.json', result)
    print('PHASE3_SOURCE_STREAM_INPUTS_EXPORTED', config['row_id'], 1050, 5400, flush=True)
    return result


build = export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = export(_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], row_id=result['config']['row_id'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
