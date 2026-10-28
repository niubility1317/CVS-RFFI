"""Frozen six-class PAMF source/OE branch export; no fitting or truth access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import runner
from .branch_export import DIMENSIONS, BRANCHES


KEYS = runner.COMMON | runner.SOURCE | {'oe_data_root'}
COUNTS = dict(source_L=6300, source_V=27000, exposure_train=4200, novel_validation=3150)
FEATURE_REFS = dict(source_L='source_L/features.npz', source_V='source_V/features.npz',
    exposure_train='exposure_train/public/features.npz', novel_validation='novel_validation/public/features.npz')
MAIN_RX = [1, 3, 4, 6, 8]
RX_NAMES = ['1-19', '18-2', '19-2', '2-19', '3-19']
DATES = ['2021_03_08', '2021_03_15', '2021_03_23']
_read, _write = runner._read, runner._write


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the declared six-class source open export configuration is accepted')
    if not isinstance(config['oe_data_root'], str) or not config['oe_data_root']:
        raise ValueError('The validated source OE root is required')
    runner.validate_config('calibrate', {key: config[key] for key in runner.COMMON | runner.SOURCE})
    return config


def _validate_model(model):
    import torch
    head = model.core.id_backbone.cls_head.weight
    if (tuple(head.shape) != (6, 160) or sum(p.numel() for p in model.parameters()) != 232993
            or any(module.training for module in model.modules())
            or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('Frozen current full six-class FP32 PAMF required')


def extract_branches(model, x):
    """Capture eight raw branches and the complete six-class head in one forward."""
    import torch
    if (model.training or not torch.is_tensor(x) or x.ndim != 3 or tuple(x.shape[1:]) != (2, 256)
            or not len(x) or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
        raise ValueError('Frozen model and finite FP32[B,2,256] input required')
    captured, handles = {}, []
    def record(name, value):
        if name in captured:
            raise ValueError('A branch ran more than once in the complete forward: ' + name)
        captured[name] = value
    def tensor_hook(name):
        def hook(module, inputs, value):
            if not torch.is_tensor(value):
                raise ValueError('Actual tensor branch required: ' + name)
            record(name, value.detach().clone())
        return hook
    def core_hook(module, inputs, value):
        if not isinstance(value, dict) or not {'t_emb', 'f_emb', 'feat_pa'}.issubset(value):
            raise ValueError('Actual complete core auxiliary maps missing')
        record('core', {key: value[key].detach().clone() for key in ('t_emb', 'f_emb', 'feat_pa')})
    def spectrum_hook(module, inputs):
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise ValueError('Actual raw FFT gate input missing')
        record('spectrum', inputs[0].detach().clone())
    try:
        handles.append(model.core.id_backbone.register_forward_hook(core_hook))
        handles.append(model.response.register_forward_hook(tensor_hook('reference')))
        handles.append(model.core.id_backbone.freq_gate.register_forward_pre_hook(spectrum_hook))
        handles.append(model.core.id_backbone.pa_lift.register_forward_hook(tensor_hook('pa_lift')))
        for index in range(3):
            handles.append(model.sequence_pool.views[index].register_forward_hook(tensor_hook('signature' + str(index))))
        with torch.no_grad():
            fused = model.features(x)
            logits = model.classify_features(fused)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {'core', 'reference', 'spectrum', 'pa_lift', 'signature0', 'signature1', 'signature2'}:
        raise ValueError('All raw branches must be from the same complete forward')
    count = len(x)
    if (tuple(captured['spectrum'].shape) != (count, 4, 32)
            or tuple(captured['pa_lift'].shape) != (count, 24, 256)
            or any(tuple(captured['signature' + str(i)].shape) != (count, 112) for i in range(3))):
        raise ValueError('Raw FFT, PA basis or three signature views differ')
    pa = captured['pa_lift']
    branches = dict(fused160=fused.detach().clone(), time160=captured['core']['t_emb'],
        frequency160=captured['core']['f_emb'], pa160=captured['core']['feat_pa'],
        reference29=captured['reference'],
        signature336=torch.cat([captured['signature' + str(i)] for i in range(3)], dim=1),
        spectrum128=captured['spectrum'].flatten(1),
        pa_moments48=torch.cat((pa.mean(-1), pa.std(-1, unbiased=False)), dim=1))
    for name, dimension in DIMENSIONS.items():
        value = branches[name]
        if tuple(value.shape) != (count, dimension) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite FP32 raw branch shape differs: ' + name)
    if bool((branches['reference29'][:, 27] != 0).any()):
        raise ValueError('NoExplicitCFO response coordinate 27 must remain strictly zero')
    if tuple(logits.shape) != (count, 6) or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise ValueError('The full six-class default head is required; core auxiliary logits are insufficient')
    return branches, logits.detach().clone()


def _domains(meta, count):
    for name in ('rx_i', 'day_i', 'domain'):
        values = np.asarray(meta[name])
        if values.shape != (count,) or values.dtype.kind not in 'iu':
            raise ValueError('Aligned source domain integers required: ' + name)
    rx, day = np.asarray(meta['rx_i']), np.asarray(meta['day_i'])
    if not np.isin(rx, MAIN_RX).all() or not np.isin(day, [1, 2, 3]).all():
        raise ValueError('Only original source Main RX/day domains are permitted')
    if not np.array_equal(meta['domain'], np.searchsorted(MAIN_RX, rx) * 3 + day - 1):
        raise ValueError('Source Main RX/day domain mapping differs')


def _source_meta(dataset, role, contract):
    import torch
    count = COUNTS['source_L' if role == 'L_s' else 'source_V']
    meta = dataset.metadata
    if (not torch.is_tensor(dataset.x) or tuple(dataset.x.shape) != (count, 2, 256)
            or dataset.x.dtype != torch.float32 or not isinstance(meta, dict)
            or not {'ids', 'y', 'rx_i', 'day_i', 'domain', 'eq_i'}.issubset(meta)):
        raise ValueError('The matching complete original source role must be retained')
    ids, y = np.asarray(meta['ids']), np.asarray(meta['y'])
    if (ids.shape != (count,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != count
            or np.any(ids == '') or not np.array_equal(ids, np.asarray(contract['role_ids'][role]))
            or y.shape != (count,) or y.dtype.kind not in 'iu' or set(y.tolist()) != set(range(6))
            or np.asarray(meta['eq_i']).shape != (count,) or not np.all(np.asarray(meta['eq_i']) == 1)):
        raise ValueError('Original source physical order, representation or six labels differ')
    _domains(meta, count)
    return {name: np.asarray(meta[name]).copy() for name in ('ids', 'y', 'rx_i', 'day_i', 'domain')}


def _oe_handles(root):
    root = Path(root)
    complete = _read(root / 'complete.json')
    if (not isinstance(complete, dict) or complete.get('status') != 'SOURCE_OE_DATA_VALIDATED'
            or complete.get('data_status') != 'VALIDATED_ONCE' or complete.get('source_only') is not True
            or any(complete.get(name) != 0 for name in ('target_reads', 'original_L_s_reads', 'original_U_s_reads',
                'original_V_reads', 'model_reads', 'model_predictions', 'optimizer_updates', 'channel_observations_generated'))
            or complete.get('counts') != {role: COUNTS[role] for role in ('exposure_train', 'novel_validation')}):
        raise ValueError('Completed model-free/source-only OE data is required')
    domains = complete.get('source_domains', {})
    if (domains.get('receiver_names') != RX_NAMES or domains.get('receiver_indices') != MAIN_RX
            or domains.get('dates') != DATES or domains.get('day_indices') != [1, 2, 3] or domains.get('domain_count') != 15):
        raise ValueError('Source OE domains must use the original Main RX/day mapping')
    handles = {}
    for role in ('exposure_train', 'novel_validation'):
        handle = complete.get(role)
        if (not isinstance(handle, dict) or handle.get('count') != COUNTS[role]
                or handle.get('data_status') != 'VALIDATED_ONCE' or handle.get('iq_dtype') != 'float32'
                or handle.get('iq_shape') != [COUNTS[role], 2, 256]
                or any(not isinstance(handle.get(key), str) or not handle[key] for key in ('capsule_id', 'split_id'))):
            raise ValueError('Completed source OE public data handles differ')
        # Builder paths are provenance text. Actual reads use root/role/public,
        # including after a byte-preserving move to another machine.
        handles[role] = {key: handle[key] for key in ('capsule_id', 'split_id', 'count')}
    if handles['exposure_train']['capsule_id'] == handles['novel_validation']['capsule_id']:
        raise ValueError('The two source OE roles must retain separate capsules')
    return handles


def _oe_public(root, role):
    public = Path(root) / role / 'public'
    if {path.name for path in public.iterdir()} != {'iq.npy', 'ids.npy', 'metadata.npz'}:
        raise ValueError('Source OE public files must contain IQ, IDs and domain metadata only')
    ids = np.load(public / 'ids.npy', allow_pickle=False)
    values = np.load(public / 'iq.npy', allow_pickle=False, mmap_mode='r')
    try:
        with np.load(public / 'metadata.npz', allow_pickle=False) as stored:
            if set(stored.files) != {'rx_i', 'day_i', 'domain'}:
                raise ValueError('TX labels or additional fields appeared in source OE public metadata')
            meta = {name: stored[name].copy() for name in stored.files}
        count = COUNTS[role]
        if (ids.shape != (count,) or ids.dtype.kind != 'U' or np.any(ids == '')
                or len(set(ids.tolist())) != count or values.shape != (count, 2, 256)
                or values.dtype != np.dtype('float32')):
            raise ValueError('Source OE opaque IDs or fixed FP32 IQ dimensions differ')
        _domains(meta, count)
        meta['ids'] = ids
        return values, meta
    except BaseException:
        values._mmap.close()
        raise


def _tensor(values, device):
    import torch
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.float32)
    array = np.ascontiguousarray(values, dtype=np.float32)
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch.float32).reshape(array.shape).to(device)


def _export_role(reader, values, meta, role, config, device, output):
    parts = {name: [] for name in BRANCHES}; logits_parts = []
    for start in range(0, len(values), config['batch_size']):
        branches, logits = extract_branches(reader, _tensor(values[start:start + config['batch_size']], device))
        for name, value in branches.items():
            parts[name].append(np.asarray(value.detach().cpu().tolist(), dtype=np.float32))
        logits_parts.append(np.asarray(logits.detach().cpu().tolist(), dtype=np.float32))
        if start == 0 or start // config['batch_size'] % 32 == 0 or start + config['batch_size'] >= len(values):
            print('PHASE3_SOURCE_OPEN_EXPORT', config['row_id'], role, min(start + config['batch_size'], len(values)), len(values), flush=True)
    arrays = dict(ids=meta['ids'], logits=np.concatenate(logits_parts),
        rx_i=np.asarray(meta['rx_i'], dtype=np.int64), day_i=np.asarray(meta['day_i'], dtype=np.int64))
    arrays.update({'features__' + name: np.concatenate(parts[name]) for name in BRANCHES})
    if role in ('source_L', 'source_V'):
        arrays['y'] = np.asarray(meta['y'], dtype=np.int64)
    path = output / FEATURE_REFS[role]
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open('xb') as stream:
        np.savez(stream, **arrays)


def export(config, device='cuda:0'):
    validate_config(config)
    if Path(config['output_root']).exists():
        raise FileExistsError('A new immutable source open output root is required')
    base = {key: config[key] for key in runner.COMMON | runner.SOURCE}
    contract, output = runner._prepare('calibrate', base)
    if (contract['counts']['L_s'] != COUNTS['source_L'] or contract['counts']['V'] != COUNTS['source_V']
            or contract.get('source_rxs') != MAIN_RX or contract.get('source_days') != [1, 2, 3]):
        raise ValueError('The original six-class full source contract is required')
    reader = runner._load_model(base, contract, device)
    _validate_model(reader)
    labeled = runner._source_dataset(base, 'L_s', contract)
    lmeta = _source_meta(labeled, 'L_s', contract)
    output.mkdir(parents=True, exist_ok=False)
    branches, logits = extract_branches(reader, _tensor(labeled.x[:2], device))
    _write(output / 'smoke.json', dict(status='PASS', actual_checkpoint=True, source_samples=2,
        source_role='original_L_s', validation_reads=0, OE_reads=0, target_reads=0, optimizer_updates=0,
        model_classes=6, parameters=232993, branch_dimensions=DIMENSIONS, logit_shape=list(logits.shape)))
    print('PHASE3_SOURCE_OPEN_SMOKE_PASS', config['row_id'], flush=True)
    _export_role(reader, labeled.x, lmeta, 'source_L', config, device, output)
    seen = set(lmeta['ids'].tolist()); del labeled
    validation = runner._source_dataset(base, 'V', contract)
    vmeta = _source_meta(validation, 'V', contract)
    if seen & set(vmeta['ids'].tolist()):
        raise ValueError('Original source L/V physical IDs overlap')
    seen.update(vmeta['ids'].tolist())
    _export_role(reader, validation.x, vmeta, 'source_V', config, device, output)
    del validation
    handles = _oe_handles(config['oe_data_root'])
    for role in ('exposure_train', 'novel_validation'):
        values, meta = _oe_public(config['oe_data_root'], role)
        try:
            if seen & set(meta['ids'].tolist()):
                raise ValueError('Source/OE public IDs overlap')
            seen.update(meta['ids'].tolist())
            _export_role(reader, values, meta, role, config, device, output)
        finally:
            values._mmap.close()
    _validate_model(reader)
    result = dict(status='SOURCE_OPEN_FEATURES_EXPORTED', source_only=True, target_reads=0,
        original_U_s_reads=0, OE_truth_reads=0, optimizer_updates=0, head_changed=False,
        model_classes=6, parameters=232993, counts=COUNTS.copy(), config=config,
        feature_refs=FEATURE_REFS.copy(), source_OE_handles=handles, branch_names=list(BRANCHES), dimensions=DIMENSIONS,
        source_L_count=COUNTS['source_L'], source_V_count=COUNTS['source_V'],
        exposure_train_count=COUNTS['exposure_train'], novel_validation_count=COUNTS['novel_validation'],
        model_forward_samples=sum(COUNTS.values()) + 2, smoke_samples=2, source_validation_updates=0,
        model_source=runner._identity(base), source_OE_TX_roster_read=False,
        readout_normalization='raw branches; no source/OE statistics fitted by exporter')
    _write(output / 'complete.json', result)
    print('SOURCE_OPEN_FEATURES_EXPORTED', config['row_id'], sum(COUNTS.values()) + 2, flush=True)
    return result


build = export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = export(_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
