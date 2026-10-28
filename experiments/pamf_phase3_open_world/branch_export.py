"""Export source-only branch evidence from the fixed own four-class model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .source_development import build_model, _validate as validate_code_runtime


DIMENSIONS = dict(fused160=160, time160=160, frequency160=160, pa160=160,
                  reference29=29, signature336=336, spectrum128=128, pa_moments48=48)
BRANCHES = tuple(DIMENSIONS)
KEYS = {'run_id', 'row_id', 'code_commit', 'code_runtime', 'source_model_root', 'source_cache',
        'source_contract', 'model_seed', 'fold', 'batch_size', 'output_root'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only fixed source branch export inputs are accepted')
    for key in ('run_id', 'row_id', 'code_commit', 'source_model_root', 'source_cache', 'source_contract', 'output_root'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('Nonempty source export string required: ' + key)
    validate_code_runtime(config['code_runtime'], config['model_seed'], 4)
    if type(config['batch_size']) is not int or config['batch_size'] != 128:
        raise ValueError('Source branch export batch_size must be 128')
    fold = config['fold']
    if (not isinstance(fold, dict) or set(fold) != {'unknown_source_labels', 'heldout_source_rx'}
            or fold['unknown_source_labels'] != [0, 1]
            or any(type(value) is not int for value in fold['unknown_source_labels'])
            or type(fold['heldout_source_rx']) is not int or fold['heldout_source_rx'] != 8):
        raise ValueError('Only the own source [0,1]/RX8 fold is declared')
    return config


def _parent(config):
    from .source_train import validate_config as validate_parent_config
    root = Path(config['source_model_root'])
    done = _read(root / 'complete.json')
    if (not isinstance(done, dict) or done.get('status') != 'SOURCE_DEVELOPMENT_FROZEN'
            or done.get('source_only') is not True or done.get('target_reads') != 0 or done.get('U_reads') != 0
            or done.get('epochs') != 20 or done.get('updates') != 540
            or done.get('checkpoint_initialization') != 'scratch' or done.get('head_joint') is not True
            or done.get('train_count') != 3360 or done.get('validation_count') != 27000
            or done.get('checkpoint_selection') != 'fixed_final_epoch20'
            or done.get('checkpoint_ref') != 'source_model.pt'):
        raise ValueError('Complete source-only fixed-final four-class parent required')
    previous = done.get('config')
    validate_parent_config(previous)
    for key in ('model_seed', 'fold', 'source_cache', 'source_contract', 'code_runtime'):
        if previous[key] != config[key]:
            raise ValueError('Parent source identity differs: ' + key)
    if Path(previous['output_root']).resolve() != root.resolve():
        raise ValueError('Parent output root identity differs')
    if not isinstance(done.get('manifest'), dict) or any(done['manifest'].get(key) != config['fold'][key] for key in config['fold']):
        raise ValueError('Parent source fold manifest differs')
    return done


def _load_model(config, parent, device):
    import torch
    model = build_model(config['code_runtime'], config['model_seed'], 4)
    # This is the only weight read, after source-only parent identity checks.
    path = Path(config['source_model_root']) / 'source_model.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(payload, dict) or payload.get('config') != parent['config']
            or payload.get('manifest') != parent['manifest'] or payload.get('epoch') != 20
            or payload.get('updates') != 540 or payload.get('source_only') is not True
            or payload.get('checkpoint_initialization') != 'scratch'
            or payload.get('target_reads') != 0 or payload.get('U_reads') != 0):
        raise ValueError('Own source checkpoint payload identity differs')
    state = payload.get('state_dict')
    head = state.get('core.id_backbone.cls_head.weight') if isinstance(state, dict) else None
    if not torch.is_tensor(head) or tuple(head.shape) != (4, 160):
        raise ValueError('Only the own four-class default head may be loaded')
    if any(not torch.is_tensor(value) or (value.is_floating_point() and value.dtype != torch.float32) for value in state.values()):
        raise ValueError('Own FP32 source state required')
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError('Own four-class source state does not match its exact model') from exc
    return model.float().to(device).eval().requires_grad_(False)


def extract_branches(model, x):
    """Capture eight raw branches in one complete forward; remove local hooks."""
    import torch
    if (model.training or not torch.is_tensor(x) or x.ndim != 3 or tuple(x.shape[1:]) != (2, 256)
            or not len(x) or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
        raise ValueError('Frozen eval model and finite FP32 I/Q [B,2,256] required')
    captured, handles = {}, []
    def record(name, value):
        if name in captured:
            raise ValueError('A branch ran more than once in the complete forward: ' + name)
        captured[name] = value
    def tensor_hook(name):
        def hook(module, inputs, value):
            if not torch.is_tensor(value):
                raise ValueError('Tensor branch output required: ' + name)
            record(name, value.detach().clone())
        return hook
    def core_hook(module, inputs, value):
        if not isinstance(value, dict) or not {'t_emb', 'f_emb', 'feat_pa'}.issubset(value):
            raise ValueError('Actual core auxiliary branch tensors missing')
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
        raise ValueError('Complete same-forward branch capture required')
    count = len(x)
    if (tuple(captured['spectrum'].shape) != (count, 4, 32)
            or tuple(captured['pa_lift'].shape) != (count, 24, 256)
            or any(tuple(captured['signature' + str(index)].shape) != (count, 112) for index in range(3))):
        raise ValueError('Raw spectrum, PA basis or signature shape differs')
    pa = captured['pa_lift']
    result = dict(fused160=fused.detach().clone(), time160=captured['core']['t_emb'],
        frequency160=captured['core']['f_emb'], pa160=captured['core']['feat_pa'],
        reference29=captured['reference'],
        signature336=torch.cat([captured['signature' + str(index)] for index in range(3)], dim=1),
        spectrum128=captured['spectrum'].flatten(1),
        pa_moments48=torch.cat((pa.mean(-1), pa.std(-1, unbiased=False)), dim=1))
    for name, dimension in DIMENSIONS.items():
        value = result[name]
        if tuple(value.shape) != (count, dimension) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite raw FP32 branch shape differs: ' + name)
    if bool((result['reference29'][:, 27] != 0).any()):
        raise ValueError('NoExplicitCFO coordinate 27 must remain strictly zero')
    if tuple(logits.shape) != (count, 4) or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Complete fused default-head logits differ')
    return result, logits.detach().clone()


def _load_smoke_x(config, contract):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    import torch
    labeled = SourceDataset(config['source_cache'], 'L_s', contract)
    meta = labeled.metadata
    selected = np.flatnonzero(~np.isin(meta['y'], config['fold']['unknown_source_labels'])
                             & (meta['rx_i'] != config['fold']['heldout_source_rx']))
    if len(selected) != 3360:
        raise ValueError('The own legal source-L fold sample count differs')
    index = torch.tensor(selected[:2].tolist(), dtype=torch.long, device=labeled.x.device)
    return labeled.x.index_select(0, index)


def _load_fold(config, contract):
    from .source_fold import load_fold
    return load_fold(config['source_cache'], contract, config['fold'])


def _export_arrays(model, data, config, device):
    arrays = {key: np.asarray(data[key], dtype=np.int64 if key.endswith('_y') or key == 'validation_receivers' else None)
              for key in ('train_y', 'train_ids', 'validation_y', 'validation_ids', 'validation_receivers')}
    for role in ('train', 'validation'):
        parts = {name: [] for name in BRANCHES}
        logits_parts = []
        values = data[role + '_x']
        for start in range(0, len(values), config['batch_size']):
            x = values[start:start + config['batch_size']].to(device=device)
            branches, logits = extract_branches(model, x)
            for name, value in branches.items():
                parts[name].append(np.asarray(value.detach().cpu().tolist(), dtype=np.float32))
            if role == 'validation':
                logits_parts.append(np.asarray(logits.detach().cpu().tolist(), dtype=np.float32))
            if start == 0 or (start // config['batch_size']) % 32 == 0 or start + config['batch_size'] >= len(values):
                print('PHASE3_SOURCE_BRANCH_EXPORT', config['row_id'], role, min(start + config['batch_size'], len(values)), len(values), flush=True)
        for name in BRANCHES:
            arrays[role + '_features__' + name] = np.concatenate(parts[name])
        if role == 'validation':
            arrays['validation_logits'] = np.concatenate(logits_parts)
    if len(arrays) != 22:
        raise ValueError('The source branch export must contain its exact 22 fields')
    return arrays


def export(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Source branch output cannot be overwritten: ' + str(output))
    parent = _parent(config)
    model = _load_model(config, parent, device)
    contract = _read(config['source_contract'])
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'config.json', config)
    smoke_x = _load_smoke_x(config, contract).to(device=device)
    smoke_values, smoke_logits = extract_branches(model, smoke_x)
    _write(output / 'smoke.json', dict(status='PASS', source_samples=len(smoke_x), source_role='selected_L_s',
        validation_reads=0, target_reads=0, optimizer_updates=0, own_checkpoint=True,
        branch_dimensions={name: value.shape[1] for name, value in smoke_values.items()}, logit_shape=list(smoke_logits.shape)))
    print('PHASE3_SOURCE_BRANCH_SMOKE_PASS', config['row_id'], flush=True)
    # The full V is first opened only after the legal two-source-L smoke passes.
    data = _load_fold(config, contract)
    from .source_train import _validate_data
    _validate_data(data, config)
    if data['manifest'] != parent['manifest']:
        raise ValueError('Current legal source roles differ from the own parent fold')
    arrays = _export_arrays(model, data, config, device)
    with (output / 'source_branches.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    result = dict(status='SOURCE_BRANCHES_EXPORTED', source_only=True, target_reads=0, optimizer_updates=0,
        head_changed=False, parent_epochs=20, parent_updates=540, train_count=3360, validation_count=27000,
        heldout_source_rx=8, branch_names=list(BRANCHES), dimensions=DIMENSIONS, config=config,
        parent_run_id=parent['config']['run_id'], parent_row_id=parent['config']['row_id'],
        parent_code_commit=parent['config']['code_commit'], source_manifest=data['manifest'],
        features_ref='source_branches.npz', readout_normalization='raw exports; statistics may be fitted on train L only',
        validation_updates=0, validation_export_passes=1, original_default_head_retained=True)
    _write(output / 'complete.json', result)
    print('PHASE3_SOURCE_BRANCHES_EXPORTED', config['row_id'], 3360, 27000, flush=True)
    return result


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
