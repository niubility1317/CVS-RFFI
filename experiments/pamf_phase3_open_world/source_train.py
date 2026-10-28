"""Fixed-budget scratch CE training for source-only four-class development."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .source_development import build_model, _validate as validate_code_runtime


KEYS = {'run_id', 'row_id', 'model_seed', 'code_commit', 'code_runtime', 'source_cache', 'source_contract',
        'fold', 'output_root', 'epochs', 'batch_size', 'shuffle_seed', 'optimizer', 'schedule', 'label_smoothing'}
OPTIMIZER = dict(name='AdamW', lr=.0003, weight_decay=.0001, betas=[.9, .999], eps=1e-8)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Source training accepts only its fixed source-only configuration')
    for key in ('run_id', 'row_id', 'code_commit', 'source_cache', 'source_contract', 'output_root'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('Nonempty source configuration string required: ' + key)
    validate_code_runtime(config['code_runtime'], config['model_seed'], 4)
    fold = config['fold']
    if not isinstance(fold, dict) or set(fold) != {'unknown_source_labels', 'heldout_source_rx'}:
        raise ValueError('Source fold may contain only the held-out source labels and receiver')
    labels = fold['unknown_source_labels']
    if (not isinstance(labels, list) or len(labels) != 2
            or any(isinstance(label, bool) or not isinstance(label, int) or label not in range(6) for label in labels)
            or labels != sorted(set(labels)) or isinstance(fold['heldout_source_rx'], bool)
            or not isinstance(fold['heldout_source_rx'], int) or fold['heldout_source_rx'] not in (1, 3, 4, 6, 8)):
        raise ValueError('Two source-only held-out classes and one registered source receiver required')
    if any(isinstance(config[key], bool) or not isinstance(config[key], int) for key in ('epochs', 'batch_size', 'shuffle_seed')):
        raise ValueError('Epoch, batch and shuffle seed must retain their integer types')
    if (config['epochs'] != 20 or config['batch_size'] != 128 or config['shuffle_seed'] != 2026102802
            or config['optimizer'] != OPTIMIZER or config['schedule'] != 'cosine_per_step'
            or config['label_smoothing'] != 0. or isinstance(config['label_smoothing'], bool)):
        raise ValueError('Fixed 20-pass CE development budget and optimizer required')
    return config


def _load_fold(config, contract):
    from .source_fold import load_fold
    return load_fold(config['source_cache'], contract, config['fold'])


def _validate_data(data, config):
    import torch
    required = {'train_x', 'train_y', 'validation_x', 'validation_y', 'train_ids', 'validation_ids',
                'validation_receivers', 'manifest', 'counts'}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError('Source fold data fields missing')
    for prefix, count, classes in (('train', 3360, 4), ('validation', 27000, 6)):
        x, y, ids = (data[prefix + suffix] for suffix in ('_x', '_y', '_ids'))
        if not torch.is_tensor(x) or tuple(x.shape) != (count, 2, 256) or x.dtype != torch.float32:
            raise ValueError('Fixed clean source tensor shape/dtype differs: ' + prefix)
        if (not isinstance(y, np.ndarray) or y.shape != (count,) or y.dtype.kind not in 'iu'
                or set(y.tolist()) != set(range(classes))):
            raise ValueError('Source role contains hidden, missing or held-out training labels: ' + prefix)
        if (not isinstance(ids, np.ndarray) or ids.shape != (count,) or ids.dtype.kind != 'U'
                or np.any(ids == '') or len(set(ids.tolist())) != count):
            raise ValueError('Source role IDs differ: ' + prefix)
    if set(data['train_ids'].tolist()) & set(data['validation_ids'].tolist()):
        raise ValueError('Training and development physical IDs overlap')
    rx = data['validation_receivers']
    if not isinstance(rx, np.ndarray) or rx.shape != (27000,) or rx.dtype.kind not in 'iu':
        raise ValueError('Source validation receiver metadata differs')
    if any(data['manifest'].get(key) != config['fold'][key] for key in ('unknown_source_labels', 'heldout_source_rx')):
        raise ValueError('Loaded source fold identity differs')


def _finite_logits(logits, count):
    import torch
    if tuple(logits.shape) != (count, 4) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite four-class default-head logits required')


def _smoke(model, train_x, device):
    import torch
    model.eval()
    with torch.no_grad():
        x = train_x[:2].to(device=device, dtype=torch.float32)
        features = model.features(x)
        logits = model.classify_features(features)
        if tuple(features.shape) != (2, 160) or not bool(torch.isfinite(features).all()):
            raise ValueError('Source-only scratch smoke features differ')
        _finite_logits(logits, 2)
    return dict(status='PASS', source_samples=2, target_reads=0, validation_forwards=0,
                U_reads=0, optimizer_updates=0, checkpoint_initialization='scratch',
                feature_shape=[2, 160], logit_shape=[2, 4])


def _fit(model, x, y, *, epochs, batch_size, shuffle_seed, optimizer_options, device, log_path, row_id):
    """Train the supplied source-L tensors; small budgets are test-only callers."""
    import torch
    from torch.nn import functional as F
    if epochs < 1 or batch_size < 1 or len(x) < 2 or len(y) != len(x):
        raise ValueError('Nonempty full-pass training budget required')
    parameters = list(model.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError('All identity parameters and the default head must train jointly')
    options = dict(optimizer_options)
    if options.pop('name') != 'AdamW':
        raise ValueError('AdamW required')
    options['betas'] = tuple(options['betas'])
    optimizer = torch.optim.AdamW(parameters, **options)
    base_lr = options['lr']
    total = epochs * math.ceil(len(x) / batch_size)
    labels = torch.tensor(np.asarray(y, dtype=np.int64).tolist(), dtype=torch.long)
    generator = torch.Generator(device='cpu').manual_seed(shuffle_seed)
    updates = exposures = 0
    started = time.perf_counter()
    first_lr = last_lr = None
    with Path(log_path).open('x', encoding='utf-8') as stream:
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(x), generator=generator)
            seen = correct = steps = 0
            loss_sum = 0.
            epoch_started = time.perf_counter()
            for start in range(0, len(x), batch_size):
                selected = order[start:start + batch_size]
                batch = x.index_select(0, selected).to(device=device, dtype=torch.float32)
                if not bool(torch.isfinite(batch).all()):
                    raise ValueError('Nonfinite source-L clean IQ')
                target = labels.index_select(0, selected).to(device)
                position = updates / (total - 1) if total > 1 else 1.
                lr = base_lr * .5 * (1. + math.cos(math.pi * position))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                _finite_logits(logits, len(batch))
                loss = F.cross_entropy(logits, target, label_smoothing=0.)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite source CE')
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                    raise FloatingPointError('Nonfinite source CE gradient')
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    raise FloatingPointError('Nonfinite source model update')
                batch_count = len(batch)
                seen += batch_count
                exposures += batch_count
                correct += int((logits.detach().argmax(1) == target).sum())
                loss_sum += float(loss.detach()) * batch_count
                updates += 1
                steps += 1
                first_lr = lr if first_lr is None else first_lr
                last_lr = lr
                if updates == 1 or updates % 10 == 0:
                    print('PHASE3_SOURCE_STEP', row_id, epoch, updates, float(loss.detach()), lr, flush=True)
            if seen != len(x):
                raise ValueError('A complete source-L pass including its tail is required')
            record = dict(epoch=epoch, seen=seen, steps=steps, cumulative_updates=updates,
                          loss=loss_sum / seen, accuracy=correct / seen, lr=last_lr,
                          tail_batch_size=len(x) % batch_size or batch_size,
                          seconds=time.perf_counter() - epoch_started,
                          elapsed_seconds=time.perf_counter() - started)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            print('PHASE3_SOURCE_EPOCH', row_id, json.dumps(record), flush=True)
    if updates != total or exposures != len(x) * epochs:
        raise ValueError('Fixed full-pass source budget not completed')
    return dict(epochs=epochs, updates=updates, train_exposures=exposures,
                first_lr=first_lr, last_lr=last_lr, nonzero_lr_steps=max(0, updates - 1))


def _export_features(model, data, output, config, device):
    import torch
    model.eval().requires_grad_(False)
    def encode(values, name, include_logits):
        feature_parts, logit_parts = [], []
        with torch.no_grad():
            for start in range(0, len(values), config['batch_size']):
                x = values[start:start + config['batch_size']].to(device=device, dtype=torch.float32)
                if not bool(torch.isfinite(x).all()):
                    raise ValueError('Nonfinite source export IQ')
                features = model.features(x)
                if tuple(features.shape) != (len(x), 160) or not bool(torch.isfinite(features).all()):
                    raise ValueError('Invalid frozen source feature export')
                feature_parts.append(np.asarray(features.detach().cpu().tolist(), dtype=np.float32))
                if include_logits:
                    logits = model.classify_features(features)
                    _finite_logits(logits, len(x))
                    logit_parts.append(np.asarray(logits.detach().cpu().tolist(), dtype=np.float32))
                if start == 0 or (start // config['batch_size']) % 32 == 0 or start + config['batch_size'] >= len(values):
                    print('PHASE3_SOURCE_EXPORT', config['row_id'], name, min(start + config['batch_size'], len(values)), len(values), flush=True)
        return np.concatenate(feature_parts), np.concatenate(logit_parts) if include_logits else None
    train_features, _ = encode(data['train_x'], 'L_train', False)
    validation_features, validation_logits = encode(data['validation_x'], 'V_final_no_grad', True)
    with (Path(output) / 'source_features.npz').open('xb') as stream:
        np.savez(stream, train_features=train_features, train_y=np.asarray(data['train_y'], dtype=np.int64),
                 train_ids=data['train_ids'], validation_features=validation_features,
                 validation_logits=validation_logits, validation_y=np.asarray(data['validation_y'], dtype=np.int64),
                 validation_ids=data['validation_ids'], validation_receivers=np.asarray(data['validation_receivers'], dtype=np.int64))


def train(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Source development output cannot be overwritten: ' + str(output))
    contract = _read(config['source_contract'])
    import torch
    torch.manual_seed(config['model_seed'])
    model = build_model(config['code_runtime'], config['model_seed'], 4).float().to(device)
    # Building activates only immutable model code; the fold loader then opens L/V source roles.
    data = _load_fold(config, contract)
    _validate_data(data, config)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'config.json', config)
    _write(output / 'fold_manifest.json', data['manifest'])
    _write(output / 'smoke.json', _smoke(model, data['train_x'], device))
    budget = _fit(model, data['train_x'], data['train_y'], epochs=config['epochs'], batch_size=config['batch_size'],
                  shuffle_seed=config['shuffle_seed'], optimizer_options=config['optimizer'], device=device,
                  log_path=output / 'epochs.jsonl', row_id=config['row_id'])
    if budget['epochs'] != 20 or budget['updates'] != 540 or budget['train_exposures'] != 67200:
        raise ValueError('Twenty complete 3360-sample passes / 540 optimizer calls required')
    model.eval().requires_grad_(False)
    with (output / 'source_model.pt').open('xb') as stream:
        torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
            config=config, manifest=data['manifest'], epoch=20, updates=540, source_only=True,
            checkpoint_initialization='scratch', target_reads=0, U_reads=0), stream)
    _export_features(model, data, output, config, device)
    complete = dict(status='SOURCE_DEVELOPMENT_FROZEN', source_only=True, target_reads=0, U_reads=0,
        checkpoint_initialization='scratch', epochs=20, updates=540, train_count=3360, validation_count=27000,
        heldout_source_rx=config['fold']['heldout_source_rx'], features_ref='source_features.npz',
        checkpoint_ref='source_model.pt', head_joint=True, config=config, manifest=data['manifest'], budget=budget,
        validation_export_passes=1, validation_updates=0, checkpoint_selection='fixed_final_epoch20',
        source_augmentation=False, EMA=False, teacher_networks=0, separate_head_fits=0,
        optimizer_policy='standard AdamW on CENTER_25K parameters; cosine first LR 0.0003 and final LR 0',
        budget_basis='20 full selected source-L passes including tails; not the historical native 4440-step E20')
    _write(output / 'complete.json', complete)
    print('PHASE3_SOURCE_DEVELOPMENT_FROZEN', config['row_id'], 20, 540, flush=True)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = train(_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], row_id=result['config']['row_id'], updates=result['updates'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
