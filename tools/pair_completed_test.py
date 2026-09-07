"""Frozen completed-row diagnostic: prepare once, predict all, then score truth-last."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SCENES = ['leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def code_paths(release):
    code = Path(release) / 'code'
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / 'scripts'))


def prepare(manifest, root):
    """Data-builder alone sees labels. The predictor never opens truth.npz."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row = manifest['rows'][0]
    code_paths(row['release'])
    from SSDG.train_ssdg import build_arg_parser, _build_ssdg_wisig_data
    from cvsrffi.eval import apply_sat_channel_for_scenario
    from cvsrffi.tensors import make_torch_generator
    ckpt = torch.load(row['checkpoint'], map_location='cpu')
    args = build_arg_parser().parse_args(['--output_dir', str(root)])
    for key, value in ckpt['args'].items():
        setattr(args, key, value)
    args.eval_batch_size = manifest['batch_size']
    args.num_workers = 0
    args.prefetch_factor = 2
    args.sample_rate_hz = float(getattr(args, 'sample_rate_hz', 0) or 25e6)
    args.sat_seed = manifest['sat_seed']
    device = torch.device('cuda:0')
    context = _build_ssdg_wisig_data(args, device)
    loader = context['named_test_loaders']['test_all_day_unseen_rx']
    n = len(loader.dataset)
    assert n > 0
    data_root = root / 'inputs'
    data_root.mkdir()
    clean = np.lib.format.open_memmap(data_root / 'clean.npy', mode='w+', dtype='float32', shape=(n, 2, args.wisig_out_len))
    received = np.lib.format.open_memmap(data_root / 'received.npy', mode='w+', dtype='float32', shape=clean.shape)
    ids, ys, rxs, days, scenes = [], [], [], [], []
    generators = [make_torch_generator(device, args.sat_seed + i * 1009) for i in range(3)]
    offset = 0
    for batch in loader:
        x, y, _, extra = batch
        size = len(x)
        # base_index addresses the fixed all-day target dataset, without class interpretation.
        keys = [hashlib.sha256(('ManySig-eq1-all-day-target:' + str(int(i))).encode()).hexdigest()[:32]
                for i in extra['base_index']]
        assigned = np.array([int(k[-8:], 16) % 3 for k in keys], dtype='int8')
        clean[offset:offset+size] = x.numpy()
        x = x.to(device)
        result = torch.empty_like(x)
        for scene_index, scenario in enumerate(SCENES):
            mask = torch.tensor((assigned == scene_index).tolist(), dtype=torch.bool, device=device)
            if bool(mask.any()):
                result[mask], _ = apply_sat_channel_for_scenario(
                    x[mask], scenario, args, gen=generators[scene_index], return_meta=False)
        received[offset:offset+size] = result.cpu().numpy()
        ids.extend(keys)
        ys.extend(y.tolist())
        rxs.extend(extra['rx_i'].tolist())
        days.extend(extra['day_i'].tolist())
        scenes.extend(assigned.tolist())
        offset += size
    assert offset == n and len(set(ids)) == n
    clean.flush()
    received.flush()
    np.savez(data_root / 'public.npz', ids=np.array(ids), rx=np.array(rxs), day=np.array(days), scene=np.array(scenes))
    np.savez(root / 'truth.npz', ids=np.array(ids), y=np.array(ys))
    write_json(data_root / 'metadata.json', {
        'n': n, 'num_classes': int(args.num_classes), 'class_id_to_tx': context['class_id_to_tx'],
        'source_receivers': args.wisig_train_rxs, 'target_receivers': args.wisig_test_rxs,
        'source_days': args.wisig_train_days, 'target_days': args.wisig_test_days,
        'loader': 'test_all_day_unseen_rx', 'scenes': SCENES,
        'scene_counts': np.bincount(scenes, minlength=3).tolist(),
        'observation': 'one immutable LEO received observation per physical sample; clean diagnostic control',
        'scene_assignment': 'opaque physical ID digest modulo 3; independent of class and model',
        'sat_seed': args.sat_seed, 'batch_size': args.eval_batch_size,
    })
    print(f'PREPARED n={n}', flush=True)


def load_for_inference(manifest, index):
    import torch
    # TF32 convolution algorithms have batch-shape-dependent rounding on RTX3090.
    # Keep the declared FP32 evaluation and the unchanged independence tolerance.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row = manifest['rows'][index]
    code_paths(row['release'])
    from smoke_adv3b02_pair_reform import load_model
    from cvsrffi.tensors import numpy_to_tensor_compat
    device = torch.device('cuda:0')
    bridge = numpy_to_tensor_compat(np.arange(6, dtype=np.float32).reshape(3, 2),
                                    numpy_dtype=np.float32, torch_dtype=torch.float32)
    assert bridge.tolist() == [[0., 1.], [2., 3.], [4., 5.]]
    payload, model, model_args, compatibility = load_model(Path(row['checkpoint']), device)
    assert payload['epoch'] == 200 and payload['checkpoint_selection'] == 'final_only'
    model.eval().requires_grad_(False)
    # Real checkpoint smoke on synthetic input, before any query input is opened.
    torch.manual_seed(392005)
    with torch.inference_mode():
        x = torch.randn(3, 2, int(model_args['input_len']), device=device)
        batched = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)['tx_logits']
        alone = torch.cat([model(z[None], y_tx=None, grl_lambda=1.0, return_aux=True)['tx_logits'] for z in x])
        torch.testing.assert_close(batched, alone, atol=1e-4, rtol=1e-4)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'SMOKE_PASS {row["row_id"]}', flush=True)
    return payload, model, compatibility, state


def predict(manifest, root, index):
    import torch
    row = manifest['rows'][index]
    payload, model, compatibility, state = load_for_inference(manifest, index)
    from cvsrffi.tensors import numpy_to_tensor_compat
    device = torch.device('cuda:0')
    public = np.load(root / 'inputs/public.npz', allow_pickle=False)
    ids = public['ids']
    logits = {}
    start = time.time()
    with torch.inference_mode():
        for view in ['clean', 'received']:
            inputs = np.load(root / f'inputs/{view}.npy', mmap_mode='r')
            values = []
            for pos in range(0, len(ids), manifest['batch_size']):
                x = numpy_to_tensor_compat(inputs[pos:pos+manifest['batch_size']],
                                           numpy_dtype=np.float32, torch_dtype=torch.float32).to(device)
                out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)['tx_logits']
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError('nonfinite prediction logits')
                values.append(out.cpu().numpy())
            logits[view] = np.concatenate(values)
    for k, v in model.state_dict().items():
        if not torch.equal(v.detach().cpu(), state[k]):
            raise RuntimeError(f'query inference changed persistent state: {k}')
    target = root / 'predictions' / row['row_id']
    target.mkdir()
    np.savez(target / 'prediction.npz', ids=ids, clean=logits['clean'], received=logits['received'])
    write_json(target / 'complete.json', {
        'row_id': row['row_id'], 'checkpoint': row['checkpoint'], 'checkpoint_epoch': payload['epoch'],
        'n': len(ids), 'compatibility': compatibility, 'query_state_unchanged': True,
        'truth_access': False, 'seconds': time.time()-start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'sealed_at_unix': time.time(),
    })
    print(f'PREDICTION_COMPLETE {row["row_id"]} n={len(ids)}', flush=True)


def group_metrics(y, pred, mask, nc):
    yy, pp = y[mask], pred[mask]
    matrix = np.bincount(yy * nc + pp, minlength=nc * nc).reshape(nc, nc)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = []
    for i in range(nc):
        count = int(matrix[i].sum())
        per_class.append({'class_id': i, 'correct': int(matrix[i, i]), 'total': count,
                          'accuracy': 100 * int(matrix[i, i]) / count if count else None})
    return {'correct': correct, 'total': total, 'accuracy': 100 * correct / total if total else None,
            'per_class': per_class, 'confusion_matrix': matrix.tolist()}


def aligned_truth(pred_ids, truth_ids, truth_y):
    if len(set(pred_ids.tolist())) != len(pred_ids) or len(set(truth_ids.tolist())) != len(truth_ids):
        raise ValueError('duplicate opaque IDs')
    if set(pred_ids.tolist()) != set(truth_ids.tolist()):
        raise ValueError('prediction/truth coverage mismatch')
    lookup = dict(zip(truth_ids.tolist(), truth_y.tolist()))
    return np.array([lookup[i] for i in pred_ids.tolist()])


def score(manifest, root):
    # Check ALL frozen candidates are sealed before this process opens truth.
    for row in manifest['rows']:
        p = root / 'predictions' / row['row_id']
        if not (p / 'complete.json').is_file() or not (p / 'prediction.npz').is_file():
            raise RuntimeError('all predictions must be complete before truth access')
    public = np.load(root / 'inputs/public.npz', allow_pickle=False)
    meta = json.loads((root / 'inputs/metadata.json').read_text())
    truth = np.load(root / 'truth.npz', allow_pickle=False)
    results = []
    for row in manifest['rows']:
        p = np.load(root / 'predictions' / row['row_id'] / 'prediction.npz', allow_pickle=False)
        if not np.array_equal(p['ids'], public['ids']):
            raise ValueError('public metadata ID mismatch')
        y = aligned_truth(p['ids'], truth['ids'], truth['y'])
        record = {'row_id': row['row_id'], 'checkpoint': row['checkpoint'], 'metrics': {}}
        nc = meta['num_classes']
        for label, logits, selection in [('clean', p['clean'], np.ones(len(y), dtype=bool))] + [
                (scene, p['received'], public['scene'] == i) for i, scene in enumerate(SCENES)]:
            if logits.shape != (len(y), nc) or not np.isfinite(logits).all():
                raise ValueError('invalid prediction array')
            pred = logits.argmax(axis=1)
            m = group_metrics(y, pred, selection, nc)
            for key in ['rx', 'day']:
                m['by_' + key] = {str(int(g)): group_metrics(y, pred, selection & (public[key] == g), nc)
                                     for g in np.unique(public[key])}
            m['by_rx_day'] = {f'{int(rx)}:{int(day)}': group_metrics(y, pred, selection & (public['rx'] == rx) & (public['day'] == day), nc)
                              for rx in np.unique(public['rx']) for day in np.unique(public['day'])}
            record['metrics'][label] = m
        record['leo_equal_scene_mean'] = float(np.mean([record['metrics'][s]['accuracy'] for s in SCENES]))
        record['leo_scene_floor'] = min(record['metrics'][s]['accuracy'] for s in SCENES)
        results.append(record)
    write_json(root / 'results.json', {'status': 'SCORED', 'claim': 'USER_REQUESTED_COMPLETED_ROWS_DIAGNOSTIC_NO_PROMOTION_NO_FEEDBACK',
                                     'metadata': meta, 'rows': results, 'scored_at_unix': time.time()})
    print('SCORED', len(results), flush=True)


def queue(manifest_path, manifest, root):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'predictions').mkdir()
    write_json(root / 'manifest.json', manifest)
    script = str(Path(__file__).resolve())
    common = [sys.executable, '-u', script, '--manifest', str(manifest_path), '--root', str(root)]
    stages = [('smoke', ['--index', str(i)]) for i in range(len(manifest['rows']))]
    stages += [('prepare', [])] + [('predict', ['--index', str(i)]) for i in range(len(manifest['rows']))] + [('score', [])]
    for mode, extra in stages:
        print('START', mode, extra, flush=True)
        subprocess.run(common + ['--mode', mode] + extra, check=True)
    write_json(root / 'done.json', {'status': 'SCORED', 'rows': len(manifest['rows']), 'completed_at_unix': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['queue', 'smoke', 'prepare', 'predict', 'score'], required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.mode == 'queue':
        queue(args.manifest.resolve(), manifest, args.root)
    elif args.mode == 'smoke':
        load_for_inference(manifest, args.index)
    elif args.mode == 'prepare':
        prepare(manifest, args.root)
    elif args.mode == 'predict':
        predict(manifest, args.root, args.index)
    else:
        score(manifest, args.root)
