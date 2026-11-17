"""Controlled MoPC author-code versus paper-equation ADS-B experiment.

Training reads only train arrays. Prediction reads only query features.
Scoring is a separate command and requires frozen predictions for all stages.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.pamf_phase2_practical.mopc_author import MoPCAuthorWiSig256


def regularization(model, previous, version):
    pairs = list(zip(model.parameters(), previous.parameters()))
    out = next(model.parameters()).new_zeros(())
    for idx, (current, old) in enumerate(pairs):
        diff = current - old
        weight = 1 - idx / len(pairs)
        out = out + weight * (diff.norm(2) if version == 'author' else diff.square().sum())
    return out


def correct(old, new_before, new_after, version):
    if version == 'author':
        similarity = torch.softmax(old @ new_before.T, dim=1)
    else:  # Paper Eqs. 10--14 directly multiply cosine S; no softmax.
        similarity = F.normalize(old, dim=1) @ F.normalize(new_before, dim=1).T
    return .97 * old + .03 * (similarity @ (new_after - new_before))


def loader(x, y, classes, shuffle=True):
    mask = (y >= classes[0]) & (y < classes[1])
    xx = torch.from_numpy(np.array(x[mask], copy=True))
    yy = torch.from_numpy(np.asarray(y[mask], dtype=np.int64))
    return DataLoader(TensorDataset(xx, yy), batch_size=16, shuffle=shuffle,
                      drop_last=True, num_workers=0, pin_memory=True)


def protos(model, data, low, high, device):
    model.eval()
    sums = torch.zeros(high - low, 256, device=device)
    counts = torch.zeros(high - low, device=device)
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            f = model.feature_extract_stage2(x)
            sums.index_add_(0, y - low, f)
            counts.index_add_(0, y - low, torch.ones_like(y, dtype=counts.dtype))
    if (counts == 0).any():
        raise RuntimeError('A class has no examples for prototype extraction')
    return sums / counts[:, None]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def train(args):
    random.seed(1731); np.random.seed(1731); torch.manual_seed(1731)
    torch.cuda.manual_seed_all(1731); torch.set_num_threads(2)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = Path(args.data)
    # The trainer has no reason to open query_x/query_labels.
    x = np.load(data / 'train_x.npy', mmap_mode='r')
    y = np.load(data / 'train_labels.npy')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    model = MoPCAuthorWiSig256(num_classes=100).to(dev)
    history = []
    if args.version == 'base':
        start_stage, previous, old_proto = 0, None, None
    else:
        if not args.base: raise ValueError('Incremental run requires frozen base')
        checkpoint = torch.load(args.base, map_location=dev)
        model.load_state_dict(checkpoint['model']); old_proto = checkpoint['prototypes'].to(dev)
        previous = copy.deepcopy(model).eval().requires_grad_(False); start_stage = 1
    stages = [(0, 50), (50, 75), (75, 100)]
    for stage in range(start_stage, 1 if args.version == 'base' else 3):
        low, high = stages[stage]
        dl = loader(x, y, (low, high))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=2e-4)
        before = time.time()
        for epoch in range(20):
            model.train(); total = np.zeros(4); steps = 0
            for bx, by in dl:
                bx, by = bx.to(dev), by.to(dev)
                logits = model.fc(model.feature_extract_stage2(bx))[:, :high]
                ce = F.cross_entropy(logits, by)
                aug = reg = ce.new_zeros(())
                if stage:
                    choices = np.random.randint(0, len(old_proto), size=16)
                    noise = np.random.normal(0, .05, size=(16, 256)).astype(np.float32)
                    features = old_proto[torch.as_tensor(choices, device=dev)] + torch.from_numpy(noise).to(dev)
                    scores = model.fc(features)[:, :high]
                    if args.version == 'author': scores = scores / 2
                    aug = F.cross_entropy(scores, torch.as_tensor(choices, device=dev))
                    reg = regularization(model, previous, args.version)  # beta = 1, explicit assumption
                loss = ce + aug + reg
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(finite).all():
                    raise FloatingPointError('Non-finite gradient')
                optimizer.step(); total += [v.detach().item() for v in (loss, ce, aug, reg)]; steps += 1
            row = {'stage': stage, 'epoch': epoch + 1, 'steps': steps,
                   'loss': (total / steps).tolist(), 'elapsed_seconds': time.time() - before}
            history.append(row); print(json.dumps(row), flush=True)
        if stage:
            p_before = protos(previous, dl, low, high, dev)
            p_after = protos(model, dl, low, high, dev)
            old_proto = torch.cat((correct(old_proto, p_before, p_after, args.version), p_after), dim=0)
        else:
            old_proto = protos(model, dl, low, high, dev)
        torch.save({'model': model.state_dict(), 'prototypes': old_proto.detach().cpu(),
                    'stage': stage, 'seen': high}, out / f'stage{stage}.pt')
        previous = copy.deepcopy(model).eval().requires_grad_(False)
        write_json(out / 'history.json', history)
    write_json(out / 'train_complete.json', {'version': args.version, 'stages': [r[1] for r in stages[start_stage:(1 if args.version == 'base' else 3)]], 'epochs': 20})


def predict(args):
    torch.set_num_threads(2)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, out = Path(args.data), Path(args.output)
    manifest = json.loads((data / 'data_manifest.json').read_text(encoding='utf-8'))
    x = np.load(data / 'query_x.npy', mmap_mode='r')
    out.mkdir(parents=True, exist_ok=False)
    model = MoPCAuthorWiSig256(num_classes=100).to(dev).eval()
    checkpoints = [Path(args.base), Path(args.incremental) / 'stage1.pt', Path(args.incremental) / 'stage2.pt']
    for stage, (cp, count, seen) in enumerate(zip(checkpoints, manifest['query_prefix_lengths'], (50, 75, 100))):
        state = torch.load(cp, map_location=dev)
        if state['stage'] != stage or state['seen'] != seen: raise ValueError('Checkpoint/stage mismatch')
        model.load_state_dict(state['model']); result = []
        with torch.no_grad():
            for pos in range(0, count, 64):
                xb = torch.from_numpy(np.array(x[pos:min(pos+64, count)], copy=True)).to(dev)
                logits = model.fc(model.feature_extract_stage2(xb))[:, :seen]
                if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite prediction')
                result.append(logits.argmax(1).cpu().numpy())
        np.save(out / f'stage{stage}_pred.npy', np.concatenate(result))
    write_json(out / 'predictions_complete.json', {'stages': [0, 1, 2]})


def score(args):
    pred_dir, data, out = Path(args.predictions), Path(args.data), Path(args.output)
    if not (pred_dir / 'predictions_complete.json').is_file(): raise ValueError('Predictions not fixed')
    labels = np.load(data / 'query_labels.npy')
    manifest = json.loads((data / 'data_manifest.json').read_text(encoding='utf-8'))
    groups = [(0, 50), (50, 75), (75, 100)]; table = []
    for stage, count in enumerate(manifest['query_prefix_lengths']):
        pred = np.load(pred_dir / f'stage{stage}_pred.npy')
        if len(pred) != count: raise ValueError('Prediction count mismatch')
        yy = labels[:count]
        table.append([float((pred[(yy >= lo) & (yy < hi)] == yy[(yy >= lo) & (yy < hi)]).mean())
                      for lo, hi in groups[:stage + 1]])
    # Eq. 23 weights incremental tasks equally, not individual records.
    avg = float(np.mean(table[-1]))
    forgetting = float(np.mean([max(table[k][j] for k in range(j, 2)) - table[2][j] for j in (0, 1)]))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'metrics.json', {'task_accuracy_by_stage': table, 'paper_A_avg': avg,
                                    'average_forgetting': forgetting, 'intransigence': None,
                                    'intransigence_reason': 'Independent task reference models not run'})


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['train', 'predict', 'score'])
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--version', choices=['base', 'author', 'paper']); p.add_argument('--base')
    p.add_argument('--incremental'); p.add_argument('--predictions')
    a = p.parse_args()
    {'train': train, 'predict': predict, 'score': score}[a.action](a)


if __name__ == '__main__': main()
