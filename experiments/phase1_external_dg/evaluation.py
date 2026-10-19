"""Frozen default-head prediction and independent, all-task truth-last scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import read, write
from .plan import VIEWS, make_plan
from . import source

_PLAN = make_plan()
_ROWS = {r['row_id']: r for r in _PLAN['rows']}
_DATES = ('2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23')


def row_spec(c):
    row = _ROWS.get(c['row_id'])
    if row is None or row['model_seed'] != c['model_seed']:
        raise ValueError('Unregistered prediction/scoring identity')
    return row


def expected_tests(row):
    return ['main', 'source_unseen_date'] if row['group'] == 'chrono' else ['main']


def expected_views(row):
    return list(row.get('views', VIEWS))


def reject_truth(value):
    if isinstance(value, dict):
        if any(k in value for k in ('truth', 'target_truth', 'label', 'labels')):
            raise ValueError('Predictor rejects truth/labels')
        for item in value.values(): reject_truth(item)
    elif isinstance(value, list):
        for item in value: reject_truth(item)


def legacy_model(c, expected, device):
    """Strictly load historical scratch CE weights; never refit any head."""
    from experiments.standard_ce_baselines.model import build
    row = row_spec(c)
    if row['budget_kind'] != 'frozen_eval' or row['family'] not in ('resnet18', 'resnet50'):
        raise ValueError('Invalid historical architecture')
    root = Path(c['source_output'])
    done, initial, contract, resolved, freeze = [read(root / f) for f in (
        'completion.json', 'initialization.json', 'source_contract.json',
        'resolved_config.json', 'freeze.json')]
    if (done['status'] != 'SOURCE_TRAINED' or done['epoch'] != 200 or done['steps'] != 10000 or
        done.get('target_access') is not False):
        raise ValueError('Historical training is not complete')
    required_initial = dict(scratch_only=True, checkpoint=None, ancestors=[],
        checkpoint_sources=[], target_access=False, target_contact=False,
        model_seed=c['model_seed'], physical_roles='EXACT_MATCH', selection='fixed_last_epoch')
    if any(initial.get(k) != v for k, v in required_initial.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if (any(contract.get(k) != v for k, v in expected.items()) or
        contract.get('equalized') != 1 or contract.get('out_len') != 256 or
        contract.get('normalize') is not True or len(contract['classes']) != 6):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    required = dict(method='standard_ce_baselines', variant=row['family'], model_seed=c['model_seed'],
        epochs=200, batch_size=128, lr=.0002, lr_min=1e-6, weight_decay=.0001,
        drop_last=False, augmentation=False, domain_backbone=False, extra_losses=[],
        selection='fixed_last_epoch', split_seed=392005)
    if any(resolved.get(k) != v for k, v in required.items()):
        raise ValueError('Historical recipe mismatch')
    if any(resolved.get(k) for k in ('resume', 'teacher', 'checkpoint', 'initial_checkpoint',
                                    'target_inputs', 'target_truth', 'p1_truth', 'p1_capsule')):
        raise ValueError('Historical inherited or target inputs')
    expected_freeze = dict(status='FROZEN', epoch=200, variant=row['family'],
        model_seed=c['model_seed'], checkpoint=str(root / 'last.pt'),
        target_access=False, target_score_used=False)
    if freeze != expected_freeze:
        raise ValueError('Historical freeze identity mismatch')
    payload = torch.load(root / 'last.pt', map_location='cpu')
    expected_payload = dict(method='standard_ce_baselines', variant=row['family'], epoch=200,
        selection='fixed_last_epoch', config=resolved, initialization=initial, source_contract=contract)
    if any(payload.get(k) != v for k, v in expected_payload.items()):
        raise ValueError('Historical checkpoint metadata mismatch')
    net = build(row['family'], c['model_seed'])
    net.load_state_dict(payload['model'], strict=True); net.to(device).eval()
    # The old source code trained the original model jointly. Confirm the saved
    # training telemetry, rather than requiring fields added after that run.
    with (root / 'epoch_metrics.jsonl').open(encoding='utf-8') as stream:
        last = None
        for line in stream:
            if line.strip(): last = json.loads(line)
    count = sum(p.numel() for p in net.parameters())
    if (last is None or last['epoch'] != 200 or last['optimizer_steps_total'] != 10000 or
        last['gradient_used_parameters'] != count or resolved['trainable_parameters'] != count):
        raise ValueError('Historical joint default-head training not established')
    with torch.no_grad(): output = net(torch.zeros(2, 2, 256, device=device))
    if output.shape != (2, 6) or not torch.isfinite(output).all():
        raise ValueError('Historical real-checkpoint no-query smoke failed')
    return net, dict(row_id=c['row_id'], family=row['family'], model_seed=c['model_seed'],
        num_classes=6, classes=contract['classes'], legacy=True)


def load_model(c, device):
    expected = read(c['expected_source_contract'])
    row = row_spec(c)
    if row['budget_kind'] == 'frozen_eval':
        return legacy_model(c, expected, device)
    net, configuration = source.frozen_model(c['source_output'], expected, device)
    if configuration['row_id'] != c['row_id'] or configuration['model_seed'] != c['model_seed']:
        raise ValueError('Frozen model does not belong to this comparison row')
    return net, configuration


def handle(root, classes):
    root = Path(root); manifest = read(root / 'manifest.json')
    if (manifest['status'] != 'VALIDATED_ONCE' or manifest['classes'] != classes or
        manifest['scenes'] != list(VIEWS[1:]) or manifest['source_target_disjoint'] is not True):
        raise ValueError('Received-IQ handles differ')
    with np.load(root / 'index.npz', allow_pickle=False) as index:
        ids = index['ids'].copy()
        positions = index['positions'].copy() if 'positions' in index.files else None
    if len(ids) != manifest['count'] or len(set(ids.tolist())) != len(ids) or not len(ids):
        raise ValueError('Prediction coverage does not match validated handles')
    if positions is not None and (positions.shape != ids.shape or positions.dtype.kind not in 'iu'):
        raise ValueError('Invalid wrapper positions')
    return manifest, ids, positions


def predict(c, device='cuda:0'):
    reject_truth(c)
    row = row_spec(c)
    if sorted(c['tests']) != sorted(expected_tests(row)):
        raise ValueError('Missing or extra test tasks')
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device)
    # All checkpoint and source-contract checks finish before target IQ is read.
    net, sc = load_model(c, device)
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    views = expected_views(row)
    all_tasks = {}
    for name in expected_tests(row):
        test = c['tests'][name]
        manifest, ids, positions = handle(test['views_root'], sc['classes'])
        if name == 'source_unseen_date' and manifest.get('diagnostic_unseen_date') is not True:
            raise ValueError('Unseen-date diagnostic handle differs')
        target = out / name; target.mkdir()
        values = {}; seconds = {}
        with torch.no_grad():
            for view in views:
                path = (manifest['clean_ref'] if view == 'clean' else
                    str(Path(manifest.get('observations_root', test['views_root'])) / (view + '.npy')))
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.ndim != 3 or array.shape[1:] != (2, 256):
                    raise ValueError('Wrong received-IQ representation')
                if positions is None and len(array) != len(ids):
                    raise ValueError('Received-IQ count mismatch')
                if positions is not None and (positions.min() < 0 or positions.max() >= len(array)):
                    raise ValueError('Wrapper exceeds stored observations')
                if device.type == 'cuda': torch.cuda.synchronize()
                tic = time.monotonic(); result = []
                for start in range(0, len(ids), 128):
                    selected = slice(start, start + 128) if positions is None else positions[start:start + 128]
                    block = np.ascontiguousarray(array[selected], dtype=np.float32)
                    x = torch.frombuffer(bytearray(block.tobytes()), dtype=torch.float32).reshape(block.shape).to(device)
                    logits = net(x) if sc.get('legacy') else source.forward(net, sc, x)
                    if logits.shape != (len(x), sc['num_classes']) or not torch.isfinite(logits).all():
                        raise FloatingPointError('Invalid frozen default-head predictions')
                    result.extend(logits.argmax(1).cpu().tolist())
                if device.type == 'cuda': torch.cuda.synchronize()
                values[view] = np.asarray(result, dtype=np.int64)
                seconds[view] = time.monotonic() - tic
                print('PREDICT', c['row_id'], name, view, len(result), flush=True)
        np.savez(target / 'predictions.npz', ids=ids, **values)
        write(target / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
            views=views, num_classes=sc['num_classes'], truth_read=False, query_fit=False,
            seconds=seconds))
        all_tasks[name] = dict(count=len(ids), views=views, seconds=seconds)
    write(out / 'complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', row_id=c['row_id'],
        tests=all_tasks, views=views, truth_read=False, query_fit=False))
    write(out / 'inference.json', dict(total_resident_parameters=sum(p.numel() for p in net.parameters()),
        default_head=True, separate_head_training=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
        device=str(device)))


def score_preflight(c):
    row = row_spec(c)
    tasks = expected_tests(row); views = expected_views(row)
    if sorted(c['tests']) != sorted(tasks):
        raise ValueError('Truth closed: incomplete declared tests')
    root = Path(c['prediction_output']); complete = read(root / 'complete.json')
    if (complete.get('status') != 'ALL_PREDICTIONS_COMPLETE' or
        complete.get('row_id') != c['row_id'] or sorted(complete.get('tests', {})) != sorted(tasks) or
        complete.get('views') != views or complete.get('truth_read') is not False or
        complete.get('query_fit') is not False):
        raise ValueError('Truth closed: all-task predictions incomplete')
    prepared = {}
    for name in tasks:
        folder = root / name; done = read(folder / 'complete.json')
        manifest, expected_ids, _ = handle(c['tests'][name]['views_root'], c['classes'])
        if (done['status'] != 'PREDICTIONS_COMPLETE' or done['views'] != views or
            done['num_classes'] != len(c['classes']) or done['count'] != len(expected_ids) or
            done['truth_read'] or done['query_fit']):
            raise ValueError('Truth closed: incomplete view coverage')
        with np.load(folder / 'predictions.npz', allow_pickle=False) as predictions:
            if set(predictions.files) != {'ids', *views} or not np.array_equal(predictions['ids'], expected_ids):
                raise ValueError('Truth closed: ID alignment differs')
            values = {v: predictions[v].copy() for v in views}
        for value in values.values():
            if (value.shape != expected_ids.shape or value.dtype.kind not in 'iu' or
                value.min() < 0 or value.max() >= len(c['classes'])):
                raise ValueError('Truth closed: invalid prediction classes')
        prepared[name] = (expected_ids, values)
    return prepared


def metrics(labels, predicted, classes):
    if not len(labels): raise ValueError('Empty metric stratum')
    cm = np.bincount(classes * labels + predicted, minlength=classes ** 2).reshape(classes, classes)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(classes), where=denominator != 0)
    return dict(count=len(labels), accuracy=float(cm.trace() / len(labels)),
        macro_f1=float(f1.mean()), confusion=cm.tolist())


def score(c):
    prepared = score_preflight(c)  # No truth path is opened above this point.
    row = row_spec(c); p = _PLAN['protocols'][row['protocol_id']]
    source_dates = {_DATES[i] for i in p['source_days']}
    classes = len(c['classes']); results = []
    for name, (ids, values) in prepared.items():
        truth = read(c['tests'][name]['truth'])
        if any(str(sid) not in truth for sid in ids):
            raise ValueError('Truth join missing a predicted physical ID')
        labels = np.asarray([truth[str(sid)]['label'] for sid in ids], dtype=np.int64)
        rx = np.asarray([str(truth[str(sid)]['receiver']) for sid in ids])
        day = np.asarray([str(truth[str(sid)]['day']) for sid in ids])
        if labels.min() < 0 or labels.max() >= classes:
            raise ValueError('Truth class mapping differs')
        visibility = np.asarray(['seen_dates' if d in source_dates else 'unseen_dates' for d in day])
        for view, predicted in values.items():
            masks = [('overall', 'ALL', np.ones(len(ids), dtype=bool))]
            for dimension, groups in [('receiver', rx), ('date', day), ('date_visibility', visibility)]:
                masks.extend((dimension, stratum, groups == stratum) for stratum in sorted(set(groups)))
            masks.extend(('receiver_date', receiver + '|' + date, (rx == receiver) & (day == date))
                         for receiver in sorted(set(rx)) for date in sorted(set(day)) if ((rx == receiver) & (day == date)).any())
            for dimension, stratum, mask in masks:
                results.append(dict(test=name, view=view, dimension=dimension, stratum=stratum,
                    **metrics(labels[mask], predicted[mask], classes)))
            for label in range(classes):
                present = labels == label
                if not present.any(): continue
                hit = int(((predicted == label) & present).sum())
                predicted_count = int((predicted == label).sum())
                results.append(dict(test=name, view=view, dimension='transmitter', stratum=c['classes'][label],
                    count=int(present.sum()), recall=hit / int(present.sum()),
                    precision=hit / predicted_count if predicted_count else 0.))
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    write(out / 'scores.json', dict(status='SCORED', row_id=c['row_id'],
        model_seed=c['model_seed'], classes=c['classes'], results=results,
        macro_f1_label_policy='all_registered_classes',
        truth_last=True, target_feedback_forbidden=True))
    write(out / 'complete.json', dict(status='SCORED_COMPLETE', tests=expected_tests(row),
        views=expected_views(row), row_id=c['row_id']))
    lines = ['# ' + c['row_id'], '', '冻结默认头；全部预登记预测固定后独立truth-last评分。',
        '', '| 测试任务 | 视图 | 样本数 | Accuracy（%） | Macro-F1（%） |',
        '|---|---|---:|---:|---:|']
    for result in results:
        if result['dimension'] == 'overall':
            lines.append('| %s | %s | %d | %.4f | %.4f |' % (
                result['test'], result['view'], result['count'],
                100 * result['accuracy'], 100 * result['macro_f1']))
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SCORED_COMPLETE', c['row_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('predict', 'score'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    globals()[args.action](read(args.config))
