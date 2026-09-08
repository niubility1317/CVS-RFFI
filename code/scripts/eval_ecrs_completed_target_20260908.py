"""Frozen Phase1 closed-set target prediction, followed by a separate scorer."""
import argparse
from collections import Counter, defaultdict
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCENES = ['clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']
TARGET_RX = [0, 2, 5, 7, 9, 10, 11]


def build_model(checkpoint, device):
    from cvsrffi.checkpoint_loading import (build_exact_ecrs_model_from_checkpoint,
        strip_module_prefix, infer_num_domains_from_state)
    from model_dual_cvsincnet import build_dual_model
    args = checkpoint['args']
    if args.get('use_ecrs', False):
        return build_exact_ecrs_model_from_checkpoint(checkpoint, input_len=256, device=device)
    state = strip_module_prefix(checkpoint['model'])
    kwargs = {k: args[k] for k in inspect.signature(build_dual_model).parameters if k in args}
    kwargs.update(num_classes=int(args['num_classes']), num_domains=infer_num_domains_from_state(state),
                  input_len=256, mixstyle_on=bool(args.get('use_mixstyle', False)), use_ecrs=False)
    if isinstance(kwargs.get('pa_orders'), str):
        kwargs['pa_orders'] = [int(v.strip()) for v in kwargs['pa_orders'].split(',') if v.strip()] or None
    model = build_dual_model(**kwargs).to(device)
    model.load_state_dict(state, strict=True)
    return model, {'loader': 'exact_train_py_dual_baseline', 'checkpoint_load_strict': True,
                   'feature_schema': checkpoint.get('feature_schema')}


class TruthBlindTarget:
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        x, _, domain, meta = self.base[index]
        safe = {k: meta[k] for k in ['physical_sample_id', 'receiver_id', 'day_id']}
        return x, -1, domain, safe


def stats(correct, total):
    return {'correct': correct, 'count': total, 'accuracy': 100 * correct / total if total else None}


def score_files(truth_path, prediction_path, expected_count):
    truth = {}
    with Path(truth_path).open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['physical_sample_id'] in truth:
                raise ValueError('duplicate truth ID')
            truth[r['physical_sample_id']] = r
    if len(truth) != expected_count:
        raise ValueError('truth coverage differs from declared target index')
    seen, totals, grouped, confusion = set(), defaultdict(lambda: [0, 0]), {}, {}
    net = {'rescue': 0, 'harm': 0}
    with Path(prediction_path).open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'source_truth' in r or 'truth' in r or 'label' in r:
                raise ValueError('truth leaked into prediction artifact')
            pid = r['physical_sample_id']
            if pid in seen or pid not in truth:
                raise ValueError('duplicate or unknown prediction ID')
            seen.add(pid)
            t = truth[pid]
            if (r['receiver_id'], r['day_id']) != (t['receiver_id'], t['day_id']):
                raise ValueError('target index metadata mismatch')
            hits = {}
            for path, pred in r['predictions'].items():
                if not 0 <= int(pred) < 6:
                    raise ValueError('prediction outside frozen registered class list')
                correct = int(pred == t['truth'])
                hits[path] = correct
                totals[path][0] += correct; totals[path][1] += 1
                if path not in grouped:
                    grouped[path] = {k: defaultdict(lambda: [0, 0]) for k in ['receiver', 'day', 'receiver_day', 'tx']}
                    confusion[path] = [[0] * 6 for _ in range(6)]
                confusion[path][t['truth']][pred] += 1
                for name, key in [('receiver', str(t['receiver_id'])), ('day', str(t['day_id'])),
                                  ('receiver_day', f"{t['receiver_id']}:{t['day_id']}"), ('tx', str(t['truth']))]:
                    grouped[path][name][key][0] += correct
                    grouped[path][name][key][1] += 1
            if 'fused' in hits:
                net['rescue'] += int(not hits['raw'] and hits['fused'])
                net['harm'] += int(hits['raw'] and not hits['fused'])
    if seen != set(truth):
        raise ValueError('prediction coverage is incomplete')
    result = {'count': len(seen), 'paths': {}, **net}
    for path, count in totals.items():
        result['paths'][path] = {**stats(*count), 'confusion': confusion[path],
            'groups': {group: {key: stats(*v) for key, v in cells.items()} for group, cells in grouped[path].items()}}
        rx = result['paths'][path]['groups']['receiver']
        result['paths'][path]['receiver_floor'] = min(v['accuracy'] for v in rx.values())
        day = result['paths'][path]['groups']['day']
        result['paths'][path]['unseen_day0_accuracy'] = day['0']['accuracy'] if '0' in day else None
    result['net'] = net['rescue'] - net['harm']
    return result


def predict(manifest, output, device_name):
    import torch
    from torch.utils.data import DataLoader
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    from cvsrffi.ecrs_evaluation import evaluate_revision_paths
    from cvsrffi.eval import apply_sat_channel_for_scenario
    torch.set_num_threads(4)
    device = torch.device(device_name)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Validate every completed source choice before any target access.
    for row in manifest['rows']:
        source = json.loads(Path(row['source_result']).read_text())
        if source['status'] != 'SOURCE_SCREEN_COMPLETE' or source['best_epoch'] != row['selected_epoch']:
            raise ValueError('row has not completed with the frozen source choice')
        if Path(source['selected_checkpoint']).resolve() != Path(row['checkpoint']).resolve():
            raise ValueError('checkpoint is not the source-selected model')
        cp = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        if cp['epoch'] != row['selected_epoch'] or int(cp['args']['num_classes']) != 6:
            raise ValueError('frozen checkpoint epoch or class list differs')
        model, audit = build_model(cp, device)
        model.eval()
        with torch.no_grad():
            probe = model.forward_identity(torch.randn(2, 2, 256, device=device))['tx_logits']
        if not torch.isfinite(probe).all():
            raise ValueError('real checkpoint no-query smoke failed')
        print(json.dumps({'smoke': row['row'], 'status': 'PASS', 'audit': audit}), flush=True)
        del cp, model, probe
    ds = load_wisig_compact_pkl(manifest['wisig_pkl'])
    base = WiSigCompactDataset(ds, out_len=256, crop_mode='center', normalize=True, center=False,
        equalized=1, tx_keep=list(range(6)), rx_keep=TARGET_RX, day_keep=[0, 1, 2, 3],
        domain='rx_day', max_samples_per_combo=None)
    if len(base) != 168000:
        raise ValueError(f'expected 168000 target records, got {len(base)}')
    with (output / 'target_truth.jsonl').open('x', encoding='utf-8') as f:
        for k, item in enumerate(base.index):
            f.write(json.dumps({'physical_sample_id': f'sample:{k}', 'truth': item.tx_i,
                'receiver_id': item.rx_i, 'day_id': item.day_i}) + '\n')
    loader = DataLoader(TruthBlindTarget(base), batch_size=256, shuffle=False, num_workers=0)
    resources = {}
    for row in manifest['rows']:
        name = row['row']; folder = output / name; folder.mkdir(exist_ok=False)
        cp = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        model, audit = build_model(cp, device)
        channel_args = SimpleNamespace(**cp['args'])
        torch.cuda.reset_peak_memory_stats(device)
        resources[name] = {'selected_epoch': cp['epoch'], 'audit': audit, 'scenarios': {}}
        for index, scene in enumerate(SCENES):
            generator = torch.Generator(device=device).manual_seed(2027 + max(0, index-1) * 1009)
            def transform(x, batch_index):
                return apply_sat_channel_for_scenario(x, scene, channel_args, gen=generator, return_meta=False)[0]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = evaluate_revision_paths(model, loader, device, scenario=scene,
                transform=None if scene == 'clean' else transform,
                output_path=folder / (scene + '.jsonl'), source_metrics=False)
            torch.cuda.synchronize(device)
            if result['count'] != 168000:
                raise ValueError('incomplete prediction output')
            resources[name]['scenarios'][scene] = {'count': result['count'],
                'elapsed_s': time.perf_counter()-started, 'prediction_path': str(folder / (scene+'.jsonl'))}
            print(json.dumps({'row': name, 'scene': scene, **resources[name]['scenarios'][scene]}), flush=True)
        resources[name]['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        resources[name]['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        del cp, model
        torch.cuda.empty_cache()
    (output / 'prediction_resources.json').write_text(json.dumps(resources, indent=2), encoding='utf-8')
    print('PREDICTIONS_COMPLETE; starting independent scorer', flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'score', '--output', str(output)], check=True)


def score(output):
    manifest = json.loads((output / 'frozen_manifest.json').read_text())
    resources = json.loads((output / 'prediction_resources.json').read_text())
    summary = {'status': 'TARGET_TEST_COMPLETE', 'scope': 'Phase1 closed-set frozen target test',
               'target_used_for_selection': False, 'rows': {}, 'resources': resources}
    for row in manifest['rows']:
        name = row['row']
        summary['rows'][name] = {'selected_epoch': row['selected_epoch'], 'scenarios': {}}
        for scene in SCENES:
            summary['rows'][name]['scenarios'][scene] = score_files(output/'target_truth.jsonl',
                output/name/(scene+'.jsonl'), 168000)
    with (output / 'target_test_summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('TARGET_TEST_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['predict', 'score'])
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.mode == 'score':
        score(args.output)
    else:
        predict(json.loads(args.manifest.read_text()), args.output, args.device)


if __name__ == '__main__':
    main()
