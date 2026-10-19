"""Frozen historical E200 default-head evaluation on the original source V only."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time


PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
PARENTS = ('20261010-phase1-standard-ce-baselines-manysig-m56-r01',
           '20261012-phase1-standard-ce-seed-extension-manysig-m52-r01')
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']


def validate_metadata(config, initialization, frozen, completion):
    if (config['variant'] != 'residual_fusion' or config['split_seed'] != 392005
            or config['source_counts'] != {'L_s': 6300, 'U_s': 56700, 'V': 27000}
            or config['selection'] != 'fixed_last_epoch' or config['augmentation']
            or config['target_access'] or config['U_s_use'] != 'unused'):
        raise ValueError('Historical source contract/config mismatch')
    if (not initialization['scratch_only'] or initialization['ancestors']
            or initialization['target_access'] or initialization.get('checkpoint_sources')):
        raise ValueError('Unverified checkpoint provenance')
    if (frozen['status'] != 'FROZEN' or frozen['epoch'] != 200
            or frozen['target_access'] or frozen['target_score_used']
            or completion['epoch'] != 200 or completion['steps'] != 10000):
        raise ValueError('Expected fixed E200 source-only checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Exact historical code defines architecture, preprocessing and source roles.
    release = PROJECT/'releases'/'standard_ce_baselines_20261010_r01'
    sys.path[:0] = [str(release), str(release/'code')]
    import torch
    from experiments.standard_ce_baselines.model import build
    from experiments.cvs_identity_ce.source import source_args
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cpu')
    rows = []
    split = loader = reference_contract = reference_config = None
    for seed in range(2026092701, 2026092709):
        parent = PARENTS[0 if seed <= 2026092704 else 1]
        folder = PROJECT/'runs'/parent/f'residual_fusion-s{seed}'/'source'
        read = lambda name: json.loads((folder/name).read_text(encoding='utf-8'))
        c, init, frozen, done = [read(name) for name in
            ('resolved_config.json', 'initialization.json', 'freeze.json', 'completion.json')]
        validate_metadata(c, init, frozen, done)
        if c['model_seed'] != seed or frozen['model_seed'] != seed:
            raise ValueError('Wrong seed')
        checkpoint = folder/'last.pt'
        if str(checkpoint) != frozen['checkpoint'] or str(checkpoint) != done['checkpoint']:
            raise ValueError('Checkpoint path mismatch')
        contract = read('source_contract.json')
        if reference_contract is not None and contract != reference_contract:
            raise ValueError('Rows do not share exact source physical roles')
        if reference_config is not None and any(c[k] != reference_config[k] for k in ('dataset', 'source_contract')):
            raise ValueError('Source input path mismatch')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (payload['variant'] != 'residual_fusion' or payload['epoch'] != 200
                or payload['classes'] != CLASSES or payload['source_contract'] != contract
                or payload['selection'] != 'fixed_last_epoch' or payload['initialization'] != init):
            raise ValueError('Checkpoint metadata/provenance mismatch')
        model = build('residual_fusion').to(device)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            smoke = model(torch.zeros(2, 2, 256))
        if smoke.shape != (2, 6) or not torch.isfinite(smoke).all():
            raise ValueError('Real-checkpoint no-query smoke failed')
        print(json.dumps(dict(event='SMOKE_PASS', seed=seed, target_access=False)), flush=True)
        if split is None:
            scoped = dict(c, output_root=str(args.output/'source_split'))
            split = build_contract_split(source_args(scoped))
            if split.test or split.named_tests or split.split_info['counts'] != c['source_counts']:
                raise ValueError('Only original source V is allowed')
            reference_contract, reference_config = contract, c
            loader = make_cvs_loader(split.val, batch_size=256, shuffle=False,
                                     num_workers=0, device=device, drop_last=False)
        cm = torch.zeros(6, 6, dtype=torch.int64)
        tic = time.perf_counter()
        with (args.output/f'predictions-s{seed}.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['physical_id', 'true_index', 'predicted_index'])
            with torch.inference_mode():
                for batch in loader:
                    y = batch['label'].cpu()
                    scores = model(batch['iq'].to(device))
                    if not torch.isfinite(scores).all():
                        raise ValueError('Nonfinite source logits')
                    predicted = scores.argmax(1).cpu()
                    cm += torch.bincount(y*6+predicted, minlength=36).reshape(6, 6)
                    writer.writerows((meta['sample_id'], a, b) for meta, a, b in
                                     zip(batch['meta'], y.tolist(), predicted.tolist()))
        if int(cm.sum()) != 27000 or cm.sum(1).tolist() != [4500]*6:
            raise ValueError('Unexpected source validation coverage')
        if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
            raise ValueError('Frozen default-head evaluation mutated model state')
        accuracy = float(cm.diag().sum())/27000
        original = done['final_source_metrics']['source_val_accuracy']
        if abs(accuracy-original) > 2/27000:
            raise ValueError('Frozen source accuracy does not reproduce historical E200 result')
        row = dict(model_seed=seed, parent_run_id=parent, checkpoint=str(checkpoint),
                   original_source_accuracy=original, source_accuracy=accuracy,
                   confusion=cm.tolist(), recall_pct={label: float(cm[i,i])/4500*100
                                                     for i, label in enumerate(CLASSES)},
                   prediction_count=int(cm.sum()), seconds=time.perf_counter()-tic,
                   model_state_unchanged=True, default_head=True, separate_head_training=False)
        rows.append(row)
        (args.output/f'seed-{seed}.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(event='ROW_COMPLETE', seed=seed, **row['recall_pct'])), flush=True)
    result = dict(status='ANALYZED', role='source_V', classes=CLASSES, rows=rows,
                  source_count_per_seed=27000, count_per_class_per_seed=4500,
                  source_training=False, target_access=False, optimizer_created=False,
                  cpu_only=True, pid=os.getpid(), cwd=os.getcwd(),
                  summary={label: dict(mean_pct=statistics.mean(r['recall_pct'][label] for r in rows),
                                       sample_sd_pp=statistics.stdev(r['recall_pct'][label] for r in rows))
                           for label in CLASSES})
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('COMPLETE '+json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
