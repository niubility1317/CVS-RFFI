"""Full tail logs and frozen clean scores; no model or IQ access."""
import gzip
import json
import math
import statistics
from pathlib import Path
from . import design as d


def rows_json(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_metrics(root):
    done = d.read(root / 'scoring/complete.json')
    if done['status'] != 'SCORED_COMPLETE' or not done['truth_last'] or 'clean' not in done['views']:
        raise ValueError('Clean independent score incomplete')
    data = d.read(root / 'scoring/scores.json')
    results = data['results']
    overall = next(x for x in results if x['view'] == 'clean' and x['dimension'] == 'overall')
    receivers = [x for x in results if x['view'] == 'clean' and x['dimension'] == 'receiver']
    if len(receivers) != 7:
        raise ValueError('All seven fixed target receivers required')
    return dict(accuracy=overall['accuracy'], macro_f1=overall['macro_f1'],
                worst_RX=min(x['accuracy'] for x in receivers))


def summary(values):
    return dict(values=values, mean=statistics.mean(values), sample_SD=statistics.stdev(values), n=len(values))


def audit(row):
    root = d.BASE / row['row_id'] / 'source'
    done = d.read(root / 'completion.json')
    result = d.read(root / 'result.json')
    ce = rows_json(root / 'satellite_ce_steps.jsonl.gz')
    reference = rows_json(root / 'reference_steps.jsonl.gz')
    unit = rows_json(root / 'unit_steps.jsonl.gz')
    sequence = rows_json(root / 'training_sequence.jsonl.gz')
    epochs = [json.loads(x) for x in (root / 'epoch_metrics.jsonl').read_text().splitlines() if x.strip()]
    if done['epoch'] != 10 or done['optimizer_steps'] != 2220 or len(ce) != 2220 or len(reference) != 2220 or len(unit) != 2220:
        raise ValueError('Incomplete fixed E10 training logs')
    if done['target_access'] or result['U_truth_read'] or result['head_fit_updates']:
        raise ValueError('Illegal training access or separate head fit')
    for i, event in enumerate(ce, 1):
        if event['step'] != i or event['temperature'] != 1. or event['weight'] != .5 or not event['gradient_alive']:
            raise ValueError('CE_HALF changed during training')
        if any(not math.isfinite(v) for v in event.values() if isinstance(v, float)):
            raise ValueError('Nonfinite CE record')
    for path in sorted((d.BASE / 'logs').glob(row['row_id'] + '.log')):
        text = path.read_text(encoding='utf8', errors='replace')
        if any(marker in text for marker in ('Traceback (most recent call last)', 'CUDA out of memory', 'FloatingPointError')):
            raise ValueError('Training stdout contains technical failure')
    return dict(status='PASS', ce_steps=len(ce), reference_steps=len(reference), unit_steps=len(unit),
                epoch_records=len(epochs), sequence_records=len(sequence), source_result=result,
                parent_epochs=20, new_epochs=10, cumulative_epochs=30, new_updates=2220)


def collect():
    for row in d.rows():
        done = d.read(d.BASE / row['row_id'] / 'prediction/complete.json')
        if done['views'] != ['clean'] or done['truth_read'] or done['head_fit_updates']:
            raise ValueError('Declared clean predictions not frozen')
    seed_metrics = {2026092701: clean_metrics(d.SEED01)}
    audits = {}
    for row in d.rows():
        audits[row['row_id']] = audit(row)
        seed_metrics[row['model_seed']] = clean_metrics(d.BASE / row['row_id'])
    comparisons = {}
    for arm in ('B00', 'B01'):
        points = {}
        for seed in (2026092701, *d.SEEDS):
            if seed == 2026092701:
                path = Path(d.PROJECT) / 'runs/20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01' / (arm + '-s' + str(seed))
            else:
                path = Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN / (arm + '-s' + str(seed))
            points[seed] = clean_metrics(path)
        comparisons[arm] = points
    values = [seed_metrics[s]['accuracy'] for s in sorted(seed_metrics)]
    result = dict(status='ANALYZED', run_id=d.RUN, model_seeds=sorted(seed_metrics), test_views=['clean'],
                  CE_HALF=summary(values), per_seed=seed_metrics, controls=comparisons,
                  paired_accuracy_pp={arm: summary([(seed_metrics[s]['accuracy'] - comparisons[arm][s]['accuracy']) * 100
                                                   for s in sorted(seed_metrics)]) for arm in comparisons},
                  new_model_epochs=20, new_updates=4440, reused_seed01=True,
                  selection_context='User-requested after exposed seed01 clean result; not new independent blind test',
                  satellite_three_seed_claim=False, automatic_expansion=False)
    d.write(d.BASE / 'full_log_verification.json', audits)
    d.write(d.BASE / 'results.json', result)
    lines = ['# CE_HALF 三种子 clean 完整结果', '', '状态：ANALYZED；seed01 复用，02/03 各追加 E10、累计 E30。', '',
             '|种子|CE_HALF clean %|原 As30 %|NoAs30 %|', '|---|---:|---:|---:|']
    for seed, v in sorted(seed_metrics.items()):
        lines.append(f"|{seed}|{100*v['accuracy']:.6f}|{100*comparisons['B01'][seed]['accuracy']:.6f}|{100*comparisons['B00'][seed]['accuracy']:.6f}|")
    lines += ['', f"三种子均值：{100*result['CE_HALF']['mean']:.6f}% ± {100*result['CE_HALF']['sample_SD']:.6f}%（样本标准差）。", '',
              '新增两种子只测 clean；不得据此宣称三种子星地收益。全部固定种子保留，不挑选或重跑。',
              '这是用户在看到 seed01 成绩后指定的复现，不构成新的独立盲测。']
    (d.BASE / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return result
