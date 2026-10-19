"""Artifact-based completion audit and summaries; never feeds a training decision."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .data import read, write
from .driver import atomic
from .evaluation import metrics, score_preflight
from .plan import VIEWS, SEEDS, make_plan

_DATES = ('2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23')


def extend_frozen_scores(ids, values, truth, classes, source_dates, test='main', original=None, class_names=None):
    y = np.asarray([truth[str(x)]['label'] for x in ids], dtype=np.int64)
    rx = np.asarray([str(truth[str(x)]['receiver']) for x in ids])
    date = np.asarray([str(truth[str(x)]['day']) for x in ids])
    if y.min() < 0 or y.max() >= classes: raise ValueError('Truth mapping differs')
    results = []
    for view, predictions in values.items():
        overall = metrics(y, predictions, classes)
        if original is not None:
            expected = next(x for x in original if x['view'] == view and x['dimension'] == 'overall'
                and x.get('test', 'main') == test)
            if (overall['confusion'] != expected['confusion'] or
                abs(overall['accuracy'] - expected['accuracy']) > 1e-12 or
                abs(overall['macro_f1'] - expected['macro_f1']) > 1e-12):
                raise ValueError('Independent frozen prediction audit mismatch')
        results.append(dict(test=test, view=view, dimension='overall', stratum='ALL', **overall))
        visibility = np.asarray(['seen_dates' if d in source_dates else 'unseen_dates' for d in date])
        for dimension, values_in_dimension in [('receiver', rx), ('date', date), ('date_visibility', visibility)]:
            for stratum in sorted(set(values_in_dimension)):
                mask = values_in_dimension == stratum
                results.append(dict(test=test, view=view, dimension=dimension, stratum=stratum,
                    **metrics(y[mask], predictions[mask], classes)))
        for label in range(classes):
            mask = y == label
            if not mask.any(): continue
            hit = int(((predictions == label) & mask).sum()); predicted_count = int((predictions == label).sum())
            results.append(dict(test=test, view=view, dimension='transmitter',
                stratum=class_names[label] if class_names else str(label), count=int(mask.sum()),
                recall=hit / int(mask.sum()), precision=hit / predicted_count if predicted_count else 0.))
    return results


def check_source(root, row):
    frozen = read(Path(root) / 'freeze.json'); c = frozen['config']
    if (frozen['status'] != 'SOURCE_TRAINED_FROZEN' or frozen['target_access'] is not False or
        frozen['target_score_used'] or c['model_seed'] != row['model_seed'] or
        c['row_id'] != row['row_id'] or c['family'] != row['family'] or c['variant'] != row['variant'] or
        c.get('initialization') != 'scratch' or c.get('U_s_use') is not False or
        c.get('default_head_joint_training') is not True or c.get('separate_head_training') is not False):
        raise ValueError('Source evidence incomplete')
    payload = torch.load(frozen['checkpoint'], map_location='cpu')
    if (payload['source_contract'] != frozen['source_contract'] or payload['config'] != c or
        payload.get('scratch_only') is not True or payload.get('ancestors') != [] or
        payload.get('target_access') is not False or payload.get('default_head_joint_training') is not True or
        payload.get('separate_head_training') is not False):
        raise ValueError('Checkpoint provenance or head contract mismatch')
    selected_epoch = payload['epoch']; selected_stage = payload['stage']; del payload
    family = row['family']
    if family in ('riei', 'drift', 'ibrff', 'sdg'):
        stages = frozen['stages']
        if len(stages) != (2 if family == 'ibrff' else 1):
            raise ValueError('Native stage count differs')
        for stage in stages:
            limit = 100 if family in ('riei', 'drift') else 200
            if stage['epochs'] < 1 or stage['epochs'] > limit or (family != 'ibrff' and stage['epochs'] != limit):
                raise ValueError('Native training budget differs')
        if family in ('riei', 'drift'):
            if selected_epoch != 100 or frozen['selection'] != 'fixed_final_E100':
                raise ValueError('Fixed final checkpoint differs')
        else:
            records = [json.loads(x) for x in (Path(root) / ('stage%d' % selected_stage) /
                'epochs.jsonl').read_text().splitlines() if x.strip()]
            chosen = min(records, key=lambda x: x['source_val_loss'])
            if selected_epoch != chosen['epoch'] or frozen['selection'] != 'source_validation_nll':
                raise ValueError('Checkpoint not selected by source NLL')
        cost = dict(stages=stages, batch_steps=sum(x['updates'] for x in stages),
            optimizer_steps=sum(x['updates'] for x in stages) * (3 if family == 'riei' and row['variant'] == 'full' else 1),
            stage_seconds=sum(x['seconds'] for x in stages), selected_epoch=selected_epoch)
    else:
        expected = row['updates'] if row['budget_kind'] == 'updates' else row['epochs'] * (
            (frozen['source_contract']['counts']['L_s'] + row['batch_size'] - 1) // row['batch_size'])
        if frozen['updates'] != expected or frozen['selection'] != 'fixed_final_budget':
            raise ValueError('ResNet actual update budget differs')
        cost = dict(updates=frozen['updates'], physical_source_exposure=frozen['physical_source_exposure'],
            seconds=frozen['seconds'], selected_epoch=selected_epoch)
    if (Path(root) / 'training_cost.json').exists():
        cost.update(process_cost=read(Path(root) / 'training_cost.json'))
    return cost


def parent_audit(rt):
    root = Path(rt['parent_run_root']); state = read(root / 'pipeline_state.json')
    matrix = read(Path(rt['parent_release_root']) / 'matrix.json')['rows']
    if state['status'] != 'ANALYZED' or len(matrix) != 24:
        raise ValueError('Parent matrix incomplete')
    inspected = []
    for row in matrix:
        rid = row['row_id']; folder = root / rid
        if state['rows'][rid]['status'] != 'ANALYZED' or read(folder / 'complete.json')['status'] != 'ANALYZED':
            raise ValueError('Parent row incomplete')
        done = read(folder / 'prediction/complete.json')
        if (done['views'] != list(VIEWS) or done['truth_read'] or done['query_fit'] or
            done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != 168000):
            raise ValueError('Parent prediction scope mismatch')
        with np.load(folder / 'prediction/predictions.npz', allow_pickle=False) as pred:
            ids = pred['ids'].copy()
            if set(pred.files) != {'ids', *VIEWS} or len(ids) != 168000 or len(set(ids.tolist())) != 168000:
                raise ValueError('Parent physical coverage differs')
            for view in VIEWS:
                value = pred[view]
                if value.shape != ids.shape or value.dtype.kind not in 'iu' or value.min() < 0 or value.max() >= 6:
                    raise ValueError('Invalid parent prediction classes')
        scores = read(folder / 'scores/scores.json')
        if scores['status'] != 'SCORED' or not scores['truth_last']:
            raise ValueError('Parent scores missing')
        cost = check_source(folder / 'source', row)
        inspected.append((row, scores['results'], cost))
    # Every parent prediction has passed completeness before independent truth access.
    truth = read(rt['existing_truth']); rows = []
    names = read(Path(rt['existing_source_cache']) / 'metadata.json')['classes']
    for row, original, cost in inspected:
        with np.load(root / row['row_id'] / 'prediction/predictions.npz', allow_pickle=False) as p:
            values = {v: p[v].copy() for v in VIEWS}
            scores = extend_frozen_scores(p['ids'], values, truth, 6, set(_DATES[1:]),
                original=original, class_names=names)
        rows.append(dict(config=dict(row, group='F0', protocol_id='F0'), results=scores,
            cost=cost, independent_prediction_audit=True))
    return rows


def selected_results(score_path):
    scores = read(score_path)
    if scores['status'] != 'SCORED' or not scores['truth_last'] or not scores['target_feedback_forbidden']:
        raise ValueError('Independent score closure missing')
    return scores['results']


def summary_records(rows):
    buckets = defaultdict(list)
    for row in rows:
        c = row['config']
        version = c.get('route', c['variant'])
        for overall in row['results']:
            if overall['dimension'] != 'overall': continue
            test, view = overall.get('test', 'main'), overall['view']
            receivers = [x['accuracy'] for x in row['results'] if x['dimension'] == 'receiver'
                and x.get('test', 'main') == test and x['view'] == view]
            if not receivers: raise ValueError('Per-RX metrics missing')
            buckets[(c['group'], c['family'], version, test, view)].append(dict(
                seed=c['model_seed'], row_id=c['row_id'], accuracy=overall['accuracy'],
                macro_f1=overall['macro_f1'], equal_rx_accuracy=statistics.mean(receivers),
                worst_rx=min(receivers)))
    output = []
    for key, values in buckets.items():
        seeds = [v['seed'] for v in values]
        if len(set(seeds)) != len(seeds): raise ValueError('Model seeds duplicated within a group')
        record = dict(zip(('group', 'family', 'version', 'test', 'view'), key))
        record.update(actual_seeds=len(values), seeds=seeds, rows=values)
        for field in ('accuracy', 'equal_rx_accuracy', 'macro_f1', 'worst_rx'):
            array = [v[field] for v in values]
            record[field + '_mean'] = statistics.mean(array)
            record[field + '_sample_sd'] = statistics.stdev(array) if len(array) > 1 else None
        output.append(record)
    paired = []
    groups = {(k[0], k[1], k[3], k[4]) for k in buckets}
    for group, family, test, view in sorted(groups):
        full = {v['seed']: v for v in buckets.get((group, family, 'full', test, view), [])}
        ce = {v['seed']: v for v in buckets.get((group, family, 'ce', test, view), [])}
        seeds = sorted(set(full) & set(ce))
        if not seeds: continue
        delta = [full[s]['accuracy'] - ce[s]['accuracy'] for s in seeds]
        rx_delta = [full[s]['equal_rx_accuracy'] - ce[s]['equal_rx_accuracy'] for s in seeds]
        paired.append(dict(group=group, family=family, test=test, view=view, actual_pairs=len(seeds),
            seeds=seeds, full_minus_ce=delta, mean=statistics.mean(delta),
            sample_sd=statistics.stdev(delta) if len(delta) > 1 else None,
            positive_pairs=sum(x > 0 for x in delta),
            equal_rx_full_minus_ce=rx_delta, equal_rx_mean=statistics.mean(rx_delta),
            equal_rx_sample_sd=statistics.stdev(rx_delta) if len(rx_delta) > 1 else None))
    return output, paired


def analyze(rt):
    plan = make_plan(); root = Path(rt['run_root'])
    state = read(root / 'pipeline_state.json'); parent = parent_audit(rt)
    rows = list(parent); actual_train = actual_eval = 0
    missing = []
    for row in plan['rows']:
        folder = root / row['row_id']
        if not (folder / 'complete.json').exists():
            missing.append(row['row_id']); continue
        done = read(folder / 'complete.json')
        if done['status'] != 'ANALYZED' or done['row_id'] != row['row_id']:
            raise ValueError('Row completion differs')
        results = selected_results(folder / 'scores/scores.json')
        configuration = read(folder / 'score_config.json')
        if configuration['row_id'] != row['row_id'] or configuration['model_seed'] != row['model_seed']:
            raise ValueError('Scoring configuration differs from registered row')
        prepared = score_preflight(configuration)
        source_dates = {_DATES[i] for i in plan['protocols'][row['protocol_id']]['source_days']}
        for test, (ids, predictions) in prepared.items():
            actual_truth = read(configuration['tests'][test]['truth'])
            extend_frozen_scores(ids, predictions, actual_truth, len(configuration['classes']),
                source_dates, test=test, original=results, class_names=configuration['classes'])
        if row['budget_kind'] == 'frozen_eval':
            actual_eval += 1; cost = dict(historical_checkpoint_reused=True)
            legacy_folder = Path(rt['legacy_run_root']) / (
                '%s-s%d' % (row['family'], row['model_seed']))
            old_done = read(legacy_folder / 'prediction/complete.json')
            if old_done['status'] != 'PREDICTIONS_COMPLETE' or old_done['truth_read'] or old_done['query_fit']:
                raise ValueError('Historical clean predictions incomplete')
            with np.load(legacy_folder / 'prediction/predictions.npz', allow_pickle=False) as p:
                ids, values = p['ids'].copy(), dict(clean=p['clean'].copy())
                if (set(p.files) != {'ids', 'clean'} or len(ids) != 168000 or
                    len(set(ids.tolist())) != 168000 or values['clean'].shape != ids.shape or
                    values['clean'].dtype.kind not in 'iu' or values['clean'].min() < 0 or values['clean'].max() >= 6):
                    raise ValueError('Historical clean coverage differs')
            truth = read(rt['existing_truth'])
            original = read(legacy_folder / 'scores/scores.json')['results']
            results += extend_frozen_scores(ids, values, truth, 6, set(_DATES[1:]), original=original,
                class_names=configuration['classes'])
        else:
            actual_train += 1; cost = check_source(folder / 'source', row)
        rows.append(dict(config=row, results=results, cost=cost))
    summary, paired = summary_records(rows)
    complete = len(parent) == 24 and actual_train == 162 and actual_eval == 6 and not missing
    status = 'ANALYZED' if complete else 'PARTIAL'
    write(root / 'summary.json', dict(status=status, parent_rows=24, new_training_rows=actual_train,
        frozen_eval_rows=actual_eval, missing=missing, summary=summary, paired=paired,
        target_feedback_forbidden=True, expected_model_seeds=list(SEEDS)))
    write(root / 'audited_rows.json', dict(status=status, rows=rows, exclude_pamf=True))
    with (root / 'summary.csv').open('x', encoding='utf-8-sig', newline='') as file:
        columns = ['group', 'family', 'version', 'test', 'view', 'actual_seeds',
            'accuracy_mean', 'accuracy_sample_sd', 'equal_rx_accuracy_mean',
            'equal_rx_accuracy_sample_sd', 'macro_f1_mean', 'macro_f1_sample_sd',
            'worst_rx_mean', 'worst_rx_sample_sd']
        writer = csv.DictWriter(file, fieldnames=columns); writer.writeheader()
        for record in summary: writer.writerow({k: record[k] for k in columns})
    lines = ['# Phase1非PAMF对比实验结果', '', '状态：' + status,
        '固定设计：24个既有训练＋162个新增训练＋6个冻结模型补测；全部预登记视图固定后独立评分。',
        '', '| 实验组 | 方法 | 版本 | n seed | clean等权RX（%） | clean总体Accuracy（%） | clean Macro-F1（%） | 最差RX（%） |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['test'] == 'main' and r['view'] == 'clean':
            lines.append('| %s | %s | %s | %d | %.4f | %.4f | %.4f | %.4f |' % (
                r['group'], r['family'], r['version'], r['actual_seeds'],
                100 * r['equal_rx_accuracy_mean'], 100 * r['accuracy_mean'],
                100 * r['macro_f1_mean'], 100 * r['worst_rx_mean']))
    lines += ['', '样本SD、逐seed、七视图、日期分层、源数量链及full−CE配对差见summary.json及audited_rows.json。',
        '三seed不视为接收机总体的显著性证明；ManySig／ManyRx同属WiSig采集系统；六practical为压力代理。',
        '', '缺少行：' + str(len(missing)), '全部负结果保留，不据目标成绩改参数、选模或选择性重跑。']
    (root / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    state.update(status=status, analyzed_training=actual_train, analyzed_frozen_eval=actual_eval,
        parent_analyzed=24, missing=missing)
    state.pop('stage_pid', None)
    atomic(root / 'pipeline_state.json', state)
    write(root / 'completion.json', dict(status=status, goal_scope='non_PAMF_fixed_design',
        parent_training=24, new_training=actual_train, frozen_eval=actual_eval,
        expected_training_total=186, expected_eval=6, complete=complete))
    print('COMPARISON_AUDIT', status, len(parent), actual_train, actual_eval, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    analyze(read(parser.parse_args().config))
