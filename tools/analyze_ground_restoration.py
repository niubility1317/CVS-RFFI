"""Summarize completed fixed Main predictions without loading IQ or labels."""
import argparse
import csv
import json
from pathlib import Path

VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
STREAMS = ('raw', 'DURA_v1', 'candidate')
TASKS = {'cross_rx': 126000, 'cross_both': 42000}
F0_COMMIT = '3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def correct(row):
    matrix = row['confusion']
    if (len(matrix) != 6 or any(len(r) != 6 for r in matrix)
            or any(type(x) is not int or x < 0 for r in matrix for x in r)
            or sum(sum(r) for r in matrix) != row['count']):
        raise ValueError('Valid six-class confusion matrix and matching count required')
    return sum(matrix[i][i] for i in range(6))


def index_rows(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in ('test', 'readout', 'view', 'dimension', 'stratum'))
        if key in result:
            raise ValueError('Duplicate scored group')
        result[key] = row
    return result


def summarize(scores):
    if (scores.get('status'), scores.get('truth_last'), scores.get('target_feedback_forbidden')) != ('SCORED', True, True):
        raise ValueError('Independent completed truth-last scores required')
    scored = index_rows(scores['results'])
    paired = index_rows(scores['paired_raw'])
    raw_keys = {key for key in scored if key[1] == 'raw'}
    if not raw_keys:
        raise ValueError('Complete raw score groups required')
    expected_scored = {(test, stream, view, dim, group)
                       for test, _, view, dim, group in raw_keys for stream in STREAMS}
    expected_paired = {key for key in expected_scored if key[1] != 'raw'}
    if set(scored) != expected_scored or set(paired) != expected_paired:
        raise ValueError('All raw, DURA and candidate groups and pair counts required')
    merged, clean_by_task = {}, {}
    for test, _, view, dim, group in raw_keys:
        if test not in TASKS or view not in VIEWS:
            raise ValueError('Unexpected test population or view')
        reference = scored[test, 'raw', view, dim, group]
        n, raw = reference['count'], correct(reference)
        key = (view, dim, group)
        entry = merged.setdefault(key, dict(view=view, dimension=dim, stratum=group, count=0,
            raw_correct=0, dura_correct=0, candidate_correct=0, dura_repairs=0, dura_damage=0,
            candidate_repairs=0, candidate_damage=0))
        entry['count'] += n
        entry['raw_correct'] += raw
        for stream, prefix in (('DURA_v1', 'dura'), ('candidate', 'candidate')):
            score = scored[test, stream, view, dim, group]
            pair = paired[test, stream, view, dim, group]
            value = correct(score)
            if (score['count'] != n or pair['count'] != n or pair['net'] != pair['repairs'] - pair['damage']
                    or raw + pair['net'] != value or min(pair['repairs'], pair['damage']) < 0):
                raise ValueError('Score, group count and paired corrections must agree')
            entry[prefix+'_correct'] += value
            entry[prefix+'_repairs'] += pair['repairs']
            entry[prefix+'_damage'] += pair['damage']
        if view == 'clean' and dim == 'overall' and group == 'ALL':
            if n != TASKS[test]:
                raise ValueError('Original full Main populations required')
            clean_by_task[test] = dict(count=n, raw=raw,
                candidate=correct(scored[test, 'candidate', view, dim, group]))
    if set(clean_by_task) != set(TASKS):
        raise ValueError('Both Main clean populations required')
    if (clean_by_task['cross_rx']['raw'], clean_by_task['cross_both']['raw']) != (111019, 34366):
        raise ValueError('Original clean baseline required for each Main population')
    clean = merged['clean', 'overall', 'ALL']
    total = sum(TASKS.values())
    if clean['count'] != total or clean['raw_correct'] != 145385:
        raise ValueError('Original frozen Tri-SIG-TIME SSL raw clean reference required')
    threshold = (clean['raw_correct'] * 100 - 3 * total + 99) // 100
    scenes = []
    for view in VIEWS[1:]:
        entry = merged[view, 'overall', 'ALL']
        if entry['count'] != total:
            raise ValueError('Every practical scene must cover both Main populations')
        scenes.append(dict(entry, minimum_correct=threshold,
                           target_met=entry['candidate_correct'] >= threshold))
    preserved = {k: v['candidate'] >= v['raw'] for k, v in clean_by_task.items()}
    return dict(sample_count=total, raw_clean_correct=clean['raw_correct'], minimum_correct=threshold,
        clean_by_task=clean_by_task, clean_preserved=preserved, scenes=scenes,
        goal_met=all(r['target_met'] for r in scenes) and all(preserved.values()),
        groups=sorted(merged.values(), key=lambda r: (VIEWS.index(r['view']), r['dimension'], str(r['stratum']))))


def load_completed(run, arm):
    run = Path(run)
    row = run/'rows'/arm
    train, pred, done, scores, result, full = (read(p) for p in (
        row/'training/complete.json', row/'prediction/complete.json', row/'score/complete.json',
        row/'score/scores.json', row/'results.json', run/'complete.json'))
    identity = train['F0_identity']
    if ((train['status'], train['epoch'], train['updates'], train['TX_labels_read'], train['V_reads'],
         train['OE_reads'], train['target_training_reads']) != ('SOURCE_TRAINED', 30, 7410, 0, 0, 0, 0)
            or (identity.get('source_code_commit'), identity.get('model_seed'), identity.get('parameters'),
                identity.get('model'), identity.get('epoch'), identity.get('updates'),
                identity.get('source_run'), identity.get('source_row'))
            != (F0_COMMIT, 2026092701, 232993, 'original_F0_E50', 50, 11100,
                '20261027-phase1-pamf-repl-shuffled-views-m16-r02',
                'main_split-repl-shuffled_time-3view-s2026092701')):
        raise ValueError('Original source-only final30 and frozen Tri-SIG-TIME SSL required')
    if (pred['status'] != 'ALL_PREDICTIONS_COMPLETE' or pred['truth_read'] or pred['query_fit']
            or pred['views'] != list(VIEWS) or pred['readouts'] != list(STREAMS)
            or set(pred['tests']) != set(TASKS)
            or any(pred['tests'][k]['count'] != v or pred['tests'][k]['views'] != list(VIEWS)
                   or pred['tests'][k]['readouts'] != list(STREAMS) for k, v in TASKS.items())):
        raise ValueError('All two-population seven-view predictions must precede scoring')
    if (done['status'] != 'SCORED_COMPLETE' or done['views'] != list(VIEWS)
            or done['readouts'] != list(STREAMS) or set(done['tests']) != set(TASKS)
            or result['status'] != 'ANALYZED' or result['truth_last'] is not True
            or full['status'] != 'ARTIFACTS_COMPLETE' or not full['all_rows_trained30_and_main_seven_scored']
            or scores['model_seed'] != 2026092701):
        raise ValueError('Completed original scoring and run required')
    if (pred['row_id'] != arm or done['row_id'] != arm or result['arm'] != arm
            or scores['row_id'] != arm or any(d['run_id'] != run.name for d in (pred, done, result, scores))
            or pred['code_commit'] != train['code_commit'] or result['code_commit'] != train['code_commit']):
        raise ValueError('Matching run, row and scientific commit required')
    return scores


def write_report(output, summary):
    output = Path(output)
    output.mkdir()
    groups = summary['groups']
    with (output/'combined_group_metrics.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with (output/'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    n = summary['sample_count']
    lines = ['# 地面 Main 恢复器正式结果', '',
             f"raw clean：{summary['raw_clean_correct']}/{n}；每场景最低正确数：{summary['minimum_correct']}。",
             '', '| 场景 | 候选准确率 | DURA 准确率 | 候选净纠错 | 达到目标 |',
             '|---|---:|---:|---:|:---:|']
    for q in summary['scenes']:
        net = q['candidate_repairs'] - q['candidate_damage']
        lines.append(f"| {q['view']} | {100*q['candidate_correct']/n:.6f}% | "
                     f"{100*q['dura_correct']/n:.6f}% | {net:+d} | {'是' if q['target_met'] else '否'} |")
    lines += ['', '## Clean 分总体', '']
    for name, q in summary['clean_by_task'].items():
        lines.append(f"- {name}：raw {q['raw']}/{q['count']}，候选 {q['candidate']}/{q['count']}，"
                     f"净变化 {q['candidate']-q['raw']:+d}。")
    lines += ['', '全部六景与两个 clean 总体同时满足：' + ('是' if summary['goal_met'] else '否') + '。',
              '该结果为固定单 seed、已曝光测试集的事后评分；训练、checkpoint 和方案选择不使用目标反馈。',
              '逐 TX/RX/day 等分组及净纠错计数见 combined_group_metrics.csv。', '']
    (output/'analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    scores = load_completed(args.run, args.arm)
    summary = summarize(scores)
    write_report(args.output, summary)
    print(json.dumps({'status': 'ANALYZED', 'goal_met': summary['goal_met'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
