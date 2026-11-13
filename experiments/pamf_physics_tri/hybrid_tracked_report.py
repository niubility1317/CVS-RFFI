"""Seven-scene paired results and unlabeled packet-trajectory summaries."""
import collections
import gzip
import json
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import hybrid_tracked_plan as plan
from .target_tuning_report import load_score


def summarize_trace(paths):
    counts, classes, receivers, days = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    previous = {}
    by_epoch = collections.defaultdict(collections.Counter)
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf8') as f:
            for line in f:
                b = json.loads(line)
                e = by_epoch[b['epoch']]
                for packet, rx, day, confidence, pseudo, accepted, v1, v2 in zip(
                    b['packet'], b['rx'], b['day'], b['confidence'], b['pseudo'], b['accepted'],
                    b['aux_prediction_1'], b['aux_prediction_2']):
                    counts['exposures'] += 1; e['exposures'] += 1
                    counts['accepted'] += int(accepted); e['accepted'] += int(accepted)
                    receivers[str(rx)] += 1; days[str(day)] += 1
                    if accepted:
                        classes[str(pseudo)] += 1
                        if v1 is not None and v2 is not None:
                            counts['accepted_aux_views_observed'] += 1
                            counts['accepted_view1_agrees'] += int(v1 == pseudo)
                            counts['accepted_view2_agrees'] += int(v2 == pseudo)
                            e['accepted_views_both_agree'] += int(v1 == pseudo and v2 == pseudo)
                    if packet in previous:
                        counts['repeat_exposures'] += 1
                        counts['pseudo_changed'] += int(previous[packet] != pseudo)
                        e['repeat_exposures'] += 1; e['pseudo_changed'] += int(previous[packet] != pseudo)
                    previous[packet] = pseudo
    return dict(**counts, distinct_packets=len(previous), accepted_pseudo_classes=dict(classes),
                rx_exposures=dict(receivers), day_exposures=dict(days), by_epoch={str(k): dict(v) for k,v in sorted(by_epoch.items())},
                interpretation='pseudo-label stability and view agreement are not true-label accuracy')


def analyze(rt):
    root = Path(rt['run_root'])
    sources, controls = read(root/'sources.json'), read(root/'controls.json')
    rows, groups = [], collections.defaultdict(list)
    for row in plan.candidate_rows():
        paths = sources[row['row_id']]
        current = load_score(paths['scores'])
        pl = load_score(controls[f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]['scores'])
        old = load_score(controls[plan.baseline_row(row)['row_id']]['scores'])
        if current['counts'] != pl['counts'] or current['counts'] != old['counts']:
            raise ValueError('Matched test populations differ')
        source = Path(paths['source_output'])
        states = [read(source/seg/'repl_state.json') for seg in ('segment30','segment50')]
        trace = summarize_trace([source/seg/'U_trace.jsonl.gz' for seg in ('segment30','segment50')])
        if trace['exposures'] != sum(s['seen'] for s in states) or trace['accepted'] != sum(s['accepted'] for s in states):
            raise ValueError('Packet trajectories do not cover the training exposure')
        item = dict(row_id=row['row_id'], setting=row['setting'], mode=row['tracked_aux_mode'], model_seed=row['model_seed'],
                    current=current, ordinary_PL=pl, original_hybrid=old, trace=trace,
                    clean_delta_vs_PL=current['clean']-pl['clean'],
                    practical_delta_vs_PL=current['practical_mean']-pl['practical_mean'])
        rows.append(item);groups[row['setting'],row['tracked_aux_mode']].append(item)
    summary=[]
    for (setting,mode),values in sorted(groups.items()):
        summary.append(dict(setting=setting,mode=mode,seeds=[r['model_seed'] for r in values],
                            clean=sum(r['current']['clean'] for r in values)/len(values),
                            practical_mean=sum(r['current']['practical_mean'] for r in values)/len(values)))
    out=root/'analysis';out.mkdir(exist_ok=False)
    write(out/'results.json',dict(status='ANALYZED',rows=rows,summary=summary,
          interpretation='test-selected failing seeds from previously exposed test; exploratory, not population averages'))
    write(out/'complete.json',dict(status='ANALYZED',rows=len(rows),groups=len(summary),test_scope='clean and six practical'))
