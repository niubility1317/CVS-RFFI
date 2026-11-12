"""Sample-weighted matched comparisons on the declared seven test views."""
import csv
from collections import defaultdict
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import target_tuning_plan as plan


def pooled(records):
    scenes, counts = {}, {}
    for view in VIEWS:
        selected = [r for r in records if r['dimension'] == 'overall' and r['readout'] == 'raw'
                    and r['view'] == view and r['test'] in ('cross_rx', 'cross_both')]
        if len(selected) != 2 or {r['test'] for r in selected} != {'cross_rx', 'cross_both'}:
            raise ValueError('Both matched target populations required for every view')
        count = sum(r['count'] for r in selected)
        if not count:
            raise ValueError('Nonempty matched target populations required')
        scenes[view] = sum(round(r['count'] * r['accuracy']) for r in selected) / count
        counts[view] = count
    return dict(clean=scenes['clean'], practical_mean=sum(scenes[v] for v in VIEWS[1:]) / 6,
                scenes=scenes, counts=counts)


def load_score(folder):
    data = read(Path(folder) / 'scores.json')
    if data.get('truth_last') is not True or read(Path(folder) / 'complete.json')['status'] != 'SCORED_COMPLETE':
        raise ValueError('Independent completed score required')
    return pooled(data['results'])


def analyze(rt):
    root = Path(rt['run_root'])
    sources, controls = read(root / 'sources.json'), read(root / 'controls.json')
    records, groups = [], defaultdict(list)
    for row in plan.candidate_rows():
        current = load_score(sources[row['row_id']]['scores'])
        pl_key = f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"
        pl = load_score(controls[pl_key]['scores'])
        original = load_score(controls[plan.baseline_row(row)['row_id']]['scores'])
        if current['counts'] != pl['counts'] or current['counts'] != original['counts']:
            raise ValueError('Matched baseline and candidate sample counts differ')
        record = dict(row_id=row['row_id'], setting=row['setting'], model_seed=row['model_seed'],
                      current=current, ordinary_PL=pl, original_prototype=original,
                      clean_delta_vs_PL=current['clean'] - pl['clean'],
                      practical_delta_vs_PL=current['practical_mean'] - pl['practical_mean'])
        records.append(record)
        groups[row['row_id'].rsplit('-s', 1)[0]].append(record)
    summary = []
    for group, values in sorted(groups.items()):
        if len(values) != len(plan.SEEDS):
            raise ValueError('All three declared seeds required for the aggregate')
        summary.append(dict(group=group, setting=values[0]['setting'], seeds=len(values),
            clean=sum(v['current']['clean'] for v in values) / len(values),
            practical_mean=sum(v['current']['practical_mean'] for v in values) / len(values),
            clean_delta_vs_PL=sum(v['clean_delta_vs_PL'] for v in values) / len(values),
            practical_delta_vs_PL=sum(v['practical_delta_vs_PL'] for v in values) / len(values)))
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    write(out / 'results.json', dict(status='ANALYZED', rows=records, summary=summary,
          test_scope='direct clean and six practical', target_feedback_forbidden=True,
          interpretation='previously exposed target test; exploratory comparison'))
    with (out / 'summary.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write(out / 'complete.json', dict(status='ANALYZED', rows=len(records), groups=len(summary),
          target_feedback_forbidden=True, test_scope='direct clean and six practical'))
