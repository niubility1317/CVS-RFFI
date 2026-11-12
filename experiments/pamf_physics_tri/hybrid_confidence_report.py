"""Matched Hybrid and ordinary-PL comparisons, all seven direct test views."""
import csv
from collections import defaultdict
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import hybrid_confidence_plan as plan
from .target_tuning_report import load_score


def analyze(rt):
    root = Path(rt['run_root'])
    sources, controls = read(root / 'sources.json'), read(root / 'controls.json')
    rows, groups = [], defaultdict(list)
    for row in plan.candidate_rows():
        current = load_score(sources[row['row_id']]['scores'])
        pl = load_score(controls[f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]['scores'])
        hybrid = load_score(controls[plan.baseline_row(row)['row_id']]['scores'])
        if current['counts'] != pl['counts'] or current['counts'] != hybrid['counts']:
            raise ValueError('Candidate and matched baseline test populations differ')
        segments = [read(Path(sources[row['row_id']]['source_output']) / part / 'repl_state.json')
                    for part in ('segment30', 'segment50')]
        seen = sum(s['seen'] for s in segments)
        used = sum(s['U_confidence_only']['used'] for s in segments)
        accepted = sum(s['accepted'] for s in segments)
        if used != accepted:
            raise ValueError('Confidence-accepted U and auxiliary exposures differ')
        entry = dict(row_id=row['row_id'], setting=row['setting'], model_seed=row['model_seed'],
                     auxiliary_weight=row.get('auxiliary_weight', 1.),
                     current=current, ordinary_PL=pl, original_hybrid=hybrid,
                     coverage=dict(seen=seen, accepted=accepted, auxiliary_used=used),
                     clean_delta_vs_PL=current['clean'] - pl['clean'],
                     practical_delta_vs_PL=current['practical_mean'] - pl['practical_mean'],
                     clean_delta_vs_hybrid=current['clean'] - hybrid['clean'],
                     practical_delta_vs_hybrid=current['practical_mean'] - hybrid['practical_mean'])
        rows.append(entry)
        groups[(row['setting'], row.get('auxiliary_weight', 1.))].append(entry)
    summary = []
    for (setting, weight), values in sorted(groups.items()):
        if len(values) != len(plan.SEEDS):
            raise ValueError('Three declared seeds required per data configuration')
        n = len(values)
        average = lambda key: sum(v[key] for v in values) / n
        summary.append(dict(setting=setting, auxiliary_weight=weight, seeds=n,
            clean=sum(v['current']['clean'] for v in values) / n,
            practical_mean=sum(v['current']['practical_mean'] for v in values) / n,
            scenes={view: sum(v['current']['scenes'][view] for v in values) / n for view in VIEWS},
            clean_delta_vs_PL=average('clean_delta_vs_PL'), practical_delta_vs_PL=average('practical_delta_vs_PL'),
            clean_delta_vs_hybrid=average('clean_delta_vs_hybrid'),
            practical_delta_vs_hybrid=average('practical_delta_vs_hybrid'),
            coverage=sum(v['coverage']['auxiliary_used'] for v in values) / sum(v['coverage']['seen'] for v in values)))
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    write(out / 'results.json', dict(status='ANALYZED', rows=rows, summary=summary,
          test_scope='direct clean and six practical', interpretation='previously exposed target; exploratory'))
    with (out / 'summary.csv').open('x', encoding='utf-8', newline='') as f:
        fields = ['setting', 'auxiliary_weight', 'seeds', 'clean', 'practical_mean', *VIEWS[1:], 'coverage',
                  'clean_delta_vs_PL', 'practical_delta_vs_PL', 'clean_delta_vs_hybrid', 'practical_delta_vs_hybrid']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row['scenes'][key] if key in VIEWS[1:] else row[key] for key in fields})
    write(out / 'complete.json', dict(status='ANALYZED', rows=len(rows), groups=len(summary),
                                     test_scope='direct clean and six practical'))
