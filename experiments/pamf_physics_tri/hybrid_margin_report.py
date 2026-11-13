"""Paired margin results and source-U diagnostics after all predictions freeze."""
from pathlib import Path
import collections
import gzip
import json
from experiments.cric_dura_comparison.io import read, write
from . import hybrid_margin_plan as plan
from .hybrid_structure_report import scores
from .hybrid_tracked_report import summarize_trace


def summarize_margin(paths):
    by_epoch = collections.defaultdict(lambda: collections.Counter())
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            for line in stream:
                batch = json.loads(line)
                stats = by_epoch[batch['epoch']]
                for accepted, violated, amount, delta in zip(batch['accepted'], batch['margin_violating'],
                                                              batch['margin_violation'], batch['physical_delta_l2']):
                    if accepted:
                        stats['accepted'] += 1
                        stats['violating'] += int(violated)
                        stats['violation_amount_sum'] += amount
                        stats['actual_nonzero_physical'] += int(delta > 0)
                        stats['physical_delta_l2_sum'] += delta
    return {str(epoch): dict(values) for epoch, values in sorted(by_epoch.items())}


def analyze(rt):
    root = Path(rt['run_root'])
    sources = read(root/'sources.json')
    controls = read(root/'controls.json')
    prior = Path(rt['project'])/'runs'/'20261113-phase1-pamf-hybrid-structure-abcd-m16-r01'
    rows = []
    for row in plan.candidate_rows():
        rid = row['row_id']
        source = sources[rid]
        base = Path(source['source_output'])
        states = [read(base/seg/'repl_state.json') for seg in ('segment30', 'segment50')]
        trace = summarize_trace([base/seg/'U_trace.jsonl.gz' for seg in ('segment30', 'segment50')])
        margin_trace = summarize_margin([base/seg/'U_trace.jsonl.gz' for seg in ('segment30', 'segment50')])
        if trace['exposures'] != sum(s['seen'] for s in states) or trace['accepted'] != sum(s['accepted'] for s in states):
            raise ValueError('Source U packet trace differs from training counters')
        c_row = f"{row['setting']}-hybrid-structure-C-s{row['model_seed']}"
        fusion = {r: scores(prior/'readouts'/c_row/'scores', r) for r in ('raw', 'DURA-LU')}
        current = {r: scores(source['scores'], r) for r in ('raw', 'DURA-LU')}
        refs = {name: {route: scores(folder, route) for route in ('raw', 'DURA-LU')}
                for name, folder in [('ordinary_PL', controls[f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]['scores']),
                                     ('original_hybrid', controls[plan.baseline_row(row)['row_id']]['scores'])]}
        delta = {}
        for route in ('raw', 'DURA-LU'):
            a, b = fusion[route], current[route]
            if a['counts'] != b['counts']:
                raise ValueError('Margin and fusion-only test populations differ')
            delta[route] = dict(clean=b['clean']-a['clean'], practical_mean=b['practical_mean']-a['practical_mean'],
                                scenes={k: b['scenes'][k]-a['scenes'][k] for k in b['scenes']})
        rows.append(dict(row_id=rid, setting=row['setting'], model_seed=row['model_seed'], current=current,
            fusion_only=fusion, delta_vs_fusion_only=delta, controls=refs, trace=trace,
            source_cap=read(base/'margin_cap.json'), margin_segments=[s['U_margin'] for s in states],
            margin_trace_by_epoch=margin_trace,
            paired_net=str(root/'paired'/rid/'paired_net.json')))
    out = root/'analysis'
    out.mkdir(exist_ok=False)
    write(out/'results.json', dict(status='ANALYZED', rows=rows,
        interpretation='selected seeds and previously exposed tests; exploratory'))
    write(out/'complete.json', dict(status='ANALYZED', rows=len(rows),
          test_scope='raw and DURA-LU clean plus six practical'))
