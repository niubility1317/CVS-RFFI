"""A/B/C/D matched source-U and raw/DURA-LU exploratory results."""
from collections import defaultdict
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import hybrid_structure_plan as plan
from .hybrid_tracked_report import summarize_trace
from .target_tuning_report import pooled


def scores(folder,route):
    folder=Path(folder);data=read(folder/'scores.json')
    if data.get('truth_last') is not True or read(folder/'complete.json')['status']!='SCORED_COMPLETE':
        raise ValueError('Independent finished score required')
    rows=[dict(r,readout='raw') for r in data['results'] if r['readout']==route]
    return pooled(rows) if rows else None


def analyze(rt):
    root=Path(rt['run_root']);sources=read(root/'sources.json');controls=read(root/'controls.json');rows=[]
    for row in plan.candidate_rows():
        source=sources[row['row_id']];base=Path(source['source_output'])
        states=[read(base/part/'repl_state.json') for part in ('segment30','segment50')]
        trace=summarize_trace([base/part/'U_trace.jsonl.gz' for part in ('segment30','segment50')])
        if trace['exposures']!=sum(s['seen'] for s in states) or trace['accepted']!=sum(s['accepted'] for s in states):
            raise ValueError('Packet trace and source training counters differ')
        current={route:scores(source['scores'],route) for route in ('raw','DURA-LU')}
        if any(v is None for v in current.values()):raise ValueError('Both frozen readouts required')
        refs={name:{route:scores(folder,route) for route in ('raw','DURA-LU')} for name,folder in (
            ('ordinary_PL',controls[f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]['scores']),
            ('original_hybrid',controls[plan.baseline_row(row)['row_id']]['scores']))}
        rows.append(dict(row_id=row['row_id'],setting=row['setting'],arm=row['structure_arm'],model_seed=row['model_seed'],
                         current=current,controls=refs,trace=trace))
    groups=defaultdict(list)
    for r in rows:groups[r['setting'],r['model_seed']].append(r)
    paired=[]
    for (setting,seed),vs in sorted(groups.items()):
        arms={x['arm']:x for x in vs}
        if set(arms)!={'A','B','C','D'}:raise ValueError('All four same-seed structural conditions required')
        for route in ('raw','DURA-LU'):
            a=arms['A']['current'][route]
            for arm in ('B','C','D'):
                b=arms[arm]['current'][route]
                if b['counts']!=a['counts']:raise ValueError('Paired populations differ')
                paired.append(dict(setting=setting,seed=seed,route=route,arm=arm,
                                   clean_delta=b['clean']-a['clean'],practical_delta=b['practical_mean']-a['practical_mean'],
                                   scene_deltas={k:b['scenes'][k]-a['scenes'][k] for k in b['scenes']}))
    out=root/'analysis';out.mkdir(exist_ok=False)
    paired_net={f'{setting}-s{seed}':str(root/'paired'/f'{setting}-paired-s{seed}'/'paired_net.json')
                for setting,_,seeds in plan.MATRIX for seed in seeds}
    if any(not Path(path).is_file() for path in paired_net.values()):raise ValueError('Independent paired diagnostics required')
    write(out/'results.json',dict(status='ANALYZED',rows=rows,matched_to_A=paired,paired_net=paired_net,
          interpretation='selected seeds and repeatedly exposed tests; exploratory, not full-seed average'))
    write(out/'complete.json',dict(status='ANALYZED',rows=len(rows),seed_groups=len(groups),
                                  test_scope='raw and DURA-LU clean and six practical'))
