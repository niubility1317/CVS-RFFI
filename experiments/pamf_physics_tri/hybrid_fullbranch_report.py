"""Same-seed fusion-plus-branch versus fusion-only comparisons."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import hybrid_fullbranch_plan as plan
from .hybrid_structure_report import scores
from .hybrid_tracked_report import summarize_trace


def analyze(rt):
    root=Path(rt['run_root']);sources=read(root/'sources.json');controls=read(root/'controls.json')
    prior=Path(rt['project'])/'runs'/'20261113-phase1-pamf-hybrid-structure-abcd-m16-r01';rows=[]
    for row in plan.candidate_rows():
        rid=row['row_id'];source=sources[rid];base=Path(source['source_output'])
        states=[read(base/seg/'repl_state.json') for seg in ('segment30','segment50')]
        trace=summarize_trace([base/seg/'U_trace.jsonl.gz' for seg in ('segment30','segment50')])
        if trace['exposures']!=sum(s['seen'] for s in states) or trace['accepted']!=sum(s['accepted'] for s in states):
            raise ValueError('Source U packet trace differs from training counters')
        c_row=f"{row['setting']}-hybrid-structure-C-s{row['model_seed']}"
        c_scores=prior/'readouts'/c_row/'scores'
        current={r:scores(source['scores'],r) for r in ('raw','DURA-LU')}
        fusion={r:scores(c_scores,r) for r in ('raw','DURA-LU')}
        refs={name:{route:scores(folder,route) for route in ('raw','DURA-LU')} for name,folder in (
            ('ordinary_PL',controls[f"{row['protocol_id']}-ordinary_PL-s{row['model_seed']}"]['scores']),
            ('original_hybrid',controls[plan.baseline_row(row)['row_id']]['scores']))}
        deltas={}
        for route in ('raw','DURA-LU'):
            a,b=fusion[route],current[route]
            if a['counts']!=b['counts']:raise ValueError('Fusion-only and full-branch populations differ')
            deltas[route]=dict(clean=b['clean']-a['clean'],practical_mean=b['practical_mean']-a['practical_mean'],
                               scenes={k:b['scenes'][k]-a['scenes'][k] for k in b['scenes']})
        rows.append(dict(row_id=rid,setting=row['setting'],model_seed=row['model_seed'],current=current,
                         fusion_only=fusion,delta_vs_fusion_only=deltas,controls=refs,trace=trace,
                         paired_net=str(root/'paired'/rid/'paired_net.json')))
    out=root/'analysis';out.mkdir(exist_ok=False)
    write(out/'results.json',dict(status='ANALYZED',rows=rows,
        interpretation='selected seeds and previously exposed tests; exploratory, not full-seed averages'))
    write(out/'complete.json',dict(status='ANALYZED',rows=len(rows),test_scope='raw and DURA-LU clean plus six practical'))
