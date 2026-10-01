"""Analyze the complete aggregate diagnostic readback; never access IQ/truth."""
import argparse
import csv
import json
from pathlib import Path


def analyze(readback,output):
    d=json.loads(Path(readback).read_text(encoding='utf-8'))
    c=d['completion'];r=d['resolved'];s=d['statistics']
    if not c or d['process'] or c['counts']!={'L_s':6300,'V':27000} or c['groups']!=180:
        raise ValueError('Incomplete/active full source diagnostic')
    if r['target_constructed'] or r['checkpoint_access'] or r['training'] or not r['source_roles_exact_match']:
        raise ValueError('Actual source-only contract mismatch')
    if s['counts']!=c['counts'] or len(s['groups'])!=180:
        raise ValueError('Aggregate completion mismatch')
    cells=s['groups'];keys={(g['role'],g['tx_index'],g['receiver'],g['day']) for g in cells}
    expected={(role,t,rx,day) for role in ['L_s','V'] for t in range(6) for rx in [1,3,4,6,8] for day in [1,2,3]}
    if keys!=expected or any(g['count']!=({'L_s':70,'V':300}[g['role']]) for g in cells):
        raise ValueError('Incomplete per-role TX/RX/day cells')
    if 'COMPLETE ' not in d['log_text'] or any(v in d['log_text'] for v in ['Traceback (most recent call last)','FloatingPointError','ValueError:']):
        raise ValueError('Terminal diagnostic log not clean')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def write(name,v):
        (output/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write('source_statistics.json',s)
    write('remote_completion.json',{k:d[k] for k in ['read_at','submit','process','resolved','completion','log_bytes','log_text']})
    columns=['role','tx_index','receiver','day','count']
    metrics=sorted(cells[0]['means'])
    with (output/'source_group_statistics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns+[p+k for k in metrics for p in ['mean_','population_sd_']]);writer.writeheader()
        for g in cells:
            writer.writerow(dict({k:g[k] for k in columns},**{'mean_'+k:g['means'][k] for k in metrics},**{'population_sd_'+k:g['population_sd'][k] for k in metrics}))
    marginals=[]
    for role in ['L_s','V']:
        for axis in ['receiver','tx_index','day']:
            for value in sorted({g[axis] for g in cells}):
                group=[g for g in cells if g['role']==role and g[axis]==value];n=sum(g['count'] for g in group)
                marginals.append(dict(role=role,axis=axis,value=value,count=n,means={k:sum(g['count']*g['means'][k] for g in group)/n for k in metrics}))
    write('source_marginals.json',marginals)
    window=s['summaries']['V']['w80_160_template_correlation']
    rx=[g['means']['w80_160_template_correlation'] for g in marginals if g['role']=='V' and g['axis']=='receiver']
    tx=[g['means']['w80_160_template_correlation'] for g in marginals if g['role']=='V' and g['axis']=='tx_index']
    verdict=dict(status='ANALYZED_SOURCE_ONLY',reference_correspondence='SUPPORTED_WITH_RX_DEPENDENT_RESIDUAL',
        template_correlation_V_mean=window['mean'],template_correlation_V_median=window['quantiles']['p50'],
        receiver_marginal_range=max(rx)-min(rx),tx_marginal_range=max(tx)-min(tx),
        identified_TX_hardware=False,performance_claim=False,test_result='N/A: no classifier',target_access=False,
        genuine_RFF_physics_goal_achieved=False,
        next_requirement='Separate TX/RX forward-model interventions; do not label equalized template residual as TX hardware.')
    write('iteration_verdict.json',verdict)
    print(json.dumps(verdict))
    return d,marginals,verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--readback',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();analyze(args.readback,args.output)
