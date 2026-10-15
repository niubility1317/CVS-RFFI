"""Retrieve this completed source probe and recount all source decisions."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import numpy as np
from . import probe_design as d
from experiments.cvs_urban_sigtime.operations import remote,SSH


def harvest(output):
    root=d.PROJECT+'/runs/'+d.RUN;release=d.PROJECT+'/releases/'+d.RELEASE
    status=remote("import json\nfrom pathlib import Path\np=Path("+repr(root)+")\nread=lambda n:json.loads((p/n).read_text()) if (p/n).exists() else None\nprint(json.dumps(dict(complete=read('complete.json'),failure=read('failure.json'),progress=read('progress.json'))))")
    if status['complete'] is None:
        print(json.dumps(dict(status='FAILED' if status['failure'] else 'RUNNING',remote=status)));return
    if status['complete']['status']!='ANALYZED' or status['complete']['forward_packets']!=d.FORWARDS:
        raise ValueError('Completed source probe differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runfiles=['complete.json','results.json','resolved_config.json','source_smoke.json','finite_bank.json','sample_decisions.jsonl.gz']
    releasefiles=['preflight.json','submit.json','source_probe.log']
    for base,names in ((root,runfiles),(release,releasefiles)):
        subprocess.run(['scp',*SSH,*['N607:'+base+'/'+name for name in names],str(output)],check=True,timeout=60)
    results=d.read(output/'results.json');d.validate(results['config'])
    if (results['optimizer_updates']!=0 or results['EMA_updates']!=0 or results['head_fit_updates']!=0
        or not results['model_state_unchanged'] or not results['random_state_unchanged'] or results['target_access']):
        raise ValueError('Frozen/source-only contract differs')
    rows={k:[] for k in d.CONDITIONS};seen={k:set() for k in d.CONDITIONS}
    with gzip.open(output/'sample_decisions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            v=json.loads(line);name=v['condition'];pid=v['physical_id']
            if name not in rows or pid in seen[name] or bool(v['correct'])!=(v['TX']==v['prediction']):raise ValueError('Source decisions inconsistent')
            seen[name].add(pid);rows[name].append(v)
    ids=seen['RAW'];raw={v['physical_id']:v for v in rows['RAW']}
    table=[]
    for name,values in rows.items():
        if len(values)!=27000 or seen[name]!=ids:raise ValueError('Source physical counts differ')
        cm=np.zeros((6,6),dtype=int);correct=repairs=new_errors=0;total_margin=0.
        for v in values:
            if any(v[k]!=raw[v['physical_id']][k] for k in ('RX','day','TX')):raise ValueError('Source roles differ across views')
            cm[v['TX'],v['prediction']]+=1;correct+=int(v['correct']);total_margin+=v['margin']
            before=raw[v['physical_id']]['correct'];repairs+=int(v['correct'] and not before);new_errors+=int(not v['correct'] and before)
        saved=results['conditions'][name]['overall']
        if (correct!=saved['correct'] or repairs!=saved['repairs'] or new_errors!=saved['new_errors']
            or cm.tolist()!=saved['confusion'] or abs(total_margin/27000-saved['margin'])>1e-6):raise ValueError('Source independent recount differs')
        table.append(dict(condition=name,count=27000,correct=correct,accuracy_pct=100*correct/27000,
            macro_F1_pct=100*saved['macro_f1'],margin_scale30=saved['margin'],retention_pct=100*saved['clean_correct_retention'],
            repairs=repairs,new_errors=new_errors,**results['conditions'][name]['waveform']))
    if d.gate(results['conditions'])!=results['source_gate']:raise ValueError('Predeclared source decision differs')
    for row in results['coverage']:
        if row['count']!=300:raise ValueError('Source session/TX support differs')
    with (output/'source_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    labels=[v['condition'].replace('_','\n') for v in table]
    colors=['gray']+['#377eb8' if v['condition'].startswith('EMP') else '#ff7f00' for v in table[1:]]
    delta=[v['accuracy_pct']-table[0]['accuracy_pct'] for v in table]
    axes[0].bar(range(7),delta,color=colors);axes[0].axhline(-1,color='red',ls='--',lw=1)
    axes[0].set(xticks=range(7),xticklabels=labels,ylabel='Accuracy change vs RAW (pp)',title='Frozen source default head')
    axes[0].tick_params(axis='x',labelsize=7)
    axes[1].bar(range(7),[v['new_errors'] for v in table],color=colors)
    axes[1].bar(range(7),[-v['repairs'] for v in table],color=colors,alpha=.4)
    axes[1].set(xticks=range(7),xticklabels=labels,ylabel='New errors (+) / repairs (-)',title='Paired source errors');axes[1].tick_params(axis='x',labelsize=7)
    rx=[1,3,4,6,8]
    for name in ('RAW','EMP_E20_REFLECT','ROT_E20_REFLECT'):
        lookup={v['value']:100*v['accuracy'] for v in results['conditions'][name]['strata'] if v['dimension']=='RX'}
        axes[2].plot(rx,[lookup[v] for v in rx],marker='o',label=name)
    axes[2].set(xlabel='Source RX',ylabel='Accuracy (%)',title='Source RX strata',xticks=rx);axes[2].legend(fontsize=8)
    fig.suptitle('Source safety only: no new training, no target predictions',fontsize=12)
    fig.savefig(output/'frozen_source_response.png',dpi=180);plt.close(fig)
    value=dict(status='VERIFIED_SOURCE_ANALYZED',source_gate=results['source_gate'],conditions=table,
        recounted_records=sum(len(v) for v in rows.values()),forward_packets=results['forward_packets'],seconds=results['seconds'],
        parent_cost=results['parent_cost'],optimizer_updates=0,head_fit_updates=0,new_models=0,target_access=False,commit=results['commit'])
    d.write(output/'analysis_summary.json',value)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();harvest(a.output)
