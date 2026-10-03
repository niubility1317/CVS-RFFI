"""Model-free independent source tensor recount after all workers are terminal."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.cvs_relation_attribution.publish import inspect,ssh,RUN,PROJECT

REMOTE=r'''
from pathlib import Path
import json,numpy as np
exec(STATISTICS,globals())
root=Path(ROOT);state=json.loads((root/'pipeline_state.json').read_text())
if state['status']!='COMPLETE' or len(state['rows'])!=12:raise ValueError('Incomplete source attribution')
rows=[];metadata=None
for rid,row in state['rows'].items():
    out=Path(row['output_root']);done=json.loads((out/'completion.json').read_text());resolved=json.loads((out/'resolved_config.json').read_text())
    if done['status']!='SOURCE_GEOMETRY_COMPLETE' or done['target_access'] is not False or done['optimizer_updates']!=0 or done['source_selection_changed'] is not False or done['model_unchanged'] is not True:raise ValueError('Invalid frozen diagnostic')
    with np.load(out/'source_geometry.npz',allow_pickle=False) as value:q={k:value[k] for k in value.files}
    contract=json.loads((out/'source_loader/source_contract.json').read_text())
    if set(q['ids'].tolist())!=set(contract['role_ids']['V']):raise ValueError('Physical V differs')
    current={str(i):(int(t),int(r),int(d)) for i,t,r,d in zip(q['ids'],q['truth'],q['receiver'],q['day'])}
    if metadata is not None and current!=metadata:raise ValueError('Cross-model paired source metadata differs')
    metadata=current
    result=analyze_arrays(q)
    for name in ('all_on','auxiliary_off'):
        stored=next(r for r in done['rows'] if r['condition']==name)
        actual=next(r for r in result['groups'] if r['group']=='ALL')[name]
        byrx={r['group'][2:]:r[name]['accuracy'] for r in result['groups'] if r['group'].startswith('RX')}
        if actual['accuracy']!=stored['accuracy'] or byrx!=stored['rx_accuracy']:raise ValueError('Stored and independent prediction recount differ')
    all_on=result['groups'][0]['all_on']
    if all_on['accuracy']!=resolved['source_record']['accuracy']:raise ValueError('E200 source reproduction differs')
    rows.append(dict(row_id=rid,variant=done['variant'],seed=done['model_seed'],analysis=result))
if {(r['variant'],r['seed']) for r in rows}!={(v,s) for v in ('relation_frequency_energy','mirror_energy','mirror_subspace') for s in range(2026092701,2026092705)}:raise ValueError('Incomplete diagnostic matrix')
print(json.dumps(dict(status='VERIFIED',models=12,source_packets_per_model=27000,decisions=648000,source_cells=1080,rows=rows,target_access=False,model_fitting=False,paired_source_metadata=True),allow_nan=False))
'''


def collect(root,output):
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=12 or any(r['process'] or r['state']['status']!='COMPLETE' for r in live['rows']):raise ValueError('Workers not independently terminal')
    e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    names=['final_readback.json','source_attribution.json','per_seed.csv','summary.json']
    if any((e/name).exists() for name in names):raise FileExistsError('Preserve existing attribution evidence')
    script=REMOTE.replace('STATISTICS',repr(Path(__file__).with_name('statistics.py').read_text(encoding='utf-8'))).replace('ROOT',repr(PROJECT+'/runs/'+RUN))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));flat=[]
    for row in data['rows']:
        a=row['analysis'];v=a['groups'][0]
        r=dict(variant=row['variant'],seed=row['seed'],accuracy_on=v['all_on']['accuracy'],accuracy_off=v['auxiliary_off']['accuracy'],
            CE_on=v['all_on']['CE'],CE_off=v['auxiliary_off']['CE'],margin_on=v['all_on']['margin'],margin_off=v['auxiliary_off']['margin'],
            helped=v['helped'],hurt=v['hurt'],prediction_changes=a['prediction_changes'])
        r.update({f+'_'+k:value for f,g in a['geometry'].items() for k,value in g.items() if k!='total'})
        flat.append(r)
    summary=[]
    for variant in ('relation_frequency_energy','mirror_energy','mirror_subspace'):
        rows=[r for r in flat if r['variant']==variant]
        metrics={}
        for k in rows[0]:
            if k in ('variant','seed'):continue
            values=[r[k] for r in rows if r[k] is not None]
            metrics[k]=dict(mean=statistics.mean(values) if values else None,
                SD=statistics.stdev(values) if len(values)>1 else None,valid_seeds=len(values))
        summary.append(dict(variant=variant,**metrics))
    for name,value in [('final_readback.json',live),('source_attribution.json',data),('summary.json',summary)]:
        with (e/name).open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (e/'per_seed.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status='VERIFIED',decisions=648000,summary=summary),ensure_ascii=False));return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
