"""Read-only independent complete test recount after all304 rows are scored."""
import argparse,json,statistics,math
from pathlib import Path
from experiments.cvs_all_frozen_clean_backfill.publish import inspect,ssh
from experiments.cvs_all_frozen_clean_backfill.contracts import RUN,RELEASE,PROJECT,ROOT,read

RECOUNT=r'''
import json,numpy as np
from pathlib import Path
project=Path(PROJECT);root=project/'runs'/RUN
def read(p):return json.loads(p.read_text())
marker=read(root/'scoring_clean_complete.json');state=read(root/'pipeline_state.json')
if marker['status']!='SCORED_COMPLETE' or marker['rows']!=304 or state['status']!='ANALYZED':raise ValueError('Incomplete backfill')
spec=read(project/'releases'/RELEASE/'experiments/cvs_all_frozen_clean_backfill/configs/launch_spec.json')
with np.load(project/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
if len(ids)!=168000 or len(set(ids.tolist()))!=168000:raise ValueError('Physical IDs differ')
truth=read(Path(spec['p1_truth']));t=[truth[i] for i in ids.tolist()]
y=np.array([v['label'] for v in t]);rx=np.array([str(v['receiver']) for v in t])
stored=read(root/'clean_scored_results.json');lookup={(r['row_id'],r['group']):r for r in stored['results']}
class_lookup={(r['row_id'],r['group']):r for r in stored['class_results']}
groups=[('ALL',np.ones(len(y),dtype=bool))]+[('RX'+v,rx==v) for v in sorted(set(rx))]
count=0;class_count=0
for row in spec['rows']:
    out=Path(row['output_root']);done=read(out/'clean_complete.json')
    if done['status']!='PREDICTIONS_COMPLETE' or done['count']!=168000 or done['truth_read'] or done['query_fit']:raise ValueError('Prediction status differs')
    with np.load(out/'clean_predictions.npz',allow_pickle=False) as f:
        if set(f.files)!={'ids','clean'} or not np.array_equal(ids,f['ids']):raise ValueError('Prediction IDs differ')
        p=f['clean'].copy()
    if p.shape!=y.shape or p.dtype.kind not in 'iu' or p.min()<0 or p.max()>5:raise ValueError('Bad predictions')
    for group,mask in groups:
        cm=np.bincount(6*y[mask]+p[mask],minlength=36).reshape(6,6);a=lookup[(row['row_id'],group)]
        if a['confusion']!=cm.tolist() or a['query_count']!=int(mask.sum()):raise ValueError('Confusion/count differs')
        diag=cm.diagonal();den=cm.sum(0)+cm.sum(1)
        f1=float(np.divide(2*diag,den,out=np.zeros(6,dtype=float),where=den!=0).mean())
        recall=float(np.divide(diag,cm.sum(1),out=np.zeros(6,dtype=float),where=cm.sum(1)!=0).mean())
        for k,v in [('accuracy',float(diag.sum()/cm.sum())),('macro_f1',f1),('macro_accuracy',recall)]:
            if abs(a[k]-v)>1e-12:raise ValueError('Independent metric differs:'+k)
        count+=1
    for tx in range(6):
        mask=y==tx;a=class_lookup[(row['row_id'],'TX'+str(tx))]
        if a['count']!=int(mask.sum()) or a['accuracy']!=float((p[mask]==y[mask]).mean()):raise ValueError('TX metric differs')
        class_count+=1
if count!=2432 or class_count!=1824 or len(lookup)!=count or len(class_lookup)!=class_count:raise ValueError('Incomplete scored matrix')
print(json.dumps(dict(status='VERIFIED',rows=304,query_count=168000,decisions=51072000,receiver_records=count,class_records=class_count,new=216,reused=88)))
'''

def collect(output):
    inspect(output)
    live=read(output/'readback.json')
    if not live['pipeline'] or live['pipeline']['status']!='ANALYZED':return False
    if live['dispatcher_process'] or any(r['process'] for r in live['rows']):raise ValueError('Workers not independently terminal')
    script=RECOUNT.replace('PROJECT',repr(PROJECT)).replace('RUN',repr(RUN)).replace('RELEASE',repr(RELEASE))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    checked=json.loads(ssh(wrapper))
    results=live['results']['results'];summary=live['summary']['summary']
    for row in summary:
        records=[r for r in results if r['model_id']==row['model_id'] and r['group']==row['group']]
        if len(records)!=4:raise ValueError('Incomplete four-seed summary')
        for key in ('accuracy','macro_f1','macro_accuracy'):
            values=[r[key] for r in records]
            if not math.isclose(row[key+'_mean'],statistics.mean(values),abs_tol=1e-12) or not math.isclose(row[key+'_seed_sd'],statistics.stdev(values),abs_tol=1e-12):raise ValueError('Seed summary differs')
    evidence=ROOT/'automation_reports/CV-SincNet'/RUN/'evidence'
    for name,value in [('final_readback.json',live),('independent_test_recount.json',checked),('test_summary.json',live['summary']),('test_results.json',live['results'])]:
        with (evidence/name).open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checked));return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
