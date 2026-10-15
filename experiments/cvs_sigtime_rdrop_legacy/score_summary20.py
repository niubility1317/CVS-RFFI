"""Summarize all nine completed E20 rows without changing the running matrix."""
from pathlib import Path
import sys,json
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_sigtime_rdrop_legacy import design as d
from experiments.cvs_sigtime_rdrop_legacy.publish import PYTHON
from experiments.cvs_phase1_stack.publish import ssh
P=W/'local_artifacts'/d.RELEASE
SCRIPT=r'''
import sys,types,json,statistics,os
from pathlib import Path
import numpy as np
c=CONFIG
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
original,fold,release=(Path(c[k]) for k in ('dependency','fold','release'))
sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
for suffix,root in (('cvs_sigtime_rdrop_legacy',release),('cvs_rdrop_fold20',release),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
 name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
from experiments.cvs_sigtime_rdrop_legacy import design as d
from experiments.cvs_sigtime_rdrop_legacy.evaluate import preflight
from experiments.cvs_sigtime_rdrop_legacy.provenance import source_proof
rows=[r for r in d.rows() if r['epochs']==20];queue=d.read(d.BASE/'queue_state.json');assert len(rows)==9 and not queue['failures'] and queue['source_gates']['20']['qualified']
proofs={};ids=None
for row in rows:
 config,physical=preflight(row['row_id']);proofs[row['row_id']]=source_proof(config)
 if ids is None:ids=physical
 else:assert np.array_equal(ids,physical)
records=[];correctness={};source=[]
for row in rows:
 rid=row['row_id'];config=d.config(row);root=d.BASE/rid;done=d.read(root/'completion.json');src=d.read(proofs[rid]/'completion.json');score=d.read(root/'scores/metrics.json')
 assert done['status']=='MODEL_PREDICTED_SCORED' and done['config']==config and done['count']==168000 and done['metric_records']==18 and done['truth_last'] and done['new_training_steps']==4440
 assert score['truth_last'] and score['independent_recount']=='VERIFIED' and len(score['records'])==18
 group=score['records'];overall=next(r for r in group if r['dimension']=='overall');overall_cm=np.asarray(overall['confusion'],dtype=np.int64)
 for r in group:
  cm=np.asarray(r['confusion'],dtype=np.int64);assert cm.shape==(6,6) and np.all(cm>=0) and cm.sum()==r['query_count']
  den=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
  assert r['row_id']==rid and r['arm']==row['arm'] and r['model_seed']==row['model_seed'] and r['readout']=='NATIVE_HEAD' and r['view']=='clean'
  assert abs(r['accuracy']-np.trace(cm)/cm.sum())<=1e-12 and abs(r['macro_f1']-f1)<=1e-12
 for dim,n in (('receiver',7),('transmitter',6),('day',4)):
  subgroup=[r for r in group if r['dimension']==dim];assert len(subgroup)==n and len({r['stratum'] for r in subgroup})==n
  assert np.array_equal(sum((np.asarray(r['confusion'],dtype=np.int64) for r in subgroup),np.zeros((6,6),np.int64)),overall_cm)
 with np.load(root/'scores/correctness.npz',allow_pickle=False) as v:
  assert np.array_equal(v['ids'],ids) and v['NATIVE_HEAD'].shape==(168000,) and v['NATIVE_HEAD'].dtype==bool
  ok=v['NATIVE_HEAD'].copy()
 assert int(ok.sum())==int(np.trace(overall_cm));correctness[rid]=ok;records+=group
 source.append(dict(row_id=rid,optimizer_steps=src['optimizer_steps'],epochs=src['epoch'],label_epochs=src['label_epochs'],pseudo_epochs=src['pseudo_epochs'],source_role_counts=dict(L_s=6300,U_s=56700,V=27000),initialization=src['initialization'],checkpoint_sources=src['checkpoint_sources'],ancestors=src['ancestors'],head_fit_updates=src['head_fit_updates'],final_L_CE=src['final_training_L_CE'],final_source_V=src['final_source_metrics']))
summary={};pairs=[];paired={}
for mode in d.MODES:
 arm='E20_'+mode;group=[r for r in records if r['arm']==arm and r['dimension']=='overall'];assert len(group)==3 and {r['model_seed'] for r in group}==set(d.SEEDS)
 details=[]
 for r in group:
  strata=[a for a in records if a['row_id']==r['row_id']]
  details.append(dict(model_seed=r['model_seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],worst_RX_accuracy=min(a['accuracy'] for a in strata if a['dimension']=='receiver'),worst_day_accuracy=min(a['accuracy'] for a in strata if a['dimension']=='day')))
 summary[arm]=dict(seed_count=3,seeds=list(d.SEEDS),per_seed=details,**{k+'_mean':statistics.mean(r[k] for r in details) for k in ('accuracy','macro_f1','worst_RX_accuracy','worst_day_accuracy')},accuracy_sample_SD=statistics.stdev(r['accuracy'] for r in details))
for after,before in (('E20_RDROP','E20_ORIGINAL'),('E20_RDROP','E20_TWO_CE'),('E20_TWO_CE','E20_ORIGINAL')):
 key=after+' minus '+before
 for seed in d.SEEDS:
  a,b=(correctness[f'{arm}-s{seed}'] for arm in (after,before));repair=int((~b&a).sum());broken=int((b&~a).sum());delta=(repair-broken)/1680
  ra=next(r for r in summary[after]['per_seed'] if r['model_seed']==seed);rb=next(r for r in summary[before]['per_seed'] if r['model_seed']==seed)
  assert abs(delta-100*(ra['accuracy']-rb['accuracy']))<=1e-10;pairs.append(dict(comparison=key,model_seed=seed,corrected_errors=repair,introduced_errors=broken,accuracy_difference_pp=delta))
 values=[r['accuracy_difference_pp'] for r in pairs if r['comparison']==key];paired[key]=dict(seed_count=3,mean_pp=statistics.mean(values),sample_SD_pp=statistics.stdev(values),positive_seeds=sum(v>0 for v in values))
assert sum(r['optimizer_steps'] for r in source)==39960 and sum(r['epochs'] for r in source)==180
print(json.dumps(dict(status='VERIFIED_SCORED_ARTIFACT_SUMMARY',run_id=d.RUN,budget_epochs=20,run_primary='E50_RDROP',run_status='RUNNING',execution_commit=(release/'release_commit.txt').read_text().strip(),source_proofs=source,summary=summary,paired_errors=pairs,paired_summary=paired,records=records,actual_models=9,actual_epochs=180,actual_optimizer_steps=39960,clean_decisions=1512000,metric_records=162,E20_RDROP_goal84_check=summary['E20_RDROP']['accuracy_mean']>=.84,goal_completed=False,IQ_reads=0,new_predictions=0,truth_reads=0,checkpoint_deserializations=0,optimizer_steps=0,head_fit_updates=0,limitation='All E20 seeds complete. Checked existing independent scorer artifacts, confusion arithmetic/partition sums and paired correctness. Fresh truth-side recount/full-log/checkpoint analysis remains scheduled for all18. E50 primary and fixedmatrix/source-only expansion remain unchanged.')))
'''
def main():
 if (P/'E20_scored_summary.json').is_file():print('ALREADY_SUMMARIZED');return
 v=d.read(P/'independent_readback.json');rows=[r for r in v['rows'] if r['row_id'].startswith('E20_')]
 if len(rows)!=9 or any(not r['completion'] or r['completion']['status']!='MODEL_PREDICTED_SCORED' or r['process'] for r in rows):print('E20_NOT_COMPLETE; no score reads');return
 assert not v['failure'] and not v['queue']['failures']
 config=dict(dependency=d.DEPENDENCY_RELEASE,fold=d.FOLD_RELEASE,release=d.PROJECT+'/releases/'+d.RELEASE);payload=SCRIPT.replace('CONFIG',repr(config));compile(payload,'complete-E20-existing-scores','exec')
 outer='import subprocess,os\nr=subprocess.run('+repr([PYTHON,'-c',payload])+',capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))\nif r.returncode:raise RuntimeError(r.stderr)\nprint(r.stdout)'
 x=json.loads(ssh(outer));d.write(P/'E20_scored_summary.json',x);print(json.dumps({k:x[k] for k in ('status','summary','paired_summary','E20_RDROP_goal84_check')},ensure_ascii=False))
if __name__=='__main__':main()
