"""Recount fixed full-matrix decisions and matched errors; never train or launch."""
import argparse,json
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
REMOTE=r'''
from pathlib import Path
import sys,json,time,numpy as np,torch
root=Path(ROOT);sys.path[:0]=[str(root),str(root/'code')]
from experiments.cvs_time_basis50 import design as d
from experiments.cvs_time_basis50.predict import preflight
from experiments.cvs_time_factorial50.predict import preflight as compact_preflight
from experiments.cvs_report_ssl.evaluate import truth_arrays
from experiments.cvs_time_basis50.provenance import source_proof
from experiments.cvs_feature_restitution.features import layout
from experiments.cvs_time_basis50.model import build
torch.set_num_threads(2)
if not (d.BASE/'completion.json').is_file():raise ValueError('Registered full matrix is incomplete; no truth read')
completion=d.read(d.BASE/'completion.json');analysis=d.read(d.BASE/'analysis.json')
if completion['status']!='EXPERIMENT_COMPLETE' or completion['rows']!=6:raise ValueError('Registered full matrix is incomplete')
entries=[];ids0=None;preds={};metrics=[]
for row in d.rows():
 cfg,ids=preflight(row['row_id']);entries.append((row['arm'],row['model_seed'],d.BASE/row['row_id'],cfg,ids))
for seed in d.SEEDS:
 cfg=d.compact_config(seed);_,ids=compact_preflight(cfg['row_id']);entries.append(('COMPACT_LINEAR',seed,d.temporal.BASE/cfg['row_id'],cfg,ids))
# All nine original prediction sets have passed their own frozen provenance
# preflight before any independent truth read by this analysis process.
for arm,seed,p,cfg,ids in entries:
 if ids0 is None:ids0=ids
 if not np.array_equal(ids0,ids):raise ValueError('Matched physical IDs differ')
 with np.load(p/'prediction/predictions.npz',allow_pickle=False) as saved:
  if set(saved.files)!={'ids','NATIVE_HEAD'} or not np.array_equal(saved['ids'],ids):raise ValueError('Unexpected default predictions')
  preds[arm,seed]=saved['NATIVE_HEAD'].copy()
y,rx,day=truth_arrays(d.read(d.TRUTH),ids0)
for arm,seed,p,cfg,ids in entries:
 records=d.read(p/'scores/metrics.json')['records']
 if len(records)!=18:raise ValueError('Metric dimensions missing')
 for m in records:
  keep=np.ones(len(y),dtype=bool) if m['dimension']=='overall' else rx==m['stratum'] if m['dimension']=='receiver' else day==m['stratum'] if m['dimension']=='day' else y==list(d.CLASSES).index(m['stratum'])
  cm=np.bincount(y[keep]*6+preds[arm,seed][keep],minlength=36).reshape(6,6);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
  if cm.tolist()!=m['confusion'] or n!=m['query_count'] or float(cm.trace()/n)!=m['accuracy'] or abs(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()-m['macro_f1'])>1e-14:raise ValueError('Independent fixed decisions/CM recount differs')
 metrics.extend(records)
errors=[]
for control in ('BASE_REF','COMPACT_LINEAR'):
 for seed in d.SEEDS:
  a=preds['SINGLE_BASIS',seed]==y;b=preds[control,seed]==y
  errors.append(dict(treatment='SINGLE_BASIS',control=control,seed=seed,count=len(y),corrected_errors=int((a&~b).sum()),introduced_errors=int((~a&b).sum()),both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum())))
initial=[];updates=[]
for row in d.rows():
 if row['arm']=='BASE_REF':continue
 cfg=d.config(row);p=source_proof(cfg);diag=d.read(d.BASE/row['row_id']/'diagnostics/summary.json')
 if diag['head_fit_updates'] or not diag['model_params_buffers_unchanged'] or not all(diag['trained_export'].values()):raise ValueError('Real default feature boundary differs')
 if any(v['changed_values']<=0 for layer in diag['parameter_updates'].values() for v in layer.values()):raise ValueError('A used time parameter tensor did not update')
 states=[torch.load(Path(c['output_root'])/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for c in (cfg,d.baseline_config(row['model_seed']))]
 model=build(cfg,'cpu');core,prefix,_=layout(model)
 compared=0
 for name,value in states[0].items():
  if any(name.startswith(prefix+'.'+l+'.') for l in ('t1','t2','t3')):continue
  if name not in states[1] or not torch.equal(value,states[1][name]):raise ValueError('Actual non-time initial state differs')
  compared+=1
 initial.append(dict(seed=row['model_seed'],actual_non_time_initial_state_tensors_equal=compared,time_tensors_intentionally_different=True))
 stages=[d.read(p/f'source_stage_epoch{epoch}.json') for epoch in (32,33,34,40,50)]
 feedback=[json.loads(line) for line in (p/'source_mechanism_epochs.jsonl').read_text().splitlines()]
 if [r['epoch'] for r in feedback]!=list(range(1,51)):raise ValueError('Source U epochs missing')
 for r in feedback:
  if r['U_truth_read'] or r['gate_changed'] or r['head_fit_updates'] or sum(g['total'] for g in r['groups'].values())!=(56700 if r['epoch']>=34 else 0):raise ValueError('Source U permission/count differs')
  if any(sum(g['predicted_class'])!=g['total'] or sum(g['accepted_class'])!=g['accepted'] for g in r['groups'].values()):raise ValueError('U predicted class histogram differs')
 updates.append(dict(row_id=row['row_id'],parameter_updates=diag['parameter_updates'],kernels=diag['kernels'],export=diag['trained_export'],source_L_geometry=diag['geometry'],source_stages=stages,source_U_feedback=feedback,U_truth_read=False))
print(json.dumps(dict(status='VERIFIED',run_id=d.RUN,execution_commit=(root/'release_commit.txt').read_text().strip(),registered_rows=6,external_completed_controls=3,new_prediction_count=504000,reused_baseline_count=504000,external_compact_prediction_count=504000,total_recounted_decisions=1512000,independent_confusions_recounted=len(metrics),all_prediction_sets_fixed_before_truth=True,target_IQ_read=False,training_updates=0,head_fit_updates=0,feedback_to_training=False,error_transitions=errors,actual_initial_matches=initial,trained_feature_evidence=updates,read_at=time.time())))
'''
def collect(output):
 from experiments.cvs_time_basis50 import design as d
 payload=REMOTE.replace('ROOT',repr(d.PROJECT+'/releases/'+d.RELEASE))
 wrapper="import subprocess,os\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)"
 value=json.loads(ssh(wrapper));output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:value[k] for k in ('status','registered_rows','external_completed_controls','total_recounted_decisions','independent_confusions_recounted','head_fit_updates')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.output)
