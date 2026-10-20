"""Read-only final audit of completed-run state dictionaries; no model execution, IQ or truth.

This is a post-run delivery helper, not a new launch/permission gate. The
immutable experiment release remains identified by its original code commit.
"""
import argparse,json,math,os,subprocess
from pathlib import Path
from statistics import mean,stdev
from . import design as d
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
KEYS=('accuracy_pp','macro_f1_pp','weak_TX_mean_pp','other_TX_mean_pp','worst_RX_pp','worst_day_pp')
def near(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def metric_check(m):
 cm=m['confusion'];assert len(cm)==6 and all(len(r)==6 and all(isinstance(v,int) and v>=0 for v in r) and sum(r)==28000 for r in cm)
 assert m['count']==168000 and near(m['accuracy'],sum(cm[i][i] for i in range(6))/168000)
 recall=[cm[i][i]/28000 for i in range(6)];macro=sum(2*cm[i][i]/(28000+sum(cm[j][i] for j in range(6))) for i in range(6))/6
 assert all(near(a,b) for a,b in zip(m['per_TX_recall'],recall)) and near(m['macro_f1'],macro)
 assert len(m['receiver_accuracy'])==7 and len(m['day_accuracy'])==4 and near(m['worst_RX'],min(m['receiver_accuracy'].values())) and near(m['worst_day'],min(m['day_accuracy'].values()))
def difference(cur,old):
 per=[100*(a-b) for a,b in zip(cur['per_TX_recall'],old['per_TX_recall'])]
 return dict(accuracy_pp=100*(cur['accuracy']-old['accuracy']),macro_f1_pp=100*(cur['macro_f1']-old['macro_f1']),per_TX_pp=per,weak_TX_mean_pp=(per[1]+per[3])/2,other_TX_mean_pp=sum(per[i] for i in (0,2,4,5))/4,worst_RX_pp=100*(cur['worst_RX']-old['worst_RX']),worst_day_pp=100*(cur['worst_day']-old['worst_day']))
def stats_check(actual,values):
 assert actual['n']==3 and near(actual['mean'],mean(values)) and near(actual['sample_SD'],stdev(values)) and actual['positive_count']==sum(v>0 for v in values)
def check(base,reference,commit):
 base=Path(base).resolve();reference=Path(reference).resolve()
 if base.name!=d.RUN or reference.name!='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01':raise ValueError('Fixed run/reference only')
 progress=d.read(base/'progress.json')
 if progress['status']!='ARTIFACTS_COMPLETE' or progress['active']:return dict(status='WAIT_FIXED_MATRIX_RUNNING',checkpoint_loads=0,IQ_reads=0,truth_reads=0,output_written=False)
 matrix=d.read(base/'frozen_matrix.json');decision=d.read(base/'source_expansion.json');summary=d.read(base/'analysis_report/results_summary.json');result=d.read(base/'result.json');execution=d.read(base/'execution.json')
 assert matrix==dict(rows=d.rows(),user_fixed_extension=d.fixed_authorization(),target_feedback=False) and decision==d.fixed_authorization() and not decision['source_gate_applied']
 assert execution['commit']==commit and execution['test_views']==['clean'] and not execution['satellite_training'] and result['run_id']==d.RUN and result['truth_last'] and not result['target_feedback'] and result['all_three_seeds_required']
 assert summary['result']==result and [dict(row_id=r['row_id'],treatment=r['treatment'],model_seed=r['model_seed']) for r in result['records']]==d.rows()
 script="[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
 probe=subprocess.run([os.environ.get('SystemRoot','C:/Windows')+'/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',script],capture_output=True,text=True,encoding='utf8',check=True,timeout=15);live=json.loads(probe.stdout) or [];live=live if isinstance(live,list) else [live]
 owned=[v for v in live if d.RUN in (v.get('CommandLine') or '') and any(' -m experiments.cvs_f_centergrad50.'+role+' ' in (v.get('CommandLine') or '') for role in ('source','predict','score','driver'))]
 if owned:return dict(status='WAIT_TERMINAL_PROCESS_EXIT',live_owned_workers=len(owned),checkpoint_loads=0,IQ_reads=0,truth_reads=0,output_written=False)
 import numpy as np
 import torch
 from .provenance import source_proof,validate_checkpoint
 torch.set_num_threads(2);rows=[];expected_pairs=[]
 for row in d.rows():
  name=row['row_id'];root=base/name;src=d.read(root/'source/local_source_complete.json');c=src['config'];source_proof(c);cp=torch.load(root/'source/final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,c);initial=torch.load(root/'source/initial_smoke.pt',map_location='cpu',weights_only=False)
  assert tuple(cp['model'][W].shape)==(6,160) and torch.isfinite(cp['model'][W]).all() and not torch.equal(initial['model'][W],cp['model'][W])
  feature_keys=[k for k in cp['model'] if k.startswith('id_backbone.encoder.core.id_backbone.') and '.cls_head.' not in k and (k.endswith('.weight') or k.endswith('.bias') or k.endswith('_coordinate'))]
  changed=sum(not torch.equal(initial['model'][k],cp['model'][k]) for k in feature_keys);assert changed>0 and all(torch.isfinite(cp['model'][k]).all() for k in feature_keys)
  del cp,initial
  pred=d.read(root/'prediction/complete.json');score=d.read(root/'scores/metrics.json');ref=d.read(reference/('BASE-s'+str(row['model_seed']))/'scores/metrics.json')
  assert src['source_commit']==pred['commit']==score['prediction_commit']==commit and pred['config']==score['config']==c
  assert pred['status']=='PREDICTIONS_COMPLETE' and pred['count']==168000 and pred['views']==['clean'] and not any(pred[k] for k in ('truth_read','query_fit','head_fit_updates'))
  for v in (score,ref):assert v['status']=='MODEL_PREDICTED_SCORED' and v['count']==168000 and v['views']==['clean'] and v['truth_last'] and v['independent_CPU_score'] and not v['head_fit_updates'];metric_check(v['records']['clean'])
  assert ref['config']['model_seed']==row['model_seed'] and ref['config']['treatment']=='BASE' and ref['prediction_commit']=='31467086c3ced30237f3ac86e91eae68645425d7'
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as p,np.load(root/'scores/correctness.npz',allow_pickle=False) as q:
   assert set(p.files)==set(q.files)=={'ids','clean'} and p['ids'].shape==(168000,) and len(set(p['ids'].tolist()))==168000 and np.array_equal(p['ids'],q['ids']) and p['clean'].shape==q['clean'].shape==(168000,) and q['clean'].dtype==np.bool_ and near(float(q['clean'].mean()),score['records']['clean']['accuracy'])
  for role in ('source','predict','score'):
   start=d.read(base/(name+'-'+role+'-start.json'));exit=d.read(base/(name+'-'+role+'-exit.json'));assert exit['returncode']==0 and exit['pid']==start['pid']
  assert (base/'frozen_matrix.json').stat().st_mtime_ns<=(base/(name+'-predict-start.json')).stat().st_mtime_ns and (base/(name+'-source-exit.json')).stat().st_mtime_ns<=(base/(name+'-predict-start.json')).stat().st_mtime_ns and (root/'prediction/complete.json').stat().st_mtime_ns<=(base/(name+'-score-start.json')).stat().st_mtime_ns
  actual=next(r for r in result['records'] if r['row_id']==name);assert actual['current']==score['records']['clean'] and actual['reused_BASE']==ref['records']['clean']
  expect=difference(actual['current'],actual['reused_BASE']);pair=next(p for p in result['paired'] if p['row_id']==name)
  assert all(near(pair[k],expect[k]) for k in KEYS) and all(near(a,b) for a,b in zip(pair['per_TX_pp'],expect['per_TX_pp']));expected_pairs.append(dict(**row,**expect))
  cost=next(v for v in result['cost'] if v['row_id']==name);assert cost['epochs']==50 and cost['optimizer_steps']==11100 and all(near(cost[k],value) for k,value in (('source_seconds',src['elapsed_seconds']),('prediction_seconds',pred['seconds']),('peak_cuda_MiB',src['peak_cuda_MiB']))) and cost['source_packet_reads']==src['actual_source_packet_reads'] and cost['statistic_observation']==d.read(root/'source/statistic_gradient_observation.json')
  rows.append(dict(row_id=name,checkpoint_and_actual_args_verified=True,default_head_joint_updated=True,deployed_core_feature_parameter_tensors_changed=changed,prediction_without_truth_or_fitting=True,independent_CPU_truth_last=True,opaque_ID_coverage_and_correctness_mean_verified=True,all_worker_exits_zero=True,prediction_completed_before_score=True,source_seconds=src['elapsed_seconds'],prediction_seconds=pred['seconds'],peak_cuda_MiB=src['peak_cuda_MiB']))
 for arm in d.ARMS:
  group=[r for r in expected_pairs if r['treatment']==arm];assert [r['model_seed'] for r in group]==list(d.SEEDS)
  for k in KEYS:stats_check(result['by_arm'][arm]['paired_summary']['metrics'][k],[r[k] for r in group])
  for i in range(6):stats_check(result['by_arm'][arm]['paired_summary']['per_TX'][i],[r['per_TX_pp'][i] for r in group])
 for arm in d.ARMS:
  for k in ('accuracy','macro_f1','worst_RX','worst_day'):stats_check(result['by_arm'][arm]['absolute'][k],[100*r['current'][k] for r in result['records'] if r['treatment']==arm])
 assert [v['row_id'] for v in result['cost']]==[r['row_id'] for r in d.rows()]
 main=expected_pairs
 weak=all(mean(r['per_TX_pp'][i] for r in main)>0 and sum(r['per_TX_pp'][i]>0 for r in main)>=2 for i in (1,3));stable=all(r['weak_TX_mean_pp']>0 for r in main);health=mean(r['accuracy_pp'] for r in main)>=0 and mean(r['worst_RX_pp'] for r in main)>=-.5 and all(mean(r['per_TX_pp'][i] for r in main)>=-.25 for i in (0,2,4,5))
 expected_primary=dict(PASS=weak and stable and health,both_weak_means_positive_and_each2of3=weak,weakmean_positive3of3=stable,other_TX_overall_worstRX_health=health,all3_required=True)
 assert {k:result['scientific_verdict'][k] for k in expected_primary}==expected_primary
 secondary=result['secondary_comparisons'];assert set(secondary)=={'STAT_GRAD','NORM_ONLY'} and result['secondary_companion_run']=='20261020-phase1-FcrossRX-statGradient-e50-clean-local-m6-r01'
 companion=base.parent/result['secondary_companion_run']
 for arm in ('STAT_GRAD','NORM_ONLY'):
  value=secondary[arm]
  if value['status']=='PENDING_OTHER_FIXED_MATRIX':
   assert value==dict(status='PENDING_OTHER_FIXED_MATRIX',n=0)
   continue
  assert value['status']=='COMPLETE_3SEED_FIXED_SECONDARY' and [r['model_seed'] for r in value['paired']]==list(d.SEEDS)
  contrasts=[]
  from experiments.cvs_f_statgrad50.provenance import source_proof as other_source_proof
  for seed in d.SEEDS:
   other=d.read(companion/(arm+'-s'+str(seed))/'scores/metrics.json');other_source_proof(other['config'])
   assert other['status']=='MODEL_PREDICTED_SCORED' and other['prediction_commit']=='7b5532e9ff3845c23c7c229385ff5763254d31c3' and other['config']['model_seed']==seed and other['config']['treatment']==arm and other['truth_last'] and other['independent_CPU_score'] and other['views']==['clean'] and other['count']==168000 and not other['head_fit_updates']
   metric_check(other['records']['clean']);current=next(r['current'] for r in result['records'] if r['model_seed']==seed);expected=difference(current,other['records']['clean']);actual=next(r for r in value['paired'] if r['model_seed']==seed)
   assert all(near(actual[k],expected[k]) for k in KEYS) and all(near(a,b) for a,b in zip(actual['per_TX_pp'],expected['per_TX_pp']));contrasts.append(expected)
  for k in KEYS:stats_check(value['summary']['metrics'][k],[r[k] for r in contrasts])
  for i in range(6):stats_check(value['summary']['per_TX'][i],[r['per_TX_pp'][i] for r in contrasts])
 for v in summary['training_logs']:assert v['parsed_epochs']==50 and v['parsed_steps']==11100 and v['full_stdout_scanned'] and v['U_seen']==963900 and v['U_truth_available']==v['skipped_nonfinite_updates']==v['satellite_active_steps']==0
 for v in summary['contrastive_logs']:assert v['policy']=='CENTER_GRAD' and v['parsed_auxiliary_steps']==11100 and v['active_steps']==9990 and v['layer_forward_backward_objectives']==39960 and v['full_auxiliary_log_read'] and not v['extra_model_forwards'] and not v['head_fit_updates']
 assert len(summary['training_logs'])==len(summary['contrastive_logs'])==3
 assert {v['row_id'] for v in summary['training_logs']}=={r['row_id'] for r in d.rows()}=={v['row_id'] for v in summary['contrastive_logs']}
 return dict(status='VERIFIED',execution_code_commit=commit,run_id=d.RUN,all_three_rows_complete=True,three_seeds_each_condition=True,live_owned_workers=0,rows=rows,full50_epoch_11100_native_and_auxiliary_logs_verified=True,statistic_backward_objectives_per_model=39960,independent_confusion_metric_and_pair_statistics_check=True,IQ_reads=0,truth_reads=0,model_forwards=0,optimizer_updates=0,head_fit_updates=0,scientific_verdict=result['scientific_verdict'],secondary_status={a:v['status'] for a,v in secondary.items()},goal_completion_not_inferred_from_artifact_completion=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--reference-base',required=True);p.add_argument('--execution-commit',required=True);p.add_argument('--output',required=True);a=p.parse_args();value=check(a.base,a.reference_base,a.execution_commit)
 if value['status']=='VERIFIED':
  with Path(a.output).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(value,allow_nan=False))
