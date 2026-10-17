"""Read-only closed-row analysis;no IQ/checkpoint/NN or model selection."""
import argparse,gzip,hashlib,itertools,json,math
from collections import Counter,defaultdict
from pathlib import Path
RUN='20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01'
CONTROL='20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01'
TRAIN_CONTROL='20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03'
SEED=2026092701
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):
 opener=gzip.open if p.suffix=='.gz' else open
 with opener(p,'rt',encoding='utf8') as f:
  for line in f:
   if not line.endswith('\n'):raise ValueError('Partial record')
   if line.strip():yield json.loads(line)
def chooser(pid,epoch,step):
 raw=json.dumps([20261017991,'FIR_Ag',epoch,step,list(map(int,pid))],separators=(',',':'))
 return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8],'little')%12
def source_logs(p,control,sat,keys):
 result=read(p/'training_result.json');assert result['actual_steps']==2220 and result['cumulative_steps']==6660 and result['L_forward_packets']==1136640 and result['Ag_training_observations']==142080 and result['As_training_observations']==(142080 if sat else 0)
 assert result['source_RX_holdout'] is None and not result['U_truth_read'] and not result['target_access'] and not result['head_fit_updates']
 epochs=list(lines(p/'epoch_metrics.jsonl'));assert [e['epoch'] for e in epochs]==list(range(1,11))
 steps=list(lines(p/'step_metrics.jsonl.gz'));assert [x['step'] for x in steps]==list(range(1,2221))
 for x in steps:
  v=x['metrics'];assert math.isfinite(v['train/loss'])
  for k,expected in [('train/optimizer_step_applied',1),('train/pseudo_truth_available',0),('train/skipped_nonfinite_grad',0),('train/skipped_nonfinite_loss',0),('concat/L_forwards',4),('concat/As_packets',64 if sat else 0),('concat/SupCon_weight',.1)]:assert v[k]==expected,(k,v[k])
 details=list(lines(p/'reference_steps.jsonl.gz'));assert [x['step'] for x in details]==list(range(1,2221));objective_error=0.
 for x in details:
  assert x['L_forwards']==4 and x['Ag_packets']==64 and x['As_packets']==(64 if sat else 0) and x['same_IQ_each_pair'] and x['ordinary_redraws']==0 and x['all_satellite_L_supervised']
  objective_error=max(objective_error,abs(x['total_L']-(.5*x['common_RD']+.5*x['extra_RD']+x['SupCon_weight']*x['SupCon'])))
 assert objective_error<2e-6
 seq=list(lines(p/'training_sequence.jsonl.gz'));other=list(lines(control/'training_sequence.jsonl.gz'));assert len(seq)==4440 and seq==other
 unique=defaultdict(set);packets=Counter();kernels=Counter();by_group=Counter();per_step=[];step=0
 for x in seq:
  role=x['role'];assert role in ('L','U')
  ids=[tuple(q) for q in x['ids']];unique[role].update(ids);packets[role]+=len(ids)
  if role=='L':
   step+=1;cumstep=4440+step;ep=(cumstep-1)//222+1;cc=Counter()
   assert len(ids)==128
   for i,pid in enumerate(ids):
    assert pid[0]==1 and pid[1] in (1,3,4,6,8) and pid[2] in (1,2,3)
    if (i+step)%2==0:
     key=keys[chooser(pid,ep,cumstep)];kernels[key]+=1;cc[key]+=1;by_group[(pid[1],pid[2],key)]+=1
   per_step.append(dict(epoch=ep,step=cumstep,kernel_counts=dict(cc)))
 assert {k:len(v) for k,v in unique.items()}=={'L':6300,'U':56700} and packets==Counter(L=284160,U=567000)
 actual=result['clock'];assert dict(kernels)==actual['kernel_counts'] and per_step==actual['steps'] and step==actual['calls']==2220
 assert [dict(RX=k[0],day=k[1],kernel=k[2],count=n) for k,n in sorted(by_group.items())]==actual['per_RX_day_kernel']
 args=read(p/'actual_native_args.json');args=args.get('args',args);ref=read(control/'actual_native_args.json');ref=ref.get('args',ref)
 differences={k for k in set(args)|set(ref) if args.get(k)!=ref.get(k)};expected={'run_id','candidate_id','output_dir'};assert differences<=expected,differences
 counts={k:result[k] for k in ('actual_steps','parent_steps','cumulative_steps','parent_epochs','new_epochs','cumulative_epochs','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets','As_training_observations','Ag_training_observations','elapsed_seconds')}
 return dict(status='VERIFIED',cost=counts,epoch_metrics=epochs,step_records=len(steps),sequence_records=len(seq),physical_packet_visits=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},same_all_LU_sequence_as_control=True,actual_args_comparison=dict(count=len(args),differing_keys=sorted(differences)),objective_max_error=objective_error,filter={k:v for k,v in actual.items() if k!='steps'},source_counts={k:len(v) for k,v in read(p/'source_contract.json')['role_ids'].items()},ordinary_actual=read(p/'ordinary_augmentation_actual.json'),channel=result['channel'])
def analyze(project):
 import numpy as np
 base=project/'runs'/RUN;done=read(base/'completion.json');assert done['status']=='ARTIFACTS_COMPLETE' and done['rows']==2 and done['new_optimizer_updates']==4440
 folders={'B00':project/'runs'/CONTROL/('B00-s'+str(SEED)),'B01':project/'runs'/CONTROL/('B01-s'+str(SEED)),'B10':base/('FIR_B10-s'+str(SEED)),'B11':base/('FIR_B11-s'+str(SEED)),'FULL_AS':project/'runs'/CONTROL/('FULL_AS-s'+str(SEED))}
 summary={};correct={};strata={};CM=0;errors=[];predcost={};physical=None
 for arm,p in folders.items():
  c=read(p/'completion.json');pred=read(p/'prediction/complete.json');assert c['status']=='TRAINED_DEFAULT_PREDICTED_SCORED' and c['truth_last'] and pred['views']==list(VIEWS) and pred['count']==168000 and not pred['truth_read'] and pred['fixed_received_IQ']
  scores=read(p/'scores/metrics.json');records=scores['records'];assert len(records)==126
  with np.load(p/'scores/correctness.npz',allow_pickle=False) as a:
   if physical is None:physical=a['ids'].copy()
   assert np.array_equal(physical,a['ids']);correct[arm]={v:a[v].copy() for v in VIEWS}
  summary[arm]={};strata[arm]=records;predcost[arm]={k:pred[k] for k in ('NN_forward_packets','inference_seconds','wall_seconds','peak_cuda_allocated_bytes')};predcost[arm]['score_CPU_wall_seconds']=scores.get('CPU_wall_seconds',scores.get('CPU_seconds'))
  for r in records:
   cm=np.asarray(r['confusion']);assert cm.shape==(6,6) and (cm>=0).all() and cm.sum()==r['query_count'];den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean();acc=float(np.trace(cm)/cm.sum());errors.extend([abs(acc-r['accuracy']),abs(float(f1)-r['macro_f1'])]);CM+=1
   if r['dimension']=='overall':
    assert int(np.trace(cm))==int(correct[arm][r['view']].sum());summary[arm][r['view']]=dict(accuracy=r['accuracy'],macro_F1=r['macro_f1'],correct=int(np.trace(cm)),worst_RX=min(q['accuracy'] for q in records if q['view']==r['view'] and q['dimension']=='receiver'))
  for view in VIEWS:
   for dimension in ('receiver','transmitter','day'):
    group=[np.asarray(r['confusion']) for r in records if r['view']==view and r['dimension']==dimension];overall=next(np.asarray(r['confusion']) for r in records if r['view']==view and r['dimension']=='overall');assert np.array_equal(sum(group),overall)
 assert max(errors)<1e-12
 pairs={}
 for left,right in [('B10','B00'),('B11','B00'),('B11','B10'),('B11','B01'),('B11','FULL_AS')]:
  out={};a=correct[left];b=correct[right];cohort=a['clean']&b['clean']
  for v in VIEWS:
   x,y=a[v],b[v];jx=x&a['clean'];jy=y&b['clean'];out[v]=dict(delta_pp=float((x.astype(int)-y.astype(int)).mean()*100),repairs=int((x&~y).sum()),new_errors=int((~x&y).sum()),joint_clean_scene_delta_count=int(jx.sum())-int(jy.sum()),common_clean_count=int(cohort.sum()),repairs_common_clean=int((x&~y&cohort).sum()),new_errors_common_clean=int((~x&y&cohort).sum()))
  pairs[left+'-'+right]=out
 interactions={v:100*(summary['B11'][v]['accuracy']-summary['B10'][v]['accuracy']-summary['B01'][v]['accuracy']+summary['B00'][v]['accuracy']) for v in VIEWS}
 gaps={a:{v:100*(summary[a]['clean']['accuracy']-summary[a][v]['accuracy']) for v in VIEWS[1:4]} for a in summary}
 conditions=dict(B10_clean_vs_B00_min_point2=pairs['B10-B00']['clean']['delta_pp']>=.2-1e-10,B11_clean_vs_B00_nonnegative=pairs['B11-B00']['clean']['delta_pp']>=-1e-10,B11_low_vs_B01_min_point5=pairs['B11-B01']['practical_low_urban']['delta_pp']>=.5-1e-10,B11_mid_vs_B01_nonnegative=pairs['B11-B01']['practical_mid_urban']['delta_pp']>=-1e-10,B11_each_suburban_vs_B01_no_worse_point25=all(pairs['B11-B01'][v]['delta_pp']>=-.25-1e-10 for v in VIEWS[1:4]))
 bank=read(project/'runs/20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01/kernel_bank.json');keys=sorted(k for k in bank if k.startswith('structured_'));source={}
 for arm,ctrl in [('B10','B00'),('B11','B01')]:source[arm]=source_logs(folders[arm]/'source',project/'runs'/TRAIN_CONTROL/(ctrl+'-s'+str(SEED))/'source',arm=='B11',keys)
 result=read(base/'result.json')
 for v in VIEWS:assert abs(interactions[v]-result['interaction_pp'][v])<1e-10
 return dict(status='ANALYZED',scientific_fixed_recipe_conditions=conditions,fixed_recipe_passed=all(conditions.values()),summary=summary,paired=pairs,interaction_pp=interactions,suburban_gaps_pp=gaps,source=source,stratified_records=strata,prediction_cost=predcost,independent_recount=dict(status='VERIFIED',confusion_matrices=CM,partition_sums=5*7*3,max_metric_error=max(errors),IQ=0,CP=0,NN=0,new_updates=0,raw_truth_reads=0),source_guard=read(base/'source_input_guard.json'),cumulative_epochs=30,model_seed_count=1,target_feedback=False,automatic_seed_extension=False,full_As_difference_has_dose_confound=True,previous_PSD_feasibility_failed_unchanged=True,joint_goal_complete=False)
def report_json(value):
 if isinstance(value,float) and not math.isfinite(value):return None
 if isinstance(value,dict):return {k:report_json(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [report_json(v) for v in value]
 return value
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);a=ap.parse_args();print(json.dumps(report_json(analyze(a.project)),allow_nan=False))
if __name__=='__main__':main()
