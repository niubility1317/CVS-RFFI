"""One mandatory no-As readout after both immutable runs finish;no IQ/CP/rawtruth/NN."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from analysis import cvs_view_pcgrad_source_logs_20261018 as logs
RUN='20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01'
AS_RUN='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
MODES=('B00','ORIGINAL','MEASURE','PROJECT')
read=logs.read

def source(project):
 B=Path(project)/'runs'/RUN;S=B/'B00-s2026092701/source';old_run,old_modes=logs.RUN,logs.MODES
 try:
  logs.RUN,logs.MODES=RUN,('B00',)
  value=logs._base_run(project,True,expected_As_packets=0)
 finally:logs.RUN,logs.MODES=old_run,old_modes
 row=value['rows']['B00'];g=logs.gradient_logs(S,'MEASURE',True);row['view_gradient']=g
 truth=row['numeric_totals'].get('train/pseudo_truth_available',{})
 if truth.get('count')!=2220 or truth.get('maximum')!=0 or g['actual_projection_updates'] or g['extra_VJP_calls']!=4440:raise ValueError('Mandatory reference must have hidden U and neutral twoVJP')
 tr=row['training_cost'];c=read(S/'resolved_config.json');init=read(S/'initialization.json');parent=Path(init['ancestors'][0]);parent_tr=read(parent/'training_result.json')
 if tr['As_training_observations'] or tr['Ag_training_observations'] or c['As'] or c['new_Ag'] or parent.name!='source' or parent.parent.name!='REPEAT-s2026092701' or not init['source_roles']=='EXACT_MATCH':raise ValueError('No-As reference or original parent differs')
 comparisons={};our_pools,_=logs.jsonl(S/'sampler_pools.jsonl.gz',True);our_seq,_=logs.jsonl(S/'training_sequence.jsonl.gz',True);our_args=read(S/'actual_native_args.json');our_args=our_args.get('args',our_args)
 for mode in MODES[1:]:
  P=Path(project)/'runs'/AS_RUN/(mode+'-s2026092701')/'source';pools,info=logs.jsonl(P/'sampler_pools.jsonl.gz',True);seq,sinfo=logs.jsonl(P/'training_sequence.jsonl.gz',True);args=read(P/'actual_native_args.json');args=args.get('args',args)
  same_pools=[(p['original_L_ids'],p['visible_y']) for p in pools]==[(p['original_L_ids'],p['visible_y']) for p in our_pools];same_seq=seq==our_seq
  differences=sorted(k for k in set(args)|set(our_args) if args.get(k)!=our_args.get(k));other_tr=read(P/'training_result.json')
  same_counts=all(tr[k]==other_tr[k] for k in ('actual_steps','parent_steps','cumulative_steps','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))
  if info['partial'] or sinfo['partial'] or len(pools)!=370 or len(seq)!=4440 or not same_pools or not same_seq or not same_counts:raise ValueError('Matching all-source physical inputs/budget failed')
  if set(differences)-{'candidate_id','output_dir','baseline_ckpt','run_id'}:raise ValueError('Unexplained actual native arg difference:'+repr(differences))
  comparisons[mode]=dict(input_pools=370,L_U_sequence_records=4440,same_input_ids_and_visible_labels=same_pools,same_physical_L_U_sequence=same_seq,same_NN_budget=same_counts,different_actual_args=differences,actual_arg_count=len(args),same_initialization_rule_not_parent_weights=True)
 value['matched_As_comparisons']=comparisons;value['parent_training_cost']=parent_tr
 value['hidden_U_truth_records']=2220;value['new_Ag']=False;value['joint_goal_complete']=False
 return value

def metrics(records):
 index={};overall={};error=0.
 for r in records:
  key=(r['view'],r['dimension'],str(r['stratum']));c=r['confusion']
  if key in index or len(c)!=6 or any(len(row)!=6 or any(not isinstance(n,int) or n<0 for n in row) for row in c):raise ValueError('Duplicate or invalid sixclass CM')
  index[key]=r;rows=[sum(row) for row in c];cols=[sum(c[i][j] for i in range(6)) for j in range(6)];total=sum(rows);correct=sum(c[i][i] for i in range(6))
  if total!=r['query_count'] or total==0:raise ValueError('CM count differs')
  calc=dict(accuracy=correct/total,macro_f1=sum(2*c[i][i]/(rows[i]+cols[i]) if rows[i]+cols[i] else 0 for i in range(6))/6,macro_accuracy=sum(c[i][i]/rows[i] for i in range(6) if rows[i])/sum(n>0 for n in rows))
  for k,v in calc.items():
   delta=abs(v-r[k]);error=max(error,delta)
   if not math.isfinite(r[k]) or delta>1e-12:raise ValueError('Independent CM metric differs')
  if r['dimension']=='overall':overall[r['view']]=dict(accuracy=calc['accuracy'],macro_F1=calc['macro_f1'],correct=correct,count=total)
 if len(index)!=126 or set(overall)!=set(VIEWS):raise ValueError('All126 score strata required')
 for view in VIEWS:
  allrow=index[(view,'overall','ALL')]
  if allrow['query_count']!=168000:raise ValueError('Full168000 each view required')
  for dim,n in (('receiver',7),('transmitter',6),('day',4)):
   group=[r for (v,d,s),r in index.items() if v==view and d==dim]
   if len(group)!=n or any(sum(r['confusion'][i][j] for r in group)!=allrow['confusion'][i][j] for i in range(6) for j in range(6)):raise ValueError('Partition sum differs')
  overall[view]['worst_RX_accuracy']=min(r['accuracy'] for (v,d,s),r in index.items() if v==view and d=='receiver')
 return overall,index,error

def run(project):
 start=time.perf_counter();B=Path(project)/'runs';summary={};indexed={};formal={};correct={};fixed_ids=None;cost={};maxerr=0.
 for run_id,nrows,count in ((RUN,1,1176000),(AS_RUN,3,3528000)):
  done=read(B/run_id/'completion.json')
  if done['status']!='ARTIFACTS_COMPLETE' or done['rows']!=nrows or done['prediction_count']!=count or not done['truth_last'] or done['head_fit_updates']:raise ValueError('Both fixed runs must finish independent scores')
 src=source(project)
 for mode in MODES:
  P=B/(RUN if mode=='B00' else AS_RUN)/(mode+'-s2026092701');d=read(P/'completion.json');p=read(P/'prediction/complete.json');s=read(P/'scores/metrics.json');root_result=read(P.parent/'result.json')
  if d['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or d['views']!=list(VIEWS) or d['count']!=168000 or not d['truth_last'] or d['head_fit_updates'] or d['cumulative_epochs']!=30 or d['optimizer_steps']!=2220 or p['truth_read'] or p['query_fit'] or not p['fixed_received_IQ'] or p['head_fit_updates'] or s['status']!='SCORED' or not s['truth_last']:raise ValueError('Fixed budget/defaulthead/sevenview/truthlast differs')
  m,idx,e=metrics(s['records']);maxerr=max(maxerr,e);summary[mode]=m;indexed[mode]=idx;formal[mode]=s['records']
  if any(abs(m[v][k]-root_result['summary'][mode][v][k])>1e-12 for v in VIEWS for k in ('accuracy','macro_F1','worst_RX_accuracy')):raise ValueError('Root summary differs')
  with np.load(P/'scores/correctness.npz',allow_pickle=False) as a:
   ids=a['ids']
   if set(a.files)!={'ids',*VIEWS} or ids.shape!=(168000,) or len(set(ids.tolist()))!=168000 or fixed_ids is not None and not np.array_equal(ids,fixed_ids):raise ValueError('Fixed paired opaque IDs differ')
   fixed_ids=ids.copy();correct[mode]={v:a[v].copy() for v in VIEWS}
  for v in VIEWS:
   z=correct[mode][v]
   if z.dtype!=np.bool_ or z.shape!=(168000,) or int(z.sum())!=m[v]['correct']:raise ValueError('Correctness recount differs')
  tr=read(P/'source/training_result.json');cost[mode]={k:tr[k] for k in ('actual_steps','parent_steps','cumulative_steps','parent_epochs','new_epochs','cumulative_epochs','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets','As_training_observations','Ag_training_observations','elapsed_seconds','extra_supervised_VJP_calls','gradient_projection_nonzero')};cost[mode].update(prediction_NN=p['NN_forward_packets'],prediction_wall_seconds=p['wall_seconds'],scorer_CPU_wall_seconds=s['CPU_wall_seconds'])
 paired={};strata=[]
 for mode in MODES[1:]:
  comparison=mode+'__B00';paired[comparison]={};a=correct[mode];b=correct['B00']
  for v in VIEWS:
   repairs=int((a[v]&~b[v]).sum());broken=int((~a[v]&b[v]).sum());joint_delta=int((a[v]&a['clean']).sum())-int((b[v]&b['clean']).sum());shared=a['clean']&b['clean']
   if repairs-broken!=summary[mode][v]['correct']-summary['B00'][v]['correct']:raise ValueError('Paired repairs net differs')
   paired[comparison][v]=dict(repairs=repairs,new_errors=broken,delta_accuracy_pp=(repairs-broken)/1680,joint_clean_and_scene_delta_pp=joint_delta/1680,shared_clean_count=int(shared.sum()),repairs_when_both_clean_correct=int((a[v]&~b[v]&shared).sum()),new_errors_when_both_clean_correct=int((~a[v]&b[v]&shared).sum()))
  for key,r in indexed[mode].items():
   s=indexed['B00'][key];strata.append(dict(comparison=comparison,view=key[0],dimension=key[1],stratum=key[2],count=r['query_count'],left_accuracy=r['accuracy'],right_accuracy=s['accuracy'],delta_accuracy_pp=100*(r['accuracy']-s['accuracy']),left_macro_F1=r['macro_f1'],right_macro_F1=s['macro_f1']))
 gaps={m:{v:100*(summary[m]['clean']['accuracy']-summary[m][v]['accuracy']) for v in VIEWS[1:4]} for m in MODES}
 return dict(status='VERIFIED',run_id=RUN,reference_for_run=AS_RUN,source=src,summary=summary,paired=paired,strata=strata,formal_metrics=formal,CM_recounts=504,partition_recounts=84,paired_comparisons=21,max_metric_error=maxerr,cost=cost,suburban_gaps_pp=gaps,Ag_efficacy_proved=False,Ag_As_interaction='NOT_DEFINED_NO_NEW_AG',single_seed_only=True,joint_goal_complete=False,IQ=0,CP=0,NN=0,raw_truth_reads=0,new_updates=0,target_feedback=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);a=p.parse_args();print(json.dumps(run(a.project),allow_nan=False))
