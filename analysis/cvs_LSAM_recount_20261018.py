"""Complete same-row L-SAM log/CM/paired recount;never opens IQ,CP or raw truth."""
import argparse,collections,json,math,time
from pathlib import Path
from analysis import cvs_view_pcgrad_source_logs_20261018 as logs
RUN='20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01'
MODES=('NOAS_REPLAY','AS_REPLAY','AS_LSAM')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
read=logs.read
def replay_records(S,mode,cost):
 rows,info=logs.jsonl(S/'L_replay_steps.jsonl.gz',True)
 if info['partial'] or not info['exists'] or len(rows)!=2220 or [x['step'] for x in rows]!=list(range(1,2221)):raise ValueError('Incomplete2220 replay records')
 radius=.05 if mode=='AS_LSAM' else 0.;update='LSAM' if radius else 'REPLAY';VJP=packets=0;maxdelta=0.
 for r in rows:
  if r['mode']!=update or r['radius']!=radius or r['cumulative_step']!=r['step']+4440 or r['L_forwards']!=8 or r['extra_L_forward_packets']!=512 or r['extra_VJP_calls']!=2 or r['U_replayed'] or r['teacher_replayed'] or not all(r[k] is True for k in ('weights_exactly_restored','global_RNG_restored','persistent_buffers_unchanged')):raise ValueError('Invalid sameIQ L replay scope/cost/restoration')
  if any(not math.isfinite(r[k]) for k in ('L_loss','replay_L_loss','L_gradient_norm','replay_gradient_norm','replay_gradient_relative_difference','actual_perturbation_L2','host_enqueue_seconds')):raise ValueError('Nonfinite replay statistic')
  expected=radius if r['L_gradient_norm'] else 0.;maxdelta=max(maxdelta,abs(expected-r['actual_perturbation_L2']))
  if radius==0 and (r['actual_perturbation_L2']!=0 or abs(r['replay_L_loss']-r['L_loss'])>5e-5+1e-5*abs(r['L_loss']) or r['replay_gradient_relative_difference']>1e-4):raise ValueError('Neutral replay not equivalent')
  VJP+=r['extra_VJP_calls'];packets+=r['extra_L_forward_packets']
 if maxdelta>2e-6:raise ValueError('Actual radius differs from fixed coordinate perturbation')
 if cost['update_mode']!=update or cost['extra_supervised_VJP_calls']!=VJP or cost['replay_calls']!=2220 or cost['replay_L_forward_packets']!=packets or cost['base_L_forward_packets']!=1136640 or cost['L_forward_packets']!=2273280 or cost['perturbed_calls']!=(2220 if radius else 0):raise ValueError('Actual replay counter or total L NN mismatch')
 if cost['student_train_forward_packets']!=2840280 or cost['teacher_train_forward_packets']!=0 or cost['teacher_eval_forward_packets']!=567000:raise ValueError('Expected native U/teacher budget changed')
 epochs=[]
 for e in range(1,11):
  group=rows[(e-1)*222:e*222];epochs.append(dict(epoch=e,cumulative_epoch=e+20,**{k:logs.summary([r[k] for r in group]) for k in ('L_loss','replay_L_loss','L_gradient_norm','replay_gradient_norm','replay_gradient_relative_difference','actual_perturbation_L2','host_enqueue_seconds')}))
 return dict(records=2220,scope='Actual perstep scalar witnesses and executing tensor assertions;not independent tensor reconstruction or identity evidence',radius=radius,radius_max_abs_error=maxdelta,extra_VJP=VJP,extra_L_forward_packets=packets,extra_L_backward_packet_derivatives=VJP*512,total_L_NN=2273280,epochs=epochs,gradient_relative_difference=logs.summary([r['replay_gradient_relative_difference'] for r in rows]),L_loss_difference=logs.summary([r['replay_L_loss']-r['L_loss'] for r in rows]),host_enqueue_seconds=sum(r['host_enqueue_seconds'] for r in rows))
def source(project):
 B=Path(project)/'runs'/RUN;result={};pools={};seqs={};args={};parents={};oldrun,oldmodes=logs.RUN,logs.MODES;start=time.perf_counter()
 try:
  logs.RUN=RUN
  for mode in MODES:
   logs.MODES=(mode,);S=B/(mode+'-s2026092701')/'source';part=logs._base_run(project,True,expected_As_packets=0 if mode=='NOAS_REPLAY' else 128);r=part['rows'][mode];tr=r['training_cost'];conf=read(S/'resolved_config.json');init=read(S/'initialization.json');cp_parent=Path(init['ancestors'][0]);a=read(S/'actual_native_args.json');a=a.get('args',a);args[mode]=a
   if tr['actual_steps']!=2220 or tr['parent_steps']!=4440 or tr['cumulative_steps']!=6660 or tr['cumulative_epochs']!=30 or tr['head_fit_updates'] or tr['U_truth_read'] or tr['target_access'] or tr['source_RX_holdout'] is not None or tr['Ag_training_observations'] or tr['As_training_observations']!=(0 if mode=='NOAS_REPLAY' else 284160) or init['source_roles']!='EXACT_MATCH' or not init['optimizer_reset'] or init['teacher_origin']!='strict own parent EMA' or conf['new_Ag'] or conf['As']!=(mode!='NOAS_REPLAY'):raise ValueError('Source budget,roles,parent,U or treatment mismatch')
   parentarm='REPEAT' if mode=='NOAS_REPLAY' else 'CONCAT'
   if cp_parent.parent.name!=parentarm+'-s2026092701' or a['baseline_ckpt']!=str(cp_parent/'final_ssdg.pth') or a['teacher_ckpt'] or a['freeze_backbone'] or (a['epochs'],a['label_epochs'],a['pseudo_epochs'])!=(10,0,10) or a['use_concat_sat_channel_aug'] or a.get('source_fold_held_rx') is not None:raise ValueError('Parent initialization or native training arguments mismatch')
   truth=r['numeric_totals'].get('train/pseudo_truth_available',{})
   if truth.get('count')!=2220 or truth.get('maximum')!=0:raise ValueError('Incomplete hidden U truth logs')
   r['L_replay']=replay_records(S,mode,tr);r['parent_source_arm']=parentarm;r['head_joint_updates_checked_by_trainer']=True;r['new_IQ_Ag']=False
   pools[mode],_=logs.jsonl(S/'sampler_pools.jsonl.gz',True);seqs[mode],_=logs.jsonl(S/'training_sequence.jsonl.gz',True);parents[parentarm]=read(cp_parent/'training_result.json');result[mode]=r
 finally:logs.RUN,logs.MODES=oldrun,oldmodes
 comparisons={}
 for mode in MODES[1:]:
  base=MODES[0];samep=[(p['original_L_ids'],p['visible_y']) for p in pools[mode]]==[(p['original_L_ids'],p['visible_y']) for p in pools[base]];sames=seqs[mode]==seqs[base];diff=sorted(k for k in set(args[mode])|set(args[base]) if args[mode].get(k)!=args[base].get(k));tc=result[mode]['training_cost'];bc=result[base]['training_cost'];samecost=all(tc[k]==bc[k] for k in ('actual_steps','parent_steps','cumulative_steps','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets','extra_supervised_VJP_calls'))
  if not samep or not sames or not samecost or set(diff)-{'baseline_ckpt','candidate_id','output_dir'}:raise ValueError('Matching source inputs/U/NN/actual args mismatch')
  comparisons[mode]=dict(complete_input_pools=370,L_U_sequence_records=4440,same_input_pools_and_L_labels=samep,same_L_U_physical_sequence=sames,same_NN_and_extra_VJP_budget=samecost,different_actual_arg_keys=diff,actual_arg_count=len(args[mode]))
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',run_id=RUN,rows=result,matched_comparisons=comparisons,parent_training_cost=parents,IQ=0,CP=0,NN=0,updates=0,target_access=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
def source_qualification(project):
 B=Path(project)/'runs'/RUN;rows={};ids=None;fixed_meta=None;CM=0;maxerr=0.
 for mode in MODES:
  P=B/(mode+'-s2026092701')/'source_qualification';r=read(P/'result.json');meta=read(P/'source_ids.json');metrics={};seen=set();records=r['records']
  if r['status']!='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY' or r['source_V_unique']!=1800 or r['source_cells']!=90 or r['all_RX']!=[1,3,4,6,8] or r['source_RX_holdout'] is not None or r['CP']!=1 or r['NN']!=23400 or r['physical_channel_calls']!=21600 or r['updates'] or r['teacher_NN'] or r['head_fit_updates'] or r['U_truth_read'] or r['target'] or not r['model_state_unchanged']:raise ValueError('Source qualification scope/cost/state differs')
  if len(meta['ids'])!=1800 or len(set(meta['ids']))!=1800 or ids is not None and ids!=meta['ids'] or collections.Counter(zip(meta['TX'],meta['RX'],meta['day']))!=collections.Counter({(t,x,d):20 for t in range(6) for x in (1,3,4,6,8) for d in (1,2,3)}):raise ValueError('Fixed all90cell V1800 physical IDs differ')
  if fixed_meta is not None and meta!=fixed_meta:raise ValueError('Source metadata and label mapping differ across rows')
  ids=meta['ids'];fixed_meta=meta
  for v in VIEWS:
   group=[z for z in records if z['view']==v];expected=1 if v=='clean' else 2
   if len(group)!=expected or {z['draw'] for z in group}!=set(range(expected)):raise ValueError('Fixed source view draws differ')
   cm=[[0]*6 for _ in range(6)]
   for a in group:
    if a['count']!=1800 or len(a['confusion'])!=6 or any(len(x)!=6 or any(type(n) is not int or n<0 for n in x) for x in a['confusion']):raise ValueError('Source CM dimensions/count invalid')
    if any(sum(row)!=300 for row in a['confusion']) or sum(map(sum,a['confusion']))!=1800 or sum(a['confusion'][i][i] for i in range(6))!=a['correct'] or set(a['per_RX'])!={'1','3','4','6','8'} or sum(s['count'] for s in a['per_RX'].values())!=1800 or sum(s['correct'] for s in a['per_RX'].values())!=a['correct'] or any(s['count']!=360 or not 0<=s['correct']<=360 for s in a['per_RX'].values()):raise ValueError('Source CM or RX sums differ')
    for i in range(6):
     for j in range(6):cm[i][j]+=a['confusion'][i][j]
    CM+=1
   n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(6));acc=correct/n;error=abs(acc-r['metrics'][v]['accuracy']);maxerr=max(maxerr,error)
   if error>1e-12 or n!=r['metrics'][v]['decisions']:raise ValueError('Source accuracy recount differs')
   rs=[sum(x) for x in cm];cs=[sum(cm[i][j] for i in range(6)) for j in range(6)];metrics[v]=dict(accuracy=acc,macro_F1=sum(2*cm[i][i]/(rs[i]+cs[i]) if rs[i]+cs[i] else 0 for i in range(6))/6,count=n,correct=correct,confusion=cm)
  if len(records)!=13:raise ValueError('Extra source views or missing records')
  rows[mode]=dict(metrics=metrics,cost={k:r[k] for k in ('CP','NN','physical_channel_calls','cache_loads','wall_seconds')})
 a=rows['AS_LSAM']['metrics'];b=rows['AS_REPLAY']['metrics'];delta={v:100*(a[v]['accuracy']-b[v]['accuracy']) for v in VIEWS};checks=dict(clean=delta['clean']>=0.,midurban=delta['practical_mid_urban']>=0.,lowurban=delta['practical_low_urban']>=.5,suburban=all(delta[v]>=-.25 for v in VIEWS[1:4]));saved=read(B/'source_matrix_decision.json')
 if checks!=saved['conditions'] or any(abs(delta[v]-saved['delta_pp'][v])>1e-12 for v in VIEWS) or saved['source_pass']!=all(checks.values()) or saved['target_access']:raise ValueError('Independent source qualification verdict differs')
 return dict(status='VERIFIED_SOURCE_QUALIFICATION',source_pass=all(checks.values()),decision=saved['decision'],conditions=checks,delta_pp=delta,rows=rows,CM_recounts=CM,per_RX_partition_recounts=CM,max_metric_error=maxerr,target_access=False,IQ=0,CP=0,NN=0,updates=0)
def run(project,source_only=False):
 start=time.perf_counter();src=source(project);qualification=source_qualification(project)
 if source_only:return dict(source=src,source_qualification=qualification,IQ=0,CP=0,NN=0,updates=0,target_scores_read=False)
 from analysis.cvs_matched_noas_recount_20261018 import metrics
 import numpy as np
 B=Path(project)/'runs'/RUN;root=read(B/'result.json');done=read(B/'completion.json');summary={};indexed={};formal={};correct={};cost={};ids=None;err=0.
 if done['status']!='ARTIFACTS_COMPLETE' or done['rows']!=3 or done['new_optimizer_updates']!=6660 or done['prediction_count']!=3528000 or not done['truth_last'] or done['head_fit_updates']:raise ValueError('All fixed rows must complete full predictions/independent scoring')
 for mode in MODES:
  P=B/(mode+'-s2026092701');d=read(P/'completion.json');pred=read(P/'prediction/complete.json');score=read(P/'scores/metrics.json')
  if d['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or d['views']!=list(VIEWS) or d['count']!=168000 or not d['truth_last'] or d['head_fit_updates'] or d['cumulative_epochs']!=30 or d['optimizer_steps']!=2220 or pred['truth_read'] or pred['query_fit'] or not pred['fixed_received_IQ'] or pred['head_fit_updates'] or score['status']!='SCORED' or not score['truth_last']:raise ValueError('Formal defaulthead/budget/received/truthlast scope differs')
  m,index,e=metrics(score['records']);err=max(err,e);summary[mode]=m;indexed[mode]=index;formal[mode]=score['records']
  if any(abs(m[v][k]-root['summary'][mode][v][k])>1e-12 for v in VIEWS for k in ('accuracy','macro_F1','worst_RX_accuracy')):raise ValueError('Formal root summary differs')
  with np.load(P/'scores/correctness.npz',allow_pickle=False) as a:
   if set(a.files)!={'ids',*VIEWS} or a['ids'].shape!=(168000,) or len(set(a['ids'].tolist()))!=168000 or ids is not None and not np.array_equal(ids,a['ids']):raise ValueError('Fixed opaque physical decision IDs differ')
   ids=a['ids'].copy();correct[mode]={v:a[v].copy() for v in VIEWS}
  for v in VIEWS:
   z=correct[mode][v]
   if z.dtype!=np.bool_ or z.shape!=(168000,) or int(z.sum())!=m[v]['correct']:raise ValueError('Correctness-vector count differs')
  tr=src['rows'][mode]['training_cost'];cost[mode]=dict(training=tr,prediction=pred,scorer={k:score[k] for k in ('status','truth_last','CPU_wall_seconds')},source_qualification=qualification['rows'][mode]['cost'])
 paired={};strata=[]
 for left,right in [('AS_REPLAY','NOAS_REPLAY'),('AS_LSAM','AS_REPLAY'),('AS_LSAM','NOAS_REPLAY')]:
  key=left+'__'+right;a=correct[left];b=correct[right];paired[key]={}
  for v in VIEWS:
   repairs=int((a[v]&~b[v]).sum());broken=int((~a[v]&b[v]).sum());joint=int((a[v]&a['clean']).sum())-int((b[v]&b['clean']).sum());shared=a['clean']&b['clean'];saved=root['paired_differences'][key][v]
   p=dict(repairs=repairs,new_errors=broken,delta_accuracy_pp=(repairs-broken)/1680,joint_clean_and_scene_delta_pp=joint/1680,shared_clean_count=int(shared.sum()),repairs_when_both_clean_correct=int((a[v]&~b[v]&shared).sum()),new_errors_when_both_clean_correct=int((~a[v]&b[v]&shared).sum()))
   if repairs-broken!=summary[left][v]['correct']-summary[right][v]['correct'] or any(abs(p[k]-saved[k])>1e-12 for k in saved):raise ValueError('Paired count differs from accuracy or root')
   paired[key][v]=p
  for k,r in indexed[left].items():
   s=indexed[right][k];strata.append(dict(comparison=key,view=k[0],dimension=k[1],stratum=k[2],count=r['query_count'],left_accuracy=r['accuracy'],right_accuracy=s['accuracy'],delta_accuracy_pp=100*(r['accuracy']-s['accuracy']),left_macro_F1=r['macro_f1'],right_macro_F1=s['macro_f1']))
 return dict(status='VERIFIED',run_id=RUN,source=src,source_qualification=qualification,summary=summary,paired=paired,strata=strata,formal_metrics=formal,cost=cost,suburban_gaps_pp={k:{v:100*(m['clean']['accuracy']-m[v]['accuracy']) for v in VIEWS[1:4]} for k,m in summary.items()},CM_recounts=378,partition_recounts=63,paired_comparisons=21,max_metric_error=err,IQ=0,CP=0,NN=0,updates=0,raw_truth_reads=0,new_Ag=False,Ag_As_interaction='NOT_DEFINED_NO_NEW_AG',single_seed_only=True,joint_goal_complete=False,target_feedback=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--source-only',action='store_true');a=p.parse_args();print(json.dumps(run(a.project,a.source_only),allow_nan=False))
