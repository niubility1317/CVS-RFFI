"""Complete B00/B10 readout;all logs/CM/paired records,no IQ/CP/rawtruth access."""
import argparse,csv,json,math,time
import numpy as np
from pathlib import Path
from analysis import cvs_view_pcgrad_source_logs_20261018 as logs
RUN='20261018-phase1-F-crossRX-repeatShrink-e20plus10-s2026092701-m2-r01'
MODES=('B00','B10');SEED=2026092701
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
read=logs.read

def source(project):
 B=Path(project)/'runs'/RUN;rows={};sequences={};args={};operators={};old_run,old_modes=logs.RUN,logs.MODES;start=time.perf_counter();parent_cost=None;parent_sources=set()
 try:
  logs.RUN=RUN
  for mode in MODES:
   logs.MODES=(mode,);S=B/(mode+'-s'+str(SEED))/'source';done=read(S/'completion.json');init=read(S/'initialization.json');config=read(S/'resolved_config.json');args[mode]=read(S/'actual_native_args.json');args[mode]=args[mode].get('args',args[mode]);r=logs._base_run(project,False,expected_As_packets=0,expected_L_forwards=6)['rows'][mode];tr=r['training_cost'];enabled=mode=='B10'
   r.pop('all_current_steps_expected_As_L4_hiddenU',None);r['all_current_steps_expected_L6_hiddenU']=True
   if (tr['actual_steps'],tr['parent_steps'],tr['cumulative_steps'],tr['cumulative_epochs'],tr['L_forward_packets'])!=(2220,4440,6660,30,1704960) or tr['As_training_observations'] or tr['Ag_training_observations']!=(284160 if enabled else 0) or tr['head_fit_updates'] or tr['U_truth_read'] or tr['target_access'] or tr['source_RX_holdout'] is not None or config['As'] or config['new_Ag']!=enabled:raise ValueError('Fixed E30/6forward/allsource/Agslot scope differs')
   if (tr['student_train_forward_packets'],tr['student_eval_forward_packets'],tr['teacher_train_forward_packets'],tr['teacher_eval_forward_packets'])!=(2271960,324002,0,567000) or tr['SupCon_applied']!=2220 or tr['SupCon_layer_forward_terms']!=8880:raise ValueError('Actual original U/teacher/SupCon or NN budget differs')
   parent=Path(init['ancestors'][0]);parent_sources.add(str(parent));parent_cost=read(parent/'training_result.json')
   if parent.parent.name!='REPEAT-s'+str(SEED) or init['source_roles']!='EXACT_MATCH' or not init['optimizer_reset'] or init['teacher_origin']!='strict own parent EMA' or done['native_disabled_probe_calls']!=1 or done['head_fit_updates']:raise ValueError('Parent/head/EMA provenance differs')
   a=args[mode]
   if a['baseline_ckpt']!=str(parent/'final_ssdg.pth') or a['teacher_ckpt'] or a['freeze_backbone'] or (a['epochs'],a['label_epochs'],a['pseudo_epochs'])!=(10,0,10) or a['use_concat_sat_channel_aug'] or a.get('source_fold_held_rx') is not None:raise ValueError('Actual source args or initialization differ')
   required=('step_log','epoch_log','sequence_log','reference_log','unit_log')
   if any(not r[k]['exists'] or r[k]['partial'] for k in required) or [r[k]['records'] for k in required]!=[2220,10,4440,2220,2220] or (r['observed_L_packets'],r['observed_U_packets'],r['observed_unique_L'],r['observed_unique_U'])!=(284160,567000,6300,56700) or r['source_RX']!=[1,3,4,6,8] or r['source_days']!=[1,2,3] or any(r['error_markers'].values()):raise ValueError('Full logs/roles/original L/U membership incomplete')
   native,info=logs.jsonl(S/'metrics_epoch.jsonl')
   with (S/'metrics_epoch.csv').open(encoding='utf8',newline='') as stream:csvrows=list(csv.DictReader(stream))
   if info['partial'] or len(native)!=10 or native!=r['epoch_curves'] or len(csvrows)!=10 or any(set(x)!=set(y) or any(y[k]!=('' if v is None else str(v)) for k,v in x.items()) for x,y in zip(native,csvrows)):raise ValueError('Complete native CSV/JSONL and observer telemetry differ')
   r['native_telemetry']=dict(JSONL_epochs=10,CSV_epochs=10,observer_epochs=10,fields_each=len(native[0]),all_values_equal=True)
   truth=r['numeric_totals'].get('train/pseudo_truth_available',{})
   if truth.get('count')!=2220 or truth.get('maximum')!=0:raise ValueError('Hidden U truth telemetry incomplete')
   reference,refinfo=logs.jsonl(S/'reference_steps.jsonl.gz',True);draws=[];CPU=0.;maximum=0.;loss_error=0.
   for step,q in enumerate(reference,1):
    op=q['operator'];values=op['lambda_values']
    expected_draw=np.random.default_rng(np.random.SeedSequence([20261018104,20+(step-1)//222+1,4440+step])).uniform(0.,.1,128).tolist()
    if values!=expected_draw:raise ValueError('Actual lambda differs from declared cumulative RNG stream')
    if (q['step'],q['epoch'],q['cumulative_step'],q['L_forwards'],q['Ag_packets'],q['As_packets'])!=(step,(step-1)//222+1,4440+step,6,128 if enabled else 0,0) or q['supervised_view_weights']!=[.5,.25,.25] or q['total_L_scale']!=1. or not q['same_IQ_each_pair'] or q['ordinary_redraws'] or len(values)!=128 or op['enabled']!=enabled or any(not math.isfinite(v) or not 0<=v<.1 for v in values):raise ValueError('Actual view slots/weights/private draws differ')
    if abs(op['applied_lambda_mean']-(sum(values)/128 if enabled else 0.))>1e-15 or not math.isfinite(op['CPU_bridge_operator_seconds']) or op['CPU_bridge_operator_seconds']<0 or not 0<=op['max_relative_change']<=.100001 or (not enabled and op['max_relative_change']!=0):raise ValueError('Operator actual amplitude/cost/identity differs')
    expected=.5*q['common_RD']+.25*q['anchor_RD']+.25*q['slot_RD']+q['SupCon_weight']*q['SupCon'];loss_error=max(loss_error,abs(q['total_L']-expected));CPU+=op['CPU_bridge_operator_seconds'];maximum=max(maximum,op['max_relative_change']);draws.append(values)
   if refinfo['partial'] or len(reference)!=2220 or loss_error>2e-6:raise ValueError('Complete ordered supervised loss scale differs')
   op=tr['IQ_operator']
   if op['calls']!=2220 or op['physical_input_packets']!=284160 or op['nonzero_candidate_packets']!=(284160 if enabled else 0) or op['labels_used'] or op['target'] or abs(op['CPU_bridge_operator_seconds']-CPU)>1e-8:raise ValueError('Operator full cost or permission differs')
   operators[mode]=draws;r['operator_recount']=dict(records=2220,lambda_values=284160,loss_recount_max_error=loss_error,maximum_relative_IQ_change=maximum,CPU_bridge_operator_seconds=CPU,all_steps_sameIQpairs=True,total_L_scale=1.,view_weights=[.5,.25,.25]);sequences[mode],_=logs.jsonl(S/'training_sequence.jsonl.gz',True);rows[mode]=r
 finally:logs.RUN,logs.MODES=old_run,old_modes
 differences=sorted(k for k in set(args['B00'])|set(args['B10']) if args['B00'].get(k)!=args['B10'].get(k));costkeys=('actual_steps','parent_steps','cumulative_steps','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')
 if len(parent_sources)!=1 or sequences['B00']!=sequences['B10'] or operators['B00']!=operators['B10'] or set(differences)-{'candidate_id','output_dir'} or any(rows['B00']['training_cost'][k]!=rows['B10']['training_cost'][k] for k in costkeys):raise ValueError('Matched full source sequence/parent/draws/args/NN differ')
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',run_id=RUN,rows=rows,matched_comparisons=dict(L_U_sequence_records=4440,all_L_U_physical_sequences_equal=True,private_lambda_values_equal=284160,all_actual_NN_cost_equal=True,different_actual_arg_keys=differences,actual_arg_count=len(args['B00']),common_NoAs20_parent=True),parent_training_cost=parent_cost,IQ=0,CP=0,NN=0,updates=0,target_access=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)

def source_qualification(project):
 B=Path(project)/'runs'/RUN;rows={};metrics={}
 for mode in MODES:
  S=B/(mode+'-s'+str(SEED));q=read(S/'source_qualification/result.json');v=read(S/'source/completion.json')['final_source_metrics']
  if q['metrics']!=v or q['source_V_unique']!=27000 or q['all_RX']!=[1,3,4,6,8] or q['NN'] or q['CP'] or q['updates'] or q['head_fit_updates'] or q['teacher_NN'] or q['physical_channel_calls'] or q['U_truth_read'] or q['target'] or not q['model_state_unchanged'] or v['source_val_count']!=27000:raise ValueError('Source fullV finalreadout reuse or cost differs')
  hits=round(v['source_val_accuracy']*27000);per_rx={r:round(a*5400) for r,a in v['source_val_rx_accuracy'].items()}
  if abs(hits/27000-v['source_val_accuracy'])>1e-12 or sum(per_rx.values())!=hits or set(per_rx)!=set(map(str,(1,3,4,6,8))) or any(abs(per_rx[r]/5400-v['source_val_rx_accuracy'][r])>1e-12 for r in per_rx):raise ValueError('Exact fullV/RX count arithmetic differs')
  metrics[mode]=dict(count=27000,correct=hits,accuracy=hits/27000,per_RX_correct=per_rx);rows[mode]=dict(metrics=metrics[mode],cost={k:q[k] for k in ('CP','NN','teacher_NN','physical_channel_calls','updates')})
 a,b=metrics['B10'],metrics['B00'];delta=(a['correct']-b['correct'])*100/27000;rx={r:(a['per_RX_correct'][r]-b['per_RX_correct'][r])*100/5400 for r in a['per_RX_correct']};checks=dict(fullV_clean_gain=delta>=.5-1e-12,each_RX_protected=all(v>=-.25-1e-12 for v in rx.values()));saved=read(B/'source_matrix_decision.json')
 if saved['conditions']!=checks or saved['source_pass']!=all(checks.values()) or saved['clean_delta_pp']!=delta or saved['RX_delta_pp']!=rx or saved['target_access'] or not saved['decision_before_primary_formal_metric_read']:raise ValueError('Frozen source-only qualification differs')
 return dict(status='VERIFIED_SOURCE_QUALIFICATION',rows=rows,conditions=checks,source_pass=all(checks.values()),clean_delta_pp=delta,RX_delta_pp=rx,decision=saved['decision'],target_access=False,new_Ag_efficacy=False)
def run(project,source_only=False,cached=None):
 start=time.perf_counter();src=source(project) if cached is None else cached['source'];qualification=source_qualification(project) if cached is None else cached['source_qualification']
 if src['run_id']!=RUN or src['status']!='VERIFIED_COMPLETE_SOURCE_LOGS' or qualification['status']!='VERIFIED_SOURCE_QUALIFICATION':raise ValueError('Wrong cached source analysis scope')
 if source_only:return dict(source=src,source_qualification=qualification,IQ=0,CP=0,NN=0,updates=0,target_scores_read=False)
 from analysis.cvs_matched_noas_recount_20261018 import metrics
 import numpy as np
 B=Path(project)/'runs'/RUN;root=read(B/'result.json');done=read(B/'completion.json');summary={};indexed={};formal={};correct={};cost={};ids=None;err=0.
 if done['status']!='ARTIFACTS_COMPLETE_PENDING_ANALYSIS' or done['rows']!=2 or done['new_optimizer_updates']!=4440 or done['prediction_count']!=2352000 or not done['truth_last'] or done['head_fit_updates']:raise ValueError('All fixed rows must complete full predictions/independent scoring')
 for mode in MODES:
  P=B/(mode+'-s2026092701');d=read(P/'completion.json');pred=read(P/'prediction/complete.json');score=read(P/'scores/metrics.json')
  if d['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or d['views']!=list(VIEWS) or d['count']!=168000 or not d['truth_last'] or d['head_fit_updates'] or d['cumulative_epochs']!=30 or d['optimizer_steps']!=2220 or pred['truth_read'] or pred['query_fit'] or not pred['fixed_received_IQ'] or pred['head_fit_updates'] or score['status']!='SCORED' or not score['truth_last']:raise ValueError('Formal defaulthead/budget/received/truthlast scope differs')
  m,index,e=metrics(score['records']);err=max(err,e);summary[mode]=m;indexed[mode]=index;formal[mode]=score['records']
  with np.load(P/'scores/correctness.npz',allow_pickle=False) as a:
   if set(a.files)!={'ids',*VIEWS} or a['ids'].shape!=(168000,) or len(set(a['ids'].tolist()))!=168000 or ids is not None and not np.array_equal(ids,a['ids']):raise ValueError('Fixed opaque physical decision IDs differ')
   ids=a['ids'].copy();correct[mode]={v:a[v].copy() for v in VIEWS}
  for v in VIEWS:
   z=correct[mode][v]
   if z.dtype!=np.bool_ or z.shape!=(168000,) or int(z.sum())!=m[v]['correct']:raise ValueError('Correctness-vector count differs')
  tr=src['rows'][mode]['training_cost'];cost[mode]=dict(training=tr,prediction=pred,scorer={k:score[k] for k in ('status','truth_last','CPU_wall_seconds')},source_qualification=qualification['rows'][mode]['cost'])
 paired={};strata=[]
 for left,right in [('B10','B00')]:
  key=left+'__'+right;a=correct[left];b=correct[right];paired[key]={}
  for v in VIEWS:
   repairs=int((a[v]&~b[v]).sum());broken=int((~a[v]&b[v]).sum());joint=int((a[v]&a['clean']).sum())-int((b[v]&b['clean']).sum());shared=a['clean']&b['clean']
   p=dict(repairs=repairs,new_errors=broken,delta_accuracy_pp=(repairs-broken)/1680,joint_clean_and_scene_delta_pp=joint/1680,shared_clean_count=int(shared.sum()),repairs_when_both_clean_correct=int((a[v]&~b[v]&shared).sum()),new_errors_when_both_clean_correct=int((~a[v]&b[v]&shared).sum()))
   if repairs-broken!=summary[left][v]['correct']-summary[right][v]['correct']:raise ValueError('Paired count differs from accuracy or root')
   paired[key][v]=p
  for k,r in indexed[left].items():
   s=indexed[right][k];strata.append(dict(comparison=key,view=k[0],dimension=k[1],stratum=k[2],count=r['query_count'],left_accuracy=r['accuracy'],right_accuracy=s['accuracy'],delta_accuracy_pp=100*(r['accuracy']-s['accuracy']),left_macro_F1=r['macro_f1'],right_macro_F1=s['macro_f1']))
 return dict(status='VERIFIED',run_id=RUN,source=src,source_qualification=qualification,summary=summary,paired=paired,strata=strata,formal_metrics=formal,cost=cost,suburban_gaps_pp={k:{v:100*(m['clean']['accuracy']-m[v]['accuracy']) for v in VIEWS[1:4]} for k,m in summary.items()},CM_recounts=252,partition_recounts=42,paired_comparisons=7,max_metric_error=err,IQ=0,CP=0,NN=0,updates=0,raw_truth_reads=0,new_Ag=True,Ag_As_interaction='NOT_DEFINED_ONLY_AG_TWO_ROWS',single_seed_only=True,joint_goal_complete=False,target_feedback=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--source-only',action='store_true');parser.add_argument('--training-only',action='store_true');parser.add_argument('--source-cache');parser.add_argument('--output',required=True);a=parser.parse_args()
 if a.training_only:v=source(a.project)
 else:v=run(a.project,a.source_only,read(Path(a.source_cache)) if a.source_cache else None)
 Path(a.output).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=v.get('status','SOURCE_ONLY_COMPLETE'),IQ=0,CP=0,NN=0,updates=0,target_scores_read=not(a.training_only or a.source_only))))
