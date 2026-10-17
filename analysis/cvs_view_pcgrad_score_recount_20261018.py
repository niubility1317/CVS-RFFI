"""Independent CM/paired recount after all fixed seven-view rows;never loads IQ/models/rawtruth."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
RUN='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02'
MODES=('ORIGINAL','MEASURE','PROJECT')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def run(project):
 B=Path(project)/'runs'/RUN;terminal=read(B/'completion.json');result=read(B/'result.json');registered=read(Path(project)/'releases/cvs_F_viewPCGrad_E30_20261018_r02/registration/experiment.json');start=time.perf_counter()
 if terminal['status']!='ARTIFACTS_COMPLETE' or terminal['rows']!=3 or not terminal['truth_last'] or terminal['head_fit_updates'] or terminal['prediction_count']!=3528000 or registered['run_id']!=RUN or len(registered['rows'])!=3:raise ValueError('All fixed registered rows must complete first')
 records={};correct={};metrics={};CM=0;maxerr=0.;partitions=0;strata=[];fixed_ids=None;cost={}
 for mode in MODES:
  rid=mode+'-s2026092701';R=B/rid;done=read(R/'completion.json');p=read(R/'prediction/complete.json');score=read(R/'scores/metrics.json')
  if done['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or done['views']!=list(VIEWS) or done['count']!=168000 or not done['truth_last'] or done['head_fit_updates'] or done['cumulative_epochs']!=30 or done['optimizer_steps']!=2220 or p['truth_read'] or p['query_fit'] or not p['fixed_received_IQ'] or p['head_fit_updates'] or score['status']!='SCORED' or not score['truth_last'] or len(score['records'])!=126:raise ValueError('Defaulthead/fixedview/truthlast budget differs')
  indexed={};metrics[mode]={}
  for v in score['records']:
   key=(v['view'],v['dimension'],str(v['stratum']))
   if key in indexed:raise ValueError('Duplicate score stratum')
   indexed[key]=v;c=v['confusion']
   if len(c)!=6 or any(len(row)!=6 or any(not isinstance(x,int) or x<0 for x in row) for row in c):raise ValueError('Nonnegative6class CM required')
   rowtot=[sum(row) for row in c];coltot=[sum(c[i][j] for i in range(6)) for j in range(6)];n=sum(rowtot);ncorrect=sum(c[i][i] for i in range(6))
   if n!=v['query_count'] or not n:raise ValueError('Invalid CM size')
   calculated=dict(accuracy=ncorrect/n,macro_f1=sum(2*c[i][i]/(rowtot[i]+coltot[i]) if rowtot[i]+coltot[i] else 0. for i in range(6))/6,macro_accuracy=sum(c[i][i]/rowtot[i] for i in range(6) if rowtot[i])/sum(z>0 for z in rowtot))
   for k,x in calculated.items():
    error=abs(x-v[k]);maxerr=max(maxerr,error)
    if not math.isfinite(v[k]) or error>1e-12:raise ValueError('CM metric mismatch')
   if v['dimension']=='overall':metrics[mode][v['view']]=dict(accuracy=calculated['accuracy'],macro_F1=calculated['macro_f1'],correct=ncorrect,count=n)
   CM+=1
  for view in VIEWS:
   overall=indexed[(view,'overall','ALL')]
   if overall['query_count']!=168000:raise ValueError('Full168000 required')
   for dimension,count in [('receiver',7),('transmitter',6),('day',4)]:
    group=[v for (w,d,s),v in indexed.items() if w==view and d==dimension]
    if len(group)!=count or any(sum(v['confusion'][i][j] for v in group)!=overall['confusion'][i][j] for i in range(6) for j in range(6)):raise ValueError('Partition does not sum')
    partitions+=1
   worst=min(v['accuracy'] for (w,d,s),v in indexed.items() if w==view and d=='receiver');metrics[mode][view]['worst_RX_accuracy']=worst
   published=result['summary'][mode][view]
   if any(abs(metrics[mode][view][k]-published[k])>1e-12 for k in ('accuracy','macro_F1','worst_RX_accuracy')):raise ValueError('Published summary mismatch')
  records[mode]=indexed
  with np.load(R/'scores/correctness.npz',allow_pickle=False) as a:
   if set(a.files)!={'ids',*VIEWS} or a['ids'].shape!=(168000,) or len(set(a['ids'].tolist()))!=168000:raise ValueError('Fixed opaque IDs required')
   if fixed_ids is not None and not np.array_equal(fixed_ids,a['ids']):raise ValueError('Paired IDs differ')
   fixed_ids=a['ids'].copy();correct[mode]={v:a[v].copy() for v in VIEWS}
  for view in VIEWS:
   if correct[mode][view].dtype!=np.bool_ or correct[mode][view].shape!=(168000,) or int(correct[mode][view].sum())!=metrics[mode][view]['correct']:raise ValueError('Correctness recount differs')
  tr=read(R/'source/training_result.json');cost[mode]={k:tr[k] for k in ['actual_steps','parent_steps','cumulative_steps','parent_epochs','new_epochs','cumulative_epochs','L_forward_packets','student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets','As_training_observations','Ag_training_observations','elapsed_seconds','batch_repartition','extra_supervised_VJP_calls','gradient_projection_calls','gradient_projection_nonzero']}
  cost[mode].update(prediction_NN=p['NN_forward_packets'],prediction_wall_seconds=p['wall_seconds'],scorer_CPU_wall_seconds=score['CPU_wall_seconds'])
 paired={};criteria={}
 for left,right in [('PROJECT','ORIGINAL'),('PROJECT','MEASURE'),('MEASURE','ORIGINAL')]:
  name=left+'__'+right;paired[name]={};a=correct[left];b=correct[right]
  for view in VIEWS:
   x,y=a[view],b[view];repair=int((x&~y).sum());broken=int((~x&y).sum());lc=int(x.sum());rc=int(y.sum());delta=(repair-broken)/168000*100;lj=int((x&a['clean']).sum());rj=int((y&b['clean']).sum());shared_clean=a['clean']&b['clean']
   paired[name][view]=dict(left_correct=lc,right_correct=rc,repairs=repair,new_errors=broken,delta_accuracy_pp=delta,left_clean_and_scene=lj,right_clean_and_scene=rj,joint_clean_and_scene_delta_pp=(lj-rj)/168000*100,shared_clean_count=int(shared_clean.sum()),repairs_when_both_clean_correct=int((x&~y&shared_clean).sum()),new_errors_when_both_clean_correct=int((~x&y&shared_clean).sum()))
   if lc-rc!=repair-broken:raise ValueError('Paired net mismatch')
   if left=='PROJECT':
    old=result['paired_differences'][right][view]
    if old['repairs']!=repair or old['new_errors']!=broken or abs(old['delta_accuracy_pp']-delta)>1e-12 or abs(old['joint_clean_and_scene_delta_pp']-(lj-rj)/168000*100)>1e-12:raise ValueError('Published paired mismatch')
  if left=='PROJECT':
   p=paired[name];criteria[right]=dict(clean=p['clean']['left_correct']>=p['clean']['right_correct'],low_urban=p['practical_low_urban']['repairs']-p['practical_low_urban']['new_errors']>=840,mid_urban=p['practical_mid_urban']['left_correct']>=p['practical_mid_urban']['right_correct'],suburban=all(p[v]['repairs']-p[v]['new_errors']>=-420 for v in VIEWS[1:4]))
   if criteria[right]!=result['criteria'][right]:raise ValueError('Fixed criteria mismatch')
  for key in records[left]:
   v,dim,s=key;x=records[left][key];y=records[right][key];strata.append(dict(comparison=name,view=v,dimension=dim,stratum=s,count=x['query_count'],left_accuracy=x['accuracy'],right_accuracy=y['accuracy'],delta_pp=100*(x['accuracy']-y['accuracy']),left_macro_F1=x['macro_f1'],right_macro_F1=y['macro_f1']))
 passed=all(x for v in criteria.values() for x in v.values())
 if passed!=result['scientific_pass'] or CM!=378 or partitions!=63:raise ValueError('Final result scope mismatch')
 gaps={m:{v:100*(metrics[m]['clean']['accuracy']-metrics[m][v]['accuracy']) for v in VIEWS[1:4]} for m in MODES}
 return dict(status='VERIFIED',run_id=RUN,CM_recounts=CM,partition_recounts=partitions,paired_comparisons=21,max_metric_error=maxerr,summary=metrics,paired=paired,criteria=criteria,scientific_pass=passed,decision=result['decision'],suburban_gaps_pp=gaps,strata=strata,cost=cost,IQ=0,CP=0,NN=0,raw_truth_reads=0,new_updates=0,new_Ag=False,Ag_efficacy_proved=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);a=p.parse_args();print(json.dumps(run(a.project),allow_nan=False))
