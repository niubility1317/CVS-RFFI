"""Independent source and frozen formal recount;no IQ/checkpoint/model imports."""
import json,math,time,csv
from pathlib import Path
import numpy as np
RUN='20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SEED=2026092701;COUNT=168000
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SOURCE_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PANEL='runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
class AwaitingArtifacts(RuntimeError):pass

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows(project):
 base=Path(project)/'runs';ref=base/REFERENCE_RUN;own=base/RUN
 return [('NOAS',ref/'B00-s2026092701'),('AS_RD',ref/'B01-s2026092701'),('AS_CE',own/'AS_CE-s2026092701'),('AS_DET',own/'AS_DET-s2026092701')]

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  names=['stage20/completion.json','source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as e:raise AwaitingArtifacts(arm+'/'+name) from e
  for name in ('stage20/completion.json','source/completion.json'):
   c=found[name];expected=(20,4440) if name.startswith('stage20') else (10,2220)
   if (c['epoch'],c['optimizer_steps'])!=expected or c['target_access'] or c['config']['model_seed']!=SEED or c['config']['row_id']!=root.name or c['config']['run_id']!=(REFERENCE_RUN if arm in ('NOAS','AS_RD') else RUN):raise ValueError('Actual row/seed/budget/freeze differs')
  frozen=found['source/source_frozen.json'];q=found['source_qualification/complete.json']
  if frozen['selection']!='final_only' or frozen['target_access'] or q['target_access'] or q['NN']!=12600:raise ValueError('Fixed final30 source boundary differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed:all seven frozen predictions/independent scoring required')
  result[arm]=found
 return result

def metric(y,p):
 y=np.asarray(y);p=np.asarray(p)
 if y.ndim!=1 or p.shape!=y.shape or len(y)==0 or np.any((y<0)|(y>=6)|(p<0)|(p>=6)):raise ValueError('Nonempty sixclass decisions required')
 cm=np.bincount((y*6+p).astype(int),minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);n=int(cm.sum());correct=int(np.trace(cm))
 return dict(count=n,correct=correct,accuracy=correct/n,macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()),confusion=cm.tolist())

def stage_clean_pass(net_repairs,n):
 return int(net_repairs)*1000>=-int(n)

def source_conditions(summary):
 det=summary['AS_DET'];old=summary['AS_RD'];ce=summary['AS_CE'];no=summary['NOAS'];clean=100*(det['full_V_clean']-no['full_V_clean']);clean_old=100*(det['full_V_clean']-old['full_V_clean']);clean_ce=100*(det['full_V_clean']-ce['full_V_clean']);urban=det['mid_low_mean_pp']-old['mid_low_mean_pp'];urban_ce=det['mid_low_mean_pp']-ce['mid_low_mean_pp']
 return dict(stage_clean=clean>=-.1-1e-12,retained_As_gain=det['mid_low_mean_pp']-no['mid_low_mean_pp']>=5.,useful_vs_original=(urban>=.5 and clean_old>=-.1-1e-12) or (clean_old>=.1-1e-12 and urban>=-.1-1e-12),dropout_attribution=(urban_ce>=.5 and clean_ce>=-.1-1e-12) or (clean_ce>=.1-1e-12 and urban_ce>=-.1-1e-12))

def paired(y,control,new,cohort):
 old=np.asarray(control)==y;good=np.asarray(new)==y;repair=int((~old&good).sum());damage=int((old&~good).sum());n=len(y)
 out=dict(count=n,control_correct=int(old.sum()),new_correct=int(good.sum()),repairs=repair,new_errors=damage,net_repairs=repair-damage,accuracy_delta_pp=100*(repair-damage)/n)
 mask=np.asarray(cohort,dtype=bool);den=int(mask.sum());out.update(fixed_NOAS_clean_correct_count=den,control_retention=float(old[mask].mean()) if den else None,new_retention=float(good[mask].mean()) if den else None,cohort_repairs=int((~old&good&mask).sum()),cohort_new_errors=int((old&~good&mask).sum()))
 return out

def source_analysis(project):
 started=time.perf_counter();metadata=ensure_metadata(project,'source');project=Path(project);panel=project/PANEL
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 role=read(panel/'data_roles.json');data={};metrics=[];summary={};correctness={}
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.array_equal(x['ids'],role['physical_ids']) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Fixed original source panel/finite default logits required')
  if data and any(not np.array_equal(x[k],data['NOAS'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Paired source IDs/roles differ')
  q=metadata[arm]['source_qualification/complete.json'];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Saved source accuracy recount differs')
  data[arm]=x;m=metadata[arm]['source/completion.json']['final_source_metrics']
  if m['source_val_count']!=27000:raise ValueError('Original fullV27000 health required')
  summary[arm]=dict(full_V_clean=m['source_val_accuracy'],full_V_RX=m['source_val_rx_accuracy'],mid_low_by_draw_pp=(100*acc[:,1:]).tolist(),mid_low_mean_pp=float(100*acc[:,1:].mean()),panel_raw=metric(x['TX'],x['clean'].argmax(1)))
 for arm,x in data.items():
  y0=x['TX'];cohort=data['NOAS']['clean'].argmax(1)==y0
  obs=[('clean',-1,x['clean'],y0,x['RX'],x['day'],cohort,None,None,None)]
  for si,scene in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    ds=(0,1) if draw==-1 else (draw,)
    obs.append((scene,draw,np.concatenate([x['satellite'][j,si] for j in ds]),np.tile(y0,len(ds)),np.tile(x['RX'],len(ds)),np.tile(x['day'],len(ds)),np.tile(cohort,len(ds)),np.concatenate([physics['state'][j,si,0] for j in ds]),np.concatenate([physics['lock'][j,si,0] for j in ds]),np.concatenate([physics['signal_to_added_noise_db'][j,si,0] for j in ds])))
  for scene,draw,z,y,rx,day,mask0,state,lock,quality in obs:
   p=z.argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
   masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(v),rx==v) for v in (1,3,4,6,8)]+[('day',str(v),day==v) for v in (1,2,3)]+[('TX',str(v),y==v) for v in range(6)]
   if state is not None:
    masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]
    masks += [('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
   for dimension,stratum,mask in masks:
    if not mask.any():continue
    den=int((mask&mask0).sum());metrics.append(dict(arm=arm,scene=scene,draw=draw,dimension=dimension,stratum=stratum,**metric(y[mask],p[mask]),mean_margin_over30=float(margin[mask].mean()),fixed_NOAS_clean_correct_count=den,retention_on_fixed_NOAS=float((p==y)[mask&mask0].mean()) if den else None))
 det=summary['AS_DET'];old=summary['AS_RD'];ce=summary['AS_CE'];no=summary['NOAS'];clean=100*(det['full_V_clean']-no['full_V_clean']);clean_old=100*(det['full_V_clean']-old['full_V_clean']);clean_ce=100*(det['full_V_clean']-ce['full_V_clean']);urban=det['mid_low_mean_pp']-old['mid_low_mean_pp'];urban_ce=det['mid_low_mean_pp']-ce['mid_low_mean_pp']
 conditions=source_conditions(summary)
 saved=read(project/'runs'/RUN/'source_matrix_decision.json')
 if saved['conditions']!=conditions or saved['passed']!=all(conditions.values()):raise ValueError('Independent source decision differs')
 pairs={}
 for new,old in [('AS_RD','NOAS'),('AS_CE','NOAS'),('AS_DET','NOAS'),('AS_CE','AS_RD'),('AS_DET','AS_RD'),('AS_DET','AS_CE')]:
  key=new+'__'+old;pairs[key]={'clean':paired(data[new]['TX'],data[old]['clean'].argmax(1),data[new]['clean'].argmax(1),data['NOAS']['clean'].argmax(1)==data[new]['TX'])}
  for si,scene in enumerate(SOURCE_SCENES):
   y=np.tile(data[new]['TX'],2);p0=data[old]['satellite'][:,si].argmax(-1).reshape(-1);p1=data[new]['satellite'][:,si].argmax(-1).reshape(-1);cohort=np.tile(data['NOAS']['clean'].argmax(1)==data[new]['TX'],2);pairs[key][scene]=paired(y,p0,p1,cohort)
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,decision=dict(conditions=conditions,passed=all(conditions.values()),paired=dict(DET_minus_NOAS_clean_pp=clean,DET_minus_RD_clean_pp=clean_old,DET_minus_CE_clean_pp=clean_ce,DET_minus_RD_midlow_pp=urban,DET_minus_CE_midlow_pp=urban_ce)),source_unique_V=1800,channel_draws=2,model_seed_count=1,fullV_metrics_from_native_not_individual_prediction_recount=True,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,added_noise_ratio_is_total_SNR=False,formal_physics_labels_available=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

def formal_analysis(project,truth_path,source):
 started=time.perf_counter();metadata=ensure_metadata(project,'final')
 if source['status']!='SOURCE_ANALYZED' or source['run_id']!=RUN or source['target_scores_read']:raise ValueError('Complete independent source analysis first')
 # Only now may fixed predictions be joined to truth.
 truth=read(truth_path);results={};private={};all_metrics=[];partitions=0
 for arm,root in rows(project):
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
   if set(f.files)!={'ids',*VIEWS}:raise ValueError('Exact seven decisions required')
   ids=f['ids'].copy();pred={v:f[v].copy() for v in VIEWS}
  if len(ids)!=COUNT or len(set(ids.tolist()))!=COUNT or set(truth)!=set(ids.tolist()):raise ValueError('Exact frozen physical IDs/truth alignment required')
  y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids]);day=np.asarray([str(truth[s]['day']) for s in ids]);saved=read(root/'scoring/scores.json')
  if not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('Independent scorer boundary differs')
  if private and (not np.array_equal(ids,private['NOAS']['ids']) or not np.array_equal(y,private['NOAS']['y'])):raise ValueError('Same paired physical samples required')
  metrics=[];views={}
  for view in VIEWS:
   with np.load(root/'prediction'/(view+'_scores.npz'),allow_pickle=False) as f:
    logits=f['logits']
    if not np.array_equal(ids,f['ids']) or logits.shape!=(COUNT,6) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),pred[view]):raise ValueError('Default logits/decisions differ')
    other=logits.copy();other[np.arange(COUNT),y]=-np.inf;margin=(logits[np.arange(COUNT),y]-other.max(1))/30
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    subtotal=np.zeros((6,6),dtype=np.int64)
    for value in values:
     mask=np.ones(COUNT,dtype=bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=metric(y[mask],pred[view][mask]);old=next(r for r in saved['results'] if r['view']==view and r['dimension']==dim and r['stratum']==str(value))
     if q['confusion']!=old['confusion'] or q['count']!=old['query_count'] or abs(q['accuracy']-old['accuracy'])>1e-12 or abs(q['macro_f1']-old['macro_f1'])>1e-12:raise ValueError('Independent formal metric differs')
     metrics.append(dict(arm=arm,row_id=root.name,model_seed=SEED,view=view,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     expected=metric(y,pred[view])['confusion']
     if not np.array_equal(subtotal,expected):raise ValueError('Formal stratum partition differs')
     partitions+=1
   overall=next(q for q in metrics if q['view']==view and q['dimension']=='overall');views[view]=dict(overall,mean_true_margin_over30=float(margin.mean()),margin_quantiles_over30=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),worst_RX_accuracy=min(q['accuracy'] for q in metrics if q['view']==view and q['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=arm in ('NOAS','AS_RD'));private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
 pairs={};comparisons={};cohort=private['NOAS']['pred']['clean']==private['NOAS']['y'];stage={}
 for new,old in [('AS_RD','NOAS'),('AS_CE','NOAS'),('AS_DET','NOAS'),('AS_CE','AS_RD'),('AS_DET','AS_RD'),('AS_DET','AS_CE')]:
  key=new+'__'+old;pairs[key]={};comparisons[key]={}
  for view in VIEWS:
   pairs[key][view]=paired(private[new]['y'],private[old]['pred'][view],private[new]['pred'][view],cohort);comparisons[key][view]=dict(accuracy_delta_pp=pairs[key][view]['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[new]['views'][view]['macro_f1']-results[old]['views'][view]['macro_f1']),worst_RX_delta_pp=100*(results[new]['views'][view]['worst_RX_accuracy']-results[old]['views'][view]['worst_RX_accuracy']))
  comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]))
 for arm in ('AS_RD','AS_CE','AS_DET'):
  diff=pairs[arm+'__NOAS']['clean'];gain=diff['accuracy_delta_pp'];stage[arm]=dict(clean_delta_pp=gain,clean_absolute_0p1pp_guard_met=stage_clean_pass(diff['net_repairs'],COUNT),clean_net_repairs=diff['net_repairs'],clean_positive_observed=gain>0,positive_below_0p1pp_is_micro_observation=0<gain<.1,midurban_gain_pp=comparisons[arm+'__NOAS']['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[arm+'__NOAS']['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[arm+'__NOAS']['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 value=dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,comparisons=comparisons,stage=stage,suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,source_decision=source['decision'],independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,target_feedback=False,goal_complete=False,wall_seconds=time.perf_counter()-started)
 return value,private

def write_csv(p,records):
 fields=list(dict.fromkeys(k for r in records for k in r))
 with Path(p).open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
