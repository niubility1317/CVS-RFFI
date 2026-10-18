"""Zero-NN staged source/log/formal recount of fixed-budget U practical student input.
Saved metadata and complete logs only; no model/checkpoint/IQ loading or launch.
Source and final are separate invocations; source mode never takes a truth path.
"""
import argparse,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_analysis_20261018 as old
RUN='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02'
TRAINING_RELEASE='cvs_FcrossRX_UPractical_E30_20261019_r02'
FAILED_RUN='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m2-r01'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','MATCHED_NOAS30','UTEACHER30','UPRACTICAL30')
PAIRS=(('UPRACTICAL30','UTEACHER30'),('UPRACTICAL30','MATCHED_NOAS30'),('UTEACHER30','MATCHED_NOAS30'),('MATCHED_NOAS30','NOAS30'),('UTEACHER30','AS30'),('UPRACTICAL30','AS30'),('MATCHED_NOAS30','AS30'),('UTEACHER30','NOAS30'),('UPRACTICAL30','NOAS30'),('AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values


def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('MATCHED_NOAS30',p/RUN/'N01-s2026092701'),('UTEACHER30',p/RUN/'C01-s2026092701'),('UPRACTICAL30',p/RUN/'S01-s2026092701')]

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json']
  if (c['epoch'],c['optimizer_steps'])!=(10,2220) or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=SEED or c['config']['row_id']!=root.name or c['config']['run_id']!=root.parent.name:raise ValueError('Actual fixed row/E30/defaulthead differs')
  if c.get('cumulative_epochs',30)!=30 or c.get('cumulative_optimizer_steps',6660)!=6660:raise ValueError('Cumulative E30 differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('training_updates',0) or q.get('head_fit_updates',0):raise ValueError('Frozen final/source readout differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all5 logical rows have all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def source_conditions(summary,quality=None):
 s=summary['UPRACTICAL30'];c=summary['UTEACHER30'];n=summary['MATCHED_NOAS30'];clean=float(100*(s['full_V_clean']-n['full_V_clean']));gain=float(100*(s['mid_low_mean']-c['mid_low_mean']))
 for label,value in [('fullV S-matchedNoAs',clean),('pooled two-draw equal midlow S-C',gain)]:logs.finite(value,label)
 checks=dict(fullV_clean_vs_matchedNoAs_within_0p1pp=bool(clean>=-.1-1e-12),midlow_vs_same_budget_control_gain_at_least_0p5pp=bool(gain>=.5-1e-12));passed=bool(all(checks.values()))
 return dict(checks=checks,source_criteria_pass=passed,E50_source_eligible=passed,fixed_recipe_closed=bool(not passed),fullV_clean_S_minus_matchedNoAs_pp=clean,midlow_S_minus_C_pp=gain,only_two_fixed_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False,automatic_extension=False,channel_draws_are_not_model_seeds=True)


def source_analysis(project):
 started=time.perf_counter();meta=ensure_metadata(project,'source');data={};summary={};metrics=[];quality={}
 panel=Path(project)/old.stable.PANEL;panel_roles=read(panel/'data_roles.json')
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as z:physics={k:z[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 for key_,array in physics.items():
  if array.shape!=(2,3,11,1800):raise ValueError('Existing source physics must retain two draws/three scenes/full index0/1800 IDs')
 if not np.isfinite(physics['signal_to_added_noise_db'][:,:,0]).all():raise ValueError('Source actual added-noise ratio finite full observations required')
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
  if not np.array_equal(x['ids'],np.asarray(panel_roles['physical_ids'])):raise ValueError('Existing source physics physical ordering differs')
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite fixed source logits absent')
  if data and any(not np.array_equal(x[k],data['NOAS30']['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source physical panel required')
  if len({tuple(np.atleast_1d(v).tolist()) for v in x['ids']})!=1800:raise ValueError('Source1800 physical IDs must be unique')
  q=meta[arm]['source_qualification/complete.json'];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source readout differs')
  native=meta[arm]['source/completion.json']['final_source_metrics']
  if native['source_val_count']!=27000:raise ValueError('Native fullV count differs')
  pred=dict(clean=x['clean'].argmax(-1));quality[arm]={}
  for i,v in enumerate(SOURCE_SCENES):pred[v]=x['satellite'][:,i].argmax(-1).reshape(-1)
  data[arm]=dict(x=x,pred=pred);panel_summary={}
  observations=[('clean',-1,x['clean'],x['TX'],x['RX'],x['day'],None,None,None)]
  for si,v in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    draws=(0,1) if draw==-1 else (draw,)
    observations.append((v,draw,np.concatenate([x['satellite'][j,si] for j in draws]),np.tile(x['TX'],len(draws)),np.tile(x['RX'],len(draws)),np.tile(x['day'],len(draws)),np.concatenate([physics['state'][j,si,0] for j in draws]),np.concatenate([physics['lock'][j,si,0] for j in draws]),np.concatenate([physics['signal_to_added_noise_db'][j,si,0] for j in draws])))
  for view,draw,z,y,rx,day,state,lock,ratio in observations:
   p=z.argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
   masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(v),rx==v) for v in sorted(set(rx))]+[('day',str(v),day==v) for v in sorted(set(day))]+[('TX',str(v),y==v) for v in range(6)]
   if ratio is not None:
    masks += [('added_noise_ratio',label,(ratio>=lo)&(ratio<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    masks += [('quality_group','hard_lt8',ratio<8),('quality_group','easy_ge8',ratio>=8)]
    masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]
   for dim,stratum,mask in masks:
    if not mask.any():continue
    m=metric(y[mask],p[mask]);metrics.append(dict(arm=arm,scene=view,draw=draw,dimension=dim,stratum=stratum,mean_trueclass_margin_over30=float(margin[mask].mean()),**m))
    if dim=='overall' and draw==-1:panel_summary[view]=dict(accuracy=m['accuracy'],macro_F1=m['macro_f1'],worst_RX=min(float((p[rx==r]==y[rx==r]).mean()) for r in np.unique(rx)),mean_trueclass_margin_over30=float(margin.mean()))
    if dim=='quality_group' and draw==-1:quality[arm].setdefault(view,{})[stratum]=dict(count=m['count'],accuracy=m['accuracy'],macro_F1=m['macro_f1'],mean_trueclass_margin_over30=float(margin[mask].mean()))
   if ratio is not None and draw==-1 and set(quality[arm].get(view,{}))!={'hard_lt8','easy_ge8'}:raise ValueError('Fixed source hard/easy populations required for the declared risk comparison')
  summary[arm]=dict(full_V_clean=native['source_val_accuracy'],full_V_RX_accuracy=native['source_val_rx_accuracy'],panel=panel_summary,mid_low_mean=float(np.mean([panel_summary[v]['accuracy'] for v in SOURCE_SCENES[1:]])),mid_low_by_draw_accuracy=acc[:,1:].tolist(),source_readout_NN=q['NN'])
 pairs={};quality_pairs={};cohort=data['NOAS30']['pred']['clean']==data['NOAS30']['x']['TX']
 for a,b in PAIRS:
  key_=a+'__'+b;pairs[key_]={};quality_pairs[key_]={}
  for v in ('clean',*SOURCE_SCENES):
   y=data[a]['x']['TX'] if v=='clean' else np.tile(data[a]['x']['TX'],2);co=cohort if v=='clean' else np.tile(cohort,2);pairs[key_][v]=paired(y,data[b]['pred'][v],data[a]['pred'][v],co)
   if v!='clean':
    ratio=physics['signal_to_added_noise_db'][:,SOURCE_SCENES.index(v),0].reshape(-1);quality_pairs[key_][v]={}
    for group,mask in [('hard_lt8',ratio<8),('easy_ge8',ratio>=8)]:quality_pairs[key_][v][group]=paired(y[mask],data[b]['pred'][v][mask],data[a]['pred'][v][mask],co[mask])
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,quality_groups=quality,quality_paired=quality_pairs,paired=pairs,decision=source_conditions(summary,quality),source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=SEED,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,added_noise_ratio_is_total_SNR=False,hard_easy_accuracy_is_equal_scene_not_pooled_count_weighted=True,margin_uses_existing_logits_only=True,source_physics_full_condition_index=0,formal_physics_tags_available=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

def actual_cost(value):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if any(type(value.get(k)) is not int or value[k]<0 for k in keys):raise ValueError('Actual native four counters required')
 counters={k:value[k] for k in keys};total=sum(counters.values())
 if value.get('native_NN',total)!=total:raise ValueError('Saved native sum differs from actual counters')
 return dict(**counters,native_training_and_validation_NN=total)

def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base differs'),logs._near(r['total_L'],total,'Original full1.25L differs')]
 for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+k],r[k],'Actual full6L native scalar differs'))
 return max(errors)

def formal_analysis(project,truth_path,source):
 started=time.perf_counter();metadata=ensure_metadata(project,'final')
 if source['status']!='SOURCE_ANALYZED' or source['run_id']!=RUN or source['target_scores_read']:raise ValueError('Independent source analysis must finish first')
 truth=read(truth_path);results={};private={};all_metrics=[];partitions=0
 for arm,root in rows(project):
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
   if set(f.files)!={'ids',*VIEWS}:raise ValueError('Exact seven fixed decisions required')
   ids=f['ids'].copy();pred={v:f[v].copy() for v in VIEWS}
  if len(ids)!=COUNT or len(set(ids.tolist()))!=COUNT or set(truth)!=set(ids.tolist()):raise ValueError('Exact frozen physical ID/truth alignment required')
  y=np.asarray([truth[z]['label'] for z in ids]);rx=np.asarray([str(truth[z]['receiver']) for z in ids]);day=np.asarray([str(truth[z]['day']) for z in ids]);saved=read(root/'scoring/scores.json')
  if not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('Independent original scorer boundary differs')
  if private and (not np.array_equal(ids,private['NOAS30']['ids']) or not np.array_equal(y,private['NOAS30']['y'])):raise ValueError('All rows must share physical pairing')
  metrics=[];views={}
  for v in VIEWS:
   with np.load(root/'prediction'/(v+'_scores.npz'),allow_pickle=False) as f:
    z=f['logits']
    if not np.array_equal(ids,f['ids']) or z.shape!=(COUNT,6) or not np.isfinite(z).all() or not np.array_equal(z.argmax(1),pred[v]):raise ValueError('Default logits/fixed decisions differ')
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    subtotal=np.zeros((6,6),dtype=np.int64)
    for value in values:
     mask=np.ones(COUNT,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=metric(y[mask],pred[v][mask]);original=next(r for r in saved['results'] if r['view']==v and r['dimension']==dim and r['stratum']==str(value))
     if q['confusion']!=original['confusion'] or q['count']!=original['query_count'] or abs(q['accuracy']-original['accuracy'])>1e-12 or abs(q['macro_f1']-original['macro_f1'])>1e-12:raise ValueError('Independent CM/F1 recount differs')
     metrics.append(dict(arm=arm,row_id=root.name,model_seed=SEED,view=v,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     if not np.array_equal(subtotal,metric(y,pred[v])['confusion']):raise ValueError('Formal RX/day/TX partitions differ')
     partitions+=1
   q=next(r for r in metrics if r['view']==v and r['dimension']=='overall');views[v]=dict(q,worst_RX_accuracy=min(r['accuracy'] for r in metrics if r['view']==v and r['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=arm in CONDITIONS[:2]);private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
 paired_strata=[];pairs={};comparisons={};cohort=private['NOAS30']['pred']['clean']==private['NOAS30']['y']
 for a,b in PAIRS:
  key=a+'__'+b;pairs[key]={};comparisons[key]={}
  for v in VIEWS:
   p=paired(private[a]['y'],private[b]['pred'][v],private[a]['pred'][v],cohort);pairs[key][v]=p;comparisons[key][v]=dict(accuracy_delta_pp=p['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[a]['views'][v]['macro_f1']-results[b]['views'][v]['macro_f1']),worst_RX_delta_pp=100*(results[a]['views'][v]['worst_RX_accuracy']-results[b]['views'][v]['worst_RX_accuracy']))
  for v in VIEWS:
   for dim,values in [('receiver',sorted(set(private[a]['rx']))),('day',sorted(set(private[a]['day']))),('transmitter',range(6))]:
    for value in values:
     mask=private[a]['rx']==value if dim=='receiver' else private[a]['day']==value if dim=='day' else private[a]['y']==value
     paired_strata.append(dict(comparison=key,view=v,dimension=dim,stratum=str(value),**paired(private[a]['y'][mask],private[b]['pred'][v][mask],private[a]['pred'][v][mask],cohort[mask])))
  comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]))
 U_view_increment={v:comparisons['UPRACTICAL30__UTEACHER30'][v]['accuracy_delta_pp'] for v in VIEWS};U_view_increment['urban_mean']=comparisons['UPRACTICAL30__UTEACHER30']['urban_mean_accuracy_delta_pp']
 key='UPRACTICAL30__MATCHED_NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 stages={a:dict(clean_delta_pp=pairs[a+'__MATCHED_NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs[a+'__MATCHED_NOAS30']['clean']['net_repairs'],COUNT),lowurban_gain_pp=comparisons[a+'__MATCHED_NOAS30']['practical_low_urban']['accuracy_delta_pp'],midurban_gain_pp=comparisons[a+'__MATCHED_NOAS30']['practical_mid_urban']['accuracy_delta_pp'],repeatable_clean_gain_proven=False) for a in CONDITIONS[3:]}
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,stages=stages,source_decision=source['decision'],U_view_effect_pp=U_view_increment,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['UPRACTICAL30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['UPRACTICAL30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def metadata_paths(project,stage):
 paths=[]
 for _,root in rows(project):
  paths += [root/'source/completion.json',root/'source/source_frozen.json',root/'source_qualification/complete.json']
  if stage=='FINAL':paths += [root/'prediction/complete.json',root/'scoring/complete.json',root/'complete.json']
 return paths

def ready(paths):
 try:
  for path in paths:read(path)
 except (FileNotFoundError,json.JSONDecodeError):return False
 return True

def owned_handles(project,release):
 base=Path(project)/'runs'/RUN;ids=[];unknown=False
 for name in ('dispatcher.json','launch.json'):
  try:
   r=read(base/name);ids += [r['pid']] if name=='dispatcher.json' else [v['pid'] for v in r['rows']]
  except FileNotFoundError:pass
  except json.JSONDecodeError:unknown=True
 live=[]
 for pid in sorted(set(ids)):
  p=Path('/proc')/str(pid)
  try:
   argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_u_practical30/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  base=Path(project)/'runs'/RUN
  if (base/'failure.json').is_file():raise RuntimeError('Owned technical failure preserved')
  for arm,own in rows(project):
   if arm in CONDITIONS[2:] and (own/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Own source readout failed;formal closure remains with original runner')
  live,unknown=probe(project,release)
  if not live and not unknown:raise AwaitingArtifacts('Owned process handles absent before '+stage+' artifacts complete')
  write(out/'state.json',dict(status='WAIT_'+stage,live_owned_PIDs=live,observation_unknown=unknown,truth_opened=False,NN=0));sleep(15)

U_TIMERS=('original_strong_cpu_wall_seconds','selection_cpu_wall_seconds','simulation_adapter_cpu_wall_seconds','replacement_copy_cpu_wall_seconds','route_cpu_wall_seconds','logging_cpu_wall_seconds')

def temporal_mask(pseudo,conf,ids):
 result=np.zeros(len(pseudo),bool);groups=defaultdict(list)
 for i,z in enumerate(ids):groups[tuple(z[:3])].append(i)
 for positions in groups.values():
  ii=np.asarray(positions);p=pseudo[ii];c=conf[ii];sig=ids[ii,3];base=ids[ii,4]
  pair=(p[:,None]==p[None,:])&(c[:,None]>=.8)&(c[None,:]>=.8)&(np.abs(sig[:,None]-sig[None,:])<=2)&(np.abs(base[:,None]-base[None,:])<=2);np.fill_diagonal(pair,False);result[ii]=pair.any(1)
 return result

def check_u_step(q,arm):
 mode='QUARTER_PRACTICAL' if arm=='UPRACTICAL30' else 'GAUSSIAN_CONTROL';i=q['step'];n=124 if i%222==0 else 256;ep=(i-1)//222+1
 if q['mode']!=mode or (q['epoch'],q['cumulative_step'],q['cumulative_epoch'],q['U_packets'],q['selected_slots'],q['student_passes'],q['teacher_NN'],q['student_NN'])!=(ep,i+4440,ep+20,n,n//4,1,n,n):raise ValueError('Original U one-student/one-teacher quarter budget differs')
 positions=q['selected_positions'];selected=np.zeros(n,bool)
 if positions!=sorted(positions) or len(set(positions))!=n//4 or any(type(z) is not int or not 0<=z<n for z in positions):raise ValueError('Quarter original U positions without replacement required')
 selected[positions]=True;as_mask=selected if mode=='QUARTER_PRACTICAL' else np.zeros(n,bool);gaussian=np.flatnonzero(~as_mask).tolist()
 if q['native_losses_preserved'] is not True or q['native_feature_adjustment'] is not None or q['native_hook']!='model._source_unlabeled_feature_hook' or q['native_optimizer_step']!=i:raise ValueError('Actual native feature hook must preserve existing pseudo and entropy losses without adjustment')
 if q['Gaussian_positions']!=gaussian or q['As_packets']!=int(as_mask.sum()) or q['sample_id_fields']!=['eq_i','rx_i','day_i','sig_i','base_index'] or any(q[k] for k in ('U_truth_available','ordinary_before_practical','extra_NN','extra_CE_operator_calls','global_RNG_used_for_routing','selection_uses_TX_or_pseudo','domain_gate','strong_agreement','U_truth_used','extra_teacher')) or not all(q[k] for k in ('original_strong_before_overwrite','teacher_qualification_before_student_view','selection_without_replacement','teacher_raw_same_input','teacher_eval_no_grad','teacher_qualification_completed_before_strong','entropy_all_U','entropy_includes_unqualified','native_losses_preserved')):raise ValueError('Teacher-only qualification/hiddenU/input-routing boundary differs')
 if q['C_returns_original_strong_object']!=(mode=='GAUSSIAN_CONTROL') or q['entropy_includes_As']!=(mode=='QUARTER_PRACTICAL') or (q['original_strong_draw_packets'],q['original_strong_std'],q['CE_label_smoothing'],q['native_global_L_label_smoothing'],q['lambda_u'],q['lambda_entropy'],q['entropy_denominator'])!=(n,.015,0.,.01,.16,.01,n):raise ValueError('Native all-slot Gaussian draw/U CE/entropy weights differ')
 if q['threshold_policy']!=dict(mode='rx_day_quantile',quantile=.86,tau_min=.92,tau_max=.97) or q['temporal_policy']!=dict(mode='batch_neighbor',window=2,minimum_confidence=.8):raise ValueError('Original teacher confidence/temporal policy must remain fixed')
 ids=np.asarray(q['sample_ids']);rx=np.asarray(q['RX']);day=np.asarray(q['day']);eq=np.asarray(q['eq']);dom=np.asarray(q['domains']);pseudo=np.asarray(q['teacher_pseudo']);conf=np.asarray(q['teacher_conf'],float);z=np.asarray(q['student_logits'],float)
 if ids.shape!=(n,5) or ids.dtype.kind not in 'iu' or z.shape!=(n,6) or not np.isfinite(z).all() or any(v.shape!=(n,) for v in (rx,day,eq,dom,pseudo,conf)) or not np.array_equal(ids[:,0],eq) or not np.array_equal(ids[:,1],rx) or not np.array_equal(ids[:,2],day) or not np.isin(rx,[1,3,4,6,8]).all() or not np.isin(day,[1,2,3]).all() or not np.all(eq==1) or not np.isin(dom,np.arange(15)).all() or not np.isin(pseudo,np.arange(6)).all() or not np.isfinite(conf).all() or np.any((conf<0)|(conf>1)):raise ValueError('Original U metadata/logits/teacher pseudo shapes and roles differ')
 masks={k:np.asarray(q[k]) for k in ('conf_mask','temporal_mask','eligibility')}
 if any(v.shape!=(n,) or v.dtype.kind!='b' for v in masks.values()) or not np.array_equal(masks['temporal_mask'],temporal_mask(pseudo,conf,ids)) or not np.array_equal(masks['eligibility'],masks['conf_mask']&masks['temporal_mask']):raise ValueError('Teacher-only eligibility must be original confidence AND temporal,independent of student')
 if np.any(masks['conf_mask']&(conf<.92)) or np.any((~masks['conf_mask'])&(conf>=float(np.float32(.97)))):raise ValueError('Logged confidence selection violates original clamped thresholds')
 e=masks['eligibility'];den=int(e.sum())
 if q['CE_denominator']!=den or q['eligible_selected_slots']!=int(e[selected].sum()) or q['eligible_Gaussian_slots']!=int(e[~as_mask].sum()) or q['eligible_As_slots']!=int(e[as_mask].sum()):raise ValueError('CE selected-count denominator and route qualification differ')
 shifted=z-z.max(1,keepdims=True);lp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True));ce=-lp[np.arange(n),pseudo];entropy=-(np.exp(lp)*np.log(np.maximum(np.exp(lp),1e-8))).sum(1);mean_ce=float(ce[e].mean()) if den else 0.;mean_entropy=float(entropy.mean());errors=[logs._near(q['native_loss_u'],mean_ce,'Native U selected-mean CE differs'),logs._near(q['native_loss_entropy'],mean_entropy,'Native entropy must average all U slots')]
 physics=q['physics']
 if len(physics)!=int(as_mask.sum()) or [v['selected_position'] for v in physics]!=(positions if mode=='QUARTER_PRACTICAL' else []) or q['scene_counts']!=dict(Counter(v['scene'] for v in physics)):raise ValueError('Actual As U physics count/order differs')
 for v in physics:
  pos=v['selected_position']
  if list(v['sample_id'])!=ids[pos].tolist() or v['scene'] not in SOURCE_SCENES or v['state'] not in ('LOS','PARTIAL','BLOCKED') or v['lock'] not in ('locked','degraded','unlocked'):raise ValueError('Same raw physical U and complete original practical observation required')
 if any(not isinstance(q[k],(int,float)) or not np.isfinite(q[k]) or q[k]<0 for k in U_TIMERS):raise ValueError('Actual host timing fields required;not CUDA events')
 return dict(maximum_error=max(errors),teacher_pseudo=pseudo,teacher_conf=conf,eligibility=e,As_mask=as_mask,selected=selected,ids=ids,domains=dom,per_CE=ce,per_entropy=entropy,U_CE=mean_ce,U_entropy=mean_entropy,accepted_student_disagreement=int((e&(z.argmax(1)!=pseudo)).sum()),physics=physics)


def parse_new_logs(root,arm):
 b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];p=cfg['u_practical'];sat=arm=='UPRACTICAL30';as_l=arm!='MATCHED_NOAS30';ref='B01-s2026092701' if as_l else 'B00-s2026092701';mode='QUARTER_PRACTICAL' if sat else 'GAUSSIAN_CONTROL'
 if cfg['reference_row_id']!=ref or cfg['crossview']['mode']!='ORD128' or cfg['As']!=as_l or (p['mode'],p['slot_fraction'],p['U_packets'],p['U_As_packets'],p['selected_slots'],p['Gaussian_std'],p['lambda_unlabeled'],p['lambda_unlabeled_entropy'])!=(mode,.25,567000,141750 if sat else 0,141750,.015,.16,.01) or any(p[k] for k in ('pseudo_strong_agreement','pseudo_domain_gate','U_truth_used','extra_NN','new_losses','L_changed','formal_test_changed','ordinary_before_practical','teacher_trajectory_equality_claim')) or not all(p[k] for k in ('original_strong_draw_all_packets','teacher_only_eligibility','pseudo_temporal_gate','teacher_quantile_unchanged','batch_neighbor_unchanged','teacher_qualification_before_student_view')) or p['pseudo_temporal_mode']!='batch_neighbor':raise ValueError('Fixed matched teacher-only U input recipe differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['parent_epochs'],result['parent_optimizer_steps'],result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(20,4440,30,6660):raise ValueError('Strict corresponding own20+10 budget required')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','Ag_packets')) or (result['L_NN'],result['L_As'],result['U_As'],result['total_L_loss_scale'],result['EMA_local_successful_updates'],result['EMA_cumulative_updates'],result['U_steps'],result['U_packets'],result['U_student_NN'])!=(1704960,142080 if as_l else 0,141750 if sat else 0,1.25,2220,6660,2220,567000,567000):raise ValueError('Actual L/U/EMA and raw-practical counts differ')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own stage20 EMA' or ema['reference_row_id']!=ref or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Strict corresponding own20 EMA/reset required')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);counts=Counter();timers=Counter();sums=defaultdict(Counter);maxerr=0.;usteps=[];ordinary_mse=[];domain_map={};rxday=defaultdict(Counter);pseudo_counts=defaultdict(lambda:np.zeros(6,np.int64));physics_counts=Counter();unique_as=set()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','u_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(z is None for z in items):raise logs.IncompleteTrainingLogs('New UPractical tail stream prefix')
  if i>STEPS:raise ValueError('Extra updates outside fixed E30 tail')
  r,s,n,u,q=items;ep=(i-1)//222+1;m=n['metrics'];logs.scan_numbers(items)
  if any(z['step']!=i for z in items) or any(z['cumulative_step']!=4440+i for z in (r,s,u,q)) or n['epoch']!=ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous local/cumulative native steps required')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'],r['Ag_mode'])!=(6,64 if as_l else 0,0,128,'NONE') or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Original supervised L6/RDrop input boundary differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw shape slot differs')
  logs._near(r['SupCon_weight'],.1,'Original ORD128 coefficient differs',1e-12);anchors=logs.relation(r,True);maxerr=max(maxerr,check_native_L(r,s,m));v=check_u_step(q,arm);maxerr=max(maxerr,v['maximum_error'])
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=(64 if as_l else 0) or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')) or m['train/pseudo_total']!=q['U_packets'] or m['train/pseudo_selected']!=q['CE_denominator']:raise ValueError('Actual native U hidden/qualification/NN metrics differ')
  maxerr=max(maxerr,logs._near(m['train/loss'],r['total_L']+.16*v['U_CE']+.01*v['U_entropy'],'Actual total must be L plus original .16U CE and .01allU entropy'))
  for rx,day,domain in zip(q['RX'],q['day'],q['domains']):
   key=(rx,day)
   if key in domain_map and domain_map[key]!=domain:raise ValueError('Original RX/day domain mapping changed')
   domain_map[key]=domain
  for j in range(q['U_packets']):
   route='As' if v['As_mask'][j] else 'Gaussian';key=(q['RX'][j],q['day'][j],route);rxday[key].update(count=1,qualified=int(v['eligibility'][j]));pseudo_counts[route][v['teacher_pseudo'][j]]+=1
   if v['eligibility'][j]:pseudo_counts[route+'_qualified'][v['teacher_pseudo'][j]]+=1
  for obs in v['physics']:
   pos=obs['selected_position'];ratio=obs['signal_to_added_noise_db'];bin_='unrecorded' if ratio is None else '<3' if ratio<3 else '3-8' if ratio<8 else '8-15' if ratio<15 else '>=15';key=(q['RX'][pos],q['day'][pos],obs['scene'],obs['state'],obs['lock'],bin_);physics_counts[key]+=1;unique_as.add(tuple(obs['sample_id']))
  counts.update(steps=1,L_NN=768,L_As=64 if as_l else 0,U_packets=q['U_packets'],U_selected_slots=q['selected_slots'],U_As=q['As_packets'],qualified=q['CE_denominator'],qualified_As=q['eligible_As_slots'],qualified_Gaussian=q['eligible_Gaussian_slots'],accepted_student_disagreement=v['accepted_student_disagreement'],SupCon_valid_anchor_terms=anchors);timers.update({k:q[k] for k in U_TIMERS});usteps.append(q);ordinary_mse.append(r['ordinary_raw_MSE']);sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update(U_CE=v['U_CE'],U_entropy=v['U_entropy'],qualified=q['CE_denominator'],U_packets=q['U_packets'],U_As=q['As_packets'],native_total=m['train/loss'])
 if counts['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New UPractical tail final update absent')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U physical batch sizes differ')
  for z in ids:
   if len(z)!=5 or z[0]!=1 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source-only physical role differs')
   unique[role].add(tuple(z))
  if role=='U' and (batches['U']>=STEPS or ids!=usteps[batches['U']]['sample_ids']):raise ValueError('Actual U physical sequence and routed logits not aligned')
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or {k:len(v) for k,v in unique.items()}!=dict(L=6300,U=56700) or unique['L']&unique['U'] or not unique_as<=unique['U'] or counts['U_packets']!=567000 or counts['U_selected_slots']!=141750 or counts['U_As']!=(141750 if sat else 0) or len(domain_map)!=15 or len(set(domain_map.values()))!=15:raise ValueError('Complete unchanged L6300/hiddenU56700 and quarter U exposure required')
 if (cost['student_train_NN'],cost['student_eval_NN'],cost['teacher_train_NN'],cost['teacher_eval_NN'])!=(2271960,324002,0,567000):raise ValueError('Actual native four NN counters differ from one U student/teacher')
 actual=result['u_practical']
 if (actual['mode'],actual['calls'],actual['U_packets'],actual['selected_slots'],actual['As_packets'],actual['qualified_packets'],actual['qualified_As_packets'],actual['qualified_Gaussian_packets'],actual['original_strong_draw_packets'])!=(mode,2220,567000,141750,141750 if sat else 0,counts['qualified'],counts['qualified_As'],counts['qualified_Gaussian'],567000) or not actual['teacher_only_eligibility'] or actual['pseudo_strong_agreement'] or actual['extra_NN'] or actual['extra_CE_operator_calls'] or actual['U_truth_used'] or actual['teacher_trajectory_equality_claim'] or (actual['lambda_u'],actual['lambda_entropy'])!=(.16,.01):raise ValueError('Producer full U summary differs from complete per-step records')
 expected=[dict(RX=k[0],day=k[1],route=k[2],count=z['count'],qualified=z['qualified']) for k,z in sorted(rxday.items())]
 if actual['RX_day_route_exposure']!=expected or actual['scene_counts']!=dict(Counter({scene:sum(n for key,n in physics_counts.items() if key[2]==scene) for scene in set(k[2] for k in physics_counts)})):raise ValueError('Actual RX/day/route/scene coverage differs')
 for key in U_TIMERS:logs._near(actual['timing'][key],timers[key],'Actual U CPU wall sum differs: '+key,1e-12)
 ordinary=read(b/'ordinary_augmentation_actual.json');args=read(b/'actual_native_args.json');args=args.get('args',args)
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ'] or ordinary['U_DataAugmentation_used'] or args['label_smoothing']!=.01 or not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher'] or args['pseudo_strong_agreement'] or args['pseudo_domain_gate'] or not args['pseudo_temporal_gate'] or args['pseudo_temporal_mode']!='batch_neighbor':raise ValueError('Original ordinary/U teacher flags differ from shared teacher-only qualification')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed stdout')
 curves=[dict(epoch=z['epoch']+20,local_epoch=z['epoch'],**{k:v/222 for k,v in sums[z['epoch']].items()},**{k:z[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for z in epochs]
 summary=dict(mode=mode,teacher_qualification_policy=dict(threshold_policy=usteps[0]['threshold_policy'],temporal_policy=usteps[0]['temporal_policy'],domain_gate=False,strong_agreement=False,qualification_before_student_input=True),RX_day_route_exposure=expected,pseudo_counts={k:v.tolist() for k,v in pseudo_counts.items()},qualified_packets=counts['qualified'],qualified_As_packets=counts['qualified_As'],qualified_Gaussian_packets=counts['qualified_Gaussian'],qualified_rate=counts['qualified']/567000,accepted_student_disagreement=counts['accepted_student_disagreement'],U_As=counts['U_As'],unique_U_As_physical=len(unique_as),physics_strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],added_noise_bin=k[5],count=n) for k,n in sorted(physics_counts.items())],CE_normalization='mean across teacher-qualified U slots;zero if none',entropy_normalization='mean across all U slots with probability clamp1e-8',loss_weights=dict(U_CE=.16,U_entropy=.01),timing=dict(timers),timing_note=actual['timing_note'],extra_NN=0,extra_CE_operator_calls=0,U_truth_used=False,teacher_student_agreement_is_not_accuracy=True,teacher_trajectory_equality_claim=False,no_raw_physical_IDs=True)
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,L_As=142080 if as_l else 0),counts=dict(counts),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},u_practical=summary,EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660),epoch_curves=curves,maximum_loss_decomposition_error=maxerr,native_telemetry=telemetry,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=ordinary_mse)

def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Complete existing original30 audit required')
 result={};lookup=dict(rows(project));parents={};comparability={};unique_parents={}
 for name in CONDITIONS[:2]:
  r=prior['rows'][name]
  if r['cost']['new_updates']!=6660 or r['cost']['new_epochs']!=30 or len(r['epoch_curves'])!=30:raise ValueError('Original30 saved full audit schema differs')
  result[name]=dict(reused=True,cost=r['cost'],epoch_curves=r['epoch_curves'],evidence='Existing complete original30 audit reused;old streams not rescanned')
 for name in CONDITIONS[2:]:
  ref='NOAS30' if name=='MATCHED_NOAS30' else 'AS30';parent_result=read(lookup[ref]/'stage20/result.json');pcost=actual_cost(parent_result);candidate,private=parse_new_logs(lookup[name],name);candidate['reused']=False;result[name]=candidate;parents[name]=dict(reference_condition=ref,cost=pcost,epochs=20,updates=4440,wall_seconds=parent_result['wall_seconds'],reused=True);unique_parents[ref]=pcost
  args=read(lookup[ref]/'source/actual_native_args.json');args=args.get('args',args);oldcost=result[ref]['cost'];oldtail={k:oldcost[k]-pcost[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};comparability[name]=dict(reference_condition=ref,old_step_logs_rescanned=False,old_tail_four_counters=oldtail,new_tail_four_counters=candidate['cost'],four_counters_equal=all(oldtail[k]==candidate['cost'][k] for k in oldtail),different_actual_arg_keys=sorted(k for k in set(args)|set(private['args']) if args.get(k)!=private['args'].get(k)),common_teacher_qualification_path_unchanged=True,student_strong_agreement_removed_in_all_three_new_rows=True,common_U_or_full_IQ_trajectory_not_claimed=True)
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parents,unique_parent20_costs=unique_parents,new_model_rows=3,new_model_epochs=30,new_updates=6660,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparability,shared_parent_count=2,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

def _current_cost_summary(project,training):
 metadata=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];q=metadata[arm]['source_qualification/complete.json'];p=metadata[arm]['prediction/complete.json'];native=c['native_training_and_validation_NN'];out[arm]=dict(student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],native_NN=native,source_readout_NN=q['NN'],source_readout_wall_seconds=q.get('wall_seconds'),formal_prediction_NN=p['actual_NN'],formal_prediction_wall_seconds=p['wall_seconds'],native_wall_seconds=c['wall_seconds'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],reused=arm in CONDITIONS[:2])
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query'] or smoke['optimizer_updates'] or real['optimizer_updates'] or smoke['real_IQ'] or real['real_IQ']:raise ValueError('Actual shared no-query smoke required')
 if len(smoke['records'])!=3 or {r['row_id'] for r in smoke['records']}!={p.name for a,p in rows(project) if a in CONDITIONS[2:]} or sum(r['NN'] for r in smoke['records'])!=smoke['NN'] or smoke['total_NN']!=smoke['NN']+real['NN'] or smoke['real_checkpoint_NN']!=real['NN'] or len(real['records'])!=2 or len({r['reference_row_id'] for r in real['records']})!=2:raise ValueError('Three native probes and unique two-parent smoke must be counted once')
 for r in smoke['records']:
  if (r['L_NN'],r['U_teacher_NN'],r['U_student_NN'],r['NN'],r['optimizer_updates'])!=(768,8,8,784,0):raise ValueError('Actual synthetic NN packet counters differ')
 segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:]);shared=smoke['total_NN']
 return dict(rows=out,new_model_epochs=30,new_updates=6660,new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,shared_smoke_cost=dict(synthetic_NN=smoke['NN'],real_checkpoint_NN=real['NN'],total_NN=shared,unique_parent_count=2,not_double_charged=True),parent20_native_NN_reused={a:p['cost']['native_training_and_validation_NN'] for a,p in training['parent20'].items()},unique_parent20_native_NN_reused={a:c['native_training_and_validation_NN'] for a,c in training['unique_parent20_costs'].items()},unique_parent20_native_NN_sum=sum(c['native_training_and_validation_NN'] for c in training['unique_parent20_costs'].values()),cumulative_candidate_native_NN={a:training['parent20'][a]['cost']['native_training_and_validation_NN']+out[a]['native_NN'] for a in CONDITIONS[2:]},old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,U_physical_simulations={a:training['rows'][a]['counts']['U_As'] for a in CONDITIONS[2:]},U_CPU_host_wall={a:training['rows'][a]['u_practical']['timing'] for a in CONDITIONS[2:]},CPU_inclusive_route_not_added_to_children=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;channel/CPU metadata cost included in native wall;not isolated speed proof')


def failed_attempt_cost(project):
 c=read(Path(project)/'runs'/RUN/'technical_history/r01/cost.json')
 if c['status']!='FAILED_PREFLIGHT' or c['run_id']!=FAILED_RUN or c['NN']!=770 or any(c[k]!=0 for k in ('optimizer_updates','real_IQ','propagation_simulations','worker_count')) or c['preserved'] is not True:raise ValueError('Preserved r01 preflight cost differs from actual failed attempt')
 return {k:c[k] for k in ('status','run_id','code_commit','NN','optimizer_updates','real_IQ','propagation_simulations','worker_count','preserved','error')}


def cost_summary(project,training):
 c=_current_cost_summary(project,training);failure=failed_attempt_cost(project)
 c.update(technical_failed_attempts=[failure],technical_failed_NN=failure['NN'],candidate_whole_attempts_NN=c['new_total_NN']+failure['NN'],technical_failure_not_charged_to_training=True)
 return c


def render_report(source,formal,training,costs):
 d=source['decision'];s=formal['stage'];lines=['# U practical E30：固定三尾段结果','',RUN+'（immutable m3-r02）；model seed2026092701。N01从原NoAs20，C/S从同一原As20及各自ownEMA追加10轮。旧NoAs30/As30只读复用。','',f"源资格PASS={d['source_criteria_pass']}，固定配方关闭={d['fixed_recipe_closed']}；fullV clean S−matchedN {d['fullV_clean_S_minus_matchedNoAs_pp']:+.6f}pp（≥−.1），pooled两draw且mid/low等权 S−C {d['midlow_S_minus_C_pp']:+.6f}pp（≥+.5）。源先保存并由root读取，final是另一次显式调用；FAIL仍评分，分析不自动扩E50或seed。",'',f"正式 S−matchedN clean {s['clean_delta_pp']:+.6f}pp、净修复{s['clean_net_repairs']}，0.1绝对百分点阶段约束={s['clean_absolute_0p1pp_guard_met']}；mid/low urban {s['midurban_gain_pp']:+.6f}/{s['lowurban_gain_pp']:+.6f}pp。n=1，不证明可重复双提升。",'', 'N/C/S共同去学生strong agreement，原raw ownEMA teacher的RX/day quantile与batch_neighbor资格保留。N/C保留原Gaussian strong(.015)；S先生成整批strong，再覆盖私有均匀B//4位置为完整practical(raw同一U)：256→64，末124→31。每U只有一次teacher和student前向，原L6/1.25/ORD128保留；U CE=.16×teacher-qualified槽位均值（空选为0），entropy=.01×全部U槽均值（概率clamp1e-8），包括未资格与As槽。无额外KL、U真值或新头。S−C是U视图增量；C−旧AS含共同去agreement改变，不是纯星地收益。五条件不是2×2因子。','', '|条件|视图|ACC %|macro-F1 %|最差RX %|','|---|---|---:|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|配对|视图|ΔACC pp|修复|新增|净修复|','|---|---|---:|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['accuracy_delta_pp']:+.6f}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['','|配对|urban三场景均值Δpp|','|---|---:|']
 for a,b in PAIRS:lines.append(f"|{a}−{b}|{formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp']:+.6f}|")
 lines+=['','|条件|fullV clean %|V1800 clean %|两draw mid/low %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['','源CSV保留完整draw/pooled、RX/day/TX、state/lock、实际新增噪声四区间、F1与现有logits margin，不作附加门槛。fullV27000仅原生accuracy/RX，不伪造逐包F1；V1800两draw不是两个模型seed，源无highurban；源panel2.45GHz与训练/正式2.462GHz差异保留，signal/added-noise不是总SNR。','', '|新条件|qualified U|qualified As|qualified Gaussian|AsU|native秒|','|---|---:|---:|---:|---:|---:|']
 for a in CONDITIONS[2:]:
  r=training['rows'][a];u=r['u_practical'];lines.append(f"|{a}|{u['qualified_packets']}|{u['qualified_As_packets']}|{u['qualified_Gaussian_packets']}|{u['U_As']}|{r['cost']['wall_seconds']:.6f}|")
 lines+=['','完整u_steps重核资格、selected-mean CE/all-U entropy、teacher pseudo六类分布、RX/day/route与As state/lock/noise覆盖；pseudo分布或teacher/student agreement不是U真值精度。保持teacher代码路径不保证模型更新后teacher预测轨迹逐步相等。','', '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append('|'+a+'|'+'|'.join(str(c[k]) for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_NN','source_readout_NN','formal_prediction_NN'))+'|')
 lines+=['',f"新增3尾段30modelEpoch/6660更新；训练与读出{costs['new_training_and_readouts_NN']}NN，共享smoke{costs['new_shared_smoke_NN']}NN，新总{costs['new_total_NN']}NN。B00/B01两个真实父去重单列；C/S共享As20，旧E30不重计。各新行2220steps/10epoch、完整L6300/hiddenU56700流全扫，U学生/teacher各567000；S额外141750完整物理模拟不是新增NN，也不是零成本。CPU route包含selection/simulation/copy，不重复相加，不当CUDAevent；detach日志可同步。source prediction wall缺失保持null。",'', '|条件|suburban视图|绝对ACC %|自身clean−suburban pp|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v,g in formal['suburban_selfclean_gaps_pp'][a].items():lines.append(f"|{a}|{v}|{100*formal['summary'][a]['views'][v]['accuracy']:.6f}|{g:.6f}|")
 lines+=['',f"clean降低也会缩小gap，必须同时看绝对ACC。全部5×7固定预测和独立评分后truth join，重核{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}分区。源科学判据与正式阶段分开，默认头联合训练，无RX留出或诊断分类器。完整10配对、F1/最差RX、分层与成本保留；n1/微差不是可重复clean收益。正式不反馈参数/候选/seed或救跑，分析NN/CP/IQ/updates均0。"]
 lines+=['',f"此前r01首步接口技术失败单列{costs['technical_failed_NN']}NN（原NoAs checkpoint前向2＋synthetic L768，U前向/物理传播/更新/worker均0）；旧run/release/log保留。本run有效训练与smoke成本为{costs['new_total_NN']}NN，含失败尝试的候选实际总计{costs['candidate_whole_attempts_NN']}NN；失败不是额外科学训练，也不与本次有效smoke重复计数。"]
 return '\n'.join(lines)+'\n'

FINAL_FILES=('formal_summary.json','formal_confusions.json','formal_metrics.csv','final_report.md','costs.json','complete.json')
def validate_output(project,output,phase='source'):
 out=Path(output)
 if out.parent!=Path(project)/'runs'/RUN or re.fullmatch(r'analysis_auto_r[0-9]{2}',out.name) is None:raise ValueError('Only own immutable analysis output root')
 if phase=='source':
  if out.exists():raise FileExistsError('Existing source output;reconcile,do not duplicate')
 elif phase=='final':
  if not out.is_dir() or not (out/'source_complete.json').is_file() or any((out/n).exists() for n in (*FINAL_FILES,'failure.json')):raise FileExistsError('Final needs saved source and no final/failure outputs')
 else:raise ValueError('Explicit source/final phase required')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('source','final'),default='source');p.add_argument('--truth',type=Path);p.add_argument('--code-commit',required=True);a=p.parse_args()
 if a.release.name!=TRAINING_RELEASE or (a.phase=='source' and a.truth is not None) or (a.phase=='final' and a.truth is None):raise ValueError('Actual release and explicit truth-separated phase required')
 out=validate_output(a.project,a.output,a.phase)
 if a.phase=='source':out.mkdir()
 started=time.perf_counter();write(out/(a.phase+'_active.json'),dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=a.code_commit,phase=a.phase,NN=0,truth_opened=False))
 try:
  if a.phase=='source':
   wait_until(a.project,a.release,'SOURCE',out);training=training_analysis(a.project);write(out/'training_logs.json',training);source=source_analysis(a.project);write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,NN=0,target_scores_read=False,formal_requires_separate_explicit_invocation=True));write(out/'state.json',dict(status='SOURCE_ANALYSIS_COMPLETE',NN=0,truth_opened=False));print('SOURCE_ANALYSIS_COMPLETE;exit_without_formal',flush=True);return
  source=read(out/'source_summary.json');training=read(out/'training_logs.json');done=read(out/'source_complete.json')
  if done['status']!='SOURCE_ANALYZED' or done['training_run']!=RUN or done['target_scores_read'] or source['target_scores_read'] or training['target_scores_read'] or source['run_id']!=RUN or training['run_id']!=RUN:raise ValueError('Saved own complete source phase required before final')
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=3,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,final_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(phase=a.phase,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
