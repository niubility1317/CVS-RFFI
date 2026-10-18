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
RUN='20261019-phase1-FcrossRX-UQualified-e30-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_UQualified_E30_20261019_r01'
COMPARATOR_RUN='20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('MATCHED_NOAS30','UTEACHER30','URANDOM30','UQUALIFIED30')
PAIRS=(('UQUALIFIED30','UTEACHER30'),('UQUALIFIED30','URANDOM30'),('UQUALIFIED30','MATCHED_NOAS30'),('URANDOM30','UTEACHER30'),('URANDOM30','MATCHED_NOAS30'),('UTEACHER30','MATCHED_NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values



def rows(project):
 p=Path(project)/'runs';c=p/COMPARATOR_RUN
 return [('MATCHED_NOAS30',c/'N01-s2026092701'),('UTEACHER30',c/'C01-s2026092701'),('URANDOM30',p/RUN/'R01-s2026092701'),('UQUALIFIED30',p/RUN/'Q01-s2026092701')]

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
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all4 logical rows have all7 frozen predictions and independent scores')
  result[arm]=found
 return result


def source_conditions(summary,quality=None):
 q=summary['UQUALIFIED30'];c=summary['UTEACHER30'];n=summary['MATCHED_NOAS30'];clean=float(100*(q['full_V_clean']-n['full_V_clean']));gain=float(100*(q['mid_low_mean']-c['mid_low_mean']))
 for label,value in [('fullV Q-matchedNoAs',clean),('pooled two-draw equal midlow Q-C',gain)]:logs.finite(value,label)
 checks=dict(fullV_clean_vs_matchedNoAs_within_0p1pp=bool(clean>=-.1-1e-12),midlow_vs_Gaussian_control_gain_at_least_0p5pp=bool(gain>=.5-1e-12));passed=bool(all(checks.values()))
 return dict(checks=checks,source_criteria_pass=passed,E50_source_eligible=passed,fixed_recipe_closed=bool(not passed),fullV_clean_Q_minus_N_pp=clean,midlow_Q_minus_C_pp=gain,only_two_fixed_source_criteria=True,Q_minus_R_not_promotion_gate=True,scientific_negative_skips_formal=False,target_scores_read=False,automatic_extension=False,channel_draws_are_not_model_seeds=True)

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
  if data and any(not np.array_equal(x[k],data['MATCHED_NOAS30']['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source physical panel required')
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
 pairs={};quality_pairs={};cohort=data['MATCHED_NOAS30']['pred']['clean']==data['MATCHED_NOAS30']['x']['TX']
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
  if private and (not np.array_equal(ids,private['MATCHED_NOAS30']['ids']) or not np.array_equal(y,private['MATCHED_NOAS30']['y'])):raise ValueError('All rows must share physical pairing')
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
 paired_strata=[];pairs={};comparisons={};cohort=private['MATCHED_NOAS30']['pred']['clean']==private['MATCHED_NOAS30']['y']
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
 U_view_increment={v:comparisons['UQUALIFIED30__UTEACHER30'][v]['accuracy_delta_pp'] for v in VIEWS};U_view_increment['urban_mean']=comparisons['UQUALIFIED30__UTEACHER30']['urban_mean_accuracy_delta_pp']
 key='UQUALIFIED30__MATCHED_NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 stages={a:dict(clean_delta_pp=pairs[a+'__MATCHED_NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs[a+'__MATCHED_NOAS30']['clean']['net_repairs'],COUNT),lowurban_gain_pp=comparisons[a+'__MATCHED_NOAS30']['practical_low_urban']['accuracy_delta_pp'],midurban_gain_pp=comparisons[a+'__MATCHED_NOAS30']['practical_mid_urban']['accuracy_delta_pp'],repeatable_clean_gain_proven=False) for a in CONDITIONS[2:]}
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,stages=stages,source_decision=source['decision'],U_view_effect_pp=U_view_increment,position_effect_pp={v:comparisons['UQUALIFIED30__URANDOM30'][v]['accuracy_delta_pp'] for v in VIEWS},clean_vs_matched_NoAs=dict(accuracy_delta_pp=pairs['UQUALIFIED30__MATCHED_NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['UQUALIFIED30__MATCHED_NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_u_qualified30/runner.py' in argv:live.append(pid)
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
 mode='QUALIFIED_PRACTICAL' if arm=='UQUALIFIED30' else 'COUNT_RANDOM';i=q['step'];n=124 if i%222==0 else 256;ep=(i-1)//222+1
 if q['mode']!=mode or (q['epoch'],q['cumulative_step'],q['cumulative_epoch'],q['U_packets'],q['student_passes'],q['teacher_NN'],q['student_NN'])!=(ep,i+4440,ep+20,n,1,n,n):raise ValueError('Original one-student/one-teacher native U budget differs')
 positions=q['selected_positions'];selected=np.zeros(n,bool)
 if positions!=sorted(positions) or len(set(positions))!=len(positions) or q['selected_slots']!=len(positions) or any(type(z) is not int or not 0<=z<n for z in positions):raise ValueError('Original U positions without replacement required')
 selected[positions]=True;as_mask=selected;gaussian=np.flatnonzero(~as_mask).tolist()
 if q['native_losses_preserved'] is not True or q['native_feature_adjustment'] is not None or q['native_hook']!='model._source_unlabeled_feature_hook' or q['native_optimizer_step']!=i:raise ValueError('Actual native feature hook must preserve existing pseudo and entropy losses without adjustment')
 if q['Gaussian_positions']!=gaussian or q['As_packets']!=int(as_mask.sum()) or q['sample_id_fields']!=['eq_i','rx_i','day_i','sig_i','base_index'] or any(q[k] for k in ('U_truth_available','ordinary_before_practical','extra_NN','extra_CE_operator_calls','global_RNG_used_for_routing','selection_uses_TX_or_pseudo','domain_gate','strong_agreement','U_truth_used','extra_teacher')) or not all(q[k] for k in ('original_strong_before_overwrite','teacher_qualification_before_student_view','selection_without_replacement','teacher_raw_same_input','teacher_eval_no_grad','teacher_qualification_completed_before_strong','entropy_all_U','entropy_includes_unqualified','native_losses_preserved')):raise ValueError('Teacher-only qualification/hiddenU/input-routing boundary differs')
 if q['C_returns_original_strong_object']!=(len(positions)==0) or q['entropy_includes_As']!=bool(positions) or (q['original_strong_draw_packets'],q['original_strong_std'],q['CE_label_smoothing'],q['native_global_L_label_smoothing'],q['lambda_u'],q['lambda_entropy'],q['entropy_denominator'])!=(n,.015,0.,.01,.16,.01,n):raise ValueError('Native all-slot Gaussian draw/U CE/entropy weights differ')
 if q['threshold_policy']!=dict(mode='rx_day_quantile',quantile=.86,tau_min=.92,tau_max=.97) or q['temporal_policy']!=dict(mode='batch_neighbor',window=2,minimum_confidence=.8):raise ValueError('Original teacher confidence/temporal policy must remain fixed')
 ids=np.asarray(q['sample_ids']);rx=np.asarray(q['RX']);day=np.asarray(q['day']);eq=np.asarray(q['eq']);dom=np.asarray(q['domains']);pseudo=np.asarray(q['teacher_pseudo']);conf=np.asarray(q['teacher_conf'],float);z=np.asarray(q['student_logits'],float)
 if ids.shape!=(n,5) or ids.dtype.kind not in 'iu' or z.shape!=(n,6) or not np.isfinite(z).all() or any(v.shape!=(n,) for v in (rx,day,eq,dom,pseudo,conf)) or not np.array_equal(ids[:,0],eq) or not np.array_equal(ids[:,1],rx) or not np.array_equal(ids[:,2],day) or not np.isin(rx,[1,3,4,6,8]).all() or not np.isin(day,[1,2,3]).all() or not np.all(eq==1) or not np.isin(dom,np.arange(15)).all() or not np.isin(pseudo,np.arange(6)).all() or not np.isfinite(conf).all() or np.any((conf<0)|(conf>1)):raise ValueError('Original U metadata/logits/teacher pseudo shapes and roles differ')
 masks={k:np.asarray(q[k]) for k in ('conf_mask','temporal_mask','eligibility')}
 if any(v.shape!=(n,) or v.dtype.kind!='b' for v in masks.values()) or not np.array_equal(masks['temporal_mask'],temporal_mask(pseudo,conf,ids)) or not np.array_equal(masks['eligibility'],masks['conf_mask']&masks['temporal_mask']):raise ValueError('Teacher-only eligibility must be original confidence AND temporal,independent of student')
 if np.any(masks['conf_mask']&(conf<.92)) or np.any((~masks['conf_mask'])&(conf>=float(np.float32(.97)))):raise ValueError('Logged confidence selection violates original clamped thresholds')
 e=masks['eligibility'];den=int(e.sum())
 if len(positions)!=den or q['qualification_count']!=den or q['qualification_positions']!=np.flatnonzero(e).tolist() or q['routing_rule']!=mode or q['same_future_Ascount_claim'] or not q['selection_uses_current_teacher_qualification'] or q['selection_uses_TX_or_pseudo_class'] or (mode=='QUALIFIED_PRACTICAL' and not np.array_equal(selected,e)):raise ValueError('Qualified route must equal the teacher mask;random route must have its own qualified count')
 if not positions and (q['simulation_adapter_cpu_wall_seconds']!=0 or q['replacement_copy_cpu_wall_seconds']!=0):raise ValueError('Empty qualification must not simulate or copy an As view')
 if q['CE_denominator']!=den or q['eligible_selected_slots']!=int(e[selected].sum()) or q['eligible_Gaussian_slots']!=int(e[~as_mask].sum()) or q['eligible_As_slots']!=int(e[as_mask].sum()):raise ValueError('CE selected-count denominator and route qualification differ')
 shifted=z-z.max(1,keepdims=True);lp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True));ce=-lp[np.arange(n),pseudo];entropy=-(np.exp(lp)*np.log(np.maximum(np.exp(lp),1e-8))).sum(1);mean_ce=float(ce[e].mean()) if den else 0.;mean_entropy=float(entropy.mean());errors=[logs._near(q['native_loss_u'],mean_ce,'Native U selected-mean CE differs'),logs._near(q['native_loss_entropy'],mean_entropy,'Native entropy must average all U slots')]
 physics=q['physics']
 if len(physics)!=int(as_mask.sum()) or [v['selected_position'] for v in physics]!=positions or q['scene_counts']!=dict(Counter(v['scene'] for v in physics)):raise ValueError('Actual As U physics count/order differs')
 for v in physics:
  pos=v['selected_position']
  if list(v['sample_id'])!=ids[pos].tolist() or v['scene'] not in SOURCE_SCENES or v['state'] not in ('LOS','PARTIAL','BLOCKED') or v['lock'] not in ('locked','degraded','unlocked'):raise ValueError('Same raw physical U and complete original practical observation required')
 if any(not isinstance(q[k],(int,float)) or not np.isfinite(q[k]) or q[k]<0 for k in U_TIMERS):raise ValueError('Actual host timing fields required;not CUDA events')
 return dict(maximum_error=max(errors),teacher_pseudo=pseudo,teacher_conf=conf,eligibility=e,As_mask=as_mask,selected=selected,ids=ids,domains=dom,per_CE=ce,per_entropy=entropy,U_CE=mean_ce,U_entropy=mean_entropy,accepted_student_disagreement=int((e&(z.argmax(1)!=pseudo)).sum()),physics=physics)


def parse_new_logs(root,arm):
 b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];p=cfg['u_practical'];as_l=True;ref='B01-s2026092701';mode='QUALIFIED_PRACTICAL' if arm=='UQUALIFIED30' else 'COUNT_RANDOM'
 if cfg['reference_row_id']!=ref or cfg['crossview']['mode']!='ORD128' or cfg['As']!=as_l or (p['mode'],p['slot_fraction'],p['U_packets'],p['U_As_packets'],p['selected_slots'],p['Gaussian_std'],p['lambda_unlabeled'],p['lambda_unlabeled_entropy'])!=(mode,None,567000,None,None,.015,.16,.01) or any(p[k] for k in ('pseudo_strong_agreement','pseudo_domain_gate','U_truth_used','extra_NN','new_losses','L_changed','formal_test_changed','ordinary_before_practical','teacher_trajectory_equality_claim')) or not all(p[k] for k in ('original_strong_draw_all_packets','teacher_only_eligibility','pseudo_temporal_gate','teacher_quantile_unchanged','batch_neighbor_unchanged','teacher_qualification_before_student_view')) or p['pseudo_temporal_mode']!='batch_neighbor':raise ValueError('Fixed matched teacher-only U input recipe differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['parent_epochs'],result['parent_optimizer_steps'],result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(20,4440,30,6660):raise ValueError('Strict corresponding own20+10 budget required')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','Ag_packets')) or (result['L_NN'],result['L_As'],result['total_L_loss_scale'],result['EMA_local_successful_updates'],result['EMA_cumulative_updates'],result['U_steps'],result['U_packets'],result['U_student_NN'])!=(1704960,142080,1.25,2220,6660,2220,567000,567000):raise ValueError('Actual L/U/EMA and raw-practical counts differ')
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
  counts.update(steps=1,As_channel_calls=int(bool(q['As_packets'])),zero_As_steps=int(not q['As_packets']),L_NN=768,L_As=64 if as_l else 0,U_packets=q['U_packets'],U_selected_slots=q['selected_slots'],U_As=q['As_packets'],qualified=q['CE_denominator'],qualified_As=q['eligible_As_slots'],qualified_Gaussian=q['eligible_Gaussian_slots'],accepted_student_disagreement=v['accepted_student_disagreement'],SupCon_valid_anchor_terms=anchors);timers.update({k:q[k] for k in U_TIMERS});usteps.append(q);ordinary_mse.append(r['ordinary_raw_MSE']);sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update(U_CE=v['U_CE'],U_entropy=v['U_entropy'],qualified=q['CE_denominator'],U_packets=q['U_packets'],U_As=q['As_packets'],native_total=m['train/loss'])
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
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or {k:len(v) for k,v in unique.items()}!=dict(L=6300,U=56700) or unique['L']&unique['U'] or not unique_as<=unique['U'] or counts['U_packets']!=567000 or counts['U_selected_slots']!=counts['qualified'] or counts['U_As']!=counts['qualified'] or len(domain_map)!=15 or len(set(domain_map.values()))!=15:raise ValueError('Complete unchanged L6300/hiddenU56700 and dynamic qualified-count exposure required')
 if (cost['student_train_NN'],cost['student_eval_NN'],cost['teacher_train_NN'],cost['teacher_eval_NN'])!=(2271960,324002,0,567000):raise ValueError('Actual native four NN counters differ from one U student/teacher')
 actual=result['u_practical']
 if (actual['mode'],actual['calls'],actual['U_packets'],actual['selected_slots'],actual['As_packets'],actual['qualified_packets'],actual['qualified_As_packets'],actual['qualified_Gaussian_packets'],actual['original_strong_draw_packets'])!=(mode,2220,567000,counts['qualified'],counts['qualified'],counts['qualified'],counts['qualified_As'],counts['qualified_Gaussian'],567000) or not actual['teacher_only_eligibility'] or actual['pseudo_strong_agreement'] or actual['extra_NN'] or actual['extra_CE_operator_calls'] or actual['U_truth_used'] or actual['teacher_trajectory_equality_claim'] or (actual['lambda_u'],actual['lambda_entropy'])!=(.16,.01):raise ValueError('Producer full U summary differs from complete per-step records')
 if result['U_As']!=counts['U_As'] or actual['qualification_count']!=counts['qualified'] or actual['same_future_Ascount_claim'] or actual['channel']['calls']!=counts['As_channel_calls'] or actual['channel']['packets']!=counts['U_As']:raise ValueError('Actual nonempty channel calls and dynamic As packets differ')
 expected=[dict(RX=k[0],day=k[1],route=k[2],count=z['count'],qualified=z['qualified']) for k,z in sorted(rxday.items())]
 if actual['RX_day_route_exposure']!=expected or actual['scene_counts']!=dict(Counter({scene:sum(n for key,n in physics_counts.items() if key[2]==scene) for scene in set(k[2] for k in physics_counts)})):raise ValueError('Actual RX/day/route/scene coverage differs')
 for key in U_TIMERS:logs._near(actual['timing'][key],timers[key],'Actual U CPU wall sum differs: '+key,1e-12)
 ordinary=read(b/'ordinary_augmentation_actual.json');args=read(b/'actual_native_args.json');args=args.get('args',args)
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ'] or ordinary['U_DataAugmentation_used'] or args['label_smoothing']!=.01 or not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher'] or args['pseudo_strong_agreement'] or args['pseudo_domain_gate'] or not args['pseudo_temporal_gate'] or args['pseudo_temporal_mode']!='batch_neighbor':raise ValueError('Original ordinary/U teacher flags differ from shared teacher-only qualification')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed stdout')
 curves=[dict(epoch=z['epoch']+20,local_epoch=z['epoch'],**{k:v/222 for k,v in sums[z['epoch']].items()},**{k:z[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for z in epochs]
 summary=dict(mode=mode,As_channel_calls=counts['As_channel_calls'],zero_As_steps=counts['zero_As_steps'],same_future_qualified_count_not_claimed=True,teacher_qualification_policy=dict(threshold_policy=usteps[0]['threshold_policy'],temporal_policy=usteps[0]['temporal_policy'],domain_gate=False,strong_agreement=False,qualification_before_student_input=True),RX_day_route_exposure=expected,pseudo_counts={k:v.tolist() for k,v in pseudo_counts.items()},qualified_packets=counts['qualified'],qualified_As_packets=counts['qualified_As'],qualified_Gaussian_packets=counts['qualified_Gaussian'],qualified_rate=counts['qualified']/567000,accepted_student_disagreement=counts['accepted_student_disagreement'],U_As=counts['U_As'],unique_U_As_physical=len(unique_as),physics_strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],added_noise_bin=k[5],count=n) for k,n in sorted(physics_counts.items())],CE_normalization='mean across teacher-qualified U slots;zero if none',entropy_normalization='mean across all U slots with probability clamp1e-8',loss_weights=dict(U_CE=.16,U_entropy=.01),timing=dict(timers),timing_note=actual['timing_note'],extra_NN=0,extra_CE_operator_calls=0,U_truth_used=False,teacher_student_agreement_is_not_accuracy=True,teacher_trajectory_equality_claim=False,no_raw_physical_IDs=True)
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,L_As=142080 if as_l else 0),counts=dict(counts),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},u_practical=summary,EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660),epoch_curves=curves,maximum_loss_decomposition_error=maxerr,native_telemetry=telemetry,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=ordinary_mse)


def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/COMPARATOR_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS' or prior['run_id']!=COMPARATOR_RUN or prior['target_scores_read']:raise ValueError('Complete comparator source audit required')
 result={};comparability={};parents={};lookup=dict(rows(project));parent_result=read(base/REFERENCE_RUN/'B01-s2026092701/stage20/result.json');pcost=actual_cost(parent_result)
 for arm in CONDITIONS[:2]:
  row=prior['rows'][arm]
  if row['status']!='VERIFIED_COMPLETE_SOURCE_LOGS' or row['counts']['steps']!=2220 or len(row['epoch_curves'])!=10 or row['cost']['local_updates']!=2220 or row['cost']['cumulative_updates']!=6660:raise ValueError('Full existing N/C E30 audit required;do not rescan old stream')
  result[arm]=dict(row,reused=True,evidence='Complete comparator source audit reused;no old raw log rescan or NN')
 for arm in CONDITIONS[2:]:
  row,private=parse_new_logs(lookup[arm],arm);result[arm]=dict(row,reused=False);parents[arm]=dict(reference_condition='original_AS20',cost=pcost,epochs=20,updates=4440,wall_seconds=parent_result['wall_seconds'],reused=True)
  oldcost=result['UTEACHER30']['cost'];newcost=row['cost'];keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')
  comparability[arm]=dict(reference_condition='UTEACHER30',old_step_logs_rescanned=False,old_tail_four_counters={k:oldcost[k] for k in keys},new_tail_four_counters={k:newcost[k] for k in keys},four_counters_equal=all(oldcost[k]==newcost[k] for k in keys),common_teacher_qualification_path_unchanged=True,strong_agreement_false_in_all_four=True,common_U_or_full_IQ_trajectory_not_claimed=True,future_teacher_qualified_counts_equal_not_claimed=True,actual_route_count=row['counts']['U_As'])
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,comparator_run=COMPARATOR_RUN,rows=result,parent20=parents,comparator_parent20={arm:prior['parent20'][arm] for arm in CONDITIONS[:2]},unique_parent20_costs={'original_AS20':pcost},new_model_rows=2,new_model_epochs=20,new_updates=4440,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparability,shared_parent_count=1,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)


def cost_summary(project,training):
 metadata=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];q=metadata[arm]['source_qualification/complete.json'];p=metadata[arm]['prediction/complete.json'];native=c['native_training_and_validation_NN'];out[arm]=dict(student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],native_NN=native,source_readout_NN=q['NN'],source_readout_wall_seconds=q.get('wall_seconds'),formal_prediction_NN=p['actual_NN'],formal_prediction_wall_seconds=p['wall_seconds'],native_wall_seconds=c['wall_seconds'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],reused=arm in CONDITIONS[:2])
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query'] or smoke['optimizer_updates'] or real['optimizer_updates'] or smoke['real_IQ'] or real['real_IQ']:raise ValueError('Actual shared no-query smoke required')
 if len(smoke['records'])!=2 or {r['row_id'] for r in smoke['records']}!={p.name for a,p in rows(project) if a in CONDITIONS[2:]} or sum(r['NN'] for r in smoke['records'])!=smoke['NN'] or smoke['total_NN']!=smoke['NN']+real['NN'] or smoke['real_checkpoint_NN']!=real['NN'] or len(real['records'])!=1 or real['records'][0]['reference_row_id']!='B01-s2026092701' or real['NN']!=2:raise ValueError('Two native probes and one shared parent smoke must be counted once')
 for r in smoke['records']:
  if (r['L_NN'],r['U_teacher_NN'],r['U_student_NN'],r['NN'],r['optimizer_updates'])!=(768,8,8,784,0):raise ValueError('Actual synthetic packet counters differ')
 segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:]);shared=smoke['total_NN']
 for arm in CONDITIONS[:2]:out[arm]['parent20_native_NN_reused']=training['comparator_parent20'][arm]['cost']['native_training_and_validation_NN'];out[arm]['cumulative_E30_native_NN_reused']=out[arm]['parent20_native_NN_reused']+out[arm]['native_NN']
 return dict(rows=out,new_model_epochs=20,new_updates=4440,new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,shared_smoke_cost=dict(synthetic_NN=smoke['NN'],real_checkpoint_NN=real['NN'],total_NN=shared,unique_parent_count=1,not_double_charged=True),parent20_native_NN_reused={a:p['cost']['native_training_and_validation_NN'] for a,p in training['parent20'].items()},unique_parent20_native_NN_reused={a:c['native_training_and_validation_NN'] for a,c in training['unique_parent20_costs'].items()},unique_parent20_native_NN_sum=sum(c['native_training_and_validation_NN'] for c in training['unique_parent20_costs'].values()),cumulative_candidate_native_NN={a:training['parent20'][a]['cost']['native_training_and_validation_NN']+out[a]['native_NN'] for a in CONDITIONS[2:]},old_N_C_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,comparator_technical_failure_NN_recharged=0,U_physical_simulations={a:training['rows'][a]['counts']['U_As'] for a in CONDITIONS[2:]},U_CPU_host_wall={a:training['rows'][a]['u_practical']['timing'] for a in CONDITIONS[2:]},CPU_inclusive_route_not_added_to_children=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;CPU simulations included in native wall;not isolated speed proof')


def render_report(source,formal,training,costs):
 d=source['decision'];stage=formal['stage'];lines=['# U teacher-qualified routing E30', '',RUN+'；model seed2026092701。新R/Q各original ownAs20+10，旧N/C完整训练、源与正式读出复用。', '',f"源两条件PASS={d['source_criteria_pass']}；fullV Q−N {d['fullV_clean_Q_minus_N_pp']:+.6f}pp（≥−.1），两draw等权mid/low Q−C {d['midlow_Q_minus_C_pp']:+.6f}pp（≥+.5）。源先保存，正式另一次显式调用；FAIL关闭固定配方，仍完成all7，不自动延长/扫阈值/选种子。", '',f"正式Q−N clean {stage['clean_delta_pp']:+.6f}pp，净修复{stage['clean_net_repairs']}；0.1pp阶段约束={stage['clean_absolute_0p1pp_guard_met']}。n=1，不证明可重复双提升。", '', 'Q−C是完整route增量；Q−R只比较资格位置与同本行k规则随机位置。两行各自teacher随训练改变，未来k与累计剂量可不同，不宣称严格同剂量位置因果隔离。Q的As位置等于原conf&temporal资格；R均匀私有随机选取同本行资格数。k=0不调用channel，原Gaussian整批先生成。L6/1.25/ORD128、CE资格均值和全部U熵均不改，0额外NN；U真值始终隐藏。', '', '|条件|视图|ACC %|macro-F1 %|最差RX %|','|---|---|---:|---:|---:|']
 for arm in CONDITIONS:
  for view in VIEWS:
   q=formal['summary'][arm]['views'][view];lines.append(f"|{arm}|{view}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|配对|视图|ΔACC pp|修复|新增|净修复|','|---|---|---:|---:|---:|---:|']
 for a,b in PAIRS:
  for view in VIEWS:
   q=formal['paired'][a+'__'+b][view];lines.append(f"|{a}−{b}|{view}|{q['accuracy_delta_pp']:+.6f}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['','|配对|urban三场景均值Δpp|','|---|---:|']
 for a,b in PAIRS:lines.append(f"|{a}−{b}|{formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp']:+.6f}|")
 lines+=['','|条件|fullV clean %|V1800 clean %|两draw mid/low %|','|---|---:|---:|---:|']
 for arm in CONDITIONS:
  q=source['summary'][arm];lines.append(f"|{arm}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['','源CSV保留完整draw/RX/day/TX/state/lock/实际新增噪声分层与现有logits margin，无新NN或附加资格门槛。fullV27000只有原生accuracy/RX，不伪造逐包F1；V1800两draw不是模型seed，源无highurban，2.45GHz与训练/正式2.462GHz边界保留。', '', '|新行|资格总数|As总数|资格As|资格Gaussian|channel调用|空As step|','|---|---:|---:|---:|---:|---:|---:|']
 for arm in CONDITIONS[2:]:
  q=training['rows'][arm]['u_practical'];lines.append(f"|{arm}|{q['qualified_packets']}|{q['U_As']}|{q['qualified_As_packets']}|{q['qualified_Gaussian_packets']}|{q['As_channel_calls']}|{q['zero_As_steps']}|")
 lines+=['','完整u_steps重核原teacher资格、所选位置、按资格count平均的.16CE、全U平均的.01entropy；资格伪类计数/RXdayroute及As scene-state-lock-noise曝光是训练聚合，不是U真值准确率。Q全部资格As，R可能有无资格As参与熵；不按包数推断梯度强弱。', '', '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm in CONDITIONS:
  q=costs['rows'][arm];lines.append('|'+arm+'|'+'|'.join(str(q[k]) for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_NN','source_readout_NN','formal_prediction_NN'))+'|')
 lines+=['',f"新增2尾段20modelEpoch/4440updates；实际训练+读出{costs['new_training_and_readouts_NN']}NN，单parent共享smoke{costs['new_shared_smoke_NN']}NN，新增总{costs['new_total_NN']}NN。旧N/C及其原770NN失败均不重复计；共同As20父成本去重单列。每新行完整2220step/10epoch、L6300和hiddenU56700流；Ustudent/teacher各567000，As动态模拟不是零CPU成本。route wall含selection/simulation/copy，不相加、不当CUDAevent；source wall缺失保持null。", '', '|条件|suburban|绝对ACC %|自身clean−suburban pp|','|---|---|---:|---:|']
 for arm in CONDITIONS:
  for view,gap in formal['suburban_selfclean_gaps_pp'][arm].items():lines.append(f"|{arm}|{view}|{100*formal['summary'][arm]['views'][view]['accuracy']:.6f}|{gap:.6f}|")
 lines+=['',f"clean降低也可缩小gap，应同时看绝对ACC。全部4×7固定预测与独立评分齐后truth join，重核{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}分区；完整6配对/RXdayTX错误分层保留。默认头联合训练，无留出RX/新诊断头/新部署结构；n1微增不证明稳定双提升，正式结果不反馈机制/参数/种子或救跑。分析NN/CP/IQ/updates均0。"]
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
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,final_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(phase=a.phase,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
