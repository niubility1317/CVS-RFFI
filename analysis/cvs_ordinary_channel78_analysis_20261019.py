"""P78: saved-output recount of ordinary AWGN/MP shadow execution and bypass.

Reads saved predictions, metadata and complete structured logs only. No model,
checkpoint or IQ loading; no training mutation or automatic technical retry.
"""
import argparse,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_analysis_20261018 as old
import cvs_budget50_source_states56_20261018 as state_core
RUN='20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_ordinaryChannel_E30_20261019_r01'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','OC_NOAS30','OC_AS30')
PAIRS=(('OC_AS30','OC_NOAS30'),('OC_AS30','AS30'),('OC_NOAS30','NOAS30'),('AS30','NOAS30'),('OC_AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('OC_NOAS30',p/RUN/'B00-s2026092701'),('OC_AS30',p/RUN/'B01-s2026092701')]

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

def source_conditions(summary):
 candidate=summary['OC_AS30'];no=summary['OC_NOAS30'];oldas=summary['AS30']
 clean=float(100*(candidate['full_V_clean']-no['full_V_clean']));midlow=float(100*(candidate['mid_low_mean']-oldas['mid_low_mean']))
 logs.finite(clean,'newAs versus matched newNoAs fullV clean');logs.finite(midlow,'newAs versus originalAs midlow')
 cleanpass=bool(clean>=-.1-1e-12);starpass=bool(midlow>=.5-1e-12);passed=bool(cleanpass and starpass)
 return dict(fullV_clean_vs_matched_NoAs_pp=clean,mid_low_mean_vs_original_As30_pp=midlow,clean_vs_matched_NoAs_within_absolute_0p1pp=cleanpass,mid_low_gain_at_least_0p5pp=starpass,source_criteria_pass=passed,E50_source_eligible=passed,fixed_recipe_closed=bool(not passed),fullV_clean_vs_original_As30_pp=float(100*(candidate['full_V_clean']-oldas['full_V_clean'])),fullV_clean_vs_original_NoAs30_pp=float(100*(candidate['full_V_clean']-summary['NOAS30']['full_V_clean'])),fullV_clean_interaction_pp=float(100*((candidate['full_V_clean']-no['full_V_clean'])-(oldas['full_V_clean']-summary['NOAS30']['full_V_clean']))),mid_low_mean_interaction_pp=float(100*((candidate['mid_low_mean']-no['mid_low_mean'])-(oldas['mid_low_mean']-summary['NOAS30']['mid_low_mean']))),only_two_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False,automatic_extension=False)

def source_analysis(project):
 started=time.perf_counter();meta=ensure_metadata(project,'source');data={};summary={};metrics=[]
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite fixed source panel/logits absent')
  if data and any(not np.array_equal(x[k],data['NOAS30']['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source physical panel required')
  if len({tuple(np.atleast_1d(v).tolist()) for v in x['ids']})!=1800:raise ValueError('Source panel physical IDs must be unique')
  q=meta[arm]['source_qualification/complete.json'];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source readout differs')
  native=meta[arm]['source/completion.json']['final_source_metrics']
  if native['source_val_count']!=27000:raise ValueError('Native fullV count differs')
  pred=dict(clean=x['clean'].argmax(-1))
  for i,v in enumerate(SOURCE_SCENES):pred[v]=x['satellite'][:,i].argmax(-1).reshape(-1)
  panel={};data[arm]=dict(x=x,pred=pred)
  for v,p in pred.items():
   y=x['TX'] if v=='clean' else np.tile(x['TX'],2);rx=x['RX'] if v=='clean' else np.tile(x['RX'],2);m=metric(y,p)
   panel[v]=dict(accuracy=m['accuracy'],macro_F1=m['macro_f1'],worst_RX=min(float((p[rx==r]==y[rx==r]).mean()) for r in np.unique(rx)))
  summary[arm]=dict(full_V_clean=native['source_val_accuracy'],full_V_RX_accuracy=native['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[v]['accuracy'] for v in SOURCE_SCENES[1:]])),mid_low_by_draw_accuracy=acc[:,1:].tolist(),source_readout_NN=q['NN'])
  obs=[('clean',-1,x['clean'],x['TX'],x['RX'],x['day'])]
  for si,v in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    ds=(0,1) if draw==-1 else (draw,);obs.append((v,draw,np.concatenate([x['satellite'][d,si] for d in ds]),np.tile(x['TX'],len(ds)),np.tile(x['RX'],len(ds)),np.tile(x['day'],len(ds))))
  for view,draw,z,y,rx,day in obs:
   p=z.argmax(1)
   for dim,values in [('overall',['ALL']),('RX',sorted(set(rx))),('day',sorted(set(day))),('TX',range(6))]:
    for value in values:
     mask=np.ones(len(y),bool) if dim=='overall' else rx==value if dim=='RX' else day==value if dim=='day' else y==value
     if mask.any():metrics.append(dict(arm=arm,scene=view,draw=draw,dimension=dim,stratum=str(value),**metric(y[mask],p[mask])))
 pairs={};cohort=data['NOAS30']['pred']['clean']==data['NOAS30']['x']['TX']
 for a,b in PAIRS:
  key=a+'__'+b;pairs[key]={}
  for v in ('clean',*SOURCE_SCENES):
   y=data[a]['x']['TX'] if v=='clean' else np.tile(data[a]['x']['TX'],2);co=cohort if v=='clean' else np.tile(cohort,2)
   pairs[key][v]=paired(y,data[b]['pred'][v],data[a]['pred'][v],co)
 strata,effects,interactions,audit=source_states(project,{a:v['x'] for a,v in data.items()})
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,decision=source_conditions(summary),state_strata=strata,state_paired=effects,state_interactions=interactions,state_alignment=audit,source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=SEED,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_status='NOT_INDEPENDENTLY_VERIFIED',training_fc_hz=2462000000,formal_fc_hz=2450000000,actual_parent_clean_correct_cohort=dict(status='MISSING_MATCHING_P43_PARENT_PREDICTION_CACHE',required_parent='P43 corresponding B00/B01 stage20',SAT18_cache_used=False,new_NN=0,blocking=False),fixed_clean_cohort='NOAS30 clean-correct panel; not the E20 parent cohort',logit_scale=30.,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

def state_tables(data,physics):
 x=data['NOAS30'];cohort=x['clean'].argmax(1)==x['TX'];strata=[];effects=[];interactions=[]
 for scene in ('clean',*SOURCE_SCENES):
  si=SOURCE_SCENES.index(scene) if scene!='clean' else None
  for draw in ((-1,) if scene=='clean' else (0,1,-1)):
   ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds);y=np.tile(x['TX'],copies);co=np.tile(cohort,copies)
   z={a:v['clean'] if scene=='clean' else np.concatenate([v['satellite'][d,si] for d in ds]) for a,v in data.items()}
   groups=[('overall','ALL',np.ones(len(y),bool))] if scene=='clean' else state_core.masks({k:np.concatenate([v[d,si,0] for d in ds]) for k,v in physics.items()})
   for dim,values,allowed in [('RX',np.tile(x['RX'],copies),(1,3,4,6,8)),('day',np.tile(x['day'],copies),(1,2,3)),('TX',y,range(6))]:groups.extend((dim,str(value),values==value) for value in allowed)
   for dimension,label,mask in groups:
    context=dict(scene=scene,draw=draw,dimension=dimension,stratum=label);current={}
    for arm in CONDITIONS:
     q=state_core.metric(y[mask],z[arm][mask],co[mask]);current[arm]=q;strata.append(dict(row_id=arm,model_seed=SEED,**context,**q))
    for a,b in PAIRS:
     q=state_core.paired(y[mask],z[b][mask].argmax(1),z[a][mask].argmax(1),co[mask]);q.update(new=a,control=b,**context,macro_f1_delta_pp=100*(current[a]['macro_f1']-current[b]['macro_f1']) if mask.any() else None,mean_margin_over30_delta=current[a]['mean_margin_over30']-current[b]['mean_margin_over30'] if mask.any() else None);effects.append(q)
    net=current['OC_AS30']['correct']-current['OC_NOAS30']['correct']-current['AS30']['correct']+current['NOAS30']['correct']
    interactions.append(dict(**context,count=int(mask.sum()),ordinary_channel_x_As_net_correct=net,ordinary_channel_x_As_accuracy_pp=100*net/int(mask.sum()) if mask.any() else None,new_Ag_success_claimed=False))
 return strata,effects,interactions

def source_states(project,data):
 panel=Path(project)/'runs'/state_core.PANEL_RUN/'diagnostic';role=read(panel/'data_roles.json');ids=state_core.check_roles(role);state_core.check_completion(read(panel/'observations/completion.json'))
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in state_core.PHYSICS_FIELDS if k in f.files}
 state_core.validate_physics(physics)
 for arm,x in data.items():state_core.validate_predictions(x,ids,data['NOAS30'])
 audit=state_core.check_records(logs.records(panel/'observations/physical_records.jsonl.gz'),physics,data['NOAS30'])
 strata,effects,interactions=state_tables(data,physics)
 return strata,effects,interactions,dict(source_role='V',physical_count=1800,channel_draws=2,full_condition_only=True,record_alignment=audit,missing_physics_fields=[k for k in state_core.PHYSICS_FIELDS if k not in physics],bins=state_core.bins_description(),added_noise_SNR_is_not_total_SNR=True,source_associations_are_not_unique_physical_causality=True)

def actual_cost(value):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if any(type(value.get(k)) is not int or value[k]<0 for k in keys):raise ValueError('Actual native four counters required')
 counters={k:value[k] for k in keys};total=sum(counters.values())
 if value.get('native_NN',total)!=total:raise ValueError('Saved native sum differs from actual counters')
 return dict(**counters,native_training_and_validation_NN=total)
def exposure_summary(counts,all_ids):
 values=np.asarray([counts[x] for x in all_ids],int)
 return dict(unique_physical_packets=len(counts),all_L_physical_packets=len(all_ids),unexposed_L_packets=int((values==0).sum()),minimum=int(values.min()),maximum=int(values.max()),mean=float(values.mean()),exposure_count_histogram={str(k):int(v) for k,v in sorted(Counter(values.tolist()).items())},per_ID_matched_to_original_not_assumed=True)


OC_KEYS=('ordinary_calls','ordinary_packets','MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','returned_ordinary_raw_MSE','post_return_rng_preserved','extra_NN')

def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base reduction differs'),logs._near(r['total_L'],total,'Full original1.25L reduction differs')]
 for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+key],r[key],'Actual full6L concat telemetry differs'))
 return max(errors)

def check_ordinary(r,c,a,m):
 if r['crossview_mode']!='ORD128' or not c['same_IQ_each_RDpair'] or c['actual_ORD128_anchor_input']!='ordinary128 with native MP/AWGN output bypass':raise ValueError('Original ordinary ORD128 input/paired IQ differs')
 if (c['ordinary_calls'],c['ordinary_packets'],c['extra_NN'],a['count'],a['extra_NN'])!=(1,128,0,128,0):raise ValueError('One original L128 ordinary call, no added NN required')
 if a['enabled'] is not True or not all(a[k] for k in ('input_unchanged','primitive_restored','full_native_aug_computation','post_return_rng_preserved')) or c['post_return_rng_preserved'] is not True:raise ValueError('L primitive output bypass/shadow/restoration differs')
 for kind in ('MP','AWGN'):
  calls=a[kind+'_calls']
  if type(calls) is not int or calls not in (0,1) or a[kind+'_hit'] is not bool(calls) or a[kind+'_packets']!=calls*128 or a['shadow_'+kind+'_packets']!=calls*128:raise ValueError('Native primitive call/hit/shadow packet counts differ')
 for key in OC_KEYS:
  if key=='post_return_rng_preserved':
   if c[key] is not True or a[key] is not True or m['ordinary_channel/'+key]!=1:raise ValueError('Native/hook/draw RNG preservation evidence differs')
  else:
   if key not in ('ordinary_calls','ordinary_packets'):logs._near(c[key],a[key],'Ordinary hook/draw field differs: '+key,1e-12)
   logs._near(m['ordinary_channel/'+key],c[key],'Actual native ordinary metric differs: '+key,1e-12)
 for key in ('primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced'):
  if type(a[key]) is not int or not 0<=a[key]<=a['MP_calls']+a['AWGN_calls']:raise ValueError('Invalid primitive RNG-advance count')
 logs.finite(a['returned_ordinary_raw_MSE'],'ordinary returned input distance')
 if a['returned_ordinary_raw_MSE']<0:raise ValueError('Negative ordinary input distance')
 logs._near(r['ordinary_raw_MSE'],a['returned_ordinary_raw_MSE'],'Native used ordinary IQ distance differs',1e-12)
 return logs.relation(r,True)

def check_U(record,step,packets):
 if record['step']!=step or record['packets']!=packets or record['extra_NN']!=0 or not all(record[k] for k in ('native_output_passthrough','L_primitive_wrappers_inactive','post_return_rng_preserved')):raise ValueError('Original U strong pass-through count/scope differs')

def sequence_summary(path):
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();by_rx=Counter();by_day=Counter();repeated=Counter()
 for q in logs.records(path):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra L sequence outside budget')
   step=batches['L']+1;repeats=128-len({tuple(z) for z in ids});repeated['batches_with_repeated_physical_ID']+=bool(repeats);repeated['repeated_input_positions']+=repeats
   for j,z in enumerate(ids):
    if (j+step)%2!=0:exposure[tuple(z)]+=1;by_rx[str(z[1])]+=1;by_day[str(z[2])]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U']:raise ValueError('Full L6300/U56700 physical training sequence differs')
 return dict(loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},selected_exposure=exposure_summary(exposure,unique['L']),selected_RX=dict(by_rx),selected_day=dict(by_day),sampler_repetitions=dict(repeated))

def parse_new_logs(root,arm):
 sat=arm=='OC_AS30';b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];policy=cfg['ordinary_channel']
 if cfg['crossview']['mode']!='ORD128' or policy['removed']!=['AWGN','MP'] or (policy['ordinary_calls_per_step'],policy['ordinary_packets_per_step'])!=(1,128) or not all(policy[k] for k in ('first20_full_ordinary_retained','U_weak_strong_unchanged','As_raw_path_unchanged','other_native_operations_unchanged')) or any(policy[k] for k in ('U_augmentor_wrapped','whole_training_RNG_matched','new_losses','new_parameters','new_NN','formal_test_changed')):raise ValueError('Only final10 L ordinary MP/AWGN outputs may be bypassed')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'],result['parent_epochs'],result['parent_optimizer_steps'])!=(30,6660,20,4440):raise ValueError('Fixed own P43 20+10 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=1704960 or result['L_As']!=(142080 if sat else 0) or result['total_L_loss_scale']!=1.25:raise ValueError('Actual original six-L/As64/hidden-U budget differs')
 if (result['U_steps'],result['U_packets'],result['U_student_NN'],result['EMA_local_successful_updates'],result['EMA_cumulative_updates'])!=(2220,567000,567000,2220,6660) or not result['parent_EMA_restored']:raise ValueError('Original U consumption/own EMA continuation differs')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own original P43 stage20 EMA' or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Strict own P43 EMA initialization differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);counts=Counter();sums=defaultdict(Counter);ordinary_mse=[];maxerr=0.;scenes=Counter();states=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','ordinary_channel_steps.jsonl.gz','ordinary_draw_steps.jsonl.gz','U_passthrough_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('Incomplete P78 per-update stream')
  if i>STEPS:raise ValueError('Extra update outside final10 budget')
  r,s,n,u,c,a,ur=items;ep=(i-1)//222+1;As=64 if sat else 0;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,c)) or n['epoch']!=ep or c['epoch']!=ep or c['cumulative_epoch']!=20+ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous actual epoch/update/Sinc counters differ')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,As,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Original six-forward supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw CE/RDrop slot differs')
  logs._near(r['SupCon_weight'],.1,'Original SupCon coefficient differs',1e-12);valid=check_ordinary(r,c,a,m);maxerr=max(maxerr,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=As or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerical state differs')
  number=m['train/pseudo_total'];picked=m['train/pseudo_selected'];expected=124 if i%222==0 else 256
  if number!=expected or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  check_U(ur,i,expected);logs.finite(m['train/loss'],'native total loss')
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64 or (sat and set(r['scene_counts'])-set(SOURCE_SCENES)) or (not sat and r['scene_counts']!={'NONE':64}):raise ValueError('Full raw practical/identity selected counts differ')
  scenes.update(r['scene_counts']);states.update(r['state_counts']);counts.update(steps=1,L_NN=768,As_packets=As,U_forward_packets=number,U_selected=picked,U_passthrough_calls=1,SupCon_valid_anchor_terms=valid)
  counts.update({k:int(c[k]) for k in OC_KEYS if k!='returned_ordinary_raw_MSE'});counts.update({k:a[k] for k in ('primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced')});counts.update(MP_hit_steps=int(a['MP_hit']),AWGN_hit_steps=int(a['AWGN_hit']),both_hit_steps=int(a['MP_hit'] and a['AWGN_hit']));ordinary_mse.append(r['ordinary_raw_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep]['native_loss']+=m['train/loss'];sums[ep].update({k:c[k] for k in ('MP_calls','AWGN_calls','MP_packets','AWGN_packets')})
 if counts['steps']!=STEPS:raise logs.IncompleteTrainingLogs('Final P78 update missing')
 sequence=sequence_summary(b/'training_sequence.jsonl.gz')
 if counts['U_forward_packets']!=sequence['loader_packets']['U'] or cost['student_train_NN']!=2271960 or cost['teacher_eval_NN']!=567000 or cost['teacher_train_NN'] or cost['student_eval_NN']!=324002:raise ValueError('Actual original four native NN counters differ')
 coverage=result['coverage'];cs=Counter();ct=Counter();cr=Counter();cx=Counter()
 for q in coverage:
  if type(q['count']) is not int or q['count']<=0 or q['TX'] not in range(6) or q['RX'] not in (1,3,4,6,8):raise ValueError('Invalid source physical channel coverage')
  cs[q['scene']]+=q['count'];ct[q['state']]+=q['count'];cr[str(q['RX'])]+=q['count'];cx[str(q['TX'])]+=q['count']
 if cs!=scenes or ct!=states or cr!=sequence['selected_RX']:raise ValueError('Actual selected channel/physical exposure counts differ')
 aggregate=result['ordinary_channel'];check_aggregate(aggregate,counts,ordinary_mse)
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if (ordinary['schedule_frozen_at'],ordinary['schedule_args_epochs'],ordinary['removed_primitives'])!=(13,20,['AWGN','MP']) or not all(ordinary[k] for k in ('no_ordinary_before_practical','RDrop_same_IQ','use_aug','full_native_aug_computation','U_strong_guard_only')) or ordinary['U_augmentor_wrapped'] or ordinary['whole_training_RNG_matched']:raise ValueError('Frozen own-parent ordinary schedule/L-only bypass differs')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error in completed owned stdout')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 if not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher']:raise ValueError('Native ordinary/U/EMA must remain enabled')
 curves=[dict(epoch=q['epoch']+20,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,As_packets=142080 if sat else 0,shadow_primitive_computation_retained=True),counts=dict(counts),physical_sequence=sequence,ordinary_channel=dict(aggregate,returned_ordinary_raw_MSE_mean=float(np.mean(ordinary_mse)),returned_ordinary_raw_MSE_max=max(ordinary_mse),actual_anchor_input='native ordinary128 with MP/AWGN primitive output bypass',whole_training_RNG_equivalence_not_claimed=True),EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660,parent_restored=True),selected_exposure=dict(TX=dict(cx),RX=sequence['selected_RX'],day=sequence['selected_day'],physical=sequence['selected_exposure'],coverage=coverage,actual_As_packets=142080 if sat else 0,scene_counts_are_identity_control_not_As=not sat,TX_RX_day_joint_not_recorded=True,per_physical_scene_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True),U_counterfactual_double_execution_claimed=False),dict(args=args,ordinary_mse=ordinary_mse)

def check_aggregate(a,counts,mse):
 if a['mode']!='L_ordinary_native_AWGN_MP_execute_then_bypass' or (a['ordinary_factory_calls'],a['ordinary_calls'],a['ordinary_packets'],a['U_calls'],a['U_packets'],a['U256_calls'],a['U124_calls'])!=(1,2220,284160,2220,567000,2210,10):raise ValueError('Actual L ordinary/U factory-call totals differ')
 if not all(a[k] for k in ('U_native_output_passthrough','U_weak_strong_unchanged','As_raw_path_unchanged','first20_full_ordinary_retained','full_native_aug_computation','all_primitive_restored','all_post_return_rng_preserved')) or any(a[k] for k in ('whole_training_RNG_matched','new_losses','new_parameters','extra_NN')) or a['pair_weights']!=[.5,.5,.25] or a['total_L_scale']!=1.25:raise ValueError('Actual L-only bypass/U conservation/objective scope differs')
 for key in ('ordinary_calls','ordinary_packets','MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced','MP_hit_steps','AWGN_hit_steps','both_hit_steps'):
  if a[key]!=counts[key]:raise ValueError('Actual ordinary aggregate mismatch: '+key)
 logs._near(a['returned_ordinary_raw_MSE_mean'],float(np.mean(mse)),'Actual ordinary MSE mean differs',1e-12)

def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Existing complete original30 audit required')
 result={};parent={};comparison={};lookup=dict(rows(project));private={}
 for name in ('NOAS30','AS30'):
  a=prior['rows'][name]
  if a['cost']['new_updates']!=6660 or a['cost']['new_epochs']!=30 or len(a['epoch_curves'])!=30:raise ValueError('Saved original30 audit schema/budget differs')
  result[name]=dict(reused=True,cost=a['cost'],epoch_curves=a['epoch_curves'],evidence='Existing complete original30 audit reused;no original step stream rescanned')
 for name,oldname in [('OC_NOAS30','NOAS30'),('OC_AS30','AS30')]:
  result[name],private[name]=parse_new_logs(lookup[name],name);result[name]['reused']=False
  a=read(lookup[oldname]/'stage20/result.json');parent[name]=dict(cost=actual_cost(a),epochs=20,updates=4440,wall_seconds=a['wall_seconds'],reused=True)
  priorcost=result[oldname]['cost'];oldtail={k:priorcost[k]-parent[name]['cost'][k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')}
  args=read(lookup[oldname]/'source/actual_native_args.json');args=args.get('args',args)
  curves=result[oldname]['epoch_curves'][20:];newcurves=result[name]['epoch_curves'];fields=('ordinary_raw_MSE','common_CE','common_KL','extra_CE','extra_KL','SupCon')
  diff={k:dict(record_count=len(curves),records_different=sum(a[k]!=b[k] for a,b in zip(curves,newcurves)),maximum_absolute_difference=max(abs(a[k]-b[k]) for a,b in zip(curves,newcurves))) for k in fields if all(k in q for q in (*curves,*newcurves))}
  comparison[name]=dict(reference_condition=oldname,reference_audit_reused=True,old_step_logs_rescanned=False,old_tail_four_counters=oldtail,new_tail_four_counters=result[name]['cost'],four_counters_equal=all(oldtail[k]==result[name]['cost'][k] for k in oldtail),available_epoch_mean_comparisons=diff,different_actual_arg_keys=sorted(k for k in set(args)|set(private[name]['args']) if args.get(k)!=private[name]['args'].get(k)),old_per_step_physical_IQ_equality_not_claimed=True,U_rule_unchanged_by_recipe=True)
  result[name]['epoch_curves']=[dict(q,parent20_reused=True) for q in result[oldname]['epoch_curves'][:20]]+[dict(q,parent20_reused=False) for q in newcurves]
 a='OC_NOAS30';b='OC_AS30';sequence=logs.compare_streams(lookup[a]/'source/training_sequence.jsonl.gz',lookup[b]/'source/training_sequence.jsonl.gz');mse_a=private[a]['ordinary_mse'];mse_b=private[b]['ordinary_mse']
 comparison['between_new_rows_only']=dict(physical_sequence=sequence,ordinary_MSE_records_different=sum(x!=y for x,y in zip(mse_a,mse_b)),maximum_ordinary_MSE_difference=max(abs(x-y) for x,y in zip(mse_a,mse_b)),comparison_is_new_NoAs_vs_new_As_not_old_reference=True)
 curves=[dict(q,row_id=arm) for arm,row in result.items() for q in row['epoch_curves']]
 if len(curves)!=120:raise ValueError('Four complete logical E30 curves required')
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parent,new_model_rows=2,new_model_epochs=20,new_updates=4440,unique_model_epochs=80,unique_optimizer_updates=17760,logical_epoch_curve_rows=120,curves=curves,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparison,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

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
 interaction={view:comparisons['OC_AS30__AS30'][view]['accuracy_delta_pp']-comparisons['OC_NOAS30__NOAS30'][view]['accuracy_delta_pp'] for view in VIEWS};interaction['urban_mean']=comparisons['OC_AS30__AS30']['urban_mean_accuracy_delta_pp']-comparisons['OC_NOAS30__NOAS30']['urban_mean_accuracy_delta_pp']
 key='OC_AS30__OC_NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],ordinary_channel_interaction_pp=interaction,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['OC_AS30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['OC_AS30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,new_Ag_success_claimed=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  out[arm]=dict(native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm in CONDITIONS[:2])
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['NN']+real['NN'];parents={a:v['cost']['native_training_and_validation_NN'] for a,v in training['parent20'].items()};segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:])
 return dict(rows=out,new_model_epochs=20,new_updates=4440,new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,parent20_native_NN_reused=parents,cumulative_candidate_native_NN={a:parents[a]+out[a]['native_NN'] for a in CONDITIONS[2:]},cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;not speed evidence')

def p78_plots(out,source,formal,training):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=dict(zip(CONDITIONS,('#64748b','#d97706','#16a34a','#2563eb')));files=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   path=out/(name+'.'+ext);fig.savefig(path,dpi=170);files.append(path.name)
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(16,4.8))
 for arm in CONDITIONS:
  seq=training['rows'][arm]['epoch_curves']
  for ax,key,title in zip(axes,('val_tx_acc','native_loss','train_grad_backbone'),('Source V accuracy (%)','Native total loss','Backbone gradient norm')):
   ax.plot([q['epoch'] for q in seq],[q.get(key,np.nan) for q in seq],label=arm,color=colors[arm]);ax.set(xlabel='Cumulative epoch',title=title);ax.axvline(20.5,color='grey',ls='--',alpha=.4);ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);save(fig,'training_curves')
 fig,axes=plt.subplots(2,3,figsize=(18,9));rr=source['state_strata']
 for si,scene in enumerate(SOURCE_SCENES[1:]):
  for di,dim in enumerate(('state_lock','signal_to_added_noise_db','actual_waveform_quality_db')):
   ax=axes[si,di];ss=[r for r in rr if r['scene']==scene and r['draw']==-1 and r['dimension']==dim];labels=list(dict.fromkeys(r['stratum'] for r in ss));xx=np.arange(len(labels))
   for i,arm in enumerate(CONDITIONS):
    chosen=[next(r for r in ss if r['row_id']==arm and r['stratum']==label) for label in labels];ax.bar(xx+(i-1.5)*.2,[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in chosen],.2,label=arm,color=colors[arm])
   counts=[next(r['count'] for r in ss if r['row_id']=='OC_AS30' and r['stratum']==label) for label in labels]
   ax.set(xticks=xx,xticklabels=[label+'\nn='+str(n) for label,n in zip(labels,counts)],title=scene+' / '+dim,ylabel='Source accuracy (%)');ax.tick_params(axis='x',labelrotation=40);ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=7);save(fig,'source_quality_states')
 fig,axes=plt.subplots(1,2,figsize=(15,5));xx=np.arange(7)
 for arm in CONDITIONS:axes[0].plot(xx,[100*formal['summary'][arm]['views'][v]['accuracy'] for v in VIEWS],marker='.',label=arm,color=colors[arm])
 for a,b in (('OC_AS30','OC_NOAS30'),('OC_AS30','AS30'),('OC_NOAS30','NOAS30')):axes[1].plot(xx,[formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for v in VIEWS],marker='.',label=a+' - '+b)
 for ax,title in zip(axes,('Frozen default-head accuracy (%)','Paired absolute percentage points')):
  ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title);ax.tick_params(axis='x',labelrotation=30);ax.legend(fontsize=7);ax.grid(alpha=.2)
 axes[1].axhline(0,color='grey',lw=.7);save(fig,'formal_all7_paired')
 fig,axes=plt.subplots(1,3,figsize=(16,4.8))
 for arm in CONDITIONS[2:]:
  seq=[q for q in training['rows'][arm]['epoch_curves'] if q['epoch']>20];epochs=[q['epoch'] for q in seq]
  axes[0].plot(epochs,[q['MP_calls'] for q in seq],label=arm+' MP');axes[0].plot(epochs,[q['AWGN_calls'] for q in seq],ls='--',label=arm+' AWGN')
  axes[1].plot(epochs,[q['ordinary_raw_MSE'] for q in seq],label=arm,color=colors[arm])
  axes[2].plot(epochs,[q['total_L'] for q in seq],label=arm,color=colors[arm])
 for ax,title in zip(axes,('Native primitive calls per L step (shadow retained)','Returned ordinary-to-raw MSE','Original total labeled loss')):ax.set(xlabel='Cumulative epoch',title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
 save(fig,'ordinary_channel_coverage')
 return files

def render_report(source,formal,training,costs,figures=()):
 text=['# P78：ordinary 通用信道与完整 practical 的尾段 2×2 检验','',RUN,'',
 '两条新行分别从对应 P43 B00/B01 的 own E20 student 与 own EMA 追加 E10，累计 E30。只对 L ordinary 链中的原生 MP/AWGN 执行原计算后旁路输出；PA、幅相、时序等其余原操作按原生配置保留，DAC/class-signature 原本关闭并保持关闭。前 E20 完整 ordinary 历史保留。完整 practical 仍从 raw64 生成，原 U、全部 CE/RDrop/SupCon、默认 cosine30 分类头与六次 L 前向不变。该机制是移除可能重叠的普通信道扰动，不预先称为成功的新 Ag。','',
 '|环境|旧 NoAs %|旧 As %|OC NoAs %|OC As %|新 As−新 NoAs pp|新 As−旧 As pp|交互 pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  values=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS]
  diffs=[formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp'] if v=='urban_mean' else formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for a,b in (('OC_AS30','OC_NOAS30'),('OC_AS30','AS30'))]
  text.append('|'+v+'|'+'|'.join(f'{z:.6f}' for z in values+diffs+[formal['ordinary_channel_interaction_pp'][v]])+'|')
 s=formal['stage'];text+=['',f"新 As 相对同预算新 NoAs：clean {s['clean_delta_pp']:+.6f} 个百分点，下降不超过 0.1 个百分点={s['clean_absolute_0p1pp_guard_met']}；low/mid urban {s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f} 个百分点，urban 三场景均值 {s['urban_mean_gain_pp']:+.6f} 个百分点。",'交互=(OC As−原 As)−(OC NoAs−原 NoAs)，必须连同四行绝对性能解释；不能把 NoAs 退化算成 As 的直接改善。一个模型种子不足以证明可重复双提升。','',
 '|条件|环境|accuracy %|宏 F1 %|最差 RX %|','|---|---|---:|---:|---:|']
 for arm in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][arm]['views'][v];text.append(f"|{arm}|{v}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 text+=['','|比较|环境|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];text.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 text+=['','源决定在 formal 阶段之前保存：'+json.dumps(source['decision'],ensure_ascii=False),
 '仅两项源条件：新 As fullV clean−新 NoAs≥−0.1 个百分点，两个固定 draw 的 mid/low 等权新 As−原 As≥+0.5 个百分点。失败仍完成全部七环境，并关闭该固定配方的 E50/扩种子；正式分数不回流决定。无原语命中数、MSE、状态代理或 raw 预测不变的额外晋级条件。',
 '训练 fc=2.462 GHz，既有固定正式 received views 的 fc=2.45 GHz；源 panel 频率本分析未独立核实，没有修改正式信道。源 V1800 状态/锁定/新增噪声比/波形与白功率代理、CFO、核、着色、AGC、边界分层及 margin÷30、固定 NoAs30 clean 正确集合保持率见 CSV。它们是源域关联描述，新增噪声比不是总 SNR；正式无逐包状态标签。源没有 high urban；fullV27000 只有原生完整 accuracy/RX，不虚构逐包 F1。',
 '真实 P43 E20 父 clean 正确集合缺少匹配缓存，本轮未测；没有混用 SAT18 父缓存，也不新增推理。固定 NoAs30 集合与父集合不混称。',
 '新两行各 2220 更新全部解析。MP/AWGN 影子调用、包数、RNG 前进计数、恢复和原 U256/U124 透明传递逐条核对；运行日志不声称对 U 双执行反事实。其他 primitive 下游输入和 sigma 可因旁路而改变，不能声称整个旧训练随机轨迹或波形完全相同。','',
 '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|native wall s|','|---|---:|---:|---:|---:|---:|---:|']
 for arm in CONDITIONS:
  q=costs['rows'][arm];text.append(f"|{arm}|{q['student_train_NN']}|{q['student_eval_NN']}|{q['teacher_train_NN']}|{q['teacher_eval_NN']}|{q['native_NN']}|{q['native_wall_seconds']:.3f}|")
 text += ['',f"新增 20 模型轮/4440 更新；新增训练与读出 {costs['new_training_and_readouts_NN']} NN，共享实际 smoke {costs['new_shared_smoke_NN']} NN，新增总计 {costs['new_total_NN']} NN。两个父 E20、两个旧尾 E10 与两个新尾 E10 合计 80 唯一模型轮/17760 更新；四条完整 E30 曲线为 120 逻辑行，复用训练不重复收费。相同前向量不等于相同墙钟；影子原语计算未省去，预算结束不等于收敛。",'分析不重新预测或评分历史 NoAs30/As30。所有 RX/TX/day 配对错误、独立混淆矩阵、完整曲线、源分层与成本写入同一输出目录。分析新增 NN/IQ/checkpoint 读取/更新均为 0。','', '实际生成图：']
 text.extend('- '+v for v in figures)
 return '\n'.join(text)+'\n'

def write_source_outputs(out,source,training):
 write(out/'training_logs.json',training);write_csv(out/'training_curves.csv',training['curves'])
 write(out/'source_summary.json',{k:v for k,v in source.items() if k not in ('metrics','state_strata','state_paired','state_interactions')})
 write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']])
 write(out/'source_state_confusions.json',source['state_strata']);write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['state_strata']]);write_csv(out/'source_paired_strata.csv',source['state_paired']);write_csv(out/'source_interaction.csv',source['state_interactions'])
 write(out/'source_decision.json',dict(status='SOURCE_DECIDED',run_id=RUN,model_seed=SEED,decision=source['decision'],formal_scores_read=False,target_feedback=False,NN=0))
 write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',run_id=RUN,NN=0,target_scores_read=False))

def run(project,out,stage,truth=None):
 project=Path(project);out=Path(out)
 if stage not in ('source','final'):raise ValueError('Only source or final stage')
 if stage=='source':
  if out.exists():raise FileExistsError('Preserve existing source analysis')
  out.mkdir(parents=True);training=training_analysis(project);source=source_analysis(project);write_source_outputs(out,source,training)
  return dict(status='SOURCE_ANALYZED',run_id=RUN,decision=source['decision'],NN=0,IQ_reads=0,checkpoint_reads=0,target_scores_read=False)
 marker=read(out/'source_complete.json');stored=read(out/'source_decision.json');source=read(out/'source_summary.json')
 if marker!={'status':'SOURCE_ANALYZED','run_id':RUN,'NN':0,'target_scores_read':False} or stored['run_id']!=RUN or stored['formal_scores_read'] is not False or source['run_id']!=RUN or source['target_scores_read'] is not False or stored['decision']!=source['decision']:raise ValueError('Saved independent source decision required before formal access')
 if truth is None:raise ValueError('Final stage needs explicitly supplied independent truth path')
 if (out/'complete.json').exists():raise FileExistsError('Preserve completed analysis')
 formal=formal_analysis(project,Path(truth),source);training=read(out/'training_logs.json');source['state_strata']=read(out/'source_state_confusions.json');costs=cost_summary(project,training)
 write(out/'formal_summary.json',{k:v for k,v in formal.items() if k not in ('metrics','paired_strata')});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);write_csv(out/'formal_paired_strata.csv',formal['paired_strata']);write(out/'costs.json',costs)
 figures=p78_plots(out,source,formal,training);(out/'final_analysis.md').write_text(render_report(source,formal,training,costs,figures),encoding='utf8')
 done=dict(status='ANALYZED',run_id=RUN,source_complete=True,formal_complete=True,new_model_rows=2,reused_model_rows=2,model_seed_count=1,logical_curve_rows=120,unique_model_epochs=80,unique_optimizer_updates=17760,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,reference_reprediction=0,reference_rescoring=0,figures=figures,repeatable_clean_positive_proven=False,missing=['matching P43 E20 parent clean-correct cohort','formal per-packet physical state labels','additional model seeds'],training_processes_untouched=True)
 write(out/'complete.json',done);return done

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--stage',choices=('source','final'),required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--truth',type=Path);args=parser.parse_args()
 print(json.dumps(run(args.project,args.out,args.stage,args.truth),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
