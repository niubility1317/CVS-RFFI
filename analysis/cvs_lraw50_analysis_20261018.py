"""Zero-model eight-condition E30/E50 factorial source/formal analysis.

All old model/log audits are reused. Only new two fixed +20 segments are parsed.
No IQ/checkpoint/model imports, training mutation, selective seed or retry.
"""
import argparse,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_lraw30_analysis_20261018 as base
old=base.old
RUN='20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_LRaw_E50_20261018_r01'
REFERENCE_RUN=old.REFERENCE_RUN
ORIGINAL50_RUN=old.RUN
LRAW30_RUN=base.RUN
CONDITIONS=('NOAS30','AS30','LRAW_NOAS30','LRAW_AS30','NOAS50','AS50','LRAW_NOAS50','LRAW_AS50')
PAIRS=(('AS30','NOAS30'),('LRAW_AS30','LRAW_NOAS30'),('AS50','NOAS50'),('LRAW_AS50','LRAW_NOAS50'),('LRAW_AS30','AS30'),('LRAW_NOAS30','NOAS30'),('LRAW_AS50','AS50'),('LRAW_NOAS50','NOAS50'),('AS50','AS30'),('NOAS50','NOAS30'),('LRAW_AS50','LRAW_AS30'),('LRAW_NOAS50','LRAW_NOAS30'),('LRAW_AS50','NOAS50'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=4440;EPOCHS=20;STEP_OFFSET=6660
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values
actual_cost=base.actual_cost;exposure_summary=base.exposure_summary

def rows(project):
 p=Path(project)/'runs';result=[]
 for arm in CONDITIONS:
  budget=int(arm[-2:]);LRaw=arm.startswith('LRAW_');run=(LRAW30_RUN if budget==30 else RUN) if LRaw else REFERENCE_RUN if budget==30 else ORIGINAL50_RUN
  row=('B00' if 'NOAS' in arm else 'B01')+'-s2026092701';result.append((arm,p/run/row))
 return result

def factorial_effects(values):
 if set(values)!=set(CONDITIONS) or any(not np.isfinite(v) for v in values.values()):raise ValueError('All eight finite fixed condition values required')
 pair={a+'__'+b:100*(values[a]-values[b]) for a,b in PAIRS}
 two=dict(LRaw_x_As_at_E30=pair['LRAW_AS30__AS30']-pair['LRAW_NOAS30__NOAS30'],LRaw_x_As_at_E50=pair['LRAW_AS50__AS50']-pair['LRAW_NOAS50__NOAS50'],As_x_budget_original=pair['AS50__AS30']-pair['NOAS50__NOAS30'],As_x_budget_LRaw=pair['LRAW_AS50__LRAW_AS30']-pair['LRAW_NOAS50__LRAW_NOAS30'],LRaw_x_budget_NoAs=pair['LRAW_NOAS50__NOAS50']-pair['LRAW_NOAS30__NOAS30'],LRaw_x_budget_As=pair['LRAW_AS50__AS50']-pair['LRAW_AS30__AS30'])
 three=two['LRaw_x_As_at_E50']-two['LRaw_x_As_at_E30']
 if abs(three-(two['As_x_budget_LRaw']-two['As_x_budget_original']))>1e-10 or abs(three-(two['LRaw_x_budget_As']-two['LRaw_x_budget_NoAs']))>1e-10:raise ValueError('Three-factor algebra differs')
 return dict(paired_accuracy_pp=pair,two_factor_interactions_pp=two,three_factor_interaction_pp=float(three))

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json'];budget=int(arm[-2:]);expected=(20,4440) if budget==50 else (10,2220)
  if (c['epoch'],c['optimizer_steps'])!=expected or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=SEED or c['config']['row_id']!=root.name or c['config']['run_id']!=root.parent.name:raise ValueError('Actual fixed eight-row/seed/budget/defaulthead differs')
  if budget==50 and (c.get('cumulative_epochs'),c.get('cumulative_optimizer_steps'))!=(50,11100):raise ValueError('Actual cumulative E50 differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('training_updates',0) or q.get('head_fit_updates',0):raise ValueError('Frozen final/source readout differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all8 logical rows have all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def source_conditions(summary):
 a=summary['LRAW_AS50'];b=summary['LRAW_AS30'];n=summary['LRAW_NOAS50'];oldas=summary['AS50']
 return dict(E50_already_authorized_from_saved_E30_source=True,new_scientific_gate=False,automatic_extension=False,scientific_negative_skips_formal=False,target_scores_read=False,As50_minus_ownAs30_fullV_clean_pp=float(100*(a['full_V_clean']-b['full_V_clean'])),As50_minus_ownAs30_mid_low_mean_pp=float(100*(a['mid_low_mean']-b['mid_low_mean'])),As50_minus_matched_NoAs50_fullV_clean_pp=float(100*(a['full_V_clean']-n['full_V_clean'])),As50_minus_original_As50_mid_low_mean_pp=float(100*(a['mid_low_mean']-oldas['mid_low_mean'])),description='Fixed-recipe budget readout only;no new seed/budget promotion or target rescue')

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
 effects={view:factorial_effects({a:(q['full_V_clean'] if view=='fullV_clean' else q['mid_low_mean'] if view=='mid_low_mean' else q['panel'][view]['accuracy']) for a,q in summary.items()}) for view in ('fullV_clean','mid_low_mean','clean',*SOURCE_SCENES)}
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,decision=source_conditions(summary),factorial=effects,source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=SEED,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

LRAW_KEYS=base.LRAW_KEYS
check_native_L=base.check_native_L
check_lraw=base.check_lraw

def parse_new_logs(root,arm):
 sat=arm=='LRAW_AS50';b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];policy=cfg['lraw']
 if cfg['crossview']['mode']!='ORD128' or policy['ordinary_calls_per_step']!=1 or policy['ordinary_packets_per_step']!=128 or policy['actual_ordinary_raw_MSE']!=0. or not all(policy[k] for k in ('first20_ordinary_retained','full_native_aug_computation','RD_same_generated_IQ','U_weak_strong_unchanged','teacher_unchanged','EMA_unchanged')) or any(policy[k] for k in ('whole_trajectory_NoAug','whole_training_RNG_matched','U_augmentor_wrapped','new_losses','new_parameters','new_NN','formal_test_changed')):raise ValueError('Actual LRaw-only new20 configuration differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(20,4440,4440,4440) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(50,11100) or (result['parent_epochs'],result['parent_optimizer_steps'])!=(30,6660):raise ValueError('Fixed own LRaw30+20 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=3409920 or result['L_As']!=(284160 if sat else 0) or result['total_L_loss_scale']!=1.25:raise ValueError('Actual hiddenU/6L/As64or0 budget differs')
 if (result['U_steps'],result['U_packets'],result['U_student_NN'],result['EMA_local_successful_updates'],result['EMA_cumulative_updates'])!=(4440,1134000,1134000,4440,11100) or not result['parent_EMA_restored']:raise ValueError('Actual original U consumption and ownEMA continuation differ')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own LRaw FINAL30 EMA' or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Strict ownEMA teacher initialization differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);count=Counter();sums=defaultdict(Counter);mse=[];discard=[];maxerr=0.;scenes=Counter();states=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','lraw_steps.jsonl.gz','ordinary_draw_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New20 LRaw stream prefix')
  if i>STEPS:raise ValueError('Extra step outside own30+20 budget')
  r,s,n,u,c,a=items;ep=(i-1)//222+1;As=64 if sat else 0;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,c,a)) or n['epoch']!=ep or c['epoch']!=ep or c['cumulative_epoch']!=30+ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous actual update/Sinc/LRaw counters differ')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,As,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Actual original sixforward supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw control slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative crossRX weight differs',1e-12);valid=check_lraw(r,c,a,m);maxerr=max(maxerr,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=As or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerics differ')
  logs.finite(m['train/loss'],'native total loss');number=m['train/pseudo_total'];picked=m['train/pseudo_selected']
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64 or (sat and set(r['scene_counts'])-set(SOURCE_SCENES)) or (not sat and r['scene_counts']!={'NONE':64}):raise ValueError('Actual full channel or raw identitycontrol counts differ')
  scenes.update(r['scene_counts']);states.update(r['state_counts']);count.update(steps=1,L_NN=768,selected_packets=64,As_packets=As,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=valid,ordinary_calls=1,ordinary_packets=128,full_native_draws=1,discardedordinary_changed_packets=c['discardedordinary_changed_packets']);mse.append(r['ordinary_raw_MSE']);discard.append(c['discardedordinary_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')});sums[ep]['discardedordinary_MSE']+=c['discardedordinary_MSE'];sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New20 final LRaw update missing')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();by_rx=Counter();by_day=Counter();repeated=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra L input metadata outside budget')
   step=batches['L']+1;repeats=128-len({tuple(z) for z in ids});repeated['batches_with_repeated_physical_ID']+=bool(repeats);repeated['repeated_input_positions']+=repeats;repeated['maximum_repeated_positions_in_batch']=max(repeated['maximum_repeated_positions_in_batch'],repeats)
   for j,z in enumerate(ids):
    if (j+step)%2!=0:exposure[tuple(z)]+=1;by_rx[str(z[1])]+=1;by_day[str(z[2])]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=4440,U=4440) or packets!=dict(L=568320,U=1134000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L'] or count['U_forward_packets']!=packets['U']:raise ValueError('Full original L6300/U56700 source sequence differs')
 if cost['student_train_NN']!=3409920+1134000 or cost['teacher_eval_NN']!=1134000 or cost['teacher_train_NN'] or cost['student_eval_NN']!=594002:raise ValueError('Actual four counters/complete source loops differ')
 a=result['lraw']
 if (a['mode'],a['ordinary_factory_calls'],a['ordinary_calls'],a['ordinary_packets'],a['full_native_draws'],a['returned_raw_MSE_max'],a['used_ordinary_raw_MSE_max'],a['parent_ordinary_epochs'],a['pair_weights'],a['SupCon'],a['total_L_scale'])!=('ordinary_draw_discard_return_raw',1,4440,568320,4440,0.,0.,20,[.5,.5,.25],'ORD128;actual raw128 input',1.25) or any(a[k] for k in ('U_augmentor_wrapped','whole_trajectory_NoAug','whole_training_RNG_matched','extra_NN','new_losses')) or not a['U_weak_strong_unchanged'] or not a['full_native_aug_computation']:raise ValueError('Actual retained full augmentation draw/U rule scope differs')
 logs._near(a['discardedordinary_MSE_mean'],float(np.mean(discard)),'Actual discarded draw mean differs',1e-12);logs._near(a['discardedordinary_MSE_max'],max(discard),'Actual discarded draw maximum differs',1e-12)
 if a['discardedordinary_changed_packets']!=count['discardedordinary_changed_packets']:raise ValueError('Actual discarded changed packet count differs')
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for q in coverage:
  if type(q['count']) is not int or q['count']<=0 or q['TX'] not in range(6) or q['RX'] not in (1,3,4,6,8):raise ValueError('Invalid actual selected coverage')
  coverage_scene[q['scene']]+=q['count'];coverage_state[q['state']]+=q['count'];coverage_tx[str(q['TX'])]+=q['count'];coverage_rx[str(q['RX'])]+=q['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_rx!=by_rx:raise ValueError('Actual channel/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ'] or not ordinary['use_aug'] or not ordinary['full_native_aug_computation'] or ordinary['returned_input']!='pre-call raw clone' or ordinary['ordinary_calls']!=4440 or ordinary['ordinary_packets']!=568320 or ordinary['U_augmentor_wrapped'] or ordinary['whole_training_RNG_matched']:raise ValueError('Actual parent ordinary schedule/retained draw differs')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed own stdout')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 if not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher']:raise ValueError('Actual native ordinary/U/EMA enabled flags differ')
 curves=[dict(epoch=q['epoch']+30,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=20,local_updates=4440,parent_epochs=30,parent_updates=6660,cumulative_epochs=50,cumulative_updates=11100,L_NN=3409920,selected_packets=284160,As_packets=284160 if sat else 0,ordinary_augmentation_computation_retained=True),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},lraw=dict(a,sampler_repetitions=dict(repeated),actual_anchor_input='raw128 in existing ORD128 slot',full_training_RNG_equivalence_not_claimed=True),EMA=dict(initialization=ema,local_updates=4440,cumulative_updates=11100,parent_restored=True),selected_exposure=dict(TX=dict(coverage_tx),RX=dict(by_rx),day=dict(by_day),physical=exposure_summary(exposure,unique['L']),actual_As_packets=284160 if sat else 0,selected_actual_view='As_raw' if sat else 'raw',identity_control_selected_packets=0 if sat else 284160,scene_counts_are_identity_control_not_As=not sat,coverage=coverage,per_physical_scene_joint_not_recorded=True,TX_RX_day_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)


def cumulative_cost(prefix,tail):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN');result={key:prefix[key]+tail[key] for key in keys};result['native_training_and_validation_NN']=sum(result.values());return result

def training_analysis(project):
 base=Path(project)/'runs';old50=read(base/ORIGINAL50_RUN/'analysis_auto_r01/training_logs.json');prior=read(base/LRAW30_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS' or old50['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Completed existing original/LRaw audits required')
 result={};parents={};comparisons={};lookup=dict(rows(project))
 for name in ('NOAS30','AS30'):
  a=prior['rows'][name]
  if a['cost']['new_updates']!=6660 or len(a['epoch_curves'])!=30:raise ValueError('Existing original E30 full audit differs')
  result[name]=dict(cost=a['cost'],cumulative_native_cost={k:a['cost'][k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')},epoch_curves=a['epoch_curves'],complete_trajectory_curves=a['epoch_curves'],reused=True,evidence='Complete existing originalE30 audit;not rescanned')
 for name in ('LRAW_NOAS30','LRAW_AS30'):
  a=prior['rows'][name];prefix=prior['parent20'][name]
  if a['cost']['local_updates']!=2220 or len(a['epoch_curves'])!=10 or prefix['updates']!=4440:raise ValueError('Existing LRaw E20+10 full audit differs')
  control='NOAS30' if 'NOAS' in name else 'AS30';curves=result[control]['epoch_curves'][:20]+a['epoch_curves'];cumulative=cumulative_cost(prefix['cost'],a['cost'])
  result[name]=dict(cost=a['cost'],cumulative_native_cost=cumulative,epoch_curves=a['epoch_curves'],complete_trajectory_curves=curves,reused=True,evidence='Existing own prefix20+LRaw10 audit reused;not rescanned')
 for name in ('NOAS50','AS50'):
  a=old50['rows'][name];previous=name.replace('50','30')
  if a['cost']['local_updates']!=4440 or len(a['epoch_curves'])!=20:raise ValueError('Existing original E30+20 full audit differs')
  result[name]=dict(cost=a['cost'],cumulative_native_cost=cumulative_cost(result[previous]['cumulative_native_cost'],a['cost']),epoch_curves=a['epoch_curves'],complete_trajectory_curves=result[previous]['complete_trajectory_curves']+a['epoch_curves'],reused=True,evidence='Existing originalE50 complete extra20 audit reused;not rescanned')
 for name,control in [('LRAW_NOAS50','NOAS50'),('LRAW_AS50','AS50')]:
  a,private=parse_new_logs(lookup[name],name);previous=name.replace('50','30');parents[name]=dict(cumulative_native_cost=result[previous]['cumulative_native_cost'],epochs=30,updates=6660,own_parent_logical_row=previous,reused=True)
  a.update(reused=False,cumulative_native_cost=cumulative_cost(result[previous]['cumulative_native_cost'],a['cost']),complete_trajectory_curves=result[previous]['complete_trajectory_curves']+a['epoch_curves']);result[name]=a
  oldcurves=result[control]['epoch_curves'];newcurves=a['epoch_curves'];fields=('ordinary_raw_MSE','common_CE','common_KL','extra_CE','extra_KL','SupCon');diff={k:dict(record_count=len(oldcurves),records_different=sum(x[k]!=y[k] for x,y in zip(oldcurves,newcurves)),maximum_absolute_difference=max(abs(x[k]-y[k]) for x,y in zip(oldcurves,newcurves))) for k in fields if all(k in q for q in (*oldcurves,*newcurves))};comparisons[name]=dict(old_step_logs_rescanned=False,reference_audit_reused=True,available_epoch_mean_comparisons=diff,old_per_step_physical_IQ_equality_not_claimed=True,full_training_RNG_equality_not_claimed=True,U_rule_unchanged=True);args=read(lookup[control]/'source/actual_native_args.json');args=args.get('args',args);comparisons[name]['different_actual_arg_keys']=sorted(k for k in set(args)|set(private['args']) if args.get(k)!=private['args'].get(k))
 for name,a in result.items():
  if len(a['complete_trajectory_curves'])!=int(name[-2:]) or [q['epoch'] for q in a['complete_trajectory_curves']]!=list(range(1,int(name[-2:])+1)):raise ValueError('Complete reused/new trajectory curves differ')
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent30=parents,new_model_rows=2,new_model_epochs=40,new_updates=8880,cumulative_candidate_epochs_each=50,cumulative_candidate_updates_each=11100,physical_and_ordinary_comparison_to_original_E50_tails=comparisons,old_all_audits_reused=True,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

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
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=arm not in CONDITIONS[-2:]);private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
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
 effects={view:factorial_effects({a:(q['urban_mean_accuracy'] if view=='urban_mean' else q['views'][view]['accuracy']) for a,q in results.items()}) for view in (*VIEWS,'urban_mean')}
 key='LRAW_AS50__LRAW_NOAS50';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],factorial=effects,clean_vs_original_NoAs50=dict(accuracy_delta_pp=pairs['LRAW_AS50__NOAS50']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['LRAW_AS50__NOAS50']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)


def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  a=training['rows'][arm];c=a['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json'];group='new_LRaw_E50_segments' if arm in CONDITIONS[-2:] else 'old_original_E50_segments' if arm in ('NOAS50','AS50') else 'old_LRaw_E30_segments' if arm.startswith('LRAW_') else 'old_original_E30_models'
  out[arm]=dict(group=group,native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],cumulative_native_cost=a['cumulative_native_cost'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm not in CONDITIONS[-2:])
 groups={group:dict(rows=[a for a,q in out.items() if q['group']==group],native_NN=sum(q['native_NN'] for q in out.values() if q['group']==group),segment_and_own_readouts_NN=sum(q['segment_and_own_readouts_NN'] for q in out.values() if q['group']==group)) for group in ('new_LRaw_E50_segments','old_original_E50_segments','old_LRaw_E30_segments','old_original_E30_models')}
 base=Path(project)/'runs'/RUN;synthetic=read(base/'synthetic_smoke.json');real=read(base/'real_checkpoint_smoke.json')
 if synthetic['status']!='PASS' or real['status']!='PASS' or synthetic['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=synthetic['NN']+real['NN'];parents={a:q['cumulative_native_cost'] for a,q in training['parent30'].items()}
 return dict(rows=out,groups=groups,new_model_epochs=40,new_updates=8880,new_training_and_readouts_NN=groups['new_LRaw_E50_segments']['segment_and_own_readouts_NN'],new_shared_smoke_NN=shared,new_total_NN=groups['new_LRaw_E50_segments']['segment_and_own_readouts_NN']+shared,parent_LRaw30_native_costs_reused=parents,cumulative_candidate_epochs_each=50,cumulative_candidate_updates_each=11100,old_costs_not_recharged=True,shared_prefix20_not_double_counted_in_group_totals=True,historical_shared_smokes_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Per-segment actual model-seconds;not parallel elapsed or speed evidence')

def render_report(source,formal,training,costs):
 lines=['# LRaw固定配方 E30→E50：同种子八条件结果','',RUN+'；模型seed2026092701。只新增两个各自LRaw30 FINAL学生/ownEMA+20段；总20+10+20，新增段重置AdamW与cos20，不是连续cos50。E50已在读取E30正式结果之前依保存的E30源双规则决定，不由本次正式分数选择。','', '本轮新2个E50模型、旧6个冻结条件，合计8logical×7视图。原NoAs/As E30、LRaw NoAs/As E30及原NoAs/As E50读出与完整训练审计复用；旧original50 actual release commit=0f399c910324e21a8b1a60346d879be5c2964a9c，ownLRaw30 parent commit=d14450bf1b0d12cc281df1f4d0e79e8760e48282，原P43reference19aedc3604fed9c193237dd92366164178af0983。','', '配方保持6L/768NN每步/总L1.25、原.5/.5/.25 RDrop、As(raw64)/NoAs raw64、ORD128成员与默认cosine头联合训练。完整ordinary每步一次128包仍执行，增强结果弃用，L和既有ORD128实际输入raw128；Uweak/strong、教师前向及ownEMA仍启用。ordinary配置来自原own20 args冻结E13，而不是LRaw30的native10 args。','', '|视图|'+'|'.join(CONDITIONS)+'|','|---|'+'---:|'*8]
 for v in (*VIEWS,'urban_mean'):
  values=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS];lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in values)+'|')
 names=('LRAW_AS50__LRAW_NOAS50','LRAW_AS50__LRAW_AS30','LRAW_NOAS50__LRAW_NOAS30','LRAW_AS50__AS50','LRAW_NOAS50__NOAS50');lines+=['','|视图|'+'|'.join(names)+'|三因子交互pp|','|---|'+'---:|'*6]
 for v in (*VIEWS,'urban_mean'):
  f=formal['factorial'][v];values=[f['paired_accuracy_pp'][n] for n in names]+[f['three_factor_interaction_pp']];lines.append('|'+v+'|'+'|'.join(f'{x:+.6f}' for x in values)+'|')
 s=formal['stage'];lines+=['',f"同E50新As−新NoAs clean {s['clean_delta_pp']:+.6f}绝对百分点，阶段clean≥−.1pp={s['clean_absolute_0p1pp_guard_met']}；low/mid urban {s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f}pp。相对原NoAs50的clean变化另列{formal['clean_vs_original_NoAs50']['accuracy_delta_pp']:+.6f}pp。",'', 'LRawAs50−ownLRawAs30是固定配方预算增量；LRawAs50−原As50是同E50方法增量。匹配NoAs的相应变化独立报告，不能把NoAs退化计作纯As收益。13配对、6项二因子及三因子交互按view保存；不将不同方法/预算的增益相加预测。','', '|条件|视图|macro-F1 %|最差RX %|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|比较|视图|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['','## 源域预算结果','', 'E50来源结果只描述固定配方预算变化，不新增晋级gate、不自动扩种子/预算、不用目标微差救跑。原E30来源PASS与本次正式阶段判断分别保留；来源结果不会跳过任何既定正式收尾。','', '|条件|fullV clean %|panel clean %|mid/low等权 %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 d=source['decision'];lines+=['',f"固定LRawAs50−ownAs30 fullVclean {d['As50_minus_ownAs30_fullV_clean_pp']:+.6f}pp、mid/low {d['As50_minus_ownAs30_mid_low_mean_pp']:+.6f}pp；仅为预算读数，不赋予新的pass/fail。",'', 'fullV27000仅原生冻结accuracy/RX，不虚构全V逐包F1/修复。V1800两draw×三场独立重算，两draw不是两模型seed，源没有high urban；panel2.45GHz与训练/正式2.462GHz差异保留。全5源RX、L6300/U56700真值隐藏/V27000/256IQ/split392005/defaulthead和正式IQ/SNR/场景条件不变。','', '|条件|suburban|绝对accuracy %|自身clean−suburban pp|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v,g in formal['suburban_selfclean_gaps_pp'][a].items():lines.append(f"|{a}|{v}|{100*formal['summary'][a]['views'][v]['accuracy']:.6f}|{g:.6f}|")
 lines+=['', 'clean下降也可能缩小suburban gap，应同时看绝对准确率，不把缩差自动算保护。','', '|条件|student train|student eval|teacher train|teacher eval|native NN|native秒|','|---|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append(f"|{a}|{c['student_train_NN']}|{c['student_eval_NN']}|{c['teacher_train_NN']}|{c['teacher_eval_NN']}|{c['native_NN']}|{c['native_wall_seconds']:.3f}|")
 lines+=['','|成本组|native NN|训练段与自身源/正式读出 NN|','|---|---:|---:|']
 for g,q in costs['groups'].items():lines.append(f"|{g}|{q['native_NN']}|{q['segment_and_own_readouts_NN']}|")
 lines+=['',f"新增两模型各4440step/20epoch，全8880steps/40modelEpoch解析；共享smoke {costs['new_shared_smoke_NN']}NN，总新增训练/读出/smoke {costs['new_total_NN']}NN。父LRaw30、旧原30/50与旧读出分列复用，不双计prefix20或历史smoke。完整ordinary的计算仍在native wall内，未单独计时不编造耗时节省；source预测wall若未单列即缺失，不用离线analysiswall代替。",'', '|新条件|ordinary call/包|discarded MSE mean/max|used-raw MSE max|U包|EMA local/cum|','|---|---|---|---:|---:|---|']
 for a in CONDITIONS[-2:]:
  q=training['rows'][a];z=q['lraw'];e=q['EMA'];lines.append(f"|{a}|{z['ordinary_calls']}/{z['ordinary_packets']}|{z['discardedordinary_MSE_mean']:.8g}/{z['discardedordinary_MSE_max']:.8g}|{z['used_ordinary_raw_MSE_max']}|{q['physical_loader_packets']['U']}|{e['local_updates']}/{e['cumulative_updates']}|")
 lines+=['', 'first20原ordinary及U7历史仍保留，E21起LRaw，不能称全程NoAug。保留的是增强该call RNG消耗，不声称整个dropout/教师/伪标签轨迹逐步一致；旧stream不重扫，旧逐步IQ/物理scene匹配不假定。discarded MSE允许0，不用作候选筛选门槛。','',f"全部8×7固定预测与独立score齐备后才truth join，重核{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}分区；包括全7 accuracy/F1/worstRX、RX/day/TX配对错误与suburban差距。模型seed n=1，阶段PASS或微小clean正值不证明可重复双提升；已暴露基准背景保留，正式不反馈设计/调参/选种子/选择性重跑。分析0NN/CP/IQ/updates。"]
 return '\n'.join(lines)+'\n'

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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_lraw_tail50/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  base=Path(project)/'runs'/RUN
  if (base/'failure.json').is_file():raise RuntimeError('Owned technical failure preserved')
  for arm,own in rows(project):
   if arm in CONDITIONS[-2:] and (own/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Own source readout failed;formal closure remains with original runner')
  live,unknown=probe(project,release)
  if not live and not unknown:raise AwaitingArtifacts('Owned process handles absent before '+stage+' artifacts complete')
  write(out/'state.json',dict(status='WAIT_'+stage,live_owned_PIDs=live,observation_unknown=unknown,truth_opened=False,NN=0));sleep(15)

def validate_output(project,output):
 out=Path(output)
 if out.parent!=Path(project)/'runs'/RUN or re.fullmatch(r'analysis_auto_r[0-9]{2}',out.name) is None:raise ValueError('Only own immutable analysis output root')
 if out.exists():raise FileExistsError('Reconcile existing output;no duplicate launch')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
 if a.release.name!=TRAINING_RELEASE:raise ValueError('Only registered actual training release')
 out=validate_output(a.project,a.output);out.mkdir();started=time.perf_counter();write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=a.code_commit,NN=0,truth_opened=False))
 try:
  wait_until(a.project,a.release,'SOURCE',out);training=training_analysis(a.project);write(out/'training_logs.json',training);source=source_analysis(a.project);write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',training_run=RUN,NN=0,target_scores_read=False));print('SOURCE_ANALYSIS_COMPLETE',flush=True)
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=6,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
