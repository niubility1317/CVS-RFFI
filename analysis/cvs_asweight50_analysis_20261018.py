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
import cvs_asweight30_analysis_20261018 as w30
old=w30.old
RUN='20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_AsWeight_E50_20261018_r01'
REFERENCE_RUN=old.REFERENCE_RUN
ORIGINAL50_RUN=old.RUN
WEIGHT30_RUN=w30.RUN
CONDITIONS=('NOAS30','AS30','WEIGHT_NOAS30','WEIGHT_AS30','NOAS50','AS50','WEIGHT_NOAS50','WEIGHT_AS50')
PAIRS=(('AS30','NOAS30'),('WEIGHT_AS30','WEIGHT_NOAS30'),('AS50','NOAS50'),('WEIGHT_AS50','WEIGHT_NOAS50'),('WEIGHT_AS30','AS30'),('WEIGHT_NOAS30','NOAS30'),('WEIGHT_AS50','AS50'),('WEIGHT_NOAS50','NOAS50'),('AS50','AS30'),('NOAS50','NOAS30'),('WEIGHT_AS50','WEIGHT_AS30'),('WEIGHT_NOAS50','WEIGHT_NOAS30'),('WEIGHT_AS50','NOAS50'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=4440;EPOCHS=20;STEP_OFFSET=6660
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values
actual_cost=w30.actual_cost;exposure_summary=w30.exposure_summary

def rows(project):
 p=Path(project)/'runs';result=[]
 for arm in CONDITIONS:
  budget=int(arm[-2:]);weighted=arm.startswith('WEIGHT_');run=(WEIGHT30_RUN if budget==30 else RUN) if weighted else REFERENCE_RUN if budget==30 else ORIGINAL50_RUN
  row=('B00' if 'NOAS' in arm else 'B01')+'-s2026092701';result.append((arm,p/run/row))
 return result

def factorial_effects(values):
 if set(values)!=set(CONDITIONS) or any(not np.isfinite(v) for v in values.values()):raise ValueError('All eight finite fixed condition values required')
 pair={a+'__'+b:100*(values[a]-values[b]) for a,b in PAIRS}
 two=dict(weight_x_As_at_E30=pair['WEIGHT_AS30__AS30']-pair['WEIGHT_NOAS30__NOAS30'],weight_x_As_at_E50=pair['WEIGHT_AS50__AS50']-pair['WEIGHT_NOAS50__NOAS50'],As_x_budget_original=pair['AS50__AS30']-pair['NOAS50__NOAS30'],As_x_budget_weight=pair['WEIGHT_AS50__WEIGHT_AS30']-pair['WEIGHT_NOAS50__WEIGHT_NOAS30'],weight_x_budget_NoAs=pair['WEIGHT_NOAS50__NOAS50']-pair['WEIGHT_NOAS30__NOAS30'],weight_x_budget_As=pair['WEIGHT_AS50__AS50']-pair['WEIGHT_AS30__AS30'])
 three=two['weight_x_As_at_E50']-two['weight_x_As_at_E30']
 if abs(three-(two['As_x_budget_weight']-two['As_x_budget_original']))>1e-10 or abs(three-(two['weight_x_budget_As']-two['weight_x_budget_NoAs']))>1e-10:raise ValueError('Three-factor algebra differs')
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
 a=summary['WEIGHT_AS50'];b=summary['WEIGHT_AS30'];clean=100*(a['full_V_clean']-b['full_V_clean']);urban=100*(a['mid_low_mean']-b['mid_low_mean']);clean_pass=bool(clean>=-1e-12);urban_pass=bool(urban>=.5-1e-12);passed=bool(clean_pass and urban_pass)
 return dict(newAs50_minus_ownAs30_fullV_clean_pp=float(clean),newAs50_minus_ownAs30_mid_low_mean_pp=float(urban),clean_nonnegative=clean_pass,mid_low_gain_at_least_0p5pp=urban_pass,source_criteria_pass=passed,fixed_02_03_source_eligible=passed,automatic_multiseed_authorized=False,fixed_recipe_closed=not passed,only_two_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False)

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

WEIGHT_KEYS=('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')

def check_weighted_L(r,s,m,w):
 """All keys below are emitted by the actual full6L hook, not inferred names."""
 errors=[]
 for prefix,record in [('common',r),('extra',r),('ordinary64',r),('selected64',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'RDrop CE/KL decomposition differs'))
 if (r['ordinary_group_weight'],r['selected_group_weight'])!=(.125,.375):raise ValueError('Only fixed .125/.375 mixed group weights')
 for suffix in ('RD','CE','KL'):
  expected=.125*r['ordinary64_'+suffix]+.375*r['selected64_'+suffix]
  errors.append(logs._near(r['mixed_weighted_'+suffix],expected,'Actual mixed '+suffix+' weighted means differ'))
  errors.append(logs._near(r['extra_'+suffix],.5*(r['ordinary64_'+suffix]+r['selected64_'+suffix]),'Actual unweighted128 '+suffix+' split differs'))
  errors.append(logs._near(w['unweighted_mixed_'+suffix],r['extra_'+suffix],'Weight step unweighted mixed differs'))
 for key in WEIGHT_KEYS:
  errors.append(logs._near(w[key],r[key],'Weight step/reference scalar differs'))
  errors.append(logs._near(m['weight/'+key],r[key],'Native weight scalar differs'))
 base=.5*r['common_RD']+r['mixed_weighted_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Weighted base_L reduction differs'),logs._near(r['total_L'],total,'Full weighted1.25L reduction differs')]
 for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+key],r[key],'Actual full6L concat telemetry differs'))
 for suffix in ('RD','CE','KL'):errors.append(logs._near(m['weight/raw_'+suffix],s['shape_'+suffix],'Native raw RDrop scalar differs'))
 errors.append(logs._near(m['concat/total_L']-r['base_L_before_shape'],.25*s['shape_RD'],'Actual raw contribution differs'))
 return max(errors)

def parse_new_logs(root,arm):
 sat=arm=='WEIGHT_AS50';b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config']
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(20,4440,4440,4440) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(50,11100):raise ValueError('Fixed own30+20 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=3409920 or result['L_As']!=(284160 if sat else 0):raise ValueError('Actual hidden-U/6L/As64or0 budget differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);count=Counter();sums=defaultdict(Counter);mse=[];maxerr=0.;selected=[];scenes=Counter();states=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','weight_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New10 weighted source stream prefix')
  if i>STEPS:raise ValueError('Extra step outside own30+20 budget')
  r,s,n,u,w=items;ep=(i-1)//222+1;As=64 if sat else 0;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,w)) or n['epoch']!=ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous native/Sinc counters differ')
  if (r['L_forwards'],r['As_packets'],r['selected_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,As,64,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Actual sixforward full supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw control slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative crossRX weight differs',1e-12);valid=logs.relation(r,True);maxerr=max(maxerr,check_weighted_L(r,s,m,w))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=As or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerics differ')
  logs.finite(m['train/loss'],'native total loss');number=m['train/pseudo_total'];picked=m['train/pseudo_selected']
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64 or (sat and set(r['scene_counts'])-set(SOURCE_SCENES)) or (not sat and r['scene_counts']!={'NONE':64}):raise ValueError('Actual channel or identitycontrol counts differ')
  if (w['As_packets'],w['selected_packets'])!=(As,64) or w['selected_positions']!=[j for j in range(128) if (j+i)%2!=0] or any(len(w[k])!=64 for k in ('selected_TX','selected_RX','selected_day')):raise ValueError('Original deterministic selected64 mask differs')
  if any(t not in range(6) for t in w['selected_TX']) or any(r not in (1,3,4,6,8) for r in w['selected_RX']) or any(d not in (1,2,3) for d in w['selected_day']):raise ValueError('Selected labeled source roles differ')
  if r['selected_actual_view']!=('As_raw' if sat else 'raw') or w['selected_actual_view']!=r['selected_actual_view']:raise ValueError('Actual selected64 raw/Asraw semantics differ')
  for key in ('selected_ordinary_MSE','selected_raw_MSE'):maxerr=max(maxerr,logs._near(w[key],r[key],'Weight/reference selected view MSE differs'))
  if not sat and r['selected_raw_MSE']!=0.:raise ValueError('NoAs selected64 must retain exact original raw IQ')
  selected.append({k:w[k] for k in ('selected_positions','selected_TX','selected_RX','selected_day')});scenes.update(r['scene_counts']);states.update(r['state_counts'])
  count.update(steps=1,L_NN=768,selected_packets=64,As_packets=As,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=valid)
  mse.append(r['ordinary_raw_MSE']);sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE','selected_ordinary_MSE','selected_raw_MSE',*WEIGHT_KEYS)});sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New10 final weighted update missing')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();by_tx=Counter();by_rx=Counter();by_day=Counter();by_cell=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   w=selected[batches['L']]
   for position,t,r,d in zip(w['selected_positions'],w['selected_TX'],w['selected_RX'],w['selected_day']):
    z=ids[position]
    if (z[1],z[2])!=(r,d):raise ValueError('Actual selected exposure/physical loader RX/day differs')
    exposure[tuple(z)]+=1;by_tx[str(t)]+=1;by_rx[str(r)]+=1;by_day[str(d)]+=1;by_cell[str((t,r,d))]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=4440,U=4440) or packets!=dict(L=568320,U=1134000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L']:raise ValueError('Full actual source L6300/U56700 sequence differs')
 if cost['student_train_NN']!=3409920+1134000 or cost['teacher_eval_NN']!=1134000 or cost['teacher_train_NN'] or cost['student_eval_NN']!=594002:raise ValueError('Actual four counters/complete source loops differ')
 a=result['mixed_weight']
 if (a['ordinary'],a['selected'],a['primary_ordinary'],a['raw'],a['total_L_scale'],a['selected_packets'],a['actual_As_packets'],a['channel_calls'],a['identity_control_calls'])!=(.125,.375,.5,.25,1.25,284160,284160 if sat else 0,4440 if sat else 0,0 if sat else 4440) or not a['unchanged_IQ_and_forward_order']:raise ValueError('Actual weighting/channel/cost scope differs')
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for q in coverage:
  coverage_scene[q['scene']]+=q['count'];coverage_state[q['state']]+=q['count'];coverage_tx[str(q['TX'])]+=q['count'];coverage_rx[str(q['RX'])]+=q['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_tx!=by_tx or coverage_rx!=by_rx:raise ValueError('Actual coverage/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Actual original ordinary augmentation differs')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed own stdout')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 curves=[dict(epoch=q['epoch']+30,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=20,local_updates=4440,parent_epochs=30,parent_updates=6660,cumulative_epochs=50,cumulative_updates=11100,L_NN=3409920,selected_packets=284160,As_packets=284160 if sat else 0),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},selected_exposure=dict(TX=dict(by_tx),RX=dict(by_rx),day=dict(by_day),TX_RX_day=dict(by_cell),physical=exposure_summary(exposure,unique['L']),actual_As_packets=284160 if sat else 0,selected_actual_view='As_raw' if sat else 'raw',identity_control_selected_packets=0 if sat else 284160,scene_counts_are_identity_control_not_As=not sat,coverage=coverage,per_physical_scene_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)


def compare_old_tail(newroot,oldroot):
 sequence=logs.compare_streams(newroot/'source/training_sequence.jsonl.gz',oldroot/'source/training_sequence.jsonl.gz');counts=Counter();numeric=0.
 for a,b in zip_longest(logs.records(newroot/'source/reference_steps.jsonl.gz'),logs.records(oldroot/'source/reference_steps.jsonl.gz')):
  if a is None or b is None:raise logs.IncompleteTrainingLogs('Matched original tail reference stream prefix')
  counts['records']+=1;counts['ordinary_MSE_records_different']+=a['ordinary_raw_MSE']!=b['ordinary_raw_MSE'];counts['scene_counts_records_different']+=a['scene_counts']!=b['scene_counts'];counts['state_counts_records_different']+=a['state_counts']!=b['state_counts'];counts['actual_As_count_records_different']+=a['As_packets']!=b['As_packets'];numeric=max(numeric,abs(a['ordinary_raw_MSE']-b['ordinary_raw_MSE']))
 if counts['records']!=4440:raise logs.IncompleteTrainingLogs('Exactly4440 matched original tail records required')
 return dict(physical_sequence=sequence,**dict(counts),maximum_ordinary_MSE_difference=numeric,supported_physical_mask_and_scene_state_count_scope_only=True,per_packet_IQ_or_channel_trace_equality_not_claimed=True)

def cumulative_cost(prefix,tail):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN');result={key:prefix[key]+tail[key] for key in keys};result['native_training_and_validation_NN']=sum(result.values());return result

def training_analysis(project):
 base=Path(project)/'runs';old50=read(base/ORIGINAL50_RUN/'analysis_auto_r01/training_logs.json');prior=read(base/WEIGHT30_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS' or old50['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Completed existing original/weighted audits required')
 result={};parents={};comparisons={};lookup=dict(rows(project))
 for name in ('NOAS30','AS30'):
  a=prior['rows'][name]
  if a['cost']['new_updates']!=6660 or len(a['epoch_curves'])!=30:raise ValueError('Existing original E30 full audit differs')
  result[name]=dict(cost=a['cost'],cumulative_native_cost={k:a['cost'][k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')},epoch_curves=a['epoch_curves'],complete_trajectory_curves=a['epoch_curves'],reused=True,evidence='Complete existing originalE30 audit;not rescanned')
 for name in ('WEIGHT_NOAS30','WEIGHT_AS30'):
  a=prior['rows'][name];prefix=prior['parent20'][name]
  if a['cost']['local_updates']!=2220 or len(a['epoch_curves'])!=10 or prefix['updates']!=4440:raise ValueError('Existing weighted E20+10 full audit differs')
  control='NOAS30' if 'NOAS' in name else 'AS30';curves=result[control]['epoch_curves'][:20]+a['epoch_curves'];cumulative=cumulative_cost(prefix['cost'],a['cost'])
  result[name]=dict(cost=a['cost'],cumulative_native_cost=cumulative,epoch_curves=a['epoch_curves'],complete_trajectory_curves=curves,reused=True,evidence='Existing own prefix20+weighted10 audit reused;not rescanned')
 for name in ('NOAS50','AS50'):
  a=old50['rows'][name];previous=name.replace('50','30')
  if a['cost']['local_updates']!=4440 or len(a['epoch_curves'])!=20:raise ValueError('Existing original E30+20 full audit differs')
  result[name]=dict(cost=a['cost'],cumulative_native_cost=cumulative_cost(result[previous]['cumulative_native_cost'],a['cost']),epoch_curves=a['epoch_curves'],complete_trajectory_curves=result[previous]['complete_trajectory_curves']+a['epoch_curves'],reused=True,evidence='Existing originalE50 complete extra20 audit reused;not rescanned')
 for name,control in [('WEIGHT_NOAS50','NOAS50'),('WEIGHT_AS50','AS50')]:
  a,private=parse_new_logs(lookup[name],name);previous=name.replace('50','30');parents[name]=dict(cumulative_native_cost=result[previous]['cumulative_native_cost'],epochs=30,updates=6660,own_parent_logical_row=previous,reused=True)
  a.update(reused=False,cumulative_native_cost=cumulative_cost(result[previous]['cumulative_native_cost'],a['cost']),complete_trajectory_curves=result[previous]['complete_trajectory_curves']+a['epoch_curves']);result[name]=a
  comparisons[name]=compare_old_tail(lookup[name],lookup[control]);args=read(lookup[control]/'source/actual_native_args.json');args=args.get('args',args);comparisons[name]['different_actual_arg_keys']=sorted(k for k in set(args)|set(private['args']) if args.get(k)!=private['args'].get(k))
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
 key='WEIGHT_AS50__WEIGHT_NOAS50';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],factorial=effects,clean_vs_original_NoAs50=dict(accuracy_delta_pp=pairs['WEIGHT_AS50__NOAS50']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['WEIGHT_AS50__NOAS50']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)


def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  a=training['rows'][arm];c=a['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json'];group='new_weight_E50_segments' if arm in CONDITIONS[-2:] else 'old_original_E50_segments' if arm in ('NOAS50','AS50') else 'old_weight_E30_segments' if arm.startswith('WEIGHT_') else 'old_original_E30_models'
  out[arm]=dict(group=group,native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],cumulative_native_cost=a['cumulative_native_cost'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm not in CONDITIONS[-2:])
 groups={group:dict(rows=[a for a,q in out.items() if q['group']==group],native_NN=sum(q['native_NN'] for q in out.values() if q['group']==group),segment_and_own_readouts_NN=sum(q['segment_and_own_readouts_NN'] for q in out.values() if q['group']==group)) for group in ('new_weight_E50_segments','old_original_E50_segments','old_weight_E30_segments','old_original_E30_models')}
 base=Path(project)/'runs'/RUN;synthetic=read(base/'synthetic_smoke.json');real=read(base/'real_checkpoint_smoke.json')
 if synthetic['status']!='PASS' or real['status']!='PASS' or synthetic['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=synthetic['NN']+real['NN'];parents={a:q['cumulative_native_cost'] for a,q in training['parent30'].items()}
 return dict(rows=out,groups=groups,new_model_epochs=40,new_updates=8880,new_training_and_readouts_NN=groups['new_weight_E50_segments']['segment_and_own_readouts_NN'],new_shared_smoke_NN=shared,new_total_NN=groups['new_weight_E50_segments']['segment_and_own_readouts_NN']+shared,parent_weighted30_native_costs_reused=parents,cumulative_candidate_epochs_each=50,cumulative_candidate_updates_each=11100,old_costs_not_recharged=True,shared_prefix20_not_double_counted_in_group_totals=True,historical_shared_smokes_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Per-segment actual model-seconds;not parallel elapsed or speed evidence')

def render_report(source,formal,training,costs):
 lines=['# As 权重配方 E30→E50：八条件同种子预算分析','',RUN+'；固定seed2026092701。只新增加权NoAs/As各自 ownFINAL30 student/EMA+20；总轨迹20+10+20，每段resetAdamW/cos20（新增段），不是连续cos50。原weight30代码固定3befd47c8041d81f9430eda53562d0821624ff04。','', '既有NoAs30/As30、加权NoAs30/As30、原NoAs50/As50冻结结果全部复用。E30合法源域两条件通过并保存后才固定E50；新正式结果不参与该决定。新段6L1.25、.125ordinary64/.375selected64、默认头、原U、场景/物理和IQ处理保持；NoAs selected仍是raw64，不是假As或ordinary128。','', '|视图|NoAs30 %|As30 %|wNoAs30 %|wAs30 %|NoAs50 %|As50 %|wNoAs50 %|wAs50 %|','|---|'+'---:|'*8]
 for view in (*VIEWS,'urban_mean'):
  values=[100*(formal['summary'][arm]['urban_mean_accuracy'] if view=='urban_mean' else formal['summary'][arm]['views'][view]['accuracy']) for arm in CONDITIONS];lines.append('|'+view+'|'+'|'.join(f'{q:.6f}' for q in values)+'|')
 lines+=['', '|视图|wAs50−wNoAs50 pp|wAs50−wAs30 pp|wNoAs50−wNoAs30 pp|wAs50−As50 pp|wNoAs50−NoAs50 pp|三因子交互 pp|','|---|---:|---:|---:|---:|---:|---:|']
 names=('WEIGHT_AS50__WEIGHT_NOAS50','WEIGHT_AS50__WEIGHT_AS30','WEIGHT_NOAS50__WEIGHT_NOAS30','WEIGHT_AS50__AS50','WEIGHT_NOAS50__NOAS50')
 for view in (*VIEWS,'urban_mean'):
  f=formal['factorial'][view];values=[f['paired_accuracy_pp'][n] for n in names]+[f['three_factor_interaction_pp']];lines.append('|'+view+'|'+'|'.join(f'{q:+.6f}' for q in values)+'|')
 s=formal['stage'];lines+=['',f"同E50加权As−加权NoAs clean变化{s['clean_delta_pp']:+.6f}绝对百分点，阶段≥−.1pp={s['clean_absolute_0p1pp_guard_met']}。low/mid urban {s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f}pp。相对原NoAs50的clean变化{formal['clean_vs_original_NoAs50']['accuracy_delta_pp']:+.6f}pp另列。",'', '每一view完整保存12个单因素配对差、6个二因子交互及三因子交互。三因子=(weight×As在E50)−(weight×As在E30)，并核对等价的As×budget与weight×budget差。NoAs随预算变化与As自己预算收益分开，不能把对照退化全归因As，不能将不同method与budget收益相加预测。','', '|条件|view|macro-F1 %|最差RX %|','|---|---|---:|---:|']
 for arm in CONDITIONS:
  for view in VIEWS:
   q=formal['summary'][arm]['views'][view];lines.append(f"|{arm}|{view}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['', '|比较|view|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for view in ('clean','practical_mid_urban','practical_low_urban'):
   q=formal['paired'][a+'__'+b][view];lines.append(f"|{a}−{b}|{view}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 d=source['decision'];lines+=['',f"后续固定02/03资格只看两项源条件：wAs50−ownwAs30 fullV clean {d['newAs50_minus_ownAs30_fullV_clean_pp']:+.6f}pp（要求≥0）；V1800两draw mid/low等权{d['newAs50_minus_ownAs30_mid_low_mean_pp']:+.6f}pp（要求≥+.5）。合取PASS={d['source_criteria_pass']}，固定02/03源资格={d['fixed_02_03_source_eligible']}；当前未自动授权多seed。NoAs及原As50源差只报告不加门槛，源科学失败仍完成正式收尾。",'', '|条件|fullV clean %|V1800 clean %|源mid/low等权 %|','|---|---:|---:|---:|']
 for arm in CONDITIONS:
  q=source['summary'][arm];lines.append(f"|{arm}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['', 'fullV27000为原生冻结clean指标，无全V逐包预测，不能虚构fullV F1/修复；固定V1800两draw×三场景面板独立重算，源没有high urban。源原始opaque数组仅fullindex0由既有producer消费，分析只读保存的logits/metadata；fc2.45GHz与训练/正式2.462GHz既有差异保留。','', '新两行8880step/40模型epoch全部解析，旧20/30/原50完整audit复用；每条30/50轨迹完整曲线保留。与原E50tail的物理序列、ordinaryMSE及scene/state计数实际比较见training_logs.json，仅声称已记录范围，不声称逐点IQ或逐物理×scene实现相同。','', '|成本组|本组native NN|训练段+自身源/正式读出 NN|','|---|---:|---:|']
 for group,q in costs['groups'].items():lines.append(f"|{group}|{q['native_NN']}|{q['segment_and_own_readouts_NN']}|")
 lines+=['',f"本run共享smoke实际{costs['new_shared_smoke_NN']}NN；新增两行+读出+smoke合计{costs['new_total_NN']}NN。ownweighted30亲本及原30/原50/E30测试历史成本分列，不重计新增；组总计不重复加共享prefix20。每row学生train/eval、教师train/eval、actual wall、native累计成本见costs.json。分析0NN/0CP/0IQ/0更新，模型秒不是并行加速证据。",'',f"全部8logical×7冻结预测及独立评分齐备后才连接truth；独立核对{formal['independent_confusion_matrices']}个CM、{formal['partition_reconciliations']}次RX/day/TX分区合并。七view全部accuracy/F1/worstRX、分层配对修复/新增错误、suburban相对自身clean差距均保存。",'', '单seed观察不证明可重复；不足0.1pp的clean正变化标为微小观察，阶段基本不降不等于最终双提升。已暴露基准背景保留，不宣称新的盲测；正式IQ、SNR、信道、困难样本不改，正式结果不回流权重、种子、模型选择或选择性重跑。']
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_practical_as_weight50/runner.py' in argv:live.append(pid)
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
