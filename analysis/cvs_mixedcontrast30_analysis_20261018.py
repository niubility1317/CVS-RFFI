"""Zero-model source/log/formal recount for the fixed MIXED128 contrast E30 candidate.

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
RUN='20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_mixedContrast_E30_20261018_r01'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','MIXED_NOAS30','MIXED_AS30')
PAIRS=(('MIXED_AS30','MIXED_NOAS30'),('MIXED_AS30','AS30'),('MIXED_NOAS30','NOAS30'),('AS30','NOAS30'),('MIXED_AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('MIXED_NOAS30',p/RUN/'B00-s2026092701'),('MIXED_AS30',p/RUN/'B01-s2026092701')]

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
 candidate=summary['MIXED_AS30'];original=summary['AS30'];no=summary['NOAS30'];newno=summary['MIXED_NOAS30']
 clean=float(100*(candidate['full_V_clean']-original['full_V_clean']));midlow=float(100*(candidate['mid_low_mean']-original['mid_low_mean']))
 for key,value in [('fullV clean delta',clean),('midlow delta',midlow)]:logs.finite(value,key)
 cleanpass=bool(clean>=-.1-1e-12);starpass=bool(midlow>=.5-1e-12);passed=bool(cleanpass and starpass)
 return dict(fullV_clean_vs_original_As30_pp=clean,mid_low_mean_vs_original_As30_pp=midlow,clean_vs_original_As30_within_absolute_0p1pp=cleanpass,mid_low_gain_at_least_0p5pp=starpass,source_criteria_pass=passed,E50_source_eligible=passed,fixed_recipe_closed=bool(not passed),mixed_NoAs_vs_original_NoAs_fullV_clean_pp=float(100*(newno['full_V_clean']-no['full_V_clean'])),mixed_NoAs_vs_original_NoAs_mid_low_mean_pp=float(100*(newno['mid_low_mean']-no['mid_low_mean'])),fullV_clean_interaction_pp=float(100*((candidate['full_V_clean']-original['full_V_clean'])-(newno['full_V_clean']-no['full_V_clean']))),mid_low_mean_interaction_pp=float(100*((candidate['mid_low_mean']-original['mid_low_mean'])-(newno['mid_low_mean']-no['mid_low_mean']))),only_two_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False,per_packet_protection_gate=False,automatic_extension=False)

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
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,decision=source_conditions(summary),source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=SEED,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)

def actual_cost(value):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if any(type(value.get(k)) is not int or value[k]<0 for k in keys):raise ValueError('Actual native four counters required')
 counters={k:value[k] for k in keys};total=sum(counters.values())
 if value.get('native_NN',total)!=total:raise ValueError('Saved native sum differs from actual counters')
 return dict(**counters,native_training_and_validation_NN=total)
def exposure_summary(counts,all_ids):
 values=np.asarray([counts[x] for x in all_ids],int)
 return dict(unique_physical_packets=len(counts),all_L_physical_packets=len(all_ids),unexposed_L_packets=int((values==0).sum()),minimum=int(values.min()),maximum=int(values.max()),mean=float(values.mean()),exposure_count_histogram={str(k):int(v) for k,v in sorted(Counter(values.tolist()).items())},per_ID_matched_to_original_not_assumed=True)


CONTRAST_KEYS=('physical_inputs','anchors_per_term','anchors_per_physical_input','map_terms','raw_maps_used','extra_NN','true_As_anchors_per_term')

def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base reduction differs'),logs._near(r['total_L'],total,'Full original1.25L reduction differs')]
 for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+key],r[key],'Actual full6L concat telemetry differs'))
 errors.append(logs._near(m['concat/total_L']-r['base_L_before_shape'],.25*s['shape_RD'],'Raw .25 contribution differs'))
 return max(errors)

def check_mixed_contrast(r,c,m,sat):
 As=64 if sat else 0;step=r['step'];details=r['SupCon_details'];expected={(p,l) for p in ('first','second') for l in ('time_down','t2.pool')}
 if r['crossview_mode']!='MIXED128' or r['new_losses'] or r['extra_U_NN'] or len(details)!=4:raise ValueError('Only existing128-position mixed contrast scope')
 if (c['mode'],c['physical_inputs'],c['anchors_per_term'],c['anchors_per_physical_input'],c['map_terms'],c['selected_packets'],c['As_packets'],c['true_As_anchors_per_term'])!=('MIXED128',128,128,1,4,64,As,As):raise ValueError('MIXED128 position/anchor/As counts differ')
 if c['anchor_positions']!=list(range(128)) or c['selected_positions']!=[j for j in range(128) if (j+step)%2!=0]:raise ValueError('Reuse original128 positions and original selected64 mask')
 if any(c[k] for k in ('ordinary_maps_used','raw_maps_used','extra_NN','new_losses','ordinary_redraws')) or not c['same_IQ_each_pair'] or c['maps_source']!='existing mixed128 corresponding RD pass':raise ValueError('Contrast must reuse corresponding existing mixed forward maps')
 if c['selected_actual_view']!=('As_raw' if sat else 'raw') or c['selected_actual_view']!=r['selected_actual_view']:raise ValueError('Selected raw64/Asraw64 semantics differ')
 for k in ('selected_ordinary_MSE','selected_raw_MSE'):logs._near(c[k],r[k],'Actual selected view MSE telemetry differs')
 if not sat and c['selected_raw_MSE']!=0.:raise ValueError('NoAs selected64 must retain exact raw view')
 if any(len(c[k])!=128 for k in ('TX','RX','day')) or any(v not in range(6) for v in c['TX']) or any(v not in (1,3,4,6,8) for v in c['RX']) or any(v not in (1,2,3) for v in c['day']):raise ValueError('Only original labeled source128 input metadata')
 maps=c['pass_maps']
 if len(maps)!=4 or {(q['phase'],q['level']) for q in maps}!=expected or {(q['phase'],q['level']) for q in details}!=expected:raise ValueError('Four corresponding RD pass/level maps required')
 bymap={(q['phase'],q['level']):q['shape'] for q in maps}
 for q in maps:
  shape=q['shape']
  if len(shape)!=3 or shape[0]!=128 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('Existing mixed feature map shape differs')
 for level in ('time_down','t2.pool'):
  if bymap[('first',level)]!=bymap[('second',level)]:raise ValueError('Corresponding mixed RD passes must have matching shape')
 y=np.asarray(c['TX']);rx=np.asarray(c['RX']);positive=(y[:,None]==y[None,:])&(rx[:,None]!=rx[None,:]);negative=y[:,None]!=y[None,:];valid=positive.any(1)&negative.any(1);counts=positive.sum(1)
 for q in details:
  if (q['mode'],q['total_anchors'],q['ordinary_anchors'],q['selected_anchors'],q['anchors_per_physical_input'],q['true_satellite_anchors'])!=('MIXED128',128,64,64,1,As) or any(q[k] for k in ('additional_anchors','same_RX_positive_pairs','cross_view_positive_pairs','raw_pair_maps_used')) or q['selected_actual_view']!=c['selected_actual_view']:raise ValueError('No additional64/SAT192 or raw-pair maps permitted')
  if q['shape']!=bymap[(q['phase'],q['level'])] or q['valid_anchors']!=int(valid.sum()) or q['positive_min']!=(int(counts[valid].min()) if valid.any() else 0) or q['positive_max']!=(int(counts[valid].max()) if valid.any() else 0):raise ValueError('Actual existing crossRX anchor/map metadata differs')
 logs._near(r['SupCon'],float(np.mean([q['loss'] for q in details])),'Four existing contrast term reduction differs')
 for key in CONTRAST_KEYS:logs._near(m['contrast/'+key],c[key],'Actual contrast native scalar differs',1e-12)
 return sum(q['valid_anchors'] for q in details)

def parse_new_logs(root,arm):
 sat=arm=='MIXED_AS30';b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config']
 if cfg['crossview']['mode']!='MIXED128' or (cfg['mixed_contrast']['mode'],cfg['mixed_contrast']['anchors_per_input_position'],cfg['mixed_contrast']['physical_ID_uniqueness_required'],cfg['mixed_contrast']['total_anchors'])!=('MIXED128',1,False,128):raise ValueError('Actual candidate contrast config differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(30,6660) or (result['parent_epochs'],result['parent_optimizer_steps'])!=(20,4440):raise ValueError('Fixed own original20+10 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=1704960 or result['L_As']!=(142080 if sat else 0) or result['total_L_loss_scale']!=1.25:raise ValueError('Actual hiddenU/6L/As64or0 budget differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);count=Counter();sums=defaultdict(Counter);mse=[];maxerr=0.;selected=[];scenes=Counter();states=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','contrast_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New10 mixed contrast stream prefix')
  if i>STEPS:raise ValueError('Extra step outside own20+10 budget')
  r,s,n,u,c=items;ep=(i-1)//222+1;As=64 if sat else 0;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,c)) or n['epoch']!=ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous actual update/Sinc counters differ')
  if (r['L_forwards'],r['As_packets'],r['selected_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,As,64,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Actual original sixforward supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw control slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative crossRX weight differs',1e-12);valid=check_mixed_contrast(r,c,m,sat);maxerr=max(maxerr,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=As or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerics differ')
  logs.finite(m['train/loss'],'native total loss');number=m['train/pseudo_total'];picked=m['train/pseudo_selected']
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64 or (sat and set(r['scene_counts'])-set(SOURCE_SCENES)) or (not sat and r['scene_counts']!={'NONE':64}):raise ValueError('Actual full channel or raw identitycontrol counts differ')
  selected.append(c);scenes.update(r['scene_counts']);states.update(r['state_counts']);count.update(steps=1,L_NN=768,selected_packets=64,As_packets=As,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=valid,contrast_input_position_terms=128*4);mse.append(r['ordinary_raw_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE','selected_ordinary_MSE','selected_raw_MSE')});sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New10 final mixed contrast update missing')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();by_tx=Counter();by_rx=Counter();by_day=Counter();by_cell=Counter();repeated=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra L input metadata outside budget')
   c=selected[batches['L']]
   if any((z[1],z[2])!=(r,d) for z,r,d in zip(ids,c['RX'],c['day'])):raise ValueError('Contrast128 input metadata must retain original L positions')
   repeats=128-len({tuple(z) for z in ids});repeated['batches_with_repeated_physical_ID']+=bool(repeats);repeated['repeated_input_positions']+=repeats;repeated['maximum_repeated_positions_in_batch']=max(repeated['maximum_repeated_positions_in_batch'],repeats)
   for position in c['selected_positions']:
    z=ids[position];t=c['TX'][position];r=c['RX'][position];d=c['day'][position];exposure[tuple(z)]+=1;by_tx[str(t)]+=1;by_rx[str(r)]+=1;by_day[str(d)]+=1;by_cell[str((t,r,d))]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L']:raise ValueError('Full original L6300/U56700 source sequence differs')
 if cost['student_train_NN']!=1704960+567000 or cost['teacher_eval_NN']!=567000 or cost['teacher_train_NN'] or cost['student_eval_NN']!=324002:raise ValueError('Actual four counters/complete source loops differ')
 a=result['mixed_contrast']
 if (a['mode'],a['physical_inputs_per_step'],a['anchors_per_term'],a['anchors_per_input_position'],a['extra_anchor_positions'],a['pair_weights'],a['coefficient'],a['total_L_scale'],a['selected_packets'],a['actual_As_packets'],a['channel_calls'],a['identity_control_calls'],a['selected_actual_view'],a['map_terms_per_step'])!=('MIXED128',128,128,1,0,[.5,.5,.25],.1,1.25,142080,142080 if sat else 0,2220 if sat else 0,0 if sat else 2220,'As_raw' if sat else 'raw',4) or any(a[k] for k in ('extra_NN','raw_pair_maps_used','ordinary_maps_used','within_batch_physical_uniqueness_required')) or not a['same_original_sampler'] or not a['unchanged_IQ_and_forward_order']:raise ValueError('Actual unchanged .25/.25 mixed contrast scope differs')
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for q in coverage:
  coverage_scene[q['scene']]+=q['count'];coverage_state[q['state']]+=q['count'];coverage_tx[str(q['TX'])]+=q['count'];coverage_rx[str(q['RX'])]+=q['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_tx!=by_tx or coverage_rx!=by_rx:raise ValueError('Actual channel/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Actual original ordinary augmentation differs')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed own stdout')
 args=read(b/'actual_native_args.json');args=args.get('args',args);curves=[dict(epoch=q['epoch']+20,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,selected_packets=142080,As_packets=142080 if sat else 0),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},contrast=dict(a,input_position_semantics='one anchor per original L batch position per term;not physical-ID uniqueness',sampler_repetitions=dict(repeated)),selected_exposure=dict(TX=dict(by_tx),RX=dict(by_rx),day=dict(by_day),TX_RX_day=dict(by_cell),physical=exposure_summary(exposure,unique['L']),actual_As_packets=142080 if sat else 0,selected_actual_view='As_raw' if sat else 'raw',identity_control_selected_packets=0 if sat else 142080,scene_counts_are_identity_control_not_As=not sat,coverage=coverage,per_physical_scene_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)

def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Existing complete original30 audit required')
 result={};parent={};comparison={};lookup=dict(rows(project));private={}
 for name in ('NOAS30','AS30'):
  a=prior['rows'][name]
  if a['cost']['new_updates']!=6660 or a['cost']['new_epochs']!=30 or len(a['epoch_curves'])!=30:raise ValueError('Saved original30 audit schema/budget differs')
  result[name]=dict(reused=True,cost=a['cost'],epoch_curves=a['epoch_curves'],evidence='Existing complete original30 audit reused;no original step stream rescanned')
 for name,oldname in [('MIXED_NOAS30','NOAS30'),('MIXED_AS30','AS30')]:
  result[name],private[name]=parse_new_logs(lookup[name],name);result[name]['reused']=False
  a=read(lookup[oldname]/'stage20/result.json');parent[name]=dict(cost=actual_cost(a),epochs=20,updates=4440,wall_seconds=a['wall_seconds'],reused=True)
  priorcost=result[oldname]['cost'];oldtail={k:priorcost[k]-parent[name]['cost'][k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')}
  args=read(lookup[oldname]/'source/actual_native_args.json');args=args.get('args',args)
  curves=result[oldname]['epoch_curves'][20:];newcurves=result[name]['epoch_curves'];fields=('ordinary_raw_MSE','common_CE','common_KL','extra_CE','extra_KL','SupCon')
  diff={k:dict(record_count=len(curves),records_different=sum(a[k]!=b[k] for a,b in zip(curves,newcurves)),maximum_absolute_difference=max(abs(a[k]-b[k]) for a,b in zip(curves,newcurves))) for k in fields if all(k in q for q in (*curves,*newcurves))}
  comparison[name]=dict(reference_condition=oldname,reference_audit_reused=True,old_step_logs_rescanned=False,old_tail_four_counters=oldtail,new_tail_four_counters=result[name]['cost'],four_counters_equal=all(oldtail[k]==result[name]['cost'][k] for k in oldtail),available_epoch_mean_comparisons=diff,different_actual_arg_keys=sorted(k for k in set(args)|set(private[name]['args']) if args.get(k)!=private[name]['args'].get(k)),old_per_step_physical_IQ_equality_not_claimed=True,U_rule_unchanged_by_recipe=True)
 a='MIXED_NOAS30';b='MIXED_AS30';sequence=logs.compare_streams(lookup[a]/'source/training_sequence.jsonl.gz',lookup[b]/'source/training_sequence.jsonl.gz');mse_a=private[a]['ordinary_mse'];mse_b=private[b]['ordinary_mse']
 comparison['between_new_rows_only']=dict(physical_sequence=sequence,ordinary_MSE_records_different=sum(x!=y for x,y in zip(mse_a,mse_b)),maximum_ordinary_MSE_difference=max(abs(x-y) for x,y in zip(mse_a,mse_b)),comparison_is_new_NoAs_vs_new_As_not_old_reference=True)
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parent,new_model_rows=2,new_model_epochs=20,new_updates=4440,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparison,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

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
 interaction={view:comparisons['MIXED_AS30__AS30'][view]['accuracy_delta_pp']-comparisons['MIXED_NOAS30__NOAS30'][view]['accuracy_delta_pp'] for view in VIEWS};interaction['urban_mean']=comparisons['MIXED_AS30__AS30']['urban_mean_accuracy_delta_pp']-comparisons['MIXED_NOAS30__NOAS30']['urban_mean_accuracy_delta_pp']
 key='MIXED_AS30__MIXED_NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],contrast_interaction_pp=interaction,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['MIXED_AS30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['MIXED_AS30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  out[arm]=dict(native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm in CONDITIONS[:2])
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['NN']+real['NN'];parents={a:v['cost']['native_training_and_validation_NN'] for a,v in training['parent20'].items()};segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:])
 return dict(rows=out,new_model_epochs=20,new_updates=4440,new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,parent20_native_NN_reused=parents,cumulative_candidate_native_NN={a:parents[a]+out[a]['native_NN'] for a in CONDITIONS[2:]},cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;not speed evidence')

def render_report(source,formal,training,costs):
 lines=['# 完整 practical：mixed128 对比输入 E30 四条件对照','',RUN+'。两个新行各自从 own 原 FINAL20 student/EMA 追加10轮；原 NoAs30/As30 FINAL30 只读复用。唯一变量为已有 crossRX SupCon 的输入 maps 从 ORD128 改为 MIXED128，原 L batch 的每个位置一个 anchor，仍128；不要求128个物理ID全唯一；不新增 loss、分支、NN或U训练。','', 'P41 的 SAT192 已源失败。本次保留该负结果背景，关键区别是以 mixed128 替代原 ordinary128 的同物理槽位，而非 ordinary128 加额外64 satellite anchors；不是仅改名重跑 SAT192。原 .5 primary ordinary + .5 mixed + .25 raw、总L尺度1.25，原SupCon .1、6L前向及次序保持，未使用已关闭路线的 .125/.375。NoAs selected64 实际是 raw64，As selected64 为完整 practical(raw64)，真实As为0/64。','', '|视图|原NoAs30 %|原As30 %|MIX NoAs30 %|MIX As30 %|新As−新NoAs pp|新As−原As pp|新NoAs−原NoAs pp|交互 pp|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  vals=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS];diff=[formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp'] if v=='urban_mean' else formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for a,b in PAIRS[:3]];lines.append('|'+v+'|'+'|'.join(f'{z:.6f}' for z in vals+diff+[formal['contrast_interaction_pp'][v]])+'|')
 s=formal['stage'];lines+=['',f"相对匹配新NoAs：clean变化{s['clean_delta_pp']:+.6f}pp，clean≥−.1绝对百分点阶段约束={s['clean_absolute_0p1pp_guard_met']}；low/mid urban变化{s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f}pp。相对原NoAs clean变化另报{formal['clean_vs_original_NoAs']['accuracy_delta_pp']:+.6f}pp。交互=(新As−原As)−(新NoAs−原NoAs)；不把NoAs退化归为纯As收益。n=1无种子SD，不足.1pp的正变化为微小观察，不证明稳定双提升。",'', '|条件|view|macro-F1 %|最差RX %|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|比较|view|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 d=source['decision'];lines+=['',f"源固定资格：新As−原As fullV clean {d['fullV_clean_vs_original_As30_pp']:+.6f}pp（要求≥−.1），两draw mid/low等权{d['mid_low_mean_vs_original_As30_pp']:+.6f}pp（要求≥+.5），两项合取PASS={d['source_criteria_pass']}。NoAs源差及交互只报告，不追加门槛；科学失败仍完成全部正式。分析不启动E50或其它候选，不据正式分数救跑。",'','|条件|fullV clean %|V1800 clean %|mid/low等权 %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['','fullV27000来自原生冻结accuracy/RX，不能虚构逐包fullV F1或修复；V1800两draw三场景独立重算，两个draw不是两个modelseed。源无high urban；面板fc2.45GHz与训练/正式2.462GHz既有差异保留。已暴露基准背景保留，不宣称新的独立盲测。','',f"仅新两行各2220step/10epoch完整解析。旧原E30完整audit直接复用，不重扫旧steps；每候选累计30epoch/6660updates。对比maps/anchor组成、原L权重、sameU与原对照的可比范围：{json.dumps(training['reference_comparability'],ensure_ascii=False)}。不虚构已保存audit未提供的逐step IQ/随机路径或逐ID×scene相等。",'','|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|native wall s|','|---|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append(f"|{a}|{c['student_train_NN']}|{c['student_eval_NN']}|{c['teacher_train_NN']}|{c['teacher_eval_NN']}|{c['native_NN']}|{c['native_wall_seconds']:.3f}|")
 lines+=['',f"新增20modelEpoch/4440updates；训练+源/正式读出{costs['new_training_and_readouts_NN']}NN，共享smoke实际{costs['new_shared_smoke_NN']}NN，总新增{costs['new_total_NN']}NN。own E20前缀、原E30及其读出分别单列复用，不重计新增。student/ownEMA、源/正式前向分列在costs.json；wall为实际模型秒，不当并行总时长或加速证据。",'',f"全部4逻辑×7冻结预测及独立评分完成后才连接truth，独立重算{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}次分区，与原scorer逐项一致。所有view及RX/day/TX修复/新增错误、suburban绝对准确率及与自身clean差距完整保存。分析0NN/0IQ/0CP/0updates，不另训诊断头，正式IQ/SNR/信道/困难样本不改。"]
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_practical_mixed_contrast30/runner.py' in argv:live.append(pid)
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
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
