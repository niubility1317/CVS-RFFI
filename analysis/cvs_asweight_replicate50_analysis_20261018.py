"""Conditional fixed-seed02/03 weighted E30+E50 replication analysis, zero model work."""
import argparse,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_asweight50_analysis_20261018 as base
w30=base.w30;old=base.old;logs=base.logs
RUN='20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01'
TRAINING_RELEASE='cvs_FcrossRX_AsWeight_E50_s0203_20261018_r01'
ORIGINAL30_REPLICATION='20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'
ORIGINAL50_REPLICATION='20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01'
SEEDS=(2026092701,2026092702,2026092703);NEW_SEEDS=SEEDS[1:]
CONDITIONS=base.CONDITIONS;PAIRS=base.PAIRS;VIEWS=base.VIEWS;SOURCE_SCENES=base.SOURCE_SCENES;COUNT=168000
read=base.read;write=base.write;metric=base.metric;paired=base.paired;write_csv=base.write_csv
AwaitingArtifacts=base.AwaitingArtifacts;actual_cost=base.actual_cost;exposure_summary=base.exposure_summary;check_weighted_L=base.check_weighted_L;WEIGHT_KEYS=base.WEIGHT_KEYS

def label(condition,seed):return condition+'-s'+str(seed)
def split_label(value):
 condition,seed=value.rsplit('-s',1);return condition,int(seed)
def physical_row(condition,seed):return ('B00' if 'NOAS' in condition else 'B01')+'-s'+str(seed)
def rows(project):
 out=[];p=Path(project)/'runs';seed01=dict(base.rows(project))
 for seed in SEEDS:
  for condition in CONDITIONS:
   if seed==SEEDS[0]:root=seed01[condition]
   elif condition.startswith('WEIGHT_'):root=p/RUN/physical_row(condition,seed)/('weighted30' if condition.endswith('30') else 'weighted50')
   else:root=p/(ORIGINAL30_REPLICATION if condition.endswith('30') else ORIGINAL50_REPLICATION)/physical_row(condition,seed)
   out.append((label(condition,seed),root))
 return out

def stats3(values):
 a=np.asarray(values,float)
 if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('All three fixed finite model-seed values required')
 return dict(values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),model_seed_count=3)

def seed01_source_eligibility(project):
 path=Path(project)/'runs'/base.RUN/'analysis_auto_r01/source_summary.json';saved=read(path);d=saved['decision']
 if saved['status']!='SOURCE_ANALYZED' or saved['run_id']!=base.RUN or saved['target_scores_read'] or any(saved[k] for k in ('NN','IQ_reads','checkpoint_reads','updates')):raise ValueError('Only completed seed01 source-only qualification')
 if any(type(d[k]) is not bool or d[k] is not True for k in ('source_criteria_pass','fixed_02_03_source_eligible','clean_nonnegative','mid_low_gain_at_least_0p5pp')):raise ValueError('Conditional replication not source-eligible;do not use this adapter')
 if d['newAs50_minus_ownAs30_fullV_clean_pp']<0 or d['newAs50_minus_ownAs30_mid_low_mean_pp']<.5-1e-12:raise ValueError('Fixed source eligibility numerical conditions differ')
 return dict(status='SOURCE_ELIGIBILITY_REUSED',source_run_id=base.RUN,decision=d,target_scores_read=False,new_thresholds=False,automatic_future_candidates=False)

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  condition,seed=split_label(arm);names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json'];budget=int(condition[-2:]);expected=(10,2220) if budget==30 else (20,4440);run=RUN if seed in NEW_SEEDS and condition.startswith('WEIGHT_') else root.parent.name
  if (c['epoch'],c['optimizer_steps'])!=expected or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=seed or c['config']['row_id']!=physical_row(condition,seed) or c['config']['run_id']!=run:raise ValueError('Actual nested-stage seed/physicalrow/fixed budget differs')
  if budget==50 and (c.get('cumulative_epochs'),c.get('cumulative_optimizer_steps'))!=(50,11100):raise ValueError('Actual cumulativeE50 differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('training_updates',0) or q.get('head_fit_updates',0):raise ValueError('Frozen source/defaulthead metadata differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all24 logical rows complete all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def aggregate_factorial(per_seed):
 output={}
 for view in per_seed[str(SEEDS[0])]:
  pairs=per_seed[str(SEEDS[0])][view]['paired_accuracy_pp'];twos=per_seed[str(SEEDS[0])][view]['two_factor_interactions_pp']
  output[view]=dict(paired_accuracy_pp={key:stats3([per_seed[str(s)][view]['paired_accuracy_pp'][key] for s in SEEDS]) for key in pairs},two_factor_interactions_pp={key:stats3([per_seed[str(s)][view]['two_factor_interactions_pp'][key] for s in SEEDS]) for key in twos},three_factor_interaction_pp=stats3([per_seed[str(s)][view]['three_factor_interaction_pp'] for s in SEEDS]))
 return output

def parse_segment(root,condition,stdout_path):
 budget=int(condition[-2:]);E=10 if budget==30 else 20;N=E*222;offset=4440 if budget==30 else 6660;parent_epochs=20 if budget==30 else 30;parent_updates=offset;U=E*56700
 sat=condition in ('WEIGHT_AS30','WEIGHT_AS50');b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config']
 seed=cfg['model_seed']
 if seed not in NEW_SEEDS or cfg['target_budget']!=budget or result['actual_F_factory_seed']!=seed or cfg['seed_binding']['model_and_native_training']!=seed or cfg['seed_binding']['F_factory_model_seed']!=seed:raise ValueError('Actual new factory/native model seed or nestedstage differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(E,N,N,N) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(budget,offset+N):raise ValueError('Fixed new10/new20 stage budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=(N*768) or result['L_As']!=((N*64) if sat else 0):raise ValueError('Actual hidden-U/6L/As64or0 budget differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,E);count=Counter();sums=defaultdict(Counter);mse=[];maxerr=0.;selected=[];scenes=Counter();states=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','weight_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New10 weighted source stream prefix')
  if i>N:raise ValueError('Extra step outside own30+20 budget')
  r,s,n,u,w=items;ep=(i-1)//222+1;As=64 if sat else 0;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=offset+i for q in (r,s,u,w)) or n['epoch']!=ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous native/Sinc counters differ')
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
 if count['steps']!=N:raise logs.IncompleteTrainingLogs('New10 final weighted update missing')
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
 if batches!=dict(L=N,U=N) or packets!=dict(L=N*128,U=U) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L']:raise ValueError('Full actual source L6300/U56700 sequence differs')
 if cost['student_train_NN']!=(N*768)+U or cost['teacher_eval_NN']!=U or cost['teacher_train_NN'] or cost['student_eval_NN']!=(E*27000+54002):raise ValueError('Actual four counters/complete source loops differ')
 a=result['mixed_weight']
 if (a['ordinary'],a['selected'],a['primary_ordinary'],a['raw'],a['total_L_scale'],a['selected_packets'],a['actual_As_packets'],a['channel_calls'],a['identity_control_calls'])!=(.125,.375,.5,.25,1.25,(N*64),(N*64) if sat else 0,N if sat else 0,0 if sat else N) or not a['unchanged_IQ_and_forward_order']:raise ValueError('Actual weighting/channel/cost scope differs')
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for q in coverage:
  coverage_scene[q['scene']]+=q['count'];coverage_state[q['state']]+=q['count'];coverage_tx[str(q['TX'])]+=q['count'];coverage_rx[str(q['RX'])]+=q['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_tx!=by_tx or coverage_rx!=by_rx:raise ValueError('Actual coverage/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Actual original ordinary augmentation differs')
 text=Path(stdout_path).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed own stdout')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 curves=[dict(epoch=q['epoch']+parent_epochs,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs]
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',model_seed=seed,actual_F_factory_seed=result['actual_F_factory_seed'],target_budget=budget,cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=E,local_updates=N,parent_epochs=parent_epochs,parent_updates=parent_updates,cumulative_epochs=budget,cumulative_updates=offset+N,L_NN=(N*768),selected_packets=(N*64),As_packets=(N*64) if sat else 0),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},selected_exposure=dict(TX=dict(by_tx),RX=dict(by_rx),day=dict(by_day),TX_RX_day=dict(by_cell),physical=exposure_summary(exposure,unique['L']),actual_As_packets=(N*64) if sat else 0,selected_actual_view='As_raw' if sat else 'raw',identity_control_selected_packets=0 if sat else (N*64),scene_counts_are_identity_control_not_As=not sat,coverage=coverage,per_physical_scene_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)



def source_analysis(project):
 started=time.perf_counter();meta=ensure_metadata(project,'source');data={};summary={};metrics=[]
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite fixed source panel/logits absent')
  if data and any(not np.array_equal(x[k],data[label('NOAS30',SEEDS[0])]['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source physical panel required')
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
 pairs={};per_seed={}
 for seed in SEEDS:
  baseline=label('NOAS30',seed);cohort=data[baseline]['pred']['clean']==data[baseline]['x']['TX']
  for a,b in PAIRS:
   left,right=label(a,seed),label(b,seed);key=str(seed)+'__'+a+'__'+b;pairs[key]={}
   for view in ('clean',*SOURCE_SCENES):
    y=data[left]['x']['TX'] if view=='clean' else np.tile(data[left]['x']['TX'],2);co=cohort if view=='clean' else np.tile(cohort,2);pairs[key][view]=paired(y,data[right]['pred'][view],data[left]['pred'][view],co)
  per_seed[str(seed)]={view:base.factorial_effects({condition:(summary[label(condition,seed)]['full_V_clean'] if view=='fullV_clean' else summary[label(condition,seed)]['mid_low_mean'] if view=='mid_low_mean' else summary[label(condition,seed)]['panel'][view]['accuracy']) for condition in CONDITIONS}) for view in ('fullV_clean','mid_low_mean','clean',*SOURCE_SCENES)}
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,paired=pairs,per_seed_factorial=per_seed,factorial_three_seed=aggregate_factorial(per_seed),eligibility=seed01_source_eligibility(project),source_unique_V=1800,channel_draws=2,model_seed_count=3,model_seeds=list(SEEDS),fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,channel_draws_are_not_model_seeds=True,automatic_future_candidates=False,wall_seconds=time.perf_counter()-started)

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
  if private and (not np.array_equal(ids,private[label('NOAS30',SEEDS[0])]['ids']) or not np.array_equal(y,private[label('NOAS30',SEEDS[0])]['y'])):raise ValueError('All rows must share physical pairing')
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
     metrics.append(dict(arm=arm,row_id=physical_row(*split_label(arm)),model_seed=split_label(arm)[1],view=v,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     if not np.array_equal(subtotal,metric(y,pred[v])['confusion']):raise ValueError('Formal RX/day/TX partitions differ')
     partitions+=1
   q=next(r for r in metrics if r['view']==v and r['dimension']=='overall');views[v]=dict(q,worst_RX_accuracy=min(r['accuracy'] for r in metrics if r['view']==v and r['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=not (split_label(arm)[1] in NEW_SEEDS and split_label(arm)[0].startswith('WEIGHT_')));private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
 paired_strata=[];pairs={};comparisons={};per_seed={};stages={}
 for seed in SEEDS:
  anchor=label('NOAS30',seed);cohort=private[anchor]['pred']['clean']==private[anchor]['y'];effects={view:base.factorial_effects({condition:(results[label(condition,seed)]['urban_mean_accuracy'] if view=='urban_mean' else results[label(condition,seed)]['views'][view]['accuracy']) for condition in CONDITIONS}) for view in (*VIEWS,'urban_mean')}
  per_seed[str(seed)]=effects
  for a,b in PAIRS:
   left,right=label(a,seed),label(b,seed);key=str(seed)+'__'+a+'__'+b;pairs[key]={};comparisons[key]={}
   for view in VIEWS:
    d=paired(private[left]['y'],private[right]['pred'][view],private[left]['pred'][view],cohort);pairs[key][view]=d;comparisons[key][view]=dict(accuracy_delta_pp=d['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[left]['views'][view]['macro_f1']-results[right]['views'][view]['macro_f1']),worst_RX_delta_pp=100*(results[left]['views'][view]['worst_RX_accuracy']-results[right]['views'][view]['worst_RX_accuracy']))
    for dim,values in [('receiver',sorted(set(private[left]['rx']))),('day',sorted(set(private[left]['day']))),('transmitter',range(6))]:
     for value in values:
      mask=private[left]['rx']==value if dim=='receiver' else private[left]['day']==value if dim=='day' else private[left]['y']==value
      paired_strata.append(dict(comparison=key,view=view,dimension=dim,stratum=str(value),**paired(private[left]['y'][mask],private[right]['pred'][view][mask],private[left]['pred'][view][mask],cohort[mask])))
   comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]))
  stages[str(seed)]={}
  for budget in (30,50):
   key=str(seed)+'__WEIGHT_AS'+str(budget)+'__WEIGHT_NOAS'+str(budget);clean=pairs[key]['clean'];mid=pairs[key]['practical_mid_urban'];low=pairs[key]['practical_low_urban'];gain=clean['accuracy_delta_pp']
   stages[str(seed)][str(budget)]=dict(clean_delta_pp=gain,clean_net_repairs=clean['net_repairs'],clean_stage_within_0p1pp=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive=gain>0,positive_below_0p1pp_is_micro_observation=0<gain<.1,midurban_gain_pp=mid['accuracy_delta_pp'],lowurban_gain_pp=low['accuracy_delta_pp'],midurban_positive=mid['net_repairs']>0,lowurban_positive=low['net_repairs']>0)
 absolute={condition:{view:{metric_key:stats3([results[label(condition,s)]['views'][view][metric_key] for s in SEEDS]) for metric_key in ('accuracy','macro_f1','worst_RX_accuracy')} for view in VIEWS} for condition in CONDITIONS}
 paired_F1_worst={a+'__'+b:{view:{key:stats3([comparisons[str(seed)+'__'+a+'__'+b][view][key] for seed in SEEDS]) for key in ('macro_F1_delta_pp','worst_RX_delta_pp')} for view in VIEWS} for a,b in PAIRS}
 stage_summary={}
 for budget in ('30','50'):
  q=[stages[str(seed)][budget] for seed in SEEDS];nets=[r['clean_net_repairs'] for r in q];clean_stats=stats3([r['clean_delta_pp'] for r in q]);mean_guard=sum(nets)*1000>=-3*COUNT;all_guard=all(r['clean_stage_within_0p1pp'] for r in q);all_positive=all(r['clean_positive'] and r['midurban_positive'] and r['lowurban_positive'] for r in q)
  stage_summary[budget]=dict(clean_paired_pp=clean_stats,mean_clean_stage_within_0p1pp=bool(mean_guard),all_seeds_clean_stage_within_0p1pp=bool(all_guard),all_seeds_clean_positive=all(r['clean_positive'] for r in q),all_seeds_midlow_positive=all(r['midurban_positive'] and r['lowurban_positive'] for r in q),mean_clean_positive=clean_stats['mean']>0,mean_clean_at_least_0p1pp=clean_stats['mean']>=.1,fixed_three_seed_consistent_double_improvement_observation=bool(all_positive and sum(nets)*1000>=3*COUNT),scope='These fixed three model seeds;no seed selection,universal or independent-blind-test claim')
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,per_seed_factorial=per_seed,factorial_three_seed=aggregate_factorial(per_seed),absolute_metrics_three_seed=absolute,paired_F1_worstRX_pp_three_seed=paired_F1_worst,per_seed_stage=stages,stage_summary=stage_summary,eligibility=source['eligibility'],suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=3,model_seeds=list(SEEDS),independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)


def training_analysis(project):
 p=Path(project);lookup=dict(rows(p));original=read(p/'runs'/ORIGINAL50_REPLICATION/'analysis_auto_r01/training_logs.json');single=read(p/'runs'/base.RUN/'analysis_auto_r01/training_logs.json')
 if original['status']!='VERIFIED_COMPLETE_TRAINING_LOGS' or single['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Completed original12 and seed01weighted8 audits required')
 result={};parents={};new_steps=0
 for seed in SEEDS:
  for condition in ('NOAS30','AS30'):
   key=label(condition,seed);a=original['rows'][key];c=a['cost'];curves=a['epoch_curves']
   if len(curves)!=30 or c.get('new_updates',c.get('local_updates'))!=6660:raise ValueError('Original full E30 audit differs')
   native={k:c[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};result[key]=dict(cost=c,cumulative_native_cost=native,epoch_curves=curves,complete_trajectory_curves=curves,reused=True,group='reused_original30_models')
  for condition in ('NOAS50','AS50'):
   key=label(condition,seed);a=original['rows'][key];previous=label(condition.replace('50','30'),seed)
   if len(a['epoch_curves'])!=20 or a['cost']['local_updates']!=4440:raise ValueError('Original complete E50 tail audit differs')
   result[key]=dict(cost=a['cost'],cumulative_native_cost=base.cumulative_cost(result[previous]['cumulative_native_cost'],a['cost']),epoch_curves=a['epoch_curves'],complete_trajectory_curves=result[previous]['complete_trajectory_curves']+a['epoch_curves'],reused=True,group='reused_original50_segments')
  for prefix in ('WEIGHT_NOAS','WEIGHT_AS'):
   c30=prefix+'30';c50=prefix+'50';key30=label(c30,seed);key50=label(c50,seed)
   if seed==SEEDS[0]:
    for condition,key in ((c30,key30),(c50,key50)):
     a=single['rows'][condition]
     if len(a['complete_trajectory_curves'])!=int(condition[-2:]):raise ValueError('Existing seed01 weighted full audit differs')
     result[key]=dict(a,reused=True,group='reused_seed01_weighted'+condition[-2:]+'_segments')
   else:
    original30=label('NOAS30' if 'NOAS' in prefix else 'AS30',seed);parent=read(lookup[original30]/'stage20/result.json');parent_native=actual_cost(parent);parents[key30]=dict(cost=parent_native,epochs=20,updates=4440,reused=True,already_in_original30_group=True)
    physical=physical_row(c30,seed);stdout=p/'runs'/RUN/'logs'/(physical+'.log')
    a,_=parse_segment(lookup[key30],c30,stdout);new_steps+=a['counts']['steps'];a.update(reused=False,group='new_weighted30_segments',cumulative_native_cost=base.cumulative_cost(parent_native,a['cost']),complete_trajectory_curves=result[original30]['complete_trajectory_curves'][:20]+a['epoch_curves']);result[key30]=a
    b,_=parse_segment(lookup[key50],c50,stdout);new_steps+=b['counts']['steps'];b.update(reused=False,group='new_weighted50_segments',cumulative_native_cost=base.cumulative_cost(a['cumulative_native_cost'],b['cost']),complete_trajectory_curves=a['complete_trajectory_curves']+b['epoch_curves']);result[key50]=b
 for arm,a in result.items():
  budget=int(split_label(arm)[0][-2:])
  if [r['epoch'] for r in a['complete_trajectory_curves']]!=list(range(1,budget+1)):raise ValueError('Complete reused/new trajectory curves differ')
 if new_steps!=26640:raise logs.IncompleteTrainingLogs('All eight new stage streams must cover26640 updates')
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parents,new_model_trajectories=4,new_stage_rows=8,new_model_epochs=120,new_updates=26640,new_local_steps=new_steps,old_all_audits_reused=True,intermediate30_not_recharged_as_extra_parent=True,model_seeds=list(SEEDS),analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');output={}
 for arm,root in rows(project):
  a=training['rows'][arm];c=a['cost'];q=meta[arm]['source_qualification/complete.json'];f=meta[arm]['prediction/complete.json'];native=c['native_training_and_validation_NN']
  output[arm]=dict(group=a['group'],native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=f['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+f['actual_NN'],cumulative_native_cost=a['cumulative_native_cost'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=f['wall_seconds'],reused=a['reused'])
 names=('new_weighted30_segments','new_weighted50_segments','reused_seed01_weighted30_segments','reused_seed01_weighted50_segments','reused_original30_models','reused_original50_segments');groups={name:dict(rows=[a for a,q in output.items() if q['group']==name],native_NN=sum(q['native_NN'] for q in output.values() if q['group']==name),segment_and_own_readouts_NN=sum(q['segment_and_own_readouts_NN'] for q in output.values() if q['group']==name)) for name in names}
 root=Path(project)/'runs'/RUN;synthetic=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if synthetic['status']!='PASS' or real['status']!='PASS' or synthetic['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=synthetic['NN']+real['NN'];new=sum(groups[name]['segment_and_own_readouts_NN'] for name in names[:2])
 return dict(rows=output,groups=groups,new_shared_smoke_NN=shared,new_training_and_readouts_NN=new,new_total_NN=new+shared,new_model_epochs=120,new_updates=26640,new_model_trajectories=4,new_stage_rows=8,parent_original20_costs_reused=training['parent20'],old_costs_not_recharged=True,intermediate30_not_double_counted_as_parent=True,prefix20_already_counted_in_original30_groups=True,historical_shared_smokes_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Per-stage actual model-seconds,not parallel elapsed or speed evidence')

def render_report(source,formal,training,costs):
 lines=['# 固定02/03加权30→50复现：三种子完整24条件','',RUN+'。只有seed01既定fullV clean≥0pp、mid/low等权≥+.5pp源资格通过后才使用此适配。全部固定2026092701/02/03，seed01既有八条件和所有原模型复用；新02/03各NoAs/As own原20→weighted30→weighted50，不能按正式成绩选seed或救跑。','', '原IQ/6L/defaultcos/1.25L、.125/.375权重、NoAs selectedraw64、全部5源RX/L6300/U56700隐藏/V27000/IQ256保持。每轨迹实际20+10+20/resetcos，不冒称连续cos50。','', '|seed|view|原NoAs30 %|原As30 %|wNoAs30 %|wAs30 %|原NoAs50 %|原As50 %|wNoAs50 %|wAs50 %|','|---|---|'+'---:|'*8]
 for seed in SEEDS:
  for view in (*VIEWS,'urban_mean'):
   values=[100*(formal['summary'][label(c,seed)]['urban_mean_accuracy'] if view=='urban_mean' else formal['summary'][label(c,seed)]['views'][view]['accuracy']) for c in CONDITIONS];lines.append('|'+str(seed)+'|'+view+'|'+'|'.join(f'{x:.6f}' for x in values)+'|')
 lines+=['', '以下为逐seed配对差的均值±样本SD（ddof=1），完整values保留，不是置信区间或挑最好seed。','', '|view|wAs50−wNoAs50 pp|wAs50−wAs30 pp|wNoAs50−wNoAs30 pp|wAs50−原As50 pp|weight×As×budget pp|','|---|---:|---:|---:|---:|---:|']
 for view in (*VIEWS,'urban_mean'):
  q=formal['factorial_three_seed'][view];values=[q['paired_accuracy_pp'][key] for key in ('WEIGHT_AS50__WEIGHT_NOAS50','WEIGHT_AS50__WEIGHT_AS30','WEIGHT_NOAS50__WEIGHT_NOAS30','WEIGHT_AS50__AS50')]+[q['three_factor_interaction_pp']];lines.append('|'+view+'|'+'|'.join(f"{r['mean']:+.6f} ± {r['sample_SD']:.6f}" for r in values)+'|')
 lines+=['', '|seed|预算|clean pp|clean净修复|阶段≥−.1pp|low urban gain pp|mid urban gain pp|','|---|---|---:|---:|---|---:|---:|']
 for seed in SEEDS:
  for budget in ('30','50'):
   q=formal['per_seed_stage'][str(seed)][budget];lines.append(f"|{seed}|{budget}|{q['clean_delta_pp']:+.6f}|{q['clean_net_repairs']}|{q['clean_stage_within_0p1pp']}|{q['lowurban_gain_pp']:+.6f}|{q['midurban_gain_pp']:+.6f}|")
 for budget,q in formal['stage_summary'].items():lines+=['',f"E{budget} clean配对均值阶段={q['mean_clean_stage_within_0p1pp']}，全部seed阶段={q['all_seeds_clean_stage_within_0p1pp']}，全部seed clean正值={q['all_seeds_clean_positive']}，全部seed low/mid正值={q['all_seeds_midlow_positive']}；三固定seed一致双正且cleanmean≥+.1pp观察={q['fixed_three_seed_consistent_double_improvement_observation']}。"]
 lines+=['', '整数clean约束每seed net×1000≥−N，均值Σnet×1000≥−3N。阶段基本不降不等于双提升；不足.1pp的正变化标微小观察。全部24逻辑行accuracy/F1/worstRX、逐seed与均值/SD、全部配对与RX/day/TX修复/新增错误、suburban自身clean差距完整保存。','', '每seed与每view的12主配对、6二因子及三因子交互完整保留；NoAs预算变化单列，不把NoAs退化全归As，不相加不同实验增益预测。','', '源fullV27000仅原生冻结accuracy/RX，不能虚构逐包F1/错误修复；V1800两draw独立重算，两个draw不是两个modelseed，源无high urban。面板fc2.45GHz与训练/正式2.462GHz既有差异保留。源先完成并保存后才读取正式；该适配不再设计晋级或自动启动候选。','', '|成本组|native NN|本段+自身源/正式读出 NN|','|---|---:|---:|']
 for name,q in costs['groups'].items():lines.append(f"|{name}|{q['native_NN']}|{q['segment_and_own_readouts_NN']}|")
 lines+=['',f"新增4轨迹×30=120模型epoch/26640更新，8新stage全日志解析；共享smoke实际{costs['new_shared_smoke_NN']}NN，新增总{costs['new_total_NN']}NN。旧20/30/50审计不重扫；中间30不重复计父训练，sharedprefix20及旧NN/读出不重计新增。每段ownEMA/student/teacher四counter、wall、完整1..30/1..50曲线与累计成本保存。",'',f"全部24×7冻结预测和独立score完成后才连接truth，独立重算{formal['independent_confusion_matrices']}个CM及{formal['partition_reconciliations']}次分区合并。分析0NN/CP/IQ/更新，正式IQ/SNR/信道/样本集不改。已暴露基准背景保留；三固定seed观察不等于新的盲测或无限总体保证。"]
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_practical_as_weight_replicate50/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  base=Path(project)/'runs'/RUN
  if (base/'failure.json').is_file():raise RuntimeError('Owned technical failure preserved')
  for arm,own in rows(project):
   if split_label(arm)[1] in NEW_SEEDS and split_label(arm)[0].startswith('WEIGHT_') and (own/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Own source readout failed;formal closure remains with original runner')
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
 if a.release.name!=TRAINING_RELEASE:raise ValueError('Only registered conditional replication release')
 eligibility=seed01_source_eligibility(a.project);out=validate_output(a.project,a.output);out.mkdir();started=time.perf_counter();write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=a.code_commit,NN=0,truth_opened=False,eligibility=eligibility))
 try:
  wait_until(a.project,a.release,'SOURCE',out);training=training_analysis(a.project);write(out/'training_logs.json',training);source=source_analysis(a.project);write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',training_run=RUN,NN=0,target_scores_read=False));print('SOURCE_ANALYSIS_COMPLETE',flush=True)
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=3,new_model_trajectories=4,new_stage_rows=8,reused_model_rows=16,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
