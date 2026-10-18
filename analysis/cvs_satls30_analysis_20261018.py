"""Zero-model source/log/formal recount for the fixed As-only label-smoothed E30 candidate.

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
RUN='20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01'
TRAINING_RELEASE='cvs_FcrossRX_AsLS_E30_20261018_r02'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','SATLS_AS30')
PAIRS=(('SATLS_AS30','NOAS30'),('SATLS_AS30','AS30'),('AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('SATLS_AS30',p/RUN/'B01-s2026092701')]

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
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all3 logical rows have all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def source_conditions(summary):
 candidate=summary['SATLS_AS30'];ordinary=summary['AS30'];no=summary['NOAS30']
 clean=float(100*(candidate['full_V_clean']-no['full_V_clean']));midlow=float(100*(candidate['mid_low_mean']-ordinary['mid_low_mean']))
 for key,value in [('fullV clean versus NoAs30',clean),('midlow versus originalAs30',midlow)]:logs.finite(value,key)
 cleanpass=bool(clean>=-.1-1e-12);starpass=bool(midlow>=.5-1e-12);passed=bool(cleanpass and starpass)
 return dict(fullV_clean_vs_NoAs30_pp=clean,mid_low_mean_vs_original_As30_pp=midlow,clean_vs_NoAs30_within_absolute_0p1pp=cleanpass,mid_low_gain_at_least_0p5pp=starpass,source_criteria_pass=passed,E50_source_eligible=passed,fixed_recipe_closed=bool(not passed),fullV_clean_vs_original_As30_pp=float(100*(candidate['full_V_clean']-ordinary['full_V_clean'])),only_two_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False,per_packet_protection_gate=False,automatic_extension=False)

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


SATLS_KEYS=('selected_packets','epsilon','selected_CE_old','selected_CE_new','mixed_CE_delta','loss_delta','extra_NN')

def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base reduction differs'),logs._near(r['total_L'],total,'Full original1.25L reduction differs')]
 for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+key],r[key],'Actual full6L concat telemetry differs'))
 errors.append(logs._near(m['concat/total_L']-r['base_L_before_shape'],.25*s['shape_RD'],'Raw .25 contribution differs'))
 return max(errors)

def check_satls(r,s,c,m):
 if (c['selected_packets'],c['As_packets'],c['old_epsilon'],c['epsilon'],c['native_label_smoothing'],c['extra_NN'],c['extra_CE_operator_calls'],c['extra_CE_logit_packets'])!=(64,64,.01,.05,.01,0,4,256) or c['explicit_CUDA_synchronize'] or c['confidence_filter'] or not c['all_six_classes_same_formula'] or not c['same_IQ_each_pair'] or c['ordinary_redraws']:raise ValueError('Only fixed selectedAs64 CE05 scope and computation')
 if c['baseline_mixed_CE_operator_widths']!=[128,128] or c['extra_CE_operator_widths']!=[64,64,64,64] or c['actual_mixed_CE_operator_widths']!=[128,128,64,64,64,64]:raise ValueError('Actual baseline/extra CE widths differ')
 if c['selected_L_positions']!=[j for j in range(128) if (j+r['step'])%2!=0] or any(len(c[k])!=64 for k in ('sample_ids','TX','RX','day')):raise ValueError('Original selected64 positions/metadata required')
 if any(v not in range(6) for v in c['TX']) or any(v not in (1,3,4,6,8) for v in c['RX']) or any(v not in (1,2,3) for v in c['day']):raise ValueError('Only original full-source legalL metadata')
 if c['scene_counts']!=r['scene_counts'] or sum(c['scene_counts'].values())!=64 or set(c['scene_counts'])-set(SOURCE_SCENES):raise ValueError('Original complete practical scene assignment differs')
 e=[];old=.5*(c['selected_CE_old_pass0']+c['selected_CE_old_pass1']);new=.5*(c['selected_CE_new_pass0']+c['selected_CE_new_pass1']);ordinary=.5*(c['ordinary_CE_pass0']+c['ordinary_CE_pass1']);mixed_old=.5*(ordinary+old);mixed_new=.5*(ordinary+new);delta=mixed_new-mixed_old
 for key,expected in [('selected_CE_old',old),('selected_CE_new',new),('mixed_CE_old',mixed_old),('mixed_CE_new',mixed_new),('mixed_CE_delta',delta),('loss_delta',.5*delta),('mixed_RD_old',mixed_old+c['mixed_KL']),('mixed_RD_new',mixed_new+c['mixed_KL']),('raw_RD',s['shape_RD']),('common_RD',r['common_RD']),('SupCon',r['SupCon']),('SupCon_weight',r['SupCon_weight']),('total_L',r['total_L'])]:e.append(logs._near(c[key],expected,'Actual CE target-only reduction differs: '+key))
 for key,expected in [('extra_CE',mixed_new),('extra_KL',c['mixed_KL']),('extra_RD',c['mixed_RD_new'])]:e.append(logs._near(r[key],expected,'Actual new mixed128 CE/RD and unchanged KL differ'))
 oldfull=.5*r['common_RD']+.5*c['mixed_RD_old']+.25*s['shape_RD']+r['SupCon_weight']*r['SupCon'];e.append(logs._near(c['base_total_L'],oldfull,'Saved counterfactual original full6L total differs'));e.append(logs._near(c['total_L']-c['base_total_L'],c['loss_delta'],'Only outer.5 CE correction enters full loss'))
 if c['total_L_scale']!=1.25 or any(c[k]<0 for k in ('CE_operator_cpu_wall_seconds','scalarization_cpu_wall_seconds')):raise ValueError('Actual loss scale/timing differs')
 for k in SATLS_KEYS:e.append(logs._near(m['satls/'+k],c[k],'Native AsLS scalar differs',1e-12))
 return max(e),logs.relation(r,True)

def parse_new_logs(root,arm):
 if arm!='SATLS_AS30':raise ValueError('Only one new AsLS B01 row')
 b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];policy=cfg['satellite_label_smoothing']
 if cfg['crossview']['mode']!='ORD128' or (policy['old_epsilon'],policy['epsilon'],policy['native_label_smoothing'],policy['selected_packets_per_step'],policy['selected_total_CE_weight'],policy['extra_CE_operator_calls_per_step'],policy['extra_CE_logit_packets_per_step'])!=(.01,.05,.01,64,.25,4,256) or not all(policy[k] for k in ('mixed_KL_unchanged','common_raw_U_SupCon_unchanged','same_IQ_within_RDpair','all_six_classes_same_formula')) or any(policy[k] for k in ('confidence_filter','extra_NN','explicit_CUDA_synchronize','new_losses','new_U_views','formal_test_changed')):raise ValueError('Fixed selectedAs64 target smoothing only')
 if policy['baseline_mixed_CE_operator_widths']!=[128,128] or policy['extra_CE_operator_widths']!=[64,64,64,64] or policy['actual_mixed_CE_operator_widths']!=[128,128,64,64,64,64]:raise ValueError('Actual configured CE operator widths differ')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(30,6660) or (result['parent_epochs'],result['parent_optimizer_steps'])!=(20,4440):raise ValueError('Own originalAs20+10 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=1704960 or result['L_As']!=142080 or result['total_L_loss_scale']!=1.25 or (result['EMA_local_successful_updates'],result['EMA_cumulative_updates'])!=(2220,6660):raise ValueError('Actual hiddenU/6L/As64/ownEMA budget differs')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own stage20 EMA' or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Strict ownAs20 EMA initialization differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);count=Counter();sums=defaultdict(Counter);mse=[];maxerr=0.;selected=[];scenes=Counter();states=Counter();opwall=scalarwall=0.
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','satls_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New10 AsLS stream prefix')
  if i>STEPS:raise ValueError('Extra step outside ownAs20+10 budget')
  r,s,n,u,c=items;ep=(i-1)//222+1;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,c)) or n['epoch']!=ep or c['epoch']!=ep or c['cumulative_epoch']!=20+ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous actual update/Sinc/AsLS counters differ')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,64,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Original sixforward supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw control slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative crossRX weight differs',1e-12);ls_err,valid=check_satls(r,s,c,m);maxerr=max(maxerr,ls_err,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=64 or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerics differ')
  logs.finite(m['train/loss'],'native total loss');number=m['train/pseudo_total'];picked=m['train/pseudo_selected']
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  if sum(r['scene_counts'].values())!=64 or sum(r['state_counts'].values())!=64:raise ValueError('Actual full practical counts differ')
  selected.append(c);scenes.update(r['scene_counts']);states.update(r['state_counts']);opwall+=c['CE_operator_cpu_wall_seconds'];scalarwall+=c['scalarization_cpu_wall_seconds'];count.update(steps=1,L_NN=768,selected_packets=64,As_packets=64,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=valid,extra_CE_operator_calls=4,extra_CE_logit_packets=256);mse.append(r['ordinary_raw_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')});sums[ep].update({k:c[k] for k in ('selected_CE_old','selected_CE_new','mixed_CE_old','mixed_CE_new','mixed_CE_delta','loss_delta')});sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New10 final AsLS update missing')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();by_tx=Counter();by_rx=Counter();by_day=Counter();by_cell=Counter();repeated=Counter()
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra L metadata outside budget')
   c=selected[batches['L']]
   if [ids[j] for j in c['selected_L_positions']]!=c['sample_ids'] or any((z[1],z[2])!=(r,d) for z,r,d in zip(c['sample_ids'],c['RX'],c['day'])):raise ValueError('Actual selected64 must retain original input positions')
   repeats=128-len({tuple(z) for z in ids});repeated['batches_with_repeated_physical_ID']+=bool(repeats);repeated['repeated_input_positions']+=repeats;repeated['maximum_repeated_positions_in_batch']=max(repeated['maximum_repeated_positions_in_batch'],repeats)
   for z,t,r,d in zip(c['sample_ids'],c['TX'],c['RX'],c['day']):exposure[tuple(z)]+=1;by_tx[str(t)]+=1;by_rx[str(r)]+=1;by_day[str(d)]+=1;by_cell[str((t,r,d))]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L'] or count['U_forward_packets']!=packets['U']:raise ValueError('Full original L6300/U56700 sequence differs')
 if cost['student_train_NN']!=2271960 or cost['student_eval_NN']!=324002 or cost['teacher_train_NN'] or cost['teacher_eval_NN']!=567000:raise ValueError('Actual four counters/complete source loops differ')
 a=result['satellite_label_smoothing']
 if (a['mode'],a['old_epsilon'],a['epsilon'],a['native_label_smoothing'],a['selected_packets'],a['actual_As_packets'],a['channel_calls'],a['pair_weights'],a['SupCon'],a['total_L_scale'])!=('selectedAs_CE05',.01,.05,.01,142080,142080,2220,[.5,.5,.25],'ORD128',1.25) or not a['mixed_KL_unchanged'] or not a['common_raw_U_SupCon_unchanged'] or not a['all_six_classes_same_formula'] or any(a[k] for k in ('extra_NN','confidence_filter','new_U_views','sampler_changed')) or a['extra_CE_operator_calls']!=8880 or a['extra_CE_logit_packets']!=568320:raise ValueError('Actual fixed target-only scope/extra CE costs differ')
 if a['baseline_mixed_CE_operator_widths']!=[128,128] or a['extra_CE_operator_widths']!=[64,64,64,64] or a['actual_mixed_CE_operator_widths']!=[128,128,64,64,64,64]:raise ValueError('Saved CE operator widths differ')
 logs._near(a['CE_operator_cpu_wall_seconds'],opwall,'Extra CE operator timing sum differs',1e-12);logs._near(a['scalarization_cpu_wall_seconds'],scalarwall,'Extra scalarization timing sum differs',1e-12)
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for q in coverage:
  coverage_scene[q['scene']]+=q['count'];coverage_state[q['state']]+=q['count'];coverage_tx[str(q['TX'])]+=q['count'];coverage_rx[str(q['RX'])]+=q['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_tx!=by_tx or coverage_rx!=by_rx:raise ValueError('Actual channel/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Original parent ordinary augmentation differs')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 if args['label_smoothing']!=.01 or not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher']:raise ValueError('Actual global CE/ordinary/U/EMA flags differ')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed stdout')
 curves=[dict(epoch=q['epoch']+20,local_epoch=q['epoch'],**{k:v/222 for k,v in sums[q['epoch']].items()},**{k:q[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for q in epochs];extra=dict(extra_model_NN=0,extra_CE_operator_calls=8880,extra_CE_logit_packets=568320,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],CE_operator_cpu_wall_seconds=opwall,scalarization_cpu_wall_seconds=scalarwall,timing=a['timing'],explicit_CUDA_synchronize=False,autograd_extra_backward_time_not_isolated=True,zero_extra_model_forward_is_not_zero_compute=True)
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,selected_packets=142080,As_packets=142080),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},satls=dict(a,sampler_repetitions=dict(repeated)),extra_objective_accounting=extra,EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660),selected_exposure=dict(TX=dict(by_tx),RX=dict(by_rx),day=dict(by_day),TX_RX_day=dict(by_cell),physical=exposure_summary(exposure,unique['L']),actual_As_packets=142080,coverage=coverage,per_physical_scene_joint_not_recorded=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)

def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Existing complete original30 audit required')
 result={};lookup=dict(rows(project))
 for name in ('NOAS30','AS30'):
  r=prior['rows'][name]
  if r['cost']['new_updates']!=6660 or r['cost']['new_epochs']!=30 or len(r['epoch_curves'])!=30:raise ValueError('Saved original30 audit schema/budget differs')
  result[name]=dict(reused=True,cost=r['cost'],epoch_curves=r['epoch_curves'],evidence='Existing complete original30 audit reused;no original step stream rescanned')
 candidate,private=parse_new_logs(lookup['SATLS_AS30'],'SATLS_AS30');candidate['reused']=False;result['SATLS_AS30']=candidate
 original=read(lookup['AS30']/'stage20/result.json');parent=dict(SATLS_AS30=dict(cost=actual_cost(original),epochs=20,updates=4440,wall_seconds=original['wall_seconds'],reused=True));oldcost=result['AS30']['cost'];prefix=parent['SATLS_AS30']['cost'];oldtail={k:oldcost[k]-prefix[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')}
 args=read(lookup['AS30']/'source/actual_native_args.json');args=args.get('args',args);curves=result['AS30']['epoch_curves'][20:];fields=('ordinary_raw_MSE','common_CE','common_KL','extra_CE','extra_KL','SupCon');diff={k:dict(record_count=10,records_different=sum(x[k]!=y[k] for x,y in zip(curves,candidate['epoch_curves'])),maximum_absolute_difference=max(abs(x[k]-y[k]) for x,y in zip(curves,candidate['epoch_curves']))) for k in fields if all(k in q for q in (*curves,*candidate['epoch_curves']))}
 comparison=dict(reference_condition='AS30',reference_audit_reused=True,old_step_logs_rescanned=False,old_tail_four_counters=oldtail,new_tail_four_counters=candidate['cost'],four_counters_equal=all(oldtail[k]==candidate['cost'][k] for k in oldtail),available_epoch_mean_comparisons=diff,different_actual_arg_keys=sorted(k for k in set(args)|set(private['args']) if args.get(k)!=private['args'].get(k)),old_per_step_physical_IQ_equality_not_claimed=True,U_rule_unchanged_by_recipe=True)
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parent,new_model_rows=1,new_model_epochs=10,new_updates=2220,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparison,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)

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
 LS_increment={v:comparisons['SATLS_AS30__AS30'][v]['accuracy_delta_pp'] for v in VIEWS};LS_increment['urban_mean']=comparisons['SATLS_AS30__AS30']['urban_mean_accuracy_delta_pp']
 key='SATLS_AS30__NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],incremental_satellite_LS_effect_pp=LS_increment,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['SATLS_AS30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['SATLS_AS30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  out[arm]=dict(native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm in CONDITIONS[:2],source_readout_wall_seconds=q.get('wall_seconds'))
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['NN']+real['NN'];parents={a:v['cost']['native_training_and_validation_NN'] for a,v in training['parent20'].items()};segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:])
 return dict(rows=out,new_model_epochs=10,new_updates=2220,extra_objective_accounting={a:training['rows'][a].get('extra_objective_accounting',{}) for a in CONDITIONS[2:]},new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,parent20_native_NN_reused=parents,cumulative_candidate_native_NN={a:parents[a]+out[a]['native_NN'] for a in CONDITIONS[2:]},cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;not speed evidence')

def render_report(source,formal,training,costs):
 new='SATLS_AS30';s=formal['stage'];d=source['decision'];lines=['# As-only label smoothing .01→.05：E30三条件结果','',RUN+'；模型seed2026092701。只新增B01，从原P43 ownAs20学生/ownEMA追加10轮，总E30/6660；原NoAs30与As30固定读出复用，不使用LRaw或温度变体父权重。','',f"正式clean相对NoAs30 {s['clean_delta_pp']:+.6f}个绝对百分点，clean≥−.1pp阶段约束={s['clean_absolute_0p1pp_guard_met']}。low/mid urban相对NoAs30变化{s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f}pp；新增机制效应必须另看新As−原As，不能把普通As总收益全归于平滑。",'', '唯一变量是selected64 As两pass的CE均匀平滑从既有epsilon.01提高至固定.05，deltaepsilon.04。六分类true target由.9916667改为.9583333，其他类由.0016667改为.0083333；不是hard/onehot改soft，不改softmax温度或CE总体权重。common ordinary、mixed另一64、raw shape、U保持原.01；mixed128对称KL、ORD128 SupCon、6L/768NN、总L1.25、原As总体.25份额均保持。', '', 'mixed_CE_new=.5*meanCE(ordinary64,eps.01)+.5*meanCE(As64,eps.05)，每组为两pass CE均值；mixed项再乘原.5。只有selected64的CE差值进入原目标，无置信度门控、删样本或类专属规则，全部合法As仍监督。RDrop两次使用同一个已生成IQ，传播/接收及正式IQ/SNR/场景不变。', '', '|视图|NoAs30 %|原As30 %|AsLS30 %|新As−NoAs pp|新As−原As pp|原As−NoAs pp|','|---|---:|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  vals=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS];diff=[formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp'] if v=='urban_mean' else formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for a,b in PAIRS];lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals+diff)+'|')
 lines+=['','|条件|视图|accuracy %|macro-F1 %|最差RX %|','|---|---|---:|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|比较|视图|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['',f"来源唯一两条件：新As fullVclean−NoAs30 {d['fullV_clean_vs_NoAs30_pp']:+.6f}pp（要求≥−.1）；V1800两draw mid/low等权新As−原As {d['mid_low_mean_vs_original_As30_pp']:+.6f}pp（要求≥+.5）。合取PASS={d['source_criteria_pass']}，E50来源资格={d['E50_source_eligible']}，固定配方关闭={d['fixed_recipe_closed']}。fail关闭固定.05，不扫epsilon/温度/权重/seed救跑；仍完成全部正式，分析不自动扩预算，正式不回流资格或调参。",'', '|条件|fullV clean %|panel clean %|两draw mid/low等权 %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['', 'fullV27000为冻结原生accuracy/RX，不虚构逐包F1或修复；V1800三场两draw独立重算，两draw不是两模型seed，源无highurban。源panel2.45GHz与训练/正式2.462GHz差异保留。全5源RX、L6300/U56700真值隐藏/V27000、256点、split392005权限保持。没有RX留出。','', '|条件|suburban视图|绝对accuracy %|自身clean−suburban pp|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v,g in formal['suburban_selfclean_gaps_pp'][a].items():lines.append(f"|{a}|{v}|{100*formal['summary'][a]['views'][v]['accuracy']:.6f}|{g:.6f}|")
 lines+=['', 'clean下降本身也可能缩小suburban差距，须同时看绝对准确率，不将缩差自动算保护。单新As行只有三逻辑条件，不构造假四行因子或交互。','', '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|native秒|','|---|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append(f"|{a}|{c['student_train_NN']}|{c['student_eval_NN']}|{c['teacher_train_NN']}|{c['teacher_eval_NN']}|{c['native_NN']}|{c['native_wall_seconds']:.3f}|")
 lines+=['',f"新增1×10modelEpoch/2220updates，训练及source/formal读出{costs['new_training_and_readouts_NN']}NN、共享smoke实际{costs['new_shared_smoke_NN']}NN，总新增{costs['new_total_NN']}NN。父20、旧两E30及历史读出成本单列复用，不重计；新模型累计E30。6L/U学生/教师/validation均按实际计数器，U和ownEMA继续。额外CE与差值autograd计算虽无新NN前向，也必须按训练审计记录调用/包数和计时，不能称免费。",'', '新段完整2220步/10epoch与L/U流解析，旧original30完整audit复用不回扫stepstream。新增算子运算与计时详见training_logs.satls以及costs.extra_objective_accounting；计时是否同步按原日志语义披露，未拆分GPU反向耗时不虚构。source预测wall未单列即缺失，不用analysiswall替代。', '', f"全部3×7固定预测和独立truth-last评分齐后才truth join，重核{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}次RX/day/TX分区合并。n=1；阶段PASS或局部/微小clean增益不证明可重复双提升。已暴露基准背景保留，最终目标尚不能由单种子证明。分析0NN/CP/IQ/updates。",'', '动机仅来自一般label smoothing研究；该论文不提供CVS或星地收益证据：[When does label smoothing help?](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)。本轮必须以实际身份识别结果检验，而不是以熵/置信度/特征代理晋级。']
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_satellite_ls30/runner.py' in argv:live.append(pid)
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
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=1,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
