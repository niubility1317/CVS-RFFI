"""Zero-model source/log/formal recount for the fixed TXxRX As64 allocation E30 candidate.

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
RUN='20261018-phase1-FcrossRX-AsAlloc-e30-s2026092701-m1-r01'
TRAINING_RELEASE='cvs_FcrossRX_AsAlloc_E30_20261018_r01'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','ALLOC_AS30')
PAIRS=(('ALLOC_AS30','NOAS30'),('ALLOC_AS30','AS30'),('AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('ALLOC_AS30',p/RUN/'B01-s2026092701')]

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
 candidate=summary['ALLOC_AS30'];ordinary=summary['AS30'];no=summary['NOAS30']
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


ALLOCATION_KEYS=('selected_packets','active_cells','quota_min','quota_max','changed_positions','allocation_cpu_wall_seconds','extra_NN')

def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base reduction differs'),logs._near(r['total_L'],total,'Full original1.25L reduction differs')]
 for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+key],r[key],'Actual full6L concat telemetry differs'))
 errors.append(logs._near(m['concat/total_L']-r['base_L_before_shape'],.25*s['shape_RD'],'Raw .25 contribution differs'))
 return max(errors)

def check_allocation_positions(c):
 selected=c['selected_L_positions'];ordinary=c['ordinary_L_positions'];old=c['old_selected_L_positions']
 if any(type(v) is not int for v in selected+ordinary+old) or len(selected)!=64 or len(set(selected))!=64 or len(ordinary)!=64 or len(set(ordinary))!=64 or set(selected)&set(ordinary) or sorted(selected+ordinary)!=list(range(128)):raise ValueError('Exactly64 unique selected input positions and their64 complement required')
 if old!=[j for j in range(128) if (j+c['step'])%2!=0]:raise ValueError('Original withinbatch parity control mask differs')
 return selected,ordinary,old

def check_waterfill_cells(c,TX,RX):
 selected,ordinary,old=check_allocation_positions(c);actual=defaultdict(list)
 if len(TX)!=128 or len(RX)!=128 or any(type(t) is not int or t not in range(6) for t in TX) or any(type(r) is not int or r not in (1,3,4,6,8) for r in RX):raise ValueError('Actual full128 legal L TX/RX metadata required')
 for j,(t,r) in enumerate(zip(TX,RX)):actual[(t,r)].append(j)
 seen=set();quotas=[];unsaturated=[]
 for cell in c['cells']:
  key=(cell['TX'],cell['RX'])
  if key in seen or key not in actual:raise ValueError('Cell metadata must cover actual active cells once')
  seen.add(key);available=actual[key];picked=sorted(set(available)&set(selected));prior=sorted(set(available)&set(old));q=cell['selected'];n=len(available)
  if cell['available']!=n or cell['input_L_positions']!=available or cell['selected_L_positions']!=picked or cell['old_selected_L_positions']!=prior or cell['old_selected']!=len(prior) or type(q) is not int or not 0<=q<=n or q!=len(picked):raise ValueError('Actual waterfill quota exceeds availability or cell positions differ')
  quotas.append(q)
  if q<n:unsaturated.append(q)
 if seen!=set(actual) or sum(quotas)!=64 or not unsaturated or max(quotas)-min(unsaturated)>1:raise ValueError('Feasible64 maxmin waterfill quotas required;no class/RX whitelist')
 if (c['selected_packets'],c['active_cells'],c['quota_min'],c['quota_max'])!=(64,len(actual),min(quotas),max(quotas)):raise ValueError('Saved allocation quota summaries differ')
 return selected,ordinary,old

def check_allocation_trace(c,TX,RX):
 selected,ordinary,old=check_waterfill_cells(c,TX,RX);priorities=np.asarray(c['priorities'],float)
 if priorities.shape!=(128,) or not np.isfinite(priorities).all() or np.any((priorities<0)|(priorities>=1)):raise ValueError('One finite private uniform priority per originalposition required')
 cells=c['cells'];canonical=[];quotas=[0]*len(cells);taken=0
 for g,cell in enumerate(cells):
  if cell['canonical_index']!=g:raise ValueError('Actual canonical cells must be contiguous')
  order=sorted(cell['input_L_positions'],key=lambda j:(priorities[j],j));canonical.append((priorities[order[0]],order[0]))
  if cell['priority_ordered_L_positions']!=order or sorted(order[:cell['selected']])!=cell['selected_L_positions']:raise ValueError('Within-cell selected positions must follow recorded private priorities')
 if canonical!=sorted(canonical):raise ValueError('Canonical groups must derive from priorities,not numeric TX/RX names')
 for round_ in c['rounds']:
  active=[g for g,cell in enumerate(cells) if quotas[g]<cell['available']];order=round_['visit_order'];used=round_['selected_groups']
  if taken>=64 or round_['active_groups']!=active or sorted(order)!=active or used!=order[:min(len(active),64-taken)]:raise ValueError('Actual private waterfill rounds differ')
  for g in used:quotas[g]+=1
  taken+=len(used)
 if taken!=64 or quotas!=[cell['selected'] for cell in cells]:raise ValueError('Complete rounds must produce the recorded64 quotas')
 if c['changed_positions']!=len(set(selected)^set(old)) or c['ordinary_packets']!=64 or c['priority_ties']!=128-len(set(priorities.tolist())):raise ValueError('Actual mask-change/ordinary complement/tie counts differ')
 if (c['augmentation_seed'],c['allocation_namespace'],c['private_CPU_rng'])!=(20261016421,'FcrossRX/AsAllocation/waterfill64/v1','numpy.default_rng(PCG64)') or c['labels_used']!='trueL TX only' or c['RX_used']!='existing originalRX':raise ValueError('Original private allocation namespace/trueL label scope differs')
 if any(c[k] for k in ('global_RNG_used','sampler_changed','physical_ID_uniqueness_required','confidence_or_quality_used','U_truth_used','extra_NN','new_losses','formal_test_changed')) or c['allocation_cpu_wall_seconds']<0:raise ValueError('Only legalL position allocation;no filtering or new model work')
 return selected,ordinary,old

def check_allocation(r,s,c,m):
 positions,ordinary,prior=check_allocation_trace(c,c['TX'],c['RX'])
 if any(len(c[k])!=128 for k in ('TX','RX','day','sample_ids')) or any(v not in (1,2,3) for v in c['day']) or c['sample_id_fields']!=['eq_i','rx_i','day_i','sig_i','base_index'] or c['selected_sample_ids']!=[c['sample_ids'][j] for j in positions]:raise ValueError('Original128 metadata and actual selected64 PID positions required')
 if any(len(z)!=5 or (z[1],z[2])!=(rx,day) for z,rx,day in zip(c['sample_ids'],c['RX'],c['day'])) or c['duplicate_PID_occurrences']!=128-len({tuple(z) for z in c['sample_ids']}):raise ValueError('Original physical metadata/repeatedpositions differ;no deduplication')
 if (c['As_packets'],c['L_forwards'],c['L_NN'],c['total_L_scale'],c['extra_CE_operator_calls'])!=(64,6,768,1.25,0) or not c['same_IQ_each_pair'] or any(c[k] for k in ('ordinary_redraws','global_Torch_RNG_draws','global_CUDA_RNG_draws')):raise ValueError('Original sixforward/RDrop/globalRNG/CE execution differs')
 if any(len(c[k])!=64 for k in ('scene','state','lock')) or dict(Counter(c['scene']))!=r['scene_counts'] or c['scene_counts']!=r['scene_counts'] or dict(Counter(c['state']))!=r['state_counts'] or set(c['scene'])-set(SOURCE_SCENES):raise ValueError('Actual selected64 complete practical scene/state records differ')
 errors=[logs._near(c['total_L'],r['total_L'],'Allocation keeps original actual full6L loss')]
 for k in ALLOCATION_KEYS:errors.append(logs._near(m['allocation/'+k],c[k],'Actual allocation native scalar differs',1e-12))
 if c['allocation_hook_cpu_wall_seconds']<c['allocation_cpu_wall_seconds'] or c['allocation_cpu_wall_seconds']<0:raise ValueError('Actual allocation CPUwall scope differs')
 return max(errors),logs.relation(r,True)


def parse_new_logs(root,arm):
 if arm!='ALLOC_AS30':raise ValueError('Only one new AsAlloc B01 row')
 b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];policy=cfg['satellite_allocation']
 if cfg['crossview']['mode']!='ORD128' or (policy['mode'],policy['selected_packets_per_step'],policy['ordinary_complement_per_step'],policy['augmentation_seed'],policy['allocation_namespace'])!=('TX_RX_waterfill64',64,64,20261016421,'FcrossRX/AsAllocation/waterfill64/v1') or not all(policy[k] for k in ('original128positions_unchanged','each_physical_position_one_mixed_view','primary_ordinary128_unchanged','ORD128_anchors_unchanged')) or any(policy[k] for k in ('loader_changed','physical_ID_uniqueness_required','physical_sample_deduplication','reorder_As_to_fixed_mask','global_RNG_used','extra_NN','extra_CE_operator_calls','new_losses','new_U_views','formal_test_changed')):raise ValueError('Fixed trueL TX_RX feasible64 allocation only')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(30,6660) or (result['parent_epochs'],result['parent_optimizer_steps'])!=(20,4440):raise ValueError('Own originalAs20+10 budget differs')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or result['L_NN']!=1704960 or result['L_As']!=142080 or result['total_L_loss_scale']!=1.25 or (result['EMA_local_successful_updates'],result['EMA_cumulative_updates'])!=(2220,6660) or (result['U_steps'],result['U_packets'],result['U_student_NN'])!=(2220,567000,567000):raise ValueError('Actual hiddenU/6L/As64/ownEMA budget differs')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own stage20 EMA' or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Strict ownAs20 EMA initialization differs')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);count=Counter();sums=defaultdict(Counter);mse=[];maxerr=0.;selected=[];scenes=Counter();states=Counter();cpuwall=hookwall=0.;quota_hist=Counter();changes=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','allocation_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(q is None for q in items):raise logs.IncompleteTrainingLogs('New10 allocation stream prefix')
  if i>STEPS:raise ValueError('Extra step outside ownAs20+10 budget')
  r,s,n,u,c=items;ep=(i-1)//222+1;m=n['metrics'];logs.scan_numbers(items)
  if any(q['step']!=i for q in items) or any(q['cumulative_step']!=STEP_OFFSET+i for q in (r,s,u,c)) or n['epoch']!=ep or c['epoch']!=ep or c['cumulative_epoch']!=20+ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous actual update/Sinc/allocation counters differ')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'])!=(6,64,0,128) or r['Ag_mode']!='NONE' or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Original sixforward supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw control slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative crossRX weight differs',1e-12);allocerr,valid=check_allocation(r,s,c,m);maxerr=max(maxerr,allocerr,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=64 or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Native forward/U truth/numerics differ')
  logs.finite(m['train/loss'],'native total loss');number=m['train/pseudo_total'];picked=m['train/pseudo_selected']
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Actual original U packet budget differs')
  selected.append(c);scenes.update(r['scene_counts']);states.update(r['state_counts']);cpuwall+=c['allocation_cpu_wall_seconds'];hookwall+=c['allocation_hook_cpu_wall_seconds'];changes[c['changed_positions']]+=1;quota_hist.update(cell['selected'] for cell in c['cells']);count.update(steps=1,L_NN=768,selected_packets=64,As_packets=64,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=valid,extra_CE_operator_calls=0,extra_NN=0);mse.append(r['ordinary_raw_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')});sums[ep].update({k:c[k] for k in ALLOCATION_KEYS if k!='extra_NN'});sums[ep]['native_loss']+=m['train/loss']
 if count['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New10 final allocation update missing')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();exposure=Counter();old_exposure=Counter();by_tx=Counter();by_rx=Counter();by_day=Counter();by_cell=Counter();old_cell=Counter();available_cell=Counter();repeated=Counter();label_by_id={}
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original L/U loader packet size differs')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source physical RX/day differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra L metadata outside budget')
   c=selected[batches['L']]
   if ids!=c['sample_ids']:raise ValueError('Full allocation128 must be unchanged original L input positions')
   repeats=128-len({tuple(z) for z in ids});repeated['batches_with_repeated_physical_ID']+=bool(repeats);repeated['repeated_input_positions']+=repeats;repeated['maximum_repeated_positions_in_batch']=max(repeated['maximum_repeated_positions_in_batch'],repeats)
   for j,(z,t,r,d) in enumerate(zip(ids,c['TX'],c['RX'],c['day'])):
    pid=tuple(z)
    if pid in label_by_id and label_by_id[pid]!=t:raise ValueError('Saved trueL TX inconsistent for physical packet')
    label_by_id[pid]=t;available_cell[str((t,r,d))]+=1
    if j in c['selected_L_positions']:exposure[pid]+=1;by_tx[str(t)]+=1;by_rx[str(r)]+=1;by_day[str(d)]+=1;by_cell[str((t,r,d))]+=1
    if j in c['old_selected_L_positions']:old_exposure[pid]+=1;old_cell[str((t,r,d))]+=1
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or len(unique['L'])!=6300 or len(unique['U'])!=56700 or unique['L']&unique['U'] or set(exposure)-unique['L'] or count['U_forward_packets']!=packets['U']:raise ValueError('Full original L6300/U56700 sequence differs')
 if cost['student_train_NN']!=2271960 or cost['student_eval_NN']!=324002 or cost['teacher_train_NN'] or cost['teacher_eval_NN']!=567000:raise ValueError('Actual four counters/complete source loops differ')
 a=result['satellite_allocation']
 if (a['mode'],a['allocation_calls'],a['selected_packets'],a['ordinary_complement_packets'],a['actual_As_packets'],a['channel_calls'],a['augmentation_seed'],a['allocation_namespace'],a['pair_weights'],a['SupCon'],a['total_L_scale'],a['per_step_L_NN'])!=('TX_RX_waterfill64',2220,142080,142080,142080,2220,20261016421,'FcrossRX/AsAllocation/waterfill64/v1',[.5,.5,.25],'ORD128',1.25,768) or any(a[k] for k in ('extra_NN','extra_CE_operator_calls','pure_gradient_variance_claim','global_RNG_used','sampler_changed','physical_ID_uniqueness_required','common_U_trajectory_equality_claim')) or not a['oldmask_is_counterfactual'] or not a['each_physical_position_one_mixed_view']:raise ValueError('Actual fixed allocation scope differs')
 def strata_counter(values):
  out=Counter()
  for z in values:
   key=str((z['TX'],z['RX'],z['day']))
   if key in out or type(z['count']) is not int or z['count']<0:raise ValueError('Actual exposure stratum count differs')
   out[key]=z['count']
  return out
 for key,counter in [('availability_TX_RX_day_exposure',available_cell),('selected_TX_RX_day_exposure',by_cell),('within_current_trajectory_oldmask_TX_RX_day_exposure',old_cell)]:
  if strata_counter(a[key])!=counter:raise ValueError('Complete allocation exposure differs from saved summary: '+key)
 if a['changed_positions']!=sum(k*v for k,v in changes.items()):raise ValueError('Actual changed-position sum differs')
 logs._near(a['allocation_cpu_wall_seconds'],cpuwall,'Actual selector CPUwall sum differs',1e-12);logs._near(a['allocation_hook_cpu_wall_seconds'],hookwall,'Actual hook CPUwall sum differs',1e-12)
 coverage=result['coverage'];coverage_scene=Counter();coverage_state=Counter();coverage_tx=Counter();coverage_rx=Counter()
 for z in coverage:
  coverage_scene[z['scene']]+=z['count'];coverage_state[z['state']]+=z['count'];coverage_tx[str(z['TX'])]+=z['count'];coverage_rx[str(z['RX'])]+=z['count']
 if coverage_scene!=scenes or coverage_state!=states or coverage_tx!=by_tx or coverage_rx!=by_rx:raise ValueError('Actual channel/selected exposure summaries differ')
 ordinary=read(b/'ordinary_augmentation_actual.json')
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:raise ValueError('Original parent ordinary augmentation differs')
 args=read(b/'actual_native_args.json');args=args.get('args',args)
 if args['label_smoothing']!=.01 or not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher']:raise ValueError('Actual global CE/ordinary/U/EMA flags differ')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed stdout')
 curves=[dict(epoch=z['epoch']+20,local_epoch=z['epoch'],**{k:v/222 for k,v in sums[z['epoch']].items()},**{k:z[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for z in epochs]
 extra=dict(extra_model_NN=0,extra_CE_operator_calls=0,allocation_cpu_wall_seconds=cpuwall,allocation_hook_cpu_wall_seconds=hookwall,timing=a['timing'],selector_included_in_hook_time=True,do_not_add_selector_and_hook=True,not_CUDA_event_time=True,zero_extra_model_forward_is_not_zero_compute=True)
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,selected_packets=142080,As_packets=142080),counts=dict(count),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},allocation=dict(a,sampler_repetitions=dict(repeated),quota_histogram={str(k):v for k,v in sorted(quota_hist.items())},changed_positions_histogram={str(k):v for k,v in sorted(changes.items())}),selector_accounting=extra,EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660),selected_exposure=dict(TX=dict(by_tx),RX=dict(by_rx),day=dict(by_day),TX_RX_day=dict(by_cell),available_TX_RX_day=dict(available_cell),same_candidate_L128_old_parity_TX_RX_day=dict(old_cell),physical=exposure_summary(exposure,unique['L']),same_candidate_L128_old_parity_physical=exposure_summary(old_exposure,unique['L']),actual_As_packets=142080,coverage=coverage,old_parity_is_counterfactual_on_candidate_sequence_not_original_run_replay=True),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=mse)


def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Existing complete original30 audit required')
 result={};lookup=dict(rows(project))
 for name in ('NOAS30','AS30'):
  r=prior['rows'][name]
  if r['cost']['new_updates']!=6660 or r['cost']['new_epochs']!=30 or len(r['epoch_curves'])!=30:raise ValueError('Saved original30 audit schema/budget differs')
  result[name]=dict(reused=True,cost=r['cost'],epoch_curves=r['epoch_curves'],evidence='Existing complete original30 audit reused;no original step stream rescanned')
 candidate,private=parse_new_logs(lookup['ALLOC_AS30'],'ALLOC_AS30');candidate['reused']=False;result['ALLOC_AS30']=candidate
 original=read(lookup['AS30']/'stage20/result.json');parent=dict(ALLOC_AS30=dict(cost=actual_cost(original),epochs=20,updates=4440,wall_seconds=original['wall_seconds'],reused=True));oldcost=result['AS30']['cost'];prefix=parent['ALLOC_AS30']['cost'];oldtail={k:oldcost[k]-prefix[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')}
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
 allocation_increment={v:comparisons['ALLOC_AS30__AS30'][v]['accuracy_delta_pp'] for v in VIEWS};allocation_increment['urban_mean']=comparisons['ALLOC_AS30__AS30']['urban_mean_accuracy_delta_pp']
 key='ALLOC_AS30__NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,source_decision=source['decision'],incremental_satellite_allocation_effect_pp=allocation_increment,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['ALLOC_AS30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['ALLOC_AS30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  out[arm]=dict(native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm in CONDITIONS[:2],source_readout_wall_seconds=q.get('wall_seconds'))
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['NN']+real['NN'];parents={a:v['cost']['native_training_and_validation_NN'] for a,v in training['parent20'].items()};segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:])
 return dict(rows=out,new_model_epochs=10,new_updates=2220,selector_accounting={a:training['rows'][a].get('selector_accounting',{}) for a in CONDITIONS[2:]},new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,new_total_NN=segments+shared,parent20_native_NN_reused=parents,cumulative_candidate_native_NN={a:parents[a]+out[a]['native_NN'] for a in CONDITIONS[2:]},cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;not speed evidence')

def render_report(source,formal,training,costs):
 new='ALLOC_AS30';s=formal['stage'];d=source['decision'];audit=training['rows'][new];alloc=audit['allocation']
 lines=['# As64 TX×RX 分层选择：E30 三条件结果','',RUN+'；模型 seed 2026092701。新增一行 B01，从原 P43 ownAs20 学生与 ownEMA 追加 10 轮，累计 E30/6660 更新；原 NoAs30、As30 固定读出复用。','',f"正式 clean 相对 NoAs30 {s['clean_delta_pp']:+.6f} 个绝对百分点，clean ≥−0.1 pp 阶段约束={s['clean_absolute_0p1pp_guard_met']}。相对原 As30 的 clean 变化为 {formal['comparisons']['ALLOC_AS30__AS30']['clean']['accuracy_delta_pp']:+.6f} pp。low/mid urban 相对 NoAs30 分别 {s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f} pp。n=1，不能据阶段 PASS 或微小上涨宣称可重复双提升。",'', '唯一变量是从原 L128 位置中，按实际可用 TX×RX cell 做可行 waterfill 配额、无放回选择 64 个 As 位置。整个 L128 加载器、ordinary128 与 ORD128 SupCon 输入保持；mixed128 中进入 As 的位置及 ordinary 补集随之改变。所有位置仍有原 clean 监督，全部合法 As 均有真标签监督，不按置信度筛选，不读取 U 的 TX 真值。', '', '保留原 CE label_smoothing=.01、mixed128 对称 KL、同已生成 IQ 的两次 RDrop、原 raw shape、6L/768NN、总 L 尺度 1.25 与 As .25 份额，U weak/strong、教师与 ownEMA 不变。物理信道、接收处理、固定正式 IQ 与六场景不变。此因素改变有限 TX/RX/day/physical 的 As 曝光；不声称隔离纯方差，也不声称逐步 IQ、dropout 或整体训练轨迹相同。', '', '|视图|NoAs30 %|原 As30 %|AsAlloc30 %|新 As−NoAs pp|新 As−原 As pp|原 As−NoAs pp|','|---|---:|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  vals=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS];diff=[formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp'] if v=='urban_mean' else formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for a,b in PAIRS];lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals+diff)+'|')
 lines+=['','|条件|视图|accuracy %|macro-F1 %|最差 RX %|','|---|---|---:|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|比较|视图|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['',f"源域仅两条件：新 As fullV clean−原 NoAs30 {d['fullV_clean_vs_NoAs30_pp']:+.6f} pp（要求 ≥−0.1）；V1800 两 draw 的 mid/low 等权均值，新 As−原 As30 {d['mid_low_mean_vs_original_As30_pp']:+.6f} pp（要求 ≥+0.5）。合取 PASS={d['source_criteria_pass']}，E50 源资格={d['E50_source_eligible']}，固定配方关闭={d['fixed_recipe_closed']}。失败关闭此固定选择配方，不扫 cell/seed/配额救跑；全部正式仍完成。资格仅是 eligible，分析不自动启动 E50 或扩种子，正式结果不回流源决定。",'', '|条件|fullV clean %|panel clean %|两 draw mid/low 等权 %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['', 'fullV27000 为冻结原生 accuracy/RX，不虚构逐包 F1 或修复。V1800 三场两 draw 独立重算，两 draw 不是两个模型 seed，源面板无 high urban。源面板 2.45 GHz 与训练/正式 2.462 GHz 差异保留。全 5 源 RX、L6300/U56700 真值隐藏/V27000、256 点、split392005 权限保持，无 RX 留出。', '', '|条件|suburban 视图|绝对 accuracy %|自身 clean−suburban pp|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v,g in formal['suburban_selfclean_gaps_pp'][a].items():lines.append(f"|{a}|{v}|{100*formal['summary'][a]['views'][v]['accuracy']:.6f}|{g:.6f}|")
 lines+=['', '缩小 suburban 差距须结合 clean 与 suburban 绝对值判断，clean 下降本身也会缩差。新增操作只作用于 As 位置选择，三条件不构造假四行因子或交互；原 As 效应与新增选择效应分别报告。', '', '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|native 秒|','|---|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append(f"|{a}|{c['student_train_NN']}|{c['student_eval_NN']}|{c['teacher_train_NN']}|{c['teacher_eval_NN']}|{c['native_NN']}|{c['native_wall_seconds']:.3f}|")
 lines+=['',f"新增 10 model-epoch/2220 更新，训练及 source/formal 读出 {costs['new_training_and_readouts_NN']} NN，共享 smoke 实际 {costs['new_shared_smoke_NN']} NN，总新增 {costs['new_total_NN']} NN。父 E20、旧两 E30 及历史读出单列复用，不重计。新段 U 学生/教师与 ownEMA 的实际计数在 training_logs 中；全部 2220 步/10 轮及 L/U 流完整核对，旧审计复用不回扫。",'', 'selector 使用私有 CPU 随机流，新增 CPU 选择与标量记录耗时单列；不是 CUDA event 或额外神经前向，也不能当作零计算成本。source 预测 wall 未单列时明确缺失，不用分析 wall 替代。As 每 cell 与 physical 实际曝光分布保留，输入物理 ID 可能按原 sampler 重复，不去重、不要求批内 PID 唯一。', '', f"全部 3×7 固定预测与独立 truth-last 评分齐后才 truth join，重核 {formal['independent_confusion_matrices']} CM/{formal['partition_reconciliations']} 次 RX/day/TX 分区合并。技术缺项由完整性状态报告，不把尚未完成写成成功。n=1、已暴露基准背景保留；本轮不能证明可重复 clean 与星地双提升。分析 NN/CP/IQ/updates 均为 0。"]
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
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_satellite_allocation30/runner.py' in argv:live.append(pid)
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
