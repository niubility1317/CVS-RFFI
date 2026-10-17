"""Independent P39 recount. Source mode never opens target, IQ or checkpoint."""
import argparse,gzip,json,math,re,time
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
RUN='20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01';ROOT=base.PROJECT/'runs'/RUN
ARMS=('B0','URAW','UAS');SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PAIRS=(('URAW','B0'),('UAS','B0'),('UAS','URAW'))
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
base.ROWS={a:(RUN,a+'-s2026092701') for a in ARMS};base.COLORS.update(B0='#64748b',URAW='#2563eb',UAS='#16a34a');base.COMPARISON_PAIRS=PAIRS
read,write,write_csv=base.read,base.write,base.write_csv
PARSED={}
def lines(p):
 with gzip.open(p,'rt',encoding='utf8') if str(p).endswith('.gz') else Path(p).open(encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def numeric(v,name):
 if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Nonfinite/missing active field '+name)
 return float(v)
def required_sources():
 return [ROOT/(a+'-s2026092701')/p for a in ARMS for p in ['stage20/completion.json','source/completion.json','source_qualification/complete.json']]+[ROOT/'source_matrix_decision.json']
def wait_source(pids):
 while not all(p.is_file() for p in required_sources()):
  for p in [ROOT/'failure.json']+[ROOT/(a+'-s2026092701')/'source_qualification_failure.json' for a in ARMS]:
   if p.exists():raise RuntimeError('Known incomplete experiment;preserve outputs: '+str(p))
  if not any((Path('/proc')/str(pid)).exists() for pid in pids):raise RuntimeError('Owned processes exited before complete source artifacts')
  time.sleep(20)
def completed_training(root):
 contract=read(base.PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')['role_ids']
 for stage,e,l,u,steps in [('stage20',20,13,7,4440),('source',10,0,10,2220)]:
  b=root/stage;done=read(b/'completion.json');cfg=done['config'];init=read(b/'initialization.json');freeze=read(b/'source_frozen.json');r=read(b/'result.json')
  if (done['epoch'],done['optimizer_steps'],cfg['epochs'],cfg['label_epochs'],cfg['pseudo_epochs'],cfg['updates'])!=(e,steps,e,l,u,steps) or done['target_access'] or done['head_fit_updates'] or cfg['run_id']!=RUN or cfg['row_id']!=root.name:raise ValueError('Fixed same-row own20+10 budget differs')
  expected=[] if stage=='stage20' else [str(root/'stage20/final_ssdg.pth')]
  if init['checkpoint_sources']!=expected or init['scratch_only']!=(stage=='stage20') or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or read(b/'source_contract.json')['role_ids']!=contract:raise ValueError('Original all-source inheritance differs')
  if freeze['config']!=cfg or freeze['selection']!='final_only' or freeze['target_access'] or r['config']!=cfg or r['head_fit_updates'] or r['U_truth_read'] or r['target_access']:raise ValueError('Frozen original default head required')
 return read(root/'source/completion.json')
def completed(root):
 done=completed_training(root);p=read(root/'prediction/complete.json');s=read(root/'scoring/complete.json')
 if p.get('config')!=done['config'] or p.get('row_id')!=root.name or s.get('row_id')!=root.name or p['status']!='PREDICTIONS_COMPLETE' or p['count']!=168000 or p['views']!=list(base.VIEWS) or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or s['status']!='SCORED_COMPLETE' or s['views']!=list(base.VIEWS) or not s['truth_last']:raise ValueError('Truth remains closed:all7 frozen predictions and separate scoring required')
 return p,s,done

def parse_source(root,done=None):
 if root.name in PARSED:return PARSED[root.name]
 done=completed_training(root);arm=root.name.split('-s')[0];curves=[];totals=Counter();unique=defaultdict(set);seq_count=0;qualification=read(root/'stage20/U_qualification.json');steps_seen=0;counts=Counter();support=[];ordinary=[];cost=Counter();walls=0.;max_decomp=0.
 if qualification['NN']!=14400 or qualification['training_updates'] or qualification['target_access'] or qualification['U_truth_read'] or not qualification['state_unchanged']:raise ValueError('One fixed first-U source-only check required')
 for stage,e,l,u,offset in [('stage20',20,13,7,0),('source',10,0,10,20)]:
  b=root/stage;result=read(b/'result.json');d=read(b/'completion.json');step_offset=offset*222;expected=e*222
  if result['optimizer_steps']!=expected or result['L_NN']!=expected*512 or result['L_As']!=expected*64 or result['extra_U_NN']!=u*56700 or result['U_As']!=(u*56700 if arm=='UAS' else 0) or result['source_qualification']!=qualification:raise ValueError('Actual same-forward L/U budget differs')
  for key in ['student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN']:cost[key]+=result[key]
  walls+=d['elapsed_seconds'];support.append(result['L_channel'])
  if result['teacher_train_NN'] or result['source_RX_holdout'] is not None:raise ValueError('Teacher training or source holdout is forbidden')
  refs={};u_records={};epoch_sums=defaultdict(lambda:defaultdict(float));epoch_counts=Counter()
  for i,r in enumerate(lines(b/'reference_steps.jsonl.gz'),1):
   if r['step']!=i or r['cumulative_step']!=step_offset+i or r['L_forwards']!=4 or r['As_packets']!=64 or r['Ag_packets']!=0 or not r['same_IQ_each_pair'] or r['ordinary_redraws'] or not r['all_satellite_L_supervised']:raise ValueError('Original four-forward/.75ordinary/.25fullAs contract differs')
   for k in ['common_CE','common_KL','common_RD','extra_CE','extra_KL','extra_RD','total_L','SupCon','SupCon_weight','ordinary_raw_MSE']:numeric(r[k],k)
   epoch=(step_offset+i-1)//222+1;weight=.1*min(max((epoch-5)/10,0),1)
   if abs(r['SupCon_weight']-weight)>1e-12:raise ValueError('Cumulative original SupCon schedule differs')
   decomp=.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+r['SupCon_weight']*r['SupCon'];max_decomp=max(max_decomp,abs(decomp-r['total_L']))
   if abs(decomp-r['total_L'])>3e-5:raise ValueError('Actual labeled objective decomposition differs')
   refs[i]=r;counts['L_records']+=1;counts['SupCon_active_steps']+=int(weight>0)
   for q in r['SupCon_details']:counts['total_anchors']+=q['total_anchors'];counts['valid_anchors']+=q['valid_anchors']
   ordinary.append(dict(step=step_offset+i,epoch=epoch,MSE=r['ordinary_raw_MSE']))
  if len(refs)!=expected:raise ValueError('Incomplete labeled objective log')
  for r in lines(b/'U_steps.jsonl.gz'):
   step=r['step'];local=step-step_offset;epoch=(step-1)//222+1;n=r['packets'];qualified=qualification['source_qualified'];weight=0. if arm=='B0' or not qualified else .16*min(1.,(epoch-13)/5.)
   if local in u_records or local not in range(l*222+1,expected+1) or r['cumulative_step']!=step or r['epoch']!=epoch or r['mode']!=arm or n not in range(1,257) or r['source_qualified']!=qualified or r['U_TX_truth_read'] or r['extra_teacher_forwards'] or r['urban_confidence_filter'] or abs(r['weight']-weight)>1e-12 or abs(r['weighted_KL']-r['KL']*weight)>1e-8:raise ValueError('Actual hidden-U coefficient/teacher/view route differs')
   for k in ['KL','weighted_KL','q_mean']:numeric(r[k],k)
   if not 0<=r['q_mean']<=1 or not 0<=r['positive_q']<=n or not 0<=r['original_selected']<=n:raise ValueError('U mass/counts differ')
   u_records[local]=r;totals['extra_U_NN']+=n;totals['positive_q']+=r['positive_q'];totals['original_U_selected']+=r['original_selected']
  if len(u_records)!=u*222 or sum(r['packets'] for r in u_records.values())!=u*56700:raise ValueError('Incomplete additional U exposure')
  for i,r in enumerate(lines(b/'step_metrics.jsonl.gz'),1):
   m=r['metrics'];local_epoch=(i-1)//222+1;epoch=local_epoch+offset;ref=refs[i]
   if r['step']!=i or r['epoch']!=local_epoch or any(m.get(k) or 0 for k in ['train/skipped_nonfinite_loss','train/skipped_nonfinite_grad']):raise ValueError('Successful native updates differ')
   numeric(m['train/loss'],'train/loss')
   if local_epoch>l:
    q=u_records[i]
    if m.get('train/pseudo_truth_available',0)!=0 or m['train/pseudo_total']!=q['packets'] or m['train/pseudo_selected']!=q['original_selected']:raise ValueError('Hidden U or original selection changed')
    for k in ['KL','weighted_KL','q_mean','positive_q','packets']:epoch_sums[epoch]['U/'+k]+=q[k]
   for k in ['common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight']:epoch_sums[epoch][k]+=ref[k]
   for key in ['train/pseudo_total','train/pseudo_selected']:epoch_sums[epoch][key]+=m.get(key) or 0
   epoch_sums[epoch]['train/loss']+=m['train/loss'];epoch_counts[epoch]+=1;counts['native_step_records']+=1
  if i!=expected or set(epoch_counts.values())!={222}:raise ValueError('Incomplete native step log')
  native_epochs=list(lines(b/'epoch_metrics.jsonl'))
  if [r['epoch'] for r in native_epochs]!=list(range(1,e+1)):raise ValueError('Complete segment epochs required')
  for r in native_epochs:
   epoch=r['epoch']+offset;s=epoch_sums[epoch];curve=dict(epoch=epoch,segment=20 if stage=='stage20' else 30,lr=r['lr'],epoch_wall_seconds=r['epoch_time_s'],source_val_accuracy_pct=r['val_tx_acc'],**{k:s[k]/222 for k in ['common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight','train/loss','U/KL','U/weighted_KL','U/q_mean']},pseudo_total=s['train/pseudo_total'],pseudo_selected=s['train/pseudo_selected'],grad_backbone=r['train_grad_backbone'],grad_total=r['train_grad_total'])
   curve.update({k:r[k] for k in r if k.startswith('aug_')});curve.update({'full/RDrop_CE':.5*(s['common_CE']+s['extra_CE'])/222,'full/RDrop_KL':.5*(s['common_KL']+s['extra_KL'])/222,'full/clean_CE':s['common_CE']/222,'full/second_CE':s['extra_CE']/222});curves.append(curve)
  for i,r in enumerate(lines(b/'unit_steps.jsonl.gz'),1):
   if r['step']!=i or r['cumulative_step']!=step_offset+i or r['physical_clip_calls']!=i:raise ValueError('Sinc physical update count differs')
   counts['Sinc_updates']+=1
  if i!=expected:raise ValueError('Sinc update log incomplete')
  for r in lines(b/'training_sequence.jsonl.gz'):
   role=r['role']
   if role not in ['L','U']:raise ValueError('Only original L/U sequence roles allowed')
   totals[role]+=len(r['ids']);unique[role].update(tuple(v) for v in r['ids']);seq_count+=1
 if counts['native_step_records']!=6660 or counts['L_records']!=6660 or counts['Sinc_updates']!=6660 or counts['SupCon_active_steps']!=5550 or totals['L']!=852480 or totals['U']!=963900 or totals['extra_U_NN']!=963900 or len(unique['L'])!=6300 or len(unique['U'])!=56700 or [r['epoch'] for r in curves]!=list(range(1,31)):raise ValueError('Full E30 13L17U execution differs')
 stdout=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors=[dict(line=i,text=line[:250]) for i,line in enumerate(stdout.splitlines(),1) if re.search(r'Traceback|CUDA out of memory|\bKilled\b|RuntimeError|AssertionError|\[ERROR\]',line)]
 out=dict(curve=curves,cost=dict(source_identity_NN=sum(cost.values()),source_counters=dict(cost),source_wall_seconds=walls,L_identity_NN=3409920,extra_U_NN=963900,new_epochs=30,new_updates=6660,training_parent_epochs=0,own_stage20_is_included=True),full_records=dict(counts,sequence_records=seq_count,loader_packets={k:totals[k] for k in ['L','U']},unique_physical={k:len(v) for k,v in unique.items()},original_U_selection_fraction=totals['original_U_selected']/963900,positive_teacher_q_fraction=totals['positive_q']/963900,teacher_precision_on_U=None,U_TX_truth_hidden=True,stdout_lines=len(stdout.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',stdout,re.I)),maximum_L_loss_decomposition_error=max_decomp),source_result=dict(segments=[read(root/s/'result.json') for s in ['stage20','source']],first_U_qualification=qualification),physical_metadata=dict(not_applicable=True,reason='Legacy combined training support plot not applicable;separate L/U support below'),L_support=support,ordinary_observed=ordinary,source_qualified=qualification['source_qualified'])
 PARSED[root.name]=out;return out

def independent_source_decision(clean,urban,qualified):
 if set(clean)!=set(ARMS) or set(urban)!=set(ARMS) or set(qualified)!=set(ARMS):raise ValueError('All3 source rows mandatory')
 arr={a:np.asarray(urban[a],float) for a in ARMS}
 if any(v.shape!=(2,3) or not np.isfinite(v).all() for v in arr.values()) or any(not math.isfinite(v) for v in clean.values()):raise ValueError('Fixed2draw/3scene source matrix')
 delta=100*(arr['UAS'][:,1:]-arr['URAW'][:,1:]);checks=dict(all_own_teachers_qualified=all(qualified.values()),UAS_clean_kept_vs_base=clean['UAS']>=clean['B0']-.0025,UAS_clean_kept_vs_raw=clean['UAS']>=clean['URAW']-.0025,each_urban_draw_gain_over_raw=bool((delta>=1.-1e-10).all()))
 return dict(status='SOURCE_DECIDED',checks=checks,source_qualified=all(checks.values()),urban_deltas_pp=delta.ravel().tolist(),clean=clean,source_only=True,formal_scores_read=False,automatic_seed_extension=False,goal_complete=False)

def source_readout(out):
 data={};q={};role=read(PANEL/'data_roles.json');rows=[]
 for a in ARMS:
  b=ROOT/(a+'-s2026092701');proof=read(b/'source_qualification/complete.json')
  if proof['status']!='SOURCE_READOUT_COMPLETE' or proof['NN']!=12600 or proof['training_updates'] or proof['head_fit_updates'] or proof['target_access'] or proof['formal_scores_read']:raise ValueError('Only frozen source V readout')
  with np.load(b/'source_qualification/predictions.npz',allow_pickle=False) as f:data[a]={k:f[k].copy() for k in f.files}
  x=data[a]
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all() or not np.array_equal(x['ids'],np.asarray(role['physical_ids'])) or not np.allclose((x['satellite'].argmax(-1)==x['TX']).mean(-1),proof['source_accuracies'],atol=1e-12,rtol=0) or not np.isclose((x['clean'].argmax(1)==x['TX']).mean(),proof['clean_accuracy'],atol=1e-12,rtol=0):raise ValueError('Independent source logits or same panel differ')
  q[a]=proof
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ['state','lock','signal_to_added_noise_db']}
 y=data['B0']['TX'];rx=data['B0']['RX'];days0=data['B0']['day'];cohort=data['B0']['clean'].argmax(1)==y
 for a,x in data.items():
  if not all(np.array_equal(x[k],data['B0'][k]) for k in ['ids','TX','RX','day']):raise ValueError('Original source V roles/panel differ')
  for si,scene in enumerate(SCENES):
   for draw in [-1,0,1]:
    ds=[0,1] if draw==-1 else [draw];yy=np.tile(y,len(ds));rr=np.tile(rx,len(ds));dd=np.tile(days0,len(ds));cc=np.tile(cohort,len(ds));z=np.concatenate([x['satellite'][d,si] for d in ds]);pred=z.argmax(1);old=np.concatenate([data['B0']['satellite'][d,si].argmax(1) for d in ds]);state=np.concatenate([physics['state'][d,si,0] for d in ds]);lock=np.concatenate([physics['lock'][d,si,0] for d in ds]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds]);other=z.copy();other[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-other.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in [1,3,4,6,8]]+[('day',str(d),dd==d) for d in [1,2,3]]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==i) for i,s in enumerate(['LOS','PARTIAL','BLOCKED'])]+[('lock',s,lock==i) for i,s in enumerate(['locked','degraded','unlocked'])]+[('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dimension,stratum,mask in masks:
     if not mask.any():continue
     denominator=int((mask&cc).sum());m=base.metric(yy[mask],pred[mask]);rows.append(dict(arm=a,scene=scene,draw=draw,dimension=dimension,stratum=stratum,**m,mean_margin_over30=float(margin[mask].mean()),fixed_B0_clean_correct_count=denominator,retention_on_B0_clean_correct=float((pred==yy)[mask&cc].mean()) if denominator else None,repairs_vs_B0=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_B0=int(((pred!=yy)&(old==yy)&mask).sum())))
 summary=dict(status='SOURCE_THREE_ROW_COMPLETE',source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},rows=rows,source_qualification=q,model_seed_count=1,source_unique_V=1800,channel_draws=2,cohort='fixed new B0 clean-correct control;not a historical unaugmented father',added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False,NN=0,training_updates=0,target_access=False)
 write(out/'source_summary.json',summary);write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);return summary

def analyze_source(out):
 if not all(p.is_file() for p in required_sources()):raise ValueError('Source analysis incomplete;target remains closed')
 decision=read(ROOT/'source_matrix_decision.json');clean={};urban={};qualified={};source={}
 for a in ARMS:
  b=ROOT/(a+'-s2026092701');done=completed_training(b);source[a]=parse_source(b,done);clean[a]=done['final_source_metrics']['source_val_accuracy'];urban[a]=read(b/'source_qualification/complete.json')['source_accuracies'];qualified[a]=source[a]['source_qualified']
 independently=independent_source_decision(clean,urban,qualified)
 if independently['checks']!=decision['checks'] or independently['source_qualified']!=decision['source_qualified'] or not np.allclose(independently['urban_deltas_pp'],decision['urban_deltas_pp'],atol=1e-12,rtol=0):raise ValueError('Independent source rule differs')
 write(out/'independent_source_decision.json',independently)
 summary=source_readout(out)
 for a in ARMS:write(out/(a+'_source.json'),source[a])
 ordinary={};seq={}
 for a in ARMS[1:]:
  ordinary[a]=dict(records=len(source[a]['ordinary_observed']),mismatch=sum(x!=y for x,y in zip(source['B0']['ordinary_observed'],source[a]['ordinary_observed'])))
  seq[a]={}
  for stage in ['stage20','source']:
   pa=ROOT/(a+'-s2026092701')/stage/'training_sequence.jsonl.gz';pb=ROOT/'B0-s2026092701'/stage/'training_sequence.jsonl.gz';count=mismatch=0
   for x,y in zip_longest(lines(pa),lines(pb)):count+=1;mismatch+=x!=y
   seq[a][stage]=dict(records=count,mismatch=mismatch)
 curves=[dict(arm=a,**r) for a in ARMS for r in source[a]['curve']];write_csv(out/'training_curves.csv',curves)
 result=dict(status='SOURCE_ANALYZED',source_decision=independently,ordinary_sequence_comparison=ordinary,physical_sequence_comparison=seq,source_cost={a:source[a]['cost'] for a in ARMS},source_full_log_recount={a:source[a]['full_records'] for a in ARMS},first_U_qualification={a:source[a]['source_result']['first_U_qualification'] for a in ARMS},source_confusions=len(summary['rows']),source_cohort='B0 control clean-correct',new_NN=0,IQ_reads=0,checkpoint_reads=0,formal_scores_read=False,goal_complete=False)
 write(out/'complete.json',result);return result

def receiver_quality_policy(cfg,quality_db):
 """Independent recount of actual runtime policy;no channel or state mutation."""
 q=numeric(quality_db,'receiver quality');lo=numeric(cfg['unlocked_snr_db'],'unlocked threshold');hi=numeric(cfg['locked_snr_db'],'locked threshold');ref=numeric(cfg['quality_reference_snr_db'],'quality reference');maximum=numeric(cfg['max_residual_scale'],'residual maximum')
 if hi<=lo or maximum<1:raise ValueError('Invalid actual receiver quality policy')
 if not cfg['quality_adaptation_enabled']:return 'locked',1.
 return ('locked' if q>=hi else 'degraded' if q>=lo else 'unlocked'),10**min(max((ref-q)/20,0),math.log10(maximum))

def quality_sensitivity(record,cfg):
 """Metadata-only alternative proxies;original waveform/receiver remain fixed.

 signal_to_added_noise_db includes only simulated added noise. The reference
 ratio uses inferred post-compensation signal power / reference noise variance;
 it removes realized noise-power fluctuation but retains RX/image effects. None
 is total SNR or an acquisition estimate;no classification effect is measured.
 """
 state,scale=receiver_quality_policy(cfg,record['quality_snr_db'])
 if state!=record['receiver_lock_status'] or not math.isclose(scale,numeric(record['residual_scale'],'recorded residual scale'),rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Recorded receiver state differs from actual runtime policy')
 measured=numeric(record['measured_added_noise_power_pre_agc'],'measured added noise');variance=numeric(record['added_noise_variance_reference'],'reference noise variance');ratio=numeric(record['signal_to_added_noise_db'],'signal/added-noise ratio')
 if measured<=0 or variance<=0:raise ValueError('Positive added/reference noise powers required')
 noise_delta=10*math.log10(measured/variance);reference_ratio=ratio+noise_delta
 return dict(actual_state=state,signal_added_noise_state=receiver_quality_policy(cfg,ratio)[0],signal_reference_noise_state=receiver_quality_policy(cfg,reference_ratio)[0],signal_reference_noise_ratio_db=reference_ratio,signal_reference_noise_minus_white_proxy_db=reference_ratio-record['quality_snr_db'],measured_added_noise_reference_ratio_db=noise_delta)

def summarize_U_physics():
 packets=batches=0;strata=Counter();sessions={};receiver_variations=0;hardware_sessions={};stats=defaultdict(list);flags=Counter();sensitivity=Counter();actual_configs=None
 for stage,start,stop,expected in [('stage20',2887,4440,396900),('source',4441,6660,567000)]:
  configs=read(ROOT/'UAS-s2026092701'/stage/'actual_U_channel.json')['configurations']
  if actual_configs is not None and configs!=actual_configs:raise ValueError('Actual receiver configuration changed across own segments')
  actual_configs=configs
  hardware=read(ROOT/'UAS-s2026092701'/stage/'result.json')['U_channel']['sessions']
  for session,value in hardware.items():
   if session in hardware_sessions and hardware_sessions[session]!=value:raise ValueError('Actual stable physical hardware changed across segments')
   hardware_sessions[session]=value
  base_packets=packets;expected_step=start
  for batch in lines(ROOT/'UAS-s2026092701'/stage/'U_physics.jsonl.gz'):
   if batch['step']!=expected_step or len(batch['observations']) not in range(1,257):raise ValueError('Full U physical observations incomplete')
   expected_step+=1;batches+=1
   for r in batch['observations']:
    if r['processing_route']!='residual' or r['mode']!='post_sync' or r['channel_equalization_applied'] or r['tx_impairment_correction_applied'] or r['true_total_snr_known'] or r['scene'] not in SCENES or '/U_S' not in r['namespace']:raise ValueError('Actual full U physics/permission differs')
    diagnostic=quality_sensitivity(r,configs[r['scene']])
    for kind in ['signal_added_noise_state','signal_reference_noise_state']:sensitivity[(r['scene'],kind,diagnostic['actual_state'],diagnostic[kind])]+=1
    for k in ['signal_reference_noise_ratio_db','signal_reference_noise_minus_white_proxy_db','measured_added_noise_reference_ratio_db']:stats[k].append(diagnostic[k])
    session=r['session_id'];hardware=r['receiver']
    receiver_variations+=int(session in sessions and sessions[session]!=hardware)
    sessions[session]=hardware;pid=r['sample_id'];strata[(pid[1],pid[2],r['scene'],r['state_end'],r['receiver_lock_status'])]+=1;packets+=1
    for k in ['quality_snr_db','signal_to_added_noise_db','output_frequency_hz','residual_kernel_length','residual_kernel_energy','numerical_filter_latency_samples','agc_gain_db']:stats[k].append(numeric(r[k],k))
    stats['waveform_ratio_minus_white_proxy_db'].append(r['signal_to_added_noise_db']-r['quality_snr_db'])
    for k in ['agc_limited','boundary','noise_boundary','quality_source','channel_equalization_applied','true_total_snr_known']:flags[(k,str(r[k]))]+=1
  if expected_step!=stop+1 or packets-base_packets!=expected:raise ValueError('U generation count differs from all original U')
 if packets!=963900 or batches!=3774:raise ValueError('U full-physics budget differs')
 summary={k:dict(count=len(v),mean=float(np.mean(v)),quantiles=dict(zip(['min','q01','q10','q50','q90','q99','max'],np.quantile(v,[0,.01,.1,.5,.9,.99,1]).tolist()))) for k,v in stats.items()}
 return dict(packets=packets,batches=batches,sessions=hardware_sessions,receiver_metadata_variations=receiver_variations,statistics=summary,strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(strata.items())],flags=[dict(field=k[0],value=k[1],count=n) for k,n in sorted(flags.items())],U_TX_truth_available=False,U_accuracy_measured=False,added_noise_ratio_is_total_SNR=False,training_input_IQ_reads=0,actual_state_coverage_only_not_causal_accuracy=True,quality_policy_sensitivity=dict(rows=[dict(scene=k[0],alternative=k[1],recorded_state=k[2],alternative_state=k[3],count=n) for k,n in sorted(sensitivity.items())],runtime_configurations=actual_configs,recorded_states_match_actual_runtime_policy=True,receiver_state_and_compensation_recomputed=False,waveforms_unchanged=True,classification_effect_measured=False,alternative_ratio_is_total_SNR=False,reference_ratio_includes_post_compensation_RX_image_effects=True))

def own_plots(out,source,source_summary,physics=None):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 files=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','pdf']:
   p=out/(name+'.'+ext);fig.savefig(p,dpi=190);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7.2))
 for a in ARMS:
  c=source[a]['curve'];epoch=np.arange(1,31);color=base.COLORS[a];visible=np.asarray([r['epoch']>13 for r in c])
  for ax,key in [(axes[0,0],'U/q_mean'),(axes[0,1],'U/KL'),(axes[0,2],'U/weighted_KL')]:ax.plot(epoch,np.where(visible,[r[key] for r in c],np.nan),label=a,color=color)
  axes[1,0].plot(epoch,[r['grad_backbone'] for r in c],label=a,color=color)
  axes[1,1].plot(epoch,[r['lr'] for r in c],label=a,color=color)
  axes[1,2].plot(epoch,[r['common_CE'] for r in c],label=a+' ordinary',color=color);axes[1,2].plot(epoch,[r['extra_CE'] for r in c],label=a+' mixed',color=color,ls='--')
 for ax,title in zip(axes.flat,['Raw-teacher confidence weight; not U precision','Extra U KL readout (zero-loss B0 also measured)','Additional weighted U objective','Original backbone gradient norm','Actual native LR / optimizer reset at20','L ordinary/mixed CE; same .5/.5 pair weights']):
  ax.set(title=title,xlabel='Cumulative epoch');ax.axvline(13.5,ls=':',color='gray');ax.axvline(20.5,ls='--',color='gray');ax.grid(alpha=.18)
 axes[0,0].legend(frameon=False);axes[1,2].legend(frameon=False,fontsize=7);fig.suptitle('Full13L17U / default head / complete logs; objective values are not performance gains');save(fig,'extra_U_training_curves')
 fig,axes=plt.subplots(1,3,figsize=(14,4.6))
 for a,ax in zip(ARMS,axes):
  strata=[r for segment in source[a]['L_support'] for r in segment['strata']];matrix=np.asarray([[sum(r['count'] for r in strata if r['RX']==rx and r['scene']==scene) for scene in SCENES] for rx in [1,3,4,6,8]])
  im=ax.imshow(matrix,cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=['Low sub','Mid urban','Low urban'],yticks=range(5),yticklabels=['RX'+str(r) for r in [1,3,4,6,8]],title=a+' labeled full As')
  for j in range(5):
   for k in range(3):ax.text(k,j,str(matrix[j,k]),ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>matrix.max()*.8 else 'black')
  fig.colorbar(im,ax=ax,fraction=.04,label='Full As observations')
 fig.suptitle('Actual L coverage / no held source RX; no per-packet TX-by-state sidecar saved');save(fig,'training_L_support')
 rows=source_summary['rows'];fig,axes=plt.subplots(1,2,figsize=(11,4.5));bins=['<3','3-8','8-15','>=15']
 for ax,scene in zip(axes,SCENES[1:]):
  for a in ARMS:
   selected=[next(r for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(a,scene,-1,'added_noise_ratio',label)) for label in bins];ax.plot(range(4),[100*r['accuracy'] for r in selected],'o-',label=a,color=base.COLORS[a])
  ax.set(xticks=range(4),xticklabels=[b+' dB' for b in bins],ylabel='Source accuracy (%)',ylim=(0,100),title=scene.replace('practical_',''),xlabel='Signal / added noise ratio; not total SNR');ax.grid(alpha=.18);ax.legend(frameon=False)
 fig.suptitle('Same fixed V1800 / two channel draws / default head');save(fig,'source_added_noise_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(12,5));cells=[('LOS/locked'),('PARTIAL/locked'),('PARTIAL/degraded'),('BLOCKED/degraded'),('BLOCKED/unlocked')]
 for ax,scene in zip(axes,SCENES[1:]):
  for a in ARMS:
   selected=[next(r for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(a,scene,-1,'state_lock',label)) for label in cells];ax.plot(range(5),[100*r['accuracy'] for r in selected],'o-',label=a,color=base.COLORS[a])
  ns=[next(r['count'] for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==('B0',scene,-1,'state_lock',label)) for label in cells];ax.set(xticks=range(5),xticklabels=[s.replace('/','\n')+'\nn='+str(n) for s,n in zip(cells,ns)],ylabel='Source accuracy (%)',ylim=(0,100),title=scene.replace('practical_',''));ax.tick_params(axis='x',labelsize=7);ax.legend(frameon=False);ax.grid(alpha=.18)
 fig.suptitle('Source states are descriptive; fixed formal state labels unavailable');save(fig,'source_state_lock_accuracy')
 if physics is not None:
  fig,axes=plt.subplots(1,2,figsize=(10.5,4.5));strata=physics['strata']
  for ax,key,labels in [(axes[0],'state',['LOS','PARTIAL','BLOCKED']),(axes[1],'lock',['locked','degraded','unlocked'])]:
   bottom=np.zeros(3);total=np.asarray([sum(r['count'] for r in strata if r['scene']==s) for s in SCENES])
   for label in labels:
    vals=100*np.asarray([sum(r['count'] for r in strata if r['scene']==s and r[key]==label) for s in SCENES])/total;ax.bar(range(3),vals,bottom=bottom,label=label);bottom+=vals
   ax.set(xticks=range(3),xticklabels=['Low sub','Mid urban','Low urban'],ylim=(0,103),ylabel='Fraction of generated U observations (%)',title=key+' / U TX truth hidden');ax.legend(frameon=False,fontsize=8)
  fig.suptitle('All963900 UAS observations / support counts, not U classification accuracy');save(fig,'training_U_physical_support')
 return files

def final_analysis(out,source_dir,truth):
 prerequisite=read(source_dir/'complete.json')
 if prerequisite['status']!='SOURCE_ANALYZED' or prerequisite['formal_scores_read']:raise ValueError('Independent source analysis must finish before formal read')
 if read(source_dir/'independent_source_decision.json')['checks']!=read(ROOT/'source_matrix_decision.json')['checks']:raise ValueError('Fixed source decision differs')
 for a in ARMS:completed(ROOT/(a+'-s2026092701'))
 base.completed=completed;base.parse_source=parse_source
 results={};data={}
 for a in ARMS:results[a],data[a]=base.collect_row(a,truth)
 if any(not np.array_equal(data[a]['ids'],data['B0']['ids']) or not np.array_equal(data[a]['y'],data['B0']['y']) for a in ARMS):raise ValueError('Formal physical IDs/truth differ')
 y=data['B0']['y'];fixed=data['B0']['pred']['clean']==y;pairs=[];retention=[]
 for new,old in PAIRS:
  common=(data[new]['pred']['clean']==y)&(data[old]['pred']['clean']==y)
  for view in base.VIEWS:
   n=data[new]['pred'][view]==y;o=data[old]['pred'][view]==y;gain=int((n&~o).sum());lost=int((~n&o).sum());pairs.append(dict(new=new,old=old,view=view,repairs=gain,new_errors=lost,net=gain-lost,accuracy_delta_pp=100*(gain-lost)/len(y),common_clean_count=int(common.sum()),common_clean_repairs=int((n&~o&common).sum()),common_clean_new_errors=int((~n&o&common).sum())))
 for a in ARMS:
  for view in base.VIEWS:retention.append(dict(arm=a,view=view,fixed_B0_clean_correct_count=int(fixed.sum()),retained=int(((data[a]['pred'][view]==y)&fixed).sum()),retention=float((data[a]['pred'][view][fixed]==y[fixed]).mean())))
 physics=summarize_U_physics();write(out/'U_physics_summary.json',physics);write_csv(out/'quality_state_sensitivity.csv',physics['quality_policy_sensitivity']['rows'])
 figures=base.plots(out,results);source={a:results[a]['source'] for a in ARMS};source_summary=read(source_dir/'source_summary.json');figures+=own_plots(out,source,source_summary,physics)
 summary=dict(status='ANALYZED',run_id=RUN,model_seed=2026092701,model_seed_count=1,rows=results,pairs=pairs,fixed_B0_clean_correct_retention=retention,source_decision=read(source_dir/'independent_source_decision.json'),urban_means={a:float(np.mean([results[a]['views'][v]['accuracy'] for v in base.VIEWS[-3:]])) for a in ARMS},suburban_gaps_pp={a:{v:100*(results[a]['views']['clean']['accuracy']-results[a]['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for a in ARMS},formal_confusion_matrices=sum(len(r['metrics']) for r in results.values()),source_confusion_matrices=len(source_summary['rows']),U_TX_truth_read=False,default_head_structure_changed=False,default_head_jointly_trained=True,additional_deployment_parameters=0,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,target_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False,interpretation_limits=['B0 clean-correct cohort is an As-trained control;not unaugmented father','Source states are not formal target state labels','L TX-by-state coverage not saved perpacket;L RX/day/scene/state aggregate and fullphysicalID exposure available','No perpacket gates or perbranch gradient trajectories saved;do not infer them from loss/accuracy','Teacher source qualification is not measured U pseudo-label precision'])
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in q.items() if k!='confusion'} for r in results.values() for q in r['metrics']]);write_csv(out/'paired_repairs.csv',pairs);write_csv(out/'fixed_B0_retention.csv',retention);write(out/'figures.json',figures);write(out/'analysis_summary.json',summary);write(out/'complete.json',dict(status='ANALYZED',rows=list(ARMS),model_seed_count=1,formal_confusions=summary['formal_confusion_matrices'],source_confusions=summary['source_confusion_matrices'],figures=figures,analysis_NN=0,goal_complete=False));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path);p.add_argument('--truth',type=Path);p.add_argument('--wait-pids',nargs='*',type=int);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Reconcile own analysis output;never overwrite')
 if a.mode=='source' and a.wait_pids:wait_source(a.wait_pids)
 a.output.mkdir();started=time.perf_counter()
 if a.mode=='source':r=analyze_source(a.output)
 else:
  if a.source_analysis is None or a.truth is None:raise ValueError('Explicit completed source analysis and truth-side path required')
  r=final_analysis(a.output,a.source_analysis,a.truth)
 write(a.output/'execution.json',dict(mode=a.mode,wall_seconds=time.perf_counter()-started,NN=0,IQ_reads=0,checkpoint_reads=0,target_access=a.mode=='final',waiting_excluded_from_wall_seconds=True))
 print(json.dumps(dict(status=r['status'],mode=a.mode,output=str(a.output),NN=0)),flush=True)
if __name__=='__main__':main()
