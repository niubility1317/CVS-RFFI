"""P43 full completed-log/factorial recount. Source mode never opens target/IQ/CP."""
import argparse,gzip,json,math,re,time
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01';ROOT=base.PROJECT/'runs'/RUN
ARMS=('B00','B10_WHITE','B10_BG','B01','B11_WHITE','B11_BG')
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PAIRS=(('B10_WHITE','B00'),('B10_BG','B00'),('B01','B00'),('B11_WHITE','B00'),('B11_BG','B00'),('B11_WHITE','B01'),('B11_BG','B01'),('B10_BG','B10_WHITE'),('B11_BG','B11_WHITE'),('B11_BG','B10_BG'))
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
base.ROWS={a:(RUN,a+'-s2026092701') for a in ARMS}
base.COLORS.update(B00='#64748b',B10_WHITE='#2563eb',B10_BG='#d97706',B01='#64748b',B11_WHITE='#2563eb',B11_BG='#d97706');base.COMPARISON_PAIRS=PAIRS
read,write,write_csv=base.read,base.write,base.write_csv
PARSED={}
def mode(a):return 'BG' if a.endswith('_BG') else 'WHITE' if a.endswith('_WHITE') else 'NONE'
def satellite(a):return a.startswith('B01') or a.startswith('B11')
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

def source_cost_breakdown(counters,frozen_readout):
 keys=['student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN']
 if set(counters)!=set(keys) or any(type(counters[k]) is not int or counters[k]<0 for k in keys):raise ValueError('Actual four native forward counters required')
 if frozen_readout['status']!='SOURCE_READOUT_COMPLETE' or frozen_readout['NN']!=12600 or any(frozen_readout[k] for k in ['training_updates','head_fit_updates','target_access','formal_scores_read']):raise ValueError('Only actual frozen source readout may add12600NN')
 native=sum(counters.values())
 return dict(native_training_and_validation_identity_NN=native,frozen_source_readout_NN=frozen_readout['NN'],source_identity_NN=native+frozen_readout['NN'],source_counters=dict(counters),first_U_qualification_NN=0,extra_U_student_NN=0,frozen_source_readout_wall_seconds=None,source_wall_scope='native segments;external frozen source readout time not recorded')

def relation(r):
 active=r['SupCon_weight']>0;ds=r['SupCon_details'];n=0
 if r['crossview_mode']!='ORD128' or r['new_losses'] or r['extra_U_NN'] or len(ds)!=(4 if active else 0):raise ValueError('Only unchanged original ordinary128 SupCon')
 if active and {(q['phase'],q['level']) for q in ds}!={(p,l) for p in ('first','second') for l in ('time_down','t2.pool')}:raise ValueError('Actual depth/pass descriptors absent')
 for q in ds:
  if q['mode']!='ORD128' or q['total_anchors']!=128 or q['ordinary_anchors']!=128 or q['additional_anchors'] or q['same_RX_positive_pairs'] or q['true_satellite_anchors']:raise ValueError('Changed original SupCon membership')
  n+=q['valid_anchors']
 return n

def raw_telemetry(b,expected_epochs):
 records=list(lines(b/'metrics_epoch.jsonl'))
 if [q['epoch'] for q in records]!=list(range(1,expected_epochs+1)):raise ValueError('Native full telemetry epochs incomplete')
 nf=Counter();nulls=Counter();numbers=0
 def scan(v,path):
  nonlocal numbers
  if isinstance(v,dict):
   for k,x in v.items():scan(x,path+'.'+str(k))
  elif isinstance(v,(list,tuple)):
   for x in v:scan(x,path+'[]')
  elif isinstance(v,(int,float)):
   numbers+=1
   if not math.isfinite(v):nf[path]+=1
  elif v is None:nulls[path]+=1
 for q in records:scan(q,'epoch')
 return dict(records=len(records),numeric_values=numbers,nonfinite_field_counts=dict(nf),null_field_counts=dict(nulls))

def parse_source(root,done=None):
 if root.name in PARSED:return PARSED[root.name]
 completed_training(root);arm=root.name.split('-s')[0];curves=[];cost=Counter();counts=Counter();totals=Counter();unique=defaultdict(set);ordinary=[];support=[];rawlog={};walls=0.;maxerr=0.;native_L_error=0.;noise_summary=[]
 for stage,e,l,offset in [('stage20',20,13,0),('source',10,0,20)]:
  b=root/stage;expected=e*222;step_offset=offset*222;rawlog[stage]=raw_telemetry(b,e);r0=read(b/'result.json');done=read(b/'completion.json');cfg=done['config']
  if cfg['Ag_mode']!=mode(arm) or cfg['As']!=satellite(arm) or cfg['L_NN_per_step']!=768 or cfg['total_L_loss_scale']!=1.25 or cfg['noise']['SNR_added_db']!=30. or cfg['noise']['impulse_probability']!=.02 or cfg['noise']['impulse_excess_variance']!=49.:raise ValueError('Fixed noise/As/normalizedslot contract differs')
  if r0['optimizer_steps']!=expected or r0['L_NN']!=expected*768 or r0['L_As']!=(expected*64 if satellite(arm) else 0) or r0['Ag_packets']!=(expected*128 if mode(arm)!='NONE' else 0) or r0['Ag_control_NN']!=expected*256 or r0['extra_U_NN'] or r0['U_As'] or r0['teacher_train_NN'] or r0['source_RX_holdout'] is not None:raise ValueError('Actual sixforward/originalU/allsource budget differs')
  for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'):cost[k]+=r0[k]
  walls+=done['elapsed_seconds'];support.append(r0['L_channel']);sums=defaultdict(lambda:defaultdict(float));nume=Counter()
  streams=[lines(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
  for i,items in enumerate(zip_longest(*streams),1):
   if any(q is None for q in items):raise ValueError('Step/reference/shape/Sinc logs incomplete')
   r,s,n,u=items;epoch=offset+(i-1)//222+1;local=(i-1)//222+1;m=n['metrics'];weight=.1*min(max((epoch-5)/10,0),1)
   if any(q['step']!=i for q in items) or r['cumulative_step']!=step_offset+i or s['cumulative_step']!=step_offset+i or u['cumulative_step']!=step_offset+i or u['physical_clip_calls']!=i or n['epoch']!=local:raise ValueError('Contiguous own20+10 step/optimizer evidence differs')
   if r['L_forwards']!=6 or r['As_packets']!=(64 if satellite(arm) else 0) or r['Ag_packets']!=(128 if mode(arm)!='NONE' else 0) or r['Ag_mode']!=mode(arm) or r['extra_control_packets']!=128 or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('TrueL CE/newIQ/sameRDrop pair differs')
   if s['mode']!=mode(arm) or s['NN']!=256 or s['packets']!=128 or s['applied']!=(128 if mode(arm)!='NONE' else 0) or s['weight']!=.25 or not s['same_IQ_each_pair'] or s['new_boundary_samples']:raise ValueError('Matched raw/WHITE/BG slot differs')
   if mode(arm)=='NONE':
    if s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('NoG rawcontrol changed')
   elif s['SNR_added_db']!=30. or not s['total_SNR_unknown'] or abs(s['relative_noise_RMS']-10**(-30/20))>1e-6:raise ValueError('Realized addednoise RMS differs;not totalSNR')
   for k in ('total_L','base_L_before_shape','common_CE','common_KL','common_RD','extra_CE','extra_KL','extra_RD','SupCon','SupCon_weight','ordinary_raw_MSE'):numeric(r[k],k)
   for k in ('shape_CE','shape_KL','shape_RD','relative_noise_RMS','noise_peak_fraction_mean'):numeric(s[k],k)
   if abs(r['SupCon_weight']-weight)>1e-12 or abs(s['shape_RD']-s['shape_CE']-s['shape_KL'])>3e-5:raise ValueError('Original SupCon/RDrop coefficients differ')
   loss=.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+weight*r['SupCon']+.25*(s['shape_CE']+s['shape_KL']);err=abs(loss-r['total_L']);maxerr=max(maxerr,err)
   if err>3e-5 or abs(r['total_L']-r['base_L_before_shape']-.25*s['shape_RD'])>1e-6:raise ValueError('Total1.25 L objective not independently recovered')
   # concat/* retains old fourforward-branch telemetry. It is not total sixforward L cost.
   if m['concat/L_forwards']!=4 or m['concat/As_packets']!=64 or abs(m['concat/total_L']-r['base_L_before_shape'])>3e-5 or abs(m['shape/RD']-s['shape_RD'])>1e-6:raise ValueError('Legacy branch vs extra-slot telemetry relation differs')
   numeric(m['train/loss'],'active native loss')
   if any(m.get(k) or 0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Nonfinite native update')
   if local<=l:
    native_L_error=max(native_L_error,abs(m['train/loss']-loss))
    if abs(m['train/loss']-loss)>3e-5:raise ValueError('Real native supervised total differs from sixforward objective')
   else:
    total=int(m['train/pseudo_total']);selected=int(m['train/pseudo_selected'])
    if m.get('train/pseudo_truth_available',0)!=0 or not 0<=selected<=total<=256 or total<=0:raise ValueError('Original U hidden/selection budget differs')
    totals['native_U_forward']+=total;totals['selected_U']+=selected
   for k in ('common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight'):sums[epoch][k]+=r[k]
   for k in ('shape_CE','shape_KL','shape_RD','relative_noise_RMS','noise_peak_fraction_mean'):sums[epoch][k]+=s[k]
   sums[epoch]['weighted_SupCon']+=weight*r['SupCon'];sums[epoch]['native_total_loss']+=m['train/loss'];sums[epoch]['valid_SupCon_anchors']+=relation(r);sums[epoch]['U_total']+=m.get('train/pseudo_total') or 0;sums[epoch]['U_selected']+=m.get('train/pseudo_selected') or 0;nume[epoch]+=1
   counts['actual_steps']+=1;counts['SupCon_active_steps']+=bool(weight);counts['Ag_applied']+=s['applied'];counts['As_applied']+=r['As_packets'];ordinary.append(dict(step=step_offset+i,epoch=epoch,MSE=r['ordinary_raw_MSE']))
  if i!=expected or set(nume.values())!={222}:raise ValueError('Full segment step count differs')
  epochs=list(lines(b/'epoch_metrics.jsonl'))
  if [q['epoch'] for q in epochs]!=list(range(1,e+1)):raise ValueError('Complete epoch curves required')
  for q in epochs:
   ep=q['epoch']+offset;v=sums[ep];curve=dict(arm=arm,epoch=ep,segment=20 if stage=='stage20' else 30,lr=q['lr'],epoch_wall_seconds=q['epoch_time_s'],source_val_accuracy_pct=q['val_tx_acc'],grad_backbone=q['train_grad_backbone'],grad_total=q['train_grad_total'],pseudo_total=v['U_total'],pseudo_selected=v['U_selected'],**{k:v[k]/222 for k in ('common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight','weighted_SupCon','shape_CE','shape_KL','shape_RD','relative_noise_RMS','noise_peak_fraction_mean','native_total_loss','valid_SupCon_anchors')})
   curve.update({'full/RDrop_CE':.5*(v['common_CE']+v['extra_CE'])/222+.25*v['shape_CE']/222,'full/RDrop_KL':.5*(v['common_KL']+v['extra_KL'])/222+.25*v['shape_KL']/222,'full/clean_CE':v['common_CE']/222,'full/second_CE':v['extra_CE']/222});curve.update({k:q[k] for k in q if k.startswith('aug_')});curves.append(curve)
  for q in lines(b/'training_sequence.jsonl.gz'):
   role=q['role']
   if role not in ('L','U'):raise ValueError('Only originalL/U data sequence')
   totals[role]+=len(q['ids']);unique[role].update(tuple(v) for v in q['ids']);counts['sequence_records']+=1
 if counts['actual_steps']!=6660 or counts['SupCon_active_steps']!=5550 or counts['Ag_applied']!=(852480 if mode(arm)!='NONE' else 0) or counts['As_applied']!=(426240 if satellite(arm) else 0) or totals['L']!=852480 or totals['U']!=963900 or totals['native_U_forward']!=963900 or len(unique['L'])!=6300 or len(unique['U'])!=56700 or [q['epoch'] for q in curves]!=list(range(1,31)):raise ValueError('Actual full E30/all-source budget differs')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors=[dict(line=i,text=line[:250]) for i,line in enumerate(text.splitlines(),1) if re.search(r'Traceback|CUDA out of memory|\bKilled\b|RuntimeError|AssertionError|\[ERROR\]',line)]
 result=dict(curve=curves,cost=dict(**source_cost_breakdown(cost,read(root/'source_qualification/complete.json')),source_wall_seconds=walls,L_identity_NN=5114880,extra_Ag_control_identity_NN=1704960,extra_U_NN=0,new_epochs=30,new_updates=6660,training_parent_epochs=0,own_stage20_is_included=True),full_records=dict(counts,loader_packets={k:totals[k] for k in ('L','U')},unique_physical={k:len(v) for k,v in unique.items()},original_U_selected=totals['selected_U'],U_truth_hidden=True,stdout_lines=len(text.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',text,re.I)),native_full_epoch_telemetry=rawlog,warnings=[dict(line=i,text=line[:250]) for i,line in enumerate(text.splitlines(),1) if re.search(r'Warning|WARN|warning:',line)],maximum_L_decomposition_error=maxerr,maximum_native_supervised_loss_error=native_L_error),source_result=dict(segments=[read(root/s/'result.json') for s in ('stage20','source')]),physical_metadata=dict(not_applicable=True,reason='As RX/day/scene/state aggregates below;no real waveform replay'),L_support=support,ordinary_observed=ordinary,legacy_telemetry_note='concat total_L/L_forwards/As_packets describe inherited fourforward slot;actual total uses reference_steps+shape_steps+native counters')
 PARSED[root.name]=result;return result

def independent_source_decision(clean,urban,rx):
 if set(clean)!=set(ARMS) or set(urban)!=set(ARMS) or set(rx)!=set(ARMS):raise ValueError('Six source metrics required')
 z={a:np.asarray(urban[a],float) for a in ARMS}
 if any(set(rx[a])!={'1','3','4','6','8'} or any(type(v)not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in [clean[a],*rx[a].values()]) for a in ARMS):raise ValueError('Finite originalfiveRX/fullV rates required')
 if any(v.shape!=(2,3) or not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in z.values()):raise ValueError('Fixed2draw/3scene source panel required')
 def gain(a,b,d):return bool(np.all(z[a][:,1:]-z[b][:,1:]>=d-1e-12))
 checks=dict(Ag_clean_gain=clean['B10_BG']-clean['B00']>=.002-1e-12,joint_clean_net_gain=clean['B11_BG']-clean['B00']>=.002-1e-12,joint_clean_over_As=clean['B11_BG']-clean['B01']>=.002-1e-12,As_independent=gain('B01','B00',.03),As_with_Ag=gain('B11_BG','B10_BG',.03),Ag_urban_joint=gain('B11_BG','B01',.01),shape_over_WHITE=gain('B11_BG','B11_WHITE',.005),shape_clean_preserved=clean['B11_BG']-clean['B11_WHITE']>=-.001-1e-12,all_RX_clean_preserved=all(rx[a][r]-rx[b][r]>=-.0025-1e-12 for a,b in [('B10_BG','B00'),('B11_BG','B00'),('B11_BG','B01')] for r in ('1','3','4','6','8')),suburban_source_preserved=bool(np.all(z['B11_BG'][:,0]-z['B01'][:,0]>=-.0025-1e-12)))
 return dict(status='SOURCE_DECIDED',source_qualified=all(checks.values()),checks=checks,clean_V=clean,clean_RX=rx,urban_delta_pp={a+'-'+b:(100*(z[a][:,1:]-z[b][:,1:])).tolist() for a,b in [('B10_BG','B00'),('B10_WHITE','B00'),('B11_BG','B01'),('B11_WHITE','B01'),('B11_BG','B11_WHITE')]},source_interaction_pp=(100*(z['B11_BG'][:,1:]-z['B10_BG'][:,1:]-z['B01'][:,1:]+z['B00'][:,1:])).tolist(),source_only=True,formal_scores_read=False,automatic_seed_extension=False,Ag_proved=False,goal_complete=False)

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
 y=data['B00']['TX'];rx=data['B00']['RX'];days0=data['B00']['day'];cohort=data['B00']['clean'].argmax(1)==y
 for a,x in data.items():
  if not all(np.array_equal(x[k],data['B00'][k]) for k in ['ids','TX','RX','day']):raise ValueError('Original source V roles/panel differ')
  z=x['clean'];p=z.argmax(1);old=data['B00']['clean'].argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
  clean_masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(r),rx==r) for r in (1,3,4,6,8)]+[('day',str(d),days0==d) for d in (1,2,3)]+[('TX',str(t),y==t) for t in range(6)]
  for dim,label,mask in clean_masks:
   den=int((mask&cohort).sum());rows.append(dict(arm=a,scene='clean',draw=-1,dimension=dim,stratum=label,**base.metric(y[mask],p[mask]),mean_margin_over30=float(margin[mask].mean()),fixed_B00_clean_correct_count=den,retention_on_B00_clean_correct=float((p==y)[mask&cohort].mean()) if den else None,repairs_vs_B00=int(((p==y)&(old!=y)&mask).sum()),breaks_vs_B00=int(((p!=y)&(old==y)&mask).sum())))
  for si,scene in enumerate(SCENES):
   for draw in [-1,0,1]:
    ds=[0,1] if draw==-1 else [draw];yy=np.tile(y,len(ds));rr=np.tile(rx,len(ds));dd=np.tile(days0,len(ds));cc=np.tile(cohort,len(ds));z=np.concatenate([x['satellite'][d,si] for d in ds]);pred=z.argmax(1);old=np.concatenate([data['B00']['satellite'][d,si].argmax(1) for d in ds]);state=np.concatenate([physics['state'][d,si,0] for d in ds]);lock=np.concatenate([physics['lock'][d,si,0] for d in ds]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds]);other=z.copy();other[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-other.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in [1,3,4,6,8]]+[('day',str(d),dd==d) for d in [1,2,3]]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==i) for i,s in enumerate(['LOS','PARTIAL','BLOCKED'])]+[('lock',s,lock==i) for i,s in enumerate(['locked','degraded','unlocked'])]+[('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dimension,stratum,mask in masks:
     if not mask.any():continue
     denominator=int((mask&cc).sum());m=base.metric(yy[mask],pred[mask]);rows.append(dict(arm=a,scene=scene,draw=draw,dimension=dimension,stratum=stratum,**m,mean_margin_over30=float(margin[mask].mean()),fixed_B00_clean_correct_count=denominator,retention_on_B00_clean_correct=float((pred==yy)[mask&cc].mean()) if denominator else None,repairs_vs_B00=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_B00=int(((pred!=yy)&(old==yy)&mask).sum())))
 summary=dict(status='SOURCE_SIX_ROW_COMPLETE',source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},rows=rows,source_qualification=q,model_seed_count=1,source_unique_V=1800,channel_draws=2,cohort='fixed matched B00 FINAL30 control clean-correct;not shared E20 father',added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False,NN=0,training_updates=0,target_access=False)
 write(out/'source_summary.json',summary);write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);return summary

def analyze_source(out):
 if not all(p.is_file() for p in required_sources()):raise ValueError('Source incomplete;target stays closed')
 decision=read(ROOT/'source_matrix_decision.json');clean={};urban={};rx={};source={}
 for a in ARMS:
  b=ROOT/(a+'-s2026092701');done=completed_training(b);source[a]=parse_source(b);clean[a]=done['final_source_metrics']['source_val_accuracy'];rx[a]=done['final_source_metrics']['source_val_rx_accuracy'];urban[a]=read(b/'source_qualification/complete.json')['source_accuracies']
 independent=independent_source_decision(clean,urban,rx)
 if independent['checks']!=decision['checks'] or independent['source_qualified']!=decision['source_qualified'] or any(not np.allclose(v,decision['urban_delta_pp'][k],atol=1e-12,rtol=0) for k,v in independent['urban_delta_pp'].items()):raise ValueError('Independent prospective source decision differs')
 write(out/'independent_source_decision.json',independent);summary=source_readout(out)
 ordinary={};seq={}
 for a in ARMS:
  write(out/(a+'_source.json'),source[a])
  if a=='B00':continue
  ordinary[a]=dict(records=len(source[a]['ordinary_observed']),mismatch=sum(x!=y for x,y in zip_longest(source['B00']['ordinary_observed'],source[a]['ordinary_observed'])))
  seq[a]={}
  for stage in ('stage20','source'):
   pa=ROOT/(a+'-s2026092701')/stage/'training_sequence.jsonl.gz';pb=ROOT/'B00-s2026092701'/stage/'training_sequence.jsonl.gz';n=diff=0
   for x,y in zip_longest(lines(pa),lines(pb)):n+=1;diff+=x!=y
   seq[a][stage]=dict(records=n,mismatch=diff)
 curves=[q for a in ARMS for q in source[a]['curve']];write_csv(out/'training_curves.csv',curves);figures=source_plots(out,source,summary)
 result=dict(status='SOURCE_ANALYZED',source_decision=independent,source_cost={a:source[a]['cost'] for a in ARMS},source_full_log_recount={a:source[a]['full_records'] for a in ARMS},ordinary_sequence_comparison=ordinary,physical_sequence_comparison=seq,source_confusions=len(summary['rows']),new_NN=0,IQ_reads=0,checkpoint_reads=0,formal_scores_read=False,figures=figures,goal_complete=False)
 write(out/'complete.json',result);return result

def source_plots(out,source,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 files=[]
 def save(fig,name):
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=190);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
 for col,names in enumerate((ARMS[:3],ARMS[3:])):
  for a in names:
   c=source[a]['curve'];epoch=[q['epoch'] for q in c];color=base.COLORS[a]
   axes[0,col].plot(epoch,[q['full/RDrop_CE'] for q in c],label=a,color=color);axes[1,col].plot(epoch,[q['source_val_accuracy_pct'] for q in c],label=a,color=color);axes[2,col].plot(epoch,[q['shape_CE'] for q in c],label=a,color=color)
  for row,title,ylabel in [(0,'All L slots weighted CE','L CE / total coefficient1.25'),(1,'Full original source V health','Source V accuracy (%)'),(2,'Matched Ag or raw control slot','Unweighted slot CE')]:
   ax=axes[row,col];ax.set(title=title+(' / As off' if col==0 else ' / As on'),xlabel='Cumulative epoch',ylabel=ylabel,xlim=(1,30));ax.axvline(13.5,color='gray',ls=':');ax.axvline(20.5,color='gray',ls='--');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('All180 trained epochs; source V is not unseen-RX target performance');save(fig,'training_curves')
 fig,axes=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
 for a in ARMS:
  curve=source[a]['curve'];epoch=[r['epoch'] for r in curve];style='--' if satellite(a) else '-';color=base.COLORS[a]
  for ax,key in [(axes[0,0],'relative_noise_RMS'),(axes[0,1],'noise_peak_fraction_mean'),(axes[1,0],'grad_backbone'),(axes[1,1],'lr')]:ax.plot(epoch,[r[key] for r in curve],style,label=a,color=color)
 for ax,title,ylabel in zip(axes.flat,['Actual relative added-noise RMS','Noise temporal concentration','Original backbone gradient norm','Actual LR / reset at20'],['Noise RMS / raw RMS','Mean max-sample power / noise energy','Gradient norm','Learning rate']):ax.set(title=title,xlabel='Cumulative epoch',ylabel=ylabel);ax.axvline(20.5,color='gray',ls='--');ax.grid(alpha=.18)
 axes[0,0].legend(frameon=False,ncol=2,fontsize=7);fig.suptitle('WHITE and BG equal added-noise energy; different peak distribution; no input renormalization');save(fig,'noise_shape_and_gradient_curves')
 rows=summary['rows'];bins=['<3','3-8','8-15','>=15'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for i,scene in enumerate(SCENES[1:]):
  for j,names in enumerate((ARMS[:3],ARMS[3:])):
   ax=axes[i,j]
   for a in names:
    vals=[next((q['accuracy']*100 for q in rows if (q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==(a,scene,-1,'added_noise_ratio',label)),np.nan) for label in bins];ax.plot(range(4),vals,'o-',label=a,color=base.COLORS[a])
   ns=[next((q['count'] for q in rows if (q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==('B00',scene,-1,'added_noise_ratio',label)),0) for label in bins]
   ax.set(xticks=range(4),xticklabels=[label+' dB\nn='+str(n) for label,n in zip(bins,ns)],title=scene.replace('practical_','')+(' / As off' if j==0 else ' / As on'),ylabel='Source accuracy (%)',ylim=(0,100),xlabel='Signal / added noise ratio, not total SNR');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Same source V1800 / two draws; fixed full residual observations');save(fig,'source_added_noise_accuracy')
 cells=['LOS/locked','PARTIAL/locked','PARTIAL/degraded','BLOCKED/degraded','BLOCKED/unlocked'];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
 for i,scene in enumerate(SCENES[1:]):
  for j,names in enumerate((ARMS[:3],ARMS[3:])):
   ax=axes[i,j]
   for a in names:
    vals=[next((q['accuracy']*100 for q in rows if (q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==(a,scene,-1,'state_lock',s)),np.nan) for s in cells];ax.plot(range(len(cells)),vals,'o-',label=a,color=base.COLORS[a])
   ns=[next((q['count'] for q in rows if (q['arm'],q['scene'],q['draw'],q['dimension'],q['stratum'])==('B00',scene,-1,'state_lock',s)),0) for s in cells];ax.set(xticks=range(5),xticklabels=[s.replace('/','\n')+'\nn='+str(n) for s,n in zip(cells,ns)],title=scene.replace('practical_','')+(' / As off' if j==0 else ' / As on'),ylabel='Source accuracy (%)',ylim=(0,100));ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Descriptive source state/lock accuracy; not formal target state attribution');save(fig,'source_state_lock_accuracy')
 fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
 for a,ax in zip(ARMS[3:],axes):
  strata=[r for segment in source[a]['L_support'] for r in segment.get('strata',[])];matrix=np.asarray([[sum(r['count'] for r in strata if r['RX']==rx and r['scene']==scene) for scene in SCENES] for rx in (1,3,4,6,8)])
  if int(matrix.sum())!=426240:raise ValueError('Actual As support sum differs')
  im=ax.imshow(matrix,cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=['Low sub','Mid urban','Low urban'],yticks=range(5),yticklabels=['RX'+str(r) for r in (1,3,4,6,8)],title=a)
  for i in range(5):
   for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if im.norm(matrix[i,j])>.65 else 'black')
  fig.colorbar(im,ax=ax,fraction=.04,label='Actual labeled full As samples')
 fig.suptitle('All5 source RX; three As-off controls have exactly zero channel generation');save(fig,'training_As_support')
 return files

def effect_table(results):
 rows=[]
 for v in base.VIEWS:
  for stat in ('accuracy','macro_f1','worst_RX_accuracy'):
   x={a:float(results[a]['views'][v][stat]) for a in ARMS}
   for shape,a10,a11 in [('WHITE','B10_WHITE','B11_WHITE'),('BG','B10_BG','B11_BG')]:
    rows.append(dict(view=v,metric=stat,shape=shape,Ag_independent_pp=100*(x[a10]-x['B00']),As_independent_pp=100*(x['B01']-x['B00']),joint_net_pp=100*(x[a11]-x['B00']),As_given_Ag_pp=100*(x[a11]-x[a10]),Ag_given_As_pp=100*(x[a11]-x['B01']),interaction_pp=100*(x[a11]-x[a10]-x['B01']+x['B00']),BG_minus_WHITE_Ag_pp=100*(x['B10_BG']-x['B10_WHITE']),BG_minus_WHITE_joint_pp=100*(x['B11_BG']-x['B11_WHITE'])))
 return rows

def formal_plots(out,results,effects):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['Clean','High sub','Mid sub','Low sub','High urban','Mid urban','Low urban'];files=[]
 def save(fig,name):
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=190);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained');xx=np.arange(7)
 for ax,names,title in zip(axes[:2],(ARMS[:3],ARMS[3:]),('As off','Full As on')):
  for j,a in enumerate(names):
   vals=[100*results[a]['views'][v]['accuracy'] for v in base.VIEWS];bars=ax.bar(xx+(j-1)*.25,vals,.24,label=a,color=base.COLORS[a]);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
  ax.set(xticks=xx,xticklabels=labels,ylabel='Accuracy (%)',ylim=(0,105),title=title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,ncol=3)
 for shape,color in [('WHITE','#2563eb'),('BG','#d97706')]:
  e=[next(r for r in effects if (r['shape'],r['metric'],r['view'])==(shape,'accuracy',v)) for v in base.VIEWS];axes[2].plot(xx,[r['Ag_given_As_pp'] for r in e],'o-',label=shape+' added to As',color=color);axes[2].plot(xx,[r['Ag_independent_pp'] for r in e],'o--',label=shape+' independent',color=color)
 axes[2].axhline(0,color='gray',lw=1);axes[2].set(xticks=xx,xticklabels=labels,ylabel='Accuracy change (percentage points)',title='Ag effects under both fixed As conditions');axes[2].grid(alpha=.18);axes[2].legend(frameon=False,ncol=2);fig.suptitle('Fixed default head / same168000 packets per view / one model seed / E30');save(fig,'formal_factorial_accuracy')
 fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
 for ax,a in zip(axes.flat,ARMS):
  rows=results[a]['metrics'];rx=sorted({r['stratum'] for r in rows if r['dimension']=='receiver'});matrix=np.asarray([[100*next(r['accuracy'] for r in rows if (r['view'],r['dimension'],r['stratum'])==(v,'receiver',x)) for v in base.VIEWS] for x in rx]);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set(xticks=range(7),xticklabels=labels,yticks=range(len(rx)),yticklabels=['RX'+x for x in rx],title=a);ax.tick_params(axis='x',rotation=50,labelsize=8)
  for i in range(len(rx)):
   for j in range(7):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>65 else 'black')
 fig.colorbar(im,ax=list(axes.flat),label='Accuracy (%)',shrink=.7);fig.suptitle('All seven fixed target RX; state/lock labels are unavailable');save(fig,'formal_all_RX_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for j,(shape,color) in enumerate([('WHITE','#2563eb'),('BG','#d97706')]):
  e=[next(r for r in effects if (r['shape'],r['metric'],r['view'])==(shape,'accuracy',v)) for v in base.VIEWS];axes[0].plot(xx,[r['interaction_pp'] for r in e],'o-',color=color,label=shape)
 for a in ARMS:
  clean=results[a]['views']['clean']['accuracy'];gaps=[100*(clean-results[a]['views'][v]['accuracy']) for v in base.VIEWS[1:4]];axes[1].plot(range(3),gaps,'o--' if satellite(a) else 'o-',color=base.COLORS[a],label=a)
 axes[0].axhline(0,color='gray',lw=1);axes[0].set(xticks=xx,xticklabels=labels,ylabel='Interaction (percentage points)',title='B11 - B10 - B01 + B00');axes[0].tick_params(axis='x',rotation=45);axes[0].legend(frameon=False)
 axes[1].axhline(2,color='gray',ls=':',label='2pp exploratory goal');axes[1].set(xticks=range(3),xticklabels=['High sub','Mid sub','Low sub'],ylabel='Clean minus suburban (percentage points)',title='Gap must be read with absolute accuracy');axes[1].legend(frameon=False,fontsize=8,ncol=2)
 for ax in axes:ax.grid(alpha=.18)
 fig.suptitle('Terminal reporting only; no target feedback or model selection');save(fig,'factorial_interaction_and_suburban_gap');return files

def final_analysis(out,source_dir,truth):
 source_done=read(source_dir/'complete.json');independent=read(source_dir/'independent_source_decision.json')
 if source_done['status']!='SOURCE_ANALYZED' or source_done['formal_scores_read'] or independent['checks']!=read(ROOT/'source_matrix_decision.json')['checks']:raise ValueError('Fixed independent source decision first')
 for a in ARMS:
  completed(ROOT/(a+'-s2026092701'));PARSED[a+'-s2026092701']=read(source_dir/(a+'_source.json'))
 base.completed=completed;base.parse_source=parse_source;results={};data={}
 for a in ARMS:results[a],data[a]=base.collect_row(a,truth)
 if any(not np.array_equal(data[a]['ids'],data['B00']['ids']) or not np.array_equal(data[a]['y'],data['B00']['y']) for a in ARMS):raise ValueError('Same physical target IDs/truth required')
 y=data['B00']['y'];cohort=data['B00']['pred']['clean']==y;pairs=[];retention=[]
 for new,old in PAIRS:
  common=(data[new]['pred']['clean']==y)&(data[old]['pred']['clean']==y)
  for view in base.VIEWS:
   n=data[new]['pred'][view]==y;o=data[old]['pred'][view]==y;rep=int((n&~o).sum());br=int((~n&o).sum());pairs.append(dict(new=new,old=old,view=view,repairs=rep,new_errors=br,net=rep-br,accuracy_delta_pp=100*(rep-br)/len(y),common_clean_count=int(common.sum()),common_clean_repairs=int((n&~o&common).sum()),common_clean_new_errors=int((~n&o&common).sum())))
 for a in ARMS:
  for view in base.VIEWS:retention.append(dict(arm=a,view=view,fixed_B00_clean_correct_count=int(cohort.sum()),retained=int(((data[a]['pred'][view]==y)&cohort).sum()),retention=float((data[a]['pred'][view][cohort]==y[cohort]).mean())))
 effects=effect_table(results);figures=formal_plots(out,results,effects);smoke=read(ROOT/'real_checkpoint_smoke.json');synthetic=read(ROOT/'synthetic_smoke.json')
 if smoke['status']!='PASS' or synthetic['status']!='PASS' or smoke['NN']!=2 or synthetic['NN']!=4608 or smoke['optimizer_updates'] or synthetic['optimizer_updates']:raise ValueError('Actual shared smoke evidence required')
 costs={a:dict(source_identity_NN=results[a]['source']['cost']['source_identity_NN'],prediction_identity_NN=results[a]['source']['cost']['prediction_identity_NN'],total_identity_NN=results[a]['source']['cost']['source_identity_NN']+results[a]['source']['cost']['prediction_identity_NN'],new_epochs=30,new_updates=6660) for a in ARMS}
 gaps={a:{v:100*(results[a]['views']['clean']['accuracy']-results[a]['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for a in ARMS};goals=dict(joint_clean_positive_over_B00=results['B11_BG']['views']['clean']['accuracy']>results['B00']['views']['clean']['accuracy'],low_urban_above60=results['B11_BG']['views']['practical_low_urban']['accuracy']>.60,each_suburban_gap_at_most2pp=all(v<=2+1e-12 for v in gaps['B11_BG'].values()),crossseed_confirmed=False)
 summary=dict(status='ANALYZED',run_id=RUN,model_seed=2026092701,model_seed_count=1,rows=results,source_decision=independent,effects=effects,pairs=pairs,fixed_B00_clean_correct_retention=retention,urban_means={a:float(np.mean([results[a]['views'][v]['accuracy'] for v in base.VIEWS[-3:]])) for a in ARMS},suburban_gaps_pp=gaps,terminal_exploratory_goals=goals,identity_cost=dict(per_model=costs,shared_smoke_NN=4610,matrix_total_identity_NN=sum(c['total_identity_NN'] for c in costs.values())+4610,new_model_epochs=180,new_updates=39960,each_own20_is_included_in30=True,analysis_NN=0),formal_confusion_matrices=sum(len(r['metrics']) for r in results.values()),source_confusion_matrices=source_done['source_confusions'],analysis_NN=0,IQ_reads=0,checkpoint_reads=0,U_TX_truth_read=False,default_head_structure_changed=False,additional_deployment_parameters=0,target_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False,interpretation_limits=['One modelseed only,no crossseed proof','B00 fixed FINAL30 cohort is a matched control,not a common pretrained parent','Source states/quality are not formal target state labels or controlled causal removal effects','No perpacket branch/gate trajectories saved;U selection counts are not precision','concat step telemetry is the inherited fourforward branch;use shape/reference/counters for actual6forward total','SixL costs50%moreLNN than4Lhistory;only the sixcurrentrows match total budget'])
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in q.items() if k!='confusion'} for r in results.values() for q in r['metrics']]);write_csv(out/'factorial_effects.csv',effects);write_csv(out/'paired_repairs.csv',pairs);write_csv(out/'fixed_B00_retention.csv',retention);write(out/'figures.json',figures);write(out/'analysis_summary.json',summary);write(out/'complete.json',dict(status='ANALYZED',rows=list(ARMS),model_seed_count=1,formal_confusions=summary['formal_confusion_matrices'],source_confusions=summary['source_confusion_matrices'],figures=figures,analysis_NN=0,goal_complete=False));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path);p.add_argument('--truth',type=Path);p.add_argument('--wait-pids',nargs='*',type=int);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve immutable existing analysis')
 if a.mode=='source':
  if a.wait_pids:wait_source(a.wait_pids)
  if not all(p.is_file() for p in required_sources()):raise ValueError('Source artifacts incomplete;no output created')
 elif a.source_analysis is None or a.truth is None:raise ValueError('Explicit completedsource analysis and independenttruth-side path required')
 a.output.mkdir();tic=time.perf_counter();r=analyze_source(a.output) if a.mode=='source' else final_analysis(a.output,a.source_analysis,a.truth)
 write(a.output/'execution.json',dict(mode=a.mode,wall_seconds=time.perf_counter()-tic,NN=0,IQ_reads=0,checkpoint_reads=0,target_access=a.mode=='final',waiting_excluded_from_wall_seconds=True));print(json.dumps(dict(status=r['status'],mode=a.mode,output=str(a.output),NN=0)),flush=True)
if __name__=='__main__':main()
