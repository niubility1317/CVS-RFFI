"""Independent single-seed trajectory x As analysis;no model/IQ imports or fits."""
import argparse,csv,gzip,json,math,re,time
from collections import Counter,defaultdict
from itertools import chain,zip_longest
from pathlib import Path
import numpy as np
import cvs_impulse_factorial_E30_results_20261018 as audit
base=audit.base
PROJECT=base.PROJECT
RUN='20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
ROOT=PROJECT/'runs'/RUN
CACHED_SOURCE=PROJECT/'runs'/'20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02'/'analysis_auto_r01/source'
SEED=2026092701
VIEWS=base.VIEWS
SCENES=audit.SCENES
PANEL=audit.PANEL
PARSED={}
lines,numeric,raw_telemetry,relation,source_cost_breakdown=audit.lines,audit.numeric,audit.raw_telemetry,audit.relation,audit.source_cost_breakdown
mode,satellite=audit.mode,audit.satellite
read,write=base.read,base.write

def rows():
 return [dict(key=style+'_'+arm,style=style,arm=arm,seed=SEED,reused=style=='STAGED',run=REF if style=='STAGED' else RUN,root=PROJECT/'runs'/(REF if style=='STAGED' else RUN)/(arm+'-s'+str(SEED))) for style in ('STAGED','CONT') for arm in ('B00','B01')]

def csv_write(p,records):
 if not records:raise ValueError('No empty invented table')
 fields=list(dict.fromkeys(k for r in records for k in r))
 with Path(p).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def completed_training(root):
 p=root/'source';done=read(p/'completion.json');c=done['config'];init=read(p/'initialization.json');frozen=read(p/'source_frozen.json');r=read(p/'result.json');contract=read(PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
 if (done['epoch'],done['optimizer_steps'],c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates'])!=(30,6660,30,13,17,6660) or c['run_id']!=RUN or c['row_id']!=root.name or c['training_mode']!='single_continuous_E30' or c['epoch_offset'] or c['step_offset'] or c['parent_source'] or c['checkpoint_sources'] or c['new_Ag'] or c['Ag_mode']!='NONE':raise ValueError('Own single30 identity/budget differs')
 if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or read(p/'source_contract.json')['role_ids']!=contract['role_ids']:raise ValueError('Own fullsource scratch provenance differs')
 if done['target_access'] or done['head_fit_updates'] or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or r['config']!=c or r['head_fit_updates'] or r['U_truth_read'] or r['target_access']:raise ValueError('Own fixed defaulthead/source freeze required')
 args=read(p/'actual_native_args.json');binding=read(p/'seed_binding.json')
 if (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['seed'])!=(30,13,17,SEED) or args['baseline_ckpt'] or args['teacher_ckpt'] or binding['actual_F_factory_seed']!=SEED:raise ValueError('Actual native calendar/factory seed differs')
 return done

def source_complete(row):
 if row['reused']:
  old=audit.RUN;audit.RUN=REF
  try:done=audit.completed_training(row['root'])
  finally:audit.RUN=old
 else:done=completed_training(row['root'])
 c=done['config'];q=read(row['root']/'source_qualification/complete.json')
 if c['model_seed']!=SEED or c['arm']!=row['arm'] or c['new_Ag'] or c['Ag_mode']!='NONE' or q['config']!=c or q['status']!='SOURCE_READOUT_COMPLETE' or q['NN']!=12600 or any(q[k] for k in ['training_updates','head_fit_updates','target_access','formal_scores_read']):raise ValueError('Registered row/source-only panel differs')
 return done,q

def formal_complete(row):
 done,_=source_complete(row);root=row['root'];p=read(root/'prediction/complete.json');s=read(root/'scoring/complete.json')
 if p['config']!=done['config'] or p['row_id']!=root.name or s['row_id']!=root.name or p['status']!='PREDICTIONS_COMPLETE' or p['count']!=168000 or p['views']!=list(VIEWS) or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or s['status']!='SCORED_COMPLETE' or s['views']!=list(VIEWS) or not s['truth_last']:raise ValueError('Truth closed:all7 frozen defaulthead predictions required')
 return p,s,done

def parse_continuous(root,done=None):
 if str(root) in PARSED:return PARSED[str(root)]
 completed_training(root);arm=root.name.split('-s')[0];curves=[];cost=Counter();counts=Counter();totals=Counter();unique=defaultdict(set);ordinary=[];support=[];rawlog={};walls=0.;maxerr=0.;native_L_error=0.;noise_summary=[]
 for stage,e,l,offset in [('source',30,13,0)]:
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
 result=dict(curve=curves,cost=dict(**source_cost_breakdown(cost,read(root/'source_qualification/complete.json')),source_wall_seconds=walls,L_identity_NN=5114880,extra_Ag_control_identity_NN=1704960,extra_U_NN=0,new_epochs=30,new_updates=6660,training_parent_epochs=0,own_stage20_is_included=False),full_records=dict(counts,loader_packets={k:totals[k] for k in ('L','U')},unique_physical={k:len(v) for k,v in unique.items()},original_U_selected=totals['selected_U'],U_truth_hidden=True,stdout_lines=len(text.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',text,re.I)),native_full_epoch_telemetry=rawlog,warnings=[dict(line=i,text=line[:250]) for i,line in enumerate(text.splitlines(),1) if re.search(r'Warning|WARN|warning:',line)],maximum_L_decomposition_error=maxerr,maximum_native_supervised_loss_error=native_L_error),source_result=dict(segments=[read(root/s/'result.json') for s in ('source',)]),physical_metadata=dict(not_applicable=True,reason='As RX/day/scene/state aggregates below;no real waveform replay'),L_support=support,ordinary_observed=ordinary,legacy_telemetry_note='concat total_L/L_forwards/As_packets describe inherited fourforward slot;actual total uses reference_steps+shape_steps+native counters')
 lr=np.asarray([q['lr'] for q in curves]);
 if (np.diff(lr)>1e-15).any() or abs(lr[0]-.0002)>1e-15 or abs(lr[-1]-.000001)>1e-15:raise ValueError('Actual continuous cosine30 restarted or changed endpoints')
 result['cost']['source_wall_scope']='single native30 segment;external readout time not recorded';result['training_mode']='single_continuous_E30'
 PARSED[str(root)]=result;return result

def parse_training(row):
 if not row['reused']:r=parse_continuous(row['root'])
 else:
  proof=read(CACHED_SOURCE/'complete.json')
  if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read'] or proof['model_seed_count']!=3:raise ValueError('Prior complete source-only analysis required')
  r=read(CACHED_SOURCE/(row['root'].name+'_training.json'))
  if [q['epoch'] for q in r['curve']]!=list(range(1,31)) or r['full_records']['actual_steps']!=6660 or r['cost']['new_epochs']!=30:raise ValueError('Prior complete staged logs required')
  segments=r['source_result']['segments']
  if len(segments)!=2 or any(s['config']['run_id']!=REF or s['config']['row_id']!=row['root'].name or s['config']['model_seed']!=SEED or s['config']['new_Ag'] for s in segments):raise ValueError('Cached staged analysis row identity differs')
  r['training_mode']='staged20plus10'
 for q in r['curve']:q.update(row_id=row['key'],style=row['style'],model_seed=SEED,reused=row['reused'])
 return r

def sequence_compare(left,right):
 def stream(row):
  return chain.from_iterable(lines(row['root']/s/'training_sequence.jsonl.gz') for s in (('stage20','source') if row['reused'] else ('source',)))
 count=bad=0;first=None
 for i,(a,b) in enumerate(zip_longest(stream(left),stream(right)),1):
  count+=1
  if a!=b:bad+=1;first=first or i
 return dict(records=count,mismatched_records=bad,first_difference=first,identical=bad==0,U_labels_decoded=False,no_hash_gate=True)

def qualify(records):
 n=records['CONT_B01'];c=records['CONT_B00'];o=records['STAGED_B01'];v=n['V'];ov=o['V'];g=np.asarray(n['panel'])[:,1:]-np.asarray(o['panel'])[:,1:]
 if g.shape!=(2,2) or not np.isfinite(g).all() or not all(math.isfinite(x['V']['source_val_accuracy']) and all(math.isfinite(z) for z in x['V']['source_val_rx_accuracy'].values()) for x in records.values()):raise ValueError('Complete finite source rates required')
 checks=dict(clean_not_below_matched_NoAs=v['source_val_accuracy']>=c['V']['source_val_accuracy'],clean_vs_staged_As_within_quarter_pp=v['source_val_accuracy']>=ov['source_val_accuracy']-.0025,each_RX_vs_staged_As_within_half_pp=all(v['source_val_rx_accuracy'][str(r)]>=ov['source_val_rx_accuracy'][str(r)]-.005 for r in (1,3,4,6,8)),both_draw_mid_low_vs_staged_As_gain_one_pp=bool((g>=.01).all()))
 return dict(status='SOURCE_CRITERIA_PASS' if all(checks.values()) else 'SOURCE_CRITERIA_FAIL_FIXED_CONTINUOUS_ROUTE_CLOSED',criteria=checks,source_qualified=all(checks.values()),mid_low_gains=g.tolist(),formal_scores_read=False,model_seed_count=1,new_Ag=False,goal_complete=False)

def source_analysis(out):
 if out.exists():raise FileExistsError('Reconcile existing source analysis')
 for row in rows():source_complete(row)
 out.mkdir();started=time.perf_counter();records={};training={};curves=[];matrices=[];allx={}
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in ['state','lock','signal_to_added_noise_db']}
 role=read(PANEL/'data_roles.json')
 for row in rows():
  done,q=source_complete(row);r=parse_training(row);training[row['key']]=r;curves.extend(r['curve']);records[row['key']]=dict(V=done['final_source_metrics'],panel=q['source_accuracies'])
  with np.load(row['root']/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in f.files}
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.array_equal(x['ids'],role['physical_ids']) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all() or not np.allclose((x['satellite'].argmax(-1)==x['TX']).mean(-1),q['source_accuracies'],atol=1e-12,rtol=0):raise ValueError('Fixed source panel/argmax differs')
  allx[row['key']]=x
 reference=allx['STAGED_B00']
 for row in rows():
  x=allx[row['key']]
  if any(not np.array_equal(x[k],reference[k]) for k in ['ids','TX','RX','day']):raise ValueError('All4 source physical IDs/labels must align')
  cohort=reference['clean'].argmax(1)==x['TX'];ys=x['TX'];observations=[('clean',-1,x['clean'],ys,x['RX'],x['day'],cohort,None,None,None)]
  for si,scene in enumerate(SCENES):
   for draw in [-1,0,1]:
    ds=[0,1] if draw==-1 else [draw];n=len(ds)
    observations.append((scene,draw,np.concatenate([x['satellite'][j,si] for j in ds]),np.tile(ys,n),np.tile(x['RX'],n),np.tile(x['day'],n),np.tile(cohort,n),np.concatenate([physics['state'][j,si,0] for j in ds]),np.concatenate([physics['lock'][j,si,0] for j in ds]),np.concatenate([physics['signal_to_added_noise_db'][j,si,0] for j in ds])))
  for scene,draw,z,y,rx,day,correct0,state,lock,quality in observations:
   p=z.argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
   masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(r),rx==r) for r in [1,3,4,6,8]]+[('day',str(d),day==d) for d in [1,2,3]]+[('TX',str(t),y==t) for t in range(6)]
   if state is not None:
    masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]
    masks += [('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
   for dim,label,mask in masks:
    if not mask.any():continue
    den=int((mask&correct0).sum());m=base.metric(y[mask],p[mask]);matrices.append(dict(row_id=row['key'],style=row['style'],arm=row['arm'],model_seed=SEED,scene=scene,draw=draw,dimension=dim,stratum=label,**m,mean_margin_over30=float(margin[mask].mean()),fixed_STAGED_B00_clean_correct_count=den,satellite_retention=float((p==y)[mask&correct0].mean()) if den else None))
 decision=qualify(records);online=read(ROOT/'source_matrix_decision.json')
 if decision['criteria']!=online['criteria'] or decision['source_qualified']!=online['source_qualified'] or not np.allclose(decision['mid_low_gains'],online['mid_low_gains'],atol=1e-12,rtol=0):raise ValueError('Online and independent fixedsource decisions differ')
 comparisons=[]
 for left,right in [('CONT_B00','CONT_B01'),('STAGED_B00','CONT_B00'),('STAGED_B01','CONT_B01')]:
  a=next(r for r in rows() if r['key']==left);b=next(r for r in rows() if r['key']==right);q=sequence_compare(a,b);q.update(left=left,right=right);comparisons.append(q)
 csv_write(out/'training_curves.csv',curves);csv_write(out/'source_strata.csv',[{k:v for k,v in q.items() if k!='confusion'} for q in matrices]);write(out/'source_confusions.json',matrices);write(out/'training_stream_comparability.json',comparisons)
 for key,r in training.items():write(out/(key+'_training.json'),r)
 write(out/'source_decision.json',decision);write(out/'source_summary.json',dict(source_unique_V=1800,model_seed_count=1,channel_draws=2,source_confusions=len(matrices),records=records,decision=decision,NN=0,IQ_reads=0,checkpoint_reads=0,added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False,wall_seconds=time.perf_counter()-started));plot_source(out,matrices)
 write(out/'complete.json',dict(status='SOURCE_ANALYZED',model_seed_count=1,rows=4,new_rows=2,reused_rows=2,epoch_records=len(curves),source_confusions=len(matrices),formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0,goal_complete=False))

def plot_source(out,records):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['LOS/locked','PARTIAL/locked','PARTIAL/degraded','BLOCKED/degraded','BLOCKED/unlocked'];fig,ax=plt.subplots(figsize=(13,5));xx=np.arange(len(labels))
 for i,row in enumerate(rows()):
  vals=[]
  for label in labels:
   found=[r for r in records if (r['row_id'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(row['key'],'practical_low_urban',-1,'state_lock',label)];vals.append(100*found[0]['accuracy'] if found else np.nan)
  ax.bar(xx+(i-1.5)*.2,vals,.2,label=row['key'])
 ax.set(xticks=xx,xticklabels=labels,ylim=(0,100),ylabel='Source V accuracy (%)',title='One model seed / pooled two fixed source draws / no formal target states');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('source_lowurban_states.'+ext),dpi=170)
 plt.close(fig)

def clean_cohort_effect(y,clean_control,pred_new,pred_control):
 y=np.asarray(y);clean_control=np.asarray(clean_control);n=np.asarray(pred_new)==y;o=np.asarray(pred_control)==y;c=clean_control==y
 if any(x.shape!=y.shape for x in [clean_control,pred_new,pred_control]) or y.ndim!=1 or len(y)==0:raise ValueError('Paired decisions must align')
 den=int(c.sum());rep=int((n&~o).sum());br=int((~n&o).sum())
 return dict(repairs=rep,new_errors=br,net=rep-br,accuracy_delta_pp=100*(rep-br)/len(y),fixed_B00_clean_correct_count=den,B00_retained=int((o&c).sum()),B01_retained=int((n&c).sum()),B00_retention=float(o[c].mean()) if den else None,B01_retention=float(n[c].mean()) if den else None)

def checked_final_inputs(source_dir):
 proof=read(source_dir/'complete.json')
 if proof['status']!='SOURCE_ANALYZED' or proof['formal_scores_read'] or proof['model_seed_count']!=1:raise ValueError('Independent source-only decision required before formal aggregation')
 for row in rows():formal_complete(row)
 return proof

def interaction(values):
 if set(values)!=set(r['key'] for r in rows()) or not all(math.isfinite(v) for v in values.values()):raise ValueError('All4 finite trajectory x As values required')
 return (values['CONT_B01']-values['CONT_B00'])-(values['STAGED_B01']-values['STAGED_B00'])

def final_analysis(out,source,truth):
 proof=checked_final_inputs(source)
 if out.exists():raise FileExistsError('Immutable final analysis output')
 out.mkdir();started=time.perf_counter();byroot={str(r['root']):r for r in rows()};cached={r['key']:read(source/(r['key']+'_training.json')) for r in rows()};prior=base.completed,base.parse_source,base.ROWS
 base.completed=lambda root:formal_complete(byroot[str(root)]);base.parse_source=lambda root,done:cached[byroot[str(root)]['key']];base.ROWS={r['key']:(r['run'],r['root'].name) for r in rows()}
 try:
  results={};data={}
  for row in rows():
   r,x=base.collect_row(row['key'],truth);r.update(style=row['style'],arm=row['arm'],row_id=row['key'],model_seed=SEED,reused=row['reused'])
   for q in [*r['metrics'],*r['views'].values()]:q.update(row_id=row['key'],style=row['style'],arm=row['arm'],model_seed=SEED)
   results[row['key']]=r;data[row['key']]=x
 finally:base.completed,base.parse_source,base.ROWS=prior
 first=data['STAGED_B00']
 if any(not np.array_equal(q['ids'],first['ids']) or not np.array_equal(q['y'],first['y']) for q in data.values()):raise ValueError('All4 original formal physical samples/labels must align')
 metrics=[];overall=[];gaps=[];effects=[];retained=[];urban=[]
 for row in rows():
  name=row['key'];r=results[name];metrics.extend(r['metrics']);clean=r['views']['clean']['accuracy'];cohort=first['pred']['clean']==first['y']
  for v in VIEWS:
   q=r['views'][v];overall.append(dict(row_id=name,style=row['style'],arm=row['arm'],model_seed=SEED,view=v,accuracy=q['accuracy'],macro_f1=q['macro_f1'],worst_RX_accuracy=q['worst_RX_accuracy'],margin_over30=q['mean_true_margin_over30']));ok=data[name]['pred'][v]==first['y'];den=int(cohort.sum());retained.append(dict(row_id=name,view=v,denominator=den,retained=int((ok&cohort).sum()),retention=float(ok[cohort].mean()) if den else None))
  for v in VIEWS[1:4]:gaps.append(dict(row_id=name,view=v,clean_accuracy=clean,suburban_accuracy=r['views'][v]['accuracy'],gap_pp=100*(clean-r['views'][v]['accuracy'])))
  urban.append(dict(row_id=name,urban_mean=float(np.mean([r['views'][v]['accuracy'] for v in VIEWS[-3:]]))))
 for left,right in [('CONT_B01','CONT_B00'),('STAGED_B01','STAGED_B00'),('CONT_B01','STAGED_B01'),('CONT_B00','STAGED_B00')]:
  for v in VIEWS:
   q=clean_cohort_effect(first['y'],first['pred']['clean'],data[left]['pred'][v],data[right]['pred'][v]);q.update(left=left,right=right,view=v,macro_F1_delta_pp=100*(results[left]['views'][v]['macro_f1']-results[right]['views'][v]['macro_f1']),worst_RX_delta_pp=100*(results[left]['views'][v]['worst_RX_accuracy']-results[right]['views'][v]['worst_RX_accuracy']));effects.append(q)
 ints=[]
 for v in VIEWS:
  ints.append(dict(view=v,trajectory_x_As_accuracy_pp=100*interaction({k:r['views'][v]['accuracy'] for k,r in results.items()}),trajectory_x_As_macro_F1_pp=100*interaction({k:r['views'][v]['macro_f1'] for k,r in results.items()}),is_Ag_interaction=False))
 for name,rs in [('overall_by_row.csv',overall),('per_row_metrics.csv',[{k:v for k,v in q.items() if k!='confusion'} for q in metrics]),('paired_repairs.csv',effects),('fixed_STAGED_B00_retention.csv',retained),('suburban_gaps.csv',gaps),('urban_means.csv',urban),('trajectory_x_As_interaction.csv',ints)]:csv_write(out/name,rs)
 smoke=read(ROOT/'synthetic_smoke.json');cp=read(ROOT/'real_checkpoint_smoke.json')
 if smoke['NN']!=1536 or cp['NN']!=2 or not smoke['matched_initial_state_per_seed']:raise ValueError('Actual new-row no-query smoke evidence differs')
 costs={k:dict(reused=r['reused'],source=r['source']['cost']['source_identity_NN'],prediction=r['source']['cost']['prediction_identity_NN'],total=r['source']['cost']['source_identity_NN']+r['source']['cost']['prediction_identity_NN']) for k,r in results.items()}
 q=dict(status='ANALYZED',run_id=RUN,rows=results,paired_effects=effects,trajectory_x_As_interaction=ints,suburban_gaps=gaps,urban_means=urban,fixed_STAGED_B00_retention=retained,cost=costs,new_models=2,reused_models=2,model_seed_count=1,new_model_epochs=60,reused_model_epochs=60,new_updates=13320,reused_updates=13320,shared_new_smoke_NN=1538,new_total_NN=sum(c['total'] for c in costs.values() if not c['reused'])+1538,reused_total_NN=sum(c['total'] for c in costs.values() if c['reused']),historical_P43_sixrow_smoke_not_allocated=4610,source_decision=read(source/'source_decision.json'),formal_confusions=len(metrics),source_confusions=proof['source_confusions'],analysis_NN=0,IQ_reads=0,checkpoint_reads=0,Ag_proved=False,Ag_factorial_interaction_not_estimated=True,crossseed_confirmed=False,goal_complete=False,target_feedback=False,wall_seconds=time.perf_counter()-started,limits=['One modelseed;four rows are not fourseeds','Continuous trajectory changes optimizer/EMA/temporal cache/sampler restarts together;not isolated LR causality','No new IQAg;trajectory interaction is not AgxAs','Source state/lock/quality only;formal perpacket state unavailable','Fixed STAGED_B00 final30 clean-correct cohort;not a shared pretrainedparent','No new paired fullbranch/gate trajectory captured;no invented gate attribution'])
 write(out/'analysis_summary.json',q);plot_final(out,q);write(out/'complete.json',dict(status='ANALYZED',model_seed_count=1,new_models=2,reused_models=2,formal_confusions=len(metrics),source_confusions=proof['source_confusions'],NN=0,IQ_reads=0,checkpoint_reads=0,goal_complete=False));return q

def plot_final(out,result):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 results=result['rows'];names=[r['key'] for r in rows()];colors=dict(zip(names,['#94a3b8','#d97706','#475569','#2563eb']));labels=['Clean','High sub','Mid sub','Low sub','High urban','Mid urban','Low urban'];xx=np.arange(7)
 fig,ax=plt.subplots(figsize=(13,5.3))
 for i,name in enumerate(names):
  bars=ax.bar(xx+(i-1.5)*.19,[100*results[name]['views'][v]['accuracy'] for v in VIEWS],.19,label=name,color=colors[name]);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
 ax.set(xticks=xx,xticklabels=labels,ylim=(0,105),ylabel='Accuracy (%)',title='Single modelseed / fixed default head / fullsource E30');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('formal_accuracy.'+ext),dpi=180)
 plt.close(fig);fig,axes=plt.subplots(2,3,figsize=(15,8.3))
 panels=[('full/RDrop_CE','L paired CE'),('full/RDrop_KL','RDrop symmetric KL'),('source_val_accuracy_pct','Original source V accuracy (%)'),('lr','Learning rate'),('grad_total','Original mean total gradient norm'),('pseudo_fraction','Original U gate selected fraction')]
 for name in names:
  c=results[name]['source']['curve'];e=[q['epoch'] for q in c]
  for ax,(field,title) in zip(axes.flat,panels):
   val=[q['pseudo_selected']/q['pseudo_total'] if q['pseudo_total'] else np.nan for q in c] if field=='pseudo_fraction' else [q[field] for q in c];ax.plot(e,val,label=name,color=colors[name]);ax.set(title=title,xlabel='Epoch',xlim=(1,30));ax.grid(alpha=.2)
 for ax in axes.flat:ax.axvline(20.5,color='gray',ls=':',alpha=.6)
 axes[0,0].legend(fontsize=8);fig.suptitle('All30 epochs / dotted line marks historical restart only / V is source health');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('full_training_curves.'+ext),dpi=170)
 plt.close(fig);fig,ax=plt.subplots(figsize=(12,5.3))
 for i,(left,right) in enumerate([('CONT_B01','CONT_B00'),('STAGED_B01','STAGED_B00'),('CONT_B01','STAGED_B01'),('CONT_B00','STAGED_B00')]):
  vals=[next(q['accuracy_delta_pp'] for q in result['paired_effects'] if (q['left'],q['right'],q['view'])==(left,right,v)) for v in VIEWS];ax.plot(xx,vals,'.-',label=left+' minus '+right)
 ax.axhline(0,c='gray',lw=1);ax.set(xticks=xx,xticklabels=labels,ylabel='Paired accuracy difference (percentage points)',title='Same modelseed / no target-feedback tuning');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('paired_effects.'+ext),dpi=180)
 plt.close(fig);fig,ax=plt.subplots(figsize=(11,5))
 for i,name in enumerate(names):ax.bar(np.arange(3)+(i-1.5)*.19,[next(q['gap_pp'] for q in result['suburban_gaps'] if (q['row_id'],q['view'])==(name,v)) for v in VIEWS[1:4]],.19,label=name,color=colors[name])
 ax.axhline(2,c='red',ls='--',label='Exploratory 2pp gap');ax.set(xticks=range(3),xticklabels=labels[1:4],ylabel='Clean minus suburban accuracy (pp)',title='Absolute accuracies reported separately;lower clean cannot count as success');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('suburban_gaps.'+ext),dpi=180)
 plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-analysis',type=Path);p.add_argument('--truth',type=Path);a=p.parse_args()
 if a.mode=='source':
  if a.truth is not None:raise ValueError('Source mode cannot receive truth')
  source_analysis(a.output)
 else:
  if a.truth is None or a.source_analysis is None:raise ValueError('Final analysis needs complete source and truth-side path')
  final_analysis(a.output,a.source_analysis,a.truth)
if __name__=='__main__':main()
