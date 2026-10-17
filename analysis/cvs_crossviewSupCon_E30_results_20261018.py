"""Independent P41 recount. Source mode never opens target, IQ or checkpoint."""
import argparse,gzip,json,math,re,time
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
RUN='20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01';ROOT=base.PROJECT/'runs'/RUN
ARMS=('ORD128','DUP192','SAT192');SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
PAIRS=(('DUP192','ORD128'),('SAT192','ORD128'),('SAT192','DUP192'))
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
base.ROWS={a:(RUN,a+'-s2026092701') for a in ARMS};base.COLORS.update(ORD128='#64748b',DUP192='#2563eb',SAT192='#16a34a');base.COMPARISON_PAIRS=PAIRS
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

def source_cost_breakdown(counters,frozen_readout):
 keys=['student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN']
 if set(counters)!=set(keys) or any(type(counters[k]) is not int or counters[k]<0 for k in keys):raise ValueError('Actual four native forward counters required')
 if frozen_readout['status']!='SOURCE_READOUT_COMPLETE' or frozen_readout['NN']!=12600 or any(frozen_readout[k] for k in ['training_updates','head_fit_updates','target_access','formal_scores_read']):raise ValueError('Only actual frozen source readout may add12600NN')
 native=sum(counters.values())
 return dict(native_training_and_validation_identity_NN=native,frozen_source_readout_NN=frozen_readout['NN'],source_identity_NN=native+frozen_readout['NN'],source_counters=dict(counters),first_U_qualification_NN=0,extra_U_student_NN=0,frozen_source_readout_wall_seconds=None,source_wall_scope='native segments;external frozen source readout time not recorded')

def parse_source(root,done=None):
 if root.name in PARSED:return PARSED[root.name]
 done=completed_training(root);arm=root.name.split('-s')[0];curves=[];totals=Counter();unique=defaultdict(set);seq_count=0;steps_seen=0;counts=Counter();support=[];ordinary=[];cost=Counter();walls=0.;max_decomp=0.
 for stage,e,l,u,offset in [('stage20',20,13,7,0),('source',10,0,10,20)]:
  b=root/stage;result=read(b/'result.json');d=read(b/'completion.json');step_offset=offset*222;expected=e*222
  if result['optimizer_steps']!=expected or result['L_NN']!=expected*512 or result['L_As']!=expected*64 or result['extra_U_NN']!=0 or result['U_As']!=0 or result['source_qualification'] is not None or result['U_source_qualified'] is not None:raise ValueError('Actual same-forward L/U budget differs')
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
   relation=validate_relation_step(r,arm)
   for k,v in relation.items():epoch_sums[epoch]['aux_'+k]+=v;counts['aux_'+k]+=v
   refs[i]=r;counts['L_records']+=1;counts['SupCon_active_steps']+=int(weight>0)
   for q in r['SupCon_details']:counts['total_anchors']+=q['total_anchors'];counts['valid_anchors']+=q['valid_anchors']
   ordinary.append(dict(step=step_offset+i,epoch=epoch,MSE=r['ordinary_raw_MSE']))
  if len(refs)!=expected:raise ValueError('Incomplete labeled objective log')
  for i,r in enumerate(lines(b/'step_metrics.jsonl.gz'),1):
   m=r['metrics'];local_epoch=(i-1)//222+1;epoch=local_epoch+offset;ref=refs[i]
   if r['step']!=i or r['epoch']!=local_epoch or any(m.get(k) or 0 for k in ['train/skipped_nonfinite_loss','train/skipped_nonfinite_grad']):raise ValueError('Successful native updates differ')
   numeric(m['train/loss'],'train/loss')
   if local_epoch>l:
    n=int(m['train/pseudo_total']);selected=int(m['train/pseudo_selected'])
    if m.get('train/pseudo_truth_available',0)!=0 or not 0<=selected<=n<=256 or n<=0:raise ValueError('Original U hidden truth/selection differs')
    totals['native_U_forward']+=n;totals['original_U_selected']+=selected
   for k in ['common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight']:epoch_sums[epoch][k]+=ref[k]
   for key in ['train/pseudo_total','train/pseudo_selected']:epoch_sums[epoch][key]+=m.get(key) or 0
   epoch_sums[epoch]['weighted_SupCon']+=ref['SupCon']*ref['SupCon_weight']
   epoch_sums[epoch]['train/loss']+=m['train/loss'];epoch_counts[epoch]+=1;counts['native_step_records']+=1
  if i!=expected or set(epoch_counts.values())!={222}:raise ValueError('Incomplete native step log')
  native_epochs=list(lines(b/'epoch_metrics.jsonl'))
  if [r['epoch'] for r in native_epochs]!=list(range(1,e+1)):raise ValueError('Complete segment epochs required')
  for r in native_epochs:
   epoch=r['epoch']+offset;s=epoch_sums[epoch];curve=dict(epoch=epoch,segment=20 if stage=='stage20' else 30,lr=r['lr'],epoch_wall_seconds=r['epoch_time_s'],source_val_accuracy_pct=r['val_tx_acc'],**{k:s[k]/222 for k in ['common_CE','extra_CE','common_KL','extra_KL','total_L','SupCon','SupCon_weight','weighted_SupCon','train/loss','aux_total_anchors','aux_valid_anchors','aux_true_satellite_anchors','aux_cross_view_positive_pairs']},pseudo_total=s['train/pseudo_total'],pseudo_selected=s['train/pseudo_selected'],grad_backbone=r['train_grad_backbone'],grad_total=r['train_grad_total'])
   curve.update({k:r[k] for k in r if k.startswith('aug_')});curve.update({'full/RDrop_CE':.5*(s['common_CE']+s['extra_CE'])/222,'full/RDrop_KL':.5*(s['common_KL']+s['extra_KL'])/222,'full/clean_CE':s['common_CE']/222,'full/second_CE':s['extra_CE']/222});curves.append(curve)
  for i,r in enumerate(lines(b/'unit_steps.jsonl.gz'),1):
   if r['step']!=i or r['cumulative_step']!=step_offset+i or r['physical_clip_calls']!=i:raise ValueError('Sinc physical update count differs')
   counts['Sinc_updates']+=1
  if i!=expected:raise ValueError('Sinc update log incomplete')
  for r in lines(b/'training_sequence.jsonl.gz'):
   role=r['role']
   if role not in ['L','U']:raise ValueError('Only original L/U sequence roles allowed')
   totals[role]+=len(r['ids']);unique[role].update(tuple(v) for v in r['ids']);seq_count+=1
 if counts['native_step_records']!=6660 or counts['L_records']!=6660 or counts['Sinc_updates']!=6660 or counts['SupCon_active_steps']!=5550 or totals['L']!=852480 or totals['U']!=963900 or totals['native_U_forward']!=963900 or len(unique['L'])!=6300 or len(unique['U'])!=56700 or [r['epoch'] for r in curves]!=list(range(1,31)):raise ValueError('Full E30 13L17U execution differs')
 stdout=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors=[dict(line=i,text=line[:250]) for i,line in enumerate(stdout.splitlines(),1) if re.search(r'Traceback|CUDA out of memory|\bKilled\b|RuntimeError|AssertionError|\[ERROR\]',line)]
 out=dict(curve=curves,cost=dict(**source_cost_breakdown(cost,read(root/'source_qualification/complete.json')),source_wall_seconds=walls,L_identity_NN=3409920,extra_U_NN=0,new_epochs=30,new_updates=6660,training_parent_epochs=0,own_stage20_is_included=True),full_records=dict(counts,sequence_records=seq_count,loader_packets={k:totals[k] for k in ['L','U']},unique_physical={k:len(v) for k,v in unique.items()},original_U_selection_fraction=totals['original_U_selected']/963900,teacher_precision_on_U=None,U_TX_truth_hidden=True,stdout_lines=len(stdout.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',stdout,re.I)),maximum_L_loss_decomposition_error=max_decomp),source_result=dict(segments=[read(root/s/'result.json') for s in ['stage20','source']]),physical_metadata=dict(not_applicable=True,reason='Actual native aggregate L support below;no extra U view or perpacket sidecar'),L_support=support,ordinary_observed=ordinary)
 PARSED[root.name]=out;return out


def independent_source_decision(clean,urban):
 if set(clean)!=set(ARMS) or set(urban)!=set(ARMS):raise ValueError('Allthree source rows required')
 arr={a:np.asarray(urban[a],float) for a in ARMS}
 if any(v.shape!=(2,3) or not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in arr.values()) or any(not math.isfinite(v) or not 0<=v<=1 for v in clean.values()):raise ValueError('Two complete finite source draws')
 delta={a:(100*(arr['SAT192'][:,1:]-arr[a][:,1:])).tolist() for a in ARMS[:2]}
 checks=dict(clean_preserved_ORD=clean['SAT192']>=clean['ORD128']-.0025-1e-12,clean_preserved_DUP=clean['SAT192']>=clean['DUP192']-.0025-1e-12,allfour_urban_over_ORD=bool((np.asarray(delta['ORD128'])>=1.-1e-12).all()),allfour_urban_over_DUP=bool((np.asarray(delta['DUP192'])>=1.-1e-12).all()))
 return dict(status='SOURCE_DECIDED',source_qualified=all(checks.values()),checks=checks,clean_V=clean,urban_delta_pp=delta,source_only=True,formal_scores_read=False,automatic_seed_extension=False,Ag_proved=False,goal_complete=False)

def validate_relation_step(r,arm):
 active=r['SupCon_weight']>0;ds=r['SupCon_details'];n=128 if arm=='ORD128' else 192;stats=Counter()
 if arm not in ARMS or r['crossview_mode']!=arm or r['new_losses'] or r['extra_U_NN'] or len(ds)!=(4 if active else 0):raise ValueError('Original single objective/view activation differs')
 if active and {(q['phase'],q['level']) for q in ds}!={(p,l) for p in ('first','second') for l in ('time_down','t2.pool')}:raise ValueError('All actual depth/pass descriptors required')
 for q in ds:
  if q['mode']!=arm or q['total_anchors']!=n or q['ordinary_anchors']!=128 or q['additional_anchors']!=n-128 or q['shape'][0]!=n or q['same_RX_positive_pairs'] or q['true_satellite_anchors']!=(64 if arm=='SAT192' else 0) or not 0<=q['valid_anchors']<=n or not 0<=q['positive_min']<=q['positive_max']<n:raise ValueError('Real-view/duplicate membership or physical RX positives differ')
  for k in ['total_anchors','valid_anchors','true_satellite_anchors','cross_view_positive_pairs']:stats[k]+=q[k]
 return stats

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
 y=data['ORD128']['TX'];rx=data['ORD128']['RX'];days0=data['ORD128']['day'];cohort=data['ORD128']['clean'].argmax(1)==y
 for a,x in data.items():
  if not all(np.array_equal(x[k],data['ORD128'][k]) for k in ['ids','TX','RX','day']):raise ValueError('Original source V roles/panel differ')
  for si,scene in enumerate(SCENES):
   for draw in [-1,0,1]:
    ds=[0,1] if draw==-1 else [draw];yy=np.tile(y,len(ds));rr=np.tile(rx,len(ds));dd=np.tile(days0,len(ds));cc=np.tile(cohort,len(ds));z=np.concatenate([x['satellite'][d,si] for d in ds]);pred=z.argmax(1);old=np.concatenate([data['ORD128']['satellite'][d,si].argmax(1) for d in ds]);state=np.concatenate([physics['state'][d,si,0] for d in ds]);lock=np.concatenate([physics['lock'][d,si,0] for d in ds]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in ds]);other=z.copy();other[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-other.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in [1,3,4,6,8]]+[('day',str(d),dd==d) for d in [1,2,3]]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==i) for i,s in enumerate(['LOS','PARTIAL','BLOCKED'])]+[('lock',s,lock==i) for i,s in enumerate(['locked','degraded','unlocked'])]+[('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dimension,stratum,mask in masks:
     if not mask.any():continue
     denominator=int((mask&cc).sum());m=base.metric(yy[mask],pred[mask]);rows.append(dict(arm=a,scene=scene,draw=draw,dimension=dimension,stratum=stratum,**m,mean_margin_over30=float(margin[mask].mean()),fixed_ORD_clean_correct_count=denominator,retention_on_ORD_clean_correct=float((pred==yy)[mask&cc].mean()) if denominator else None,repairs_vs_ORD=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_ORD=int(((pred!=yy)&(old==yy)&mask).sum())))
 summary=dict(status='SOURCE_THREE_ROW_COMPLETE',source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},rows=rows,source_qualification=q,model_seed_count=1,source_unique_V=1800,channel_draws=2,cohort='fixed new ORD clean-correct control;not a historical unaugmented father',added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False,NN=0,training_updates=0,target_access=False)
 write(out/'source_summary.json',summary);write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);return summary

def analyze_source(out):
 if not all(p.is_file() for p in required_sources()):raise ValueError('Source analysis incomplete;target remains closed')
 decision=read(ROOT/'source_matrix_decision.json');clean={};urban={};source={}
 for a in ARMS:
  b=ROOT/(a+'-s2026092701');done=completed_training(b);source[a]=parse_source(b,done);clean[a]=done['final_source_metrics']['source_val_accuracy'];urban[a]=read(b/'source_qualification/complete.json')['source_accuracies']
 independently=independent_source_decision(clean,urban)
 if independently['checks']!=decision['checks'] or independently['source_qualified']!=decision['source_qualified'] or any(not np.allclose(independently['urban_delta_pp'][a],decision['urban_delta_pp'][a],atol=1e-12,rtol=0) for a in ARMS[:2]):raise ValueError('Independent source rule differs')
 write(out/'independent_source_decision.json',independently)
 summary=source_readout(out)
 for a in ARMS:write(out/(a+'_source.json'),source[a])
 ordinary={};seq={}
 for a in ARMS[1:]:
  ordinary[a]=dict(records=len(source[a]['ordinary_observed']),mismatch=sum(x!=y for x,y in zip(source['ORD128']['ordinary_observed'],source[a]['ordinary_observed'])))
  seq[a]={}
  for stage in ['stage20','source']:
   pa=ROOT/(a+'-s2026092701')/stage/'training_sequence.jsonl.gz';pb=ROOT/'ORD128-s2026092701'/stage/'training_sequence.jsonl.gz';count=mismatch=0
   for x,y in zip_longest(lines(pa),lines(pb)):count+=1;mismatch+=x!=y
   seq[a][stage]=dict(records=count,mismatch=mismatch)
 curves=[dict(arm=a,**r) for a in ARMS for r in source[a]['curve']];write_csv(out/'training_curves.csv',curves)
 result=dict(status='SOURCE_ANALYZED',source_decision=independently,ordinary_sequence_comparison=ordinary,physical_sequence_comparison=seq,source_cost={a:source[a]['cost'] for a in ARMS},source_full_log_recount={a:source[a]['full_records'] for a in ARMS},source_confusions=len(summary['rows']),source_cohort='ORD control clean-correct',new_NN=0,IQ_reads=0,checkpoint_reads=0,formal_scores_read=False,goal_complete=False)
 write(out/'complete.json',result);return result

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
  c=source[a]['curve'];epoch=np.arange(1,31);color=base.COLORS[a]
  for ax,key in [(axes[0,0],'SupCon'),(axes[0,1],'aux_total_anchors'),(axes[0,2],'aux_true_satellite_anchors'),(axes[1,0],'grad_backbone'),(axes[1,1],'lr')]:ax.plot(epoch,[r[key] for r in c],label=a,color=color)
  axes[1,2].plot(epoch,[r['common_CE'] for r in c],label=a+' ordinary',color=color);axes[1,2].plot(epoch,[r['extra_CE'] for r in c],label=a+' mixed',color=color,ls='--')
 for ax,title in zip(axes.flat,['Original SupCon objective / coefficient unchanged','Anchor entries per L step across 2 depths x 2 passes','Real As anchor entries per L step;not independent IQ','Original backbone gradient norm','Actual LR / reset at20','Original ordinary/mixed CE weights .5/.5']):
  ax.set(title=title,xlabel='Cumulative epoch');ax.axvline(5.5,ls=':',color='gray');ax.axvline(13.5,ls=':',color='gray');ax.axvline(20.5,ls='--',color='gray');ax.grid(alpha=.18)
 axes[0,0].legend(frameon=False);axes[1,2].legend(frameon=False,fontsize=7);fig.suptitle('Complete30 epochs / original U only / relationship coverage is not accuracy');save(fig,'contrastive_coverage_training_curves')
 fig,axes=plt.subplots(1,3,figsize=(14,4.6))
 for a,ax in zip(ARMS,axes):
  strata=[r for segment in source[a]['L_support'] for r in segment['strata']];matrix=np.asarray([[sum(r['count'] for r in strata if r['RX']==rx and r['scene']==scene) for scene in SCENES] for rx in [1,3,4,6,8]])
  im=ax.imshow(matrix,cmap='Blues',aspect='auto');ax.set(xticks=range(3),xticklabels=['Low sub','Mid urban','Low urban'],yticks=range(5),yticklabels=['RX'+str(r) for r in [1,3,4,6,8]],title=a+' labeled full As')
  for j in range(5):
   for k in range(3):ax.text(k,j,str(matrix[j,k]),ha='center',va='center',fontsize=8,color='white' if sum(w*c for w,c in zip((.2126,.7152,.0722),im.cmap(im.norm(matrix[j,k]))[:3]))<.45 else 'black')
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
  ns=[next(r['count'] for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==('ORD128',scene,-1,'state_lock',label)) for label in cells];ax.set(xticks=range(5),xticklabels=[s.replace('/','\n')+'\nn='+str(n) for s,n in zip(cells,ns)],ylabel='Source accuracy (%)',ylim=(0,100),title=scene.replace('practical_',''));ax.tick_params(axis='x',labelsize=7);ax.legend(frameon=False);ax.grid(alpha=.18)
 fig.suptitle('Source states are descriptive; fixed formal state labels unavailable');save(fig,'source_state_lock_accuracy')
 return files
def final_analysis(out,source_dir,truth):
 prerequisite=read(source_dir/'complete.json')
 if prerequisite['status']!='SOURCE_ANALYZED' or prerequisite['formal_scores_read']:raise ValueError('Independent source analysis must finish before formal read')
 if read(source_dir/'independent_source_decision.json')['checks']!=read(ROOT/'source_matrix_decision.json')['checks']:raise ValueError('Fixed source decision differs')
 for a in ARMS:completed(ROOT/(a+'-s2026092701'))
 base.completed=completed;base.parse_source=parse_source
 results={};data={}
 for a in ARMS:results[a],data[a]=base.collect_row(a,truth)
 if any(not np.array_equal(data[a]['ids'],data['ORD128']['ids']) or not np.array_equal(data[a]['y'],data['ORD128']['y']) for a in ARMS):raise ValueError('Formal physical IDs/truth differ')
 y=data['ORD128']['y'];fixed=data['ORD128']['pred']['clean']==y;pairs=[];retention=[]
 for new,old in PAIRS:
  common=(data[new]['pred']['clean']==y)&(data[old]['pred']['clean']==y)
  for view in base.VIEWS:
   n=data[new]['pred'][view]==y;o=data[old]['pred'][view]==y;gain=int((n&~o).sum());lost=int((~n&o).sum());pairs.append(dict(new=new,old=old,view=view,repairs=gain,new_errors=lost,net=gain-lost,accuracy_delta_pp=100*(gain-lost)/len(y),common_clean_count=int(common.sum()),common_clean_repairs=int((n&~o&common).sum()),common_clean_new_errors=int((~n&o&common).sum())))
 for a in ARMS:
  for view in base.VIEWS:retention.append(dict(arm=a,view=view,fixed_ORD_clean_correct_count=int(fixed.sum()),retained=int(((data[a]['pred'][view]==y)&fixed).sum()),retention=float((data[a]['pred'][view][fixed]==y[fixed]).mean())))
 figures=base.plots(out,results);source={a:results[a]['source'] for a in ARMS};source_summary=read(source_dir/'source_summary.json');figures+=own_plots(out,source,source_summary)
 smoke=read(ROOT/'real_checkpoint_smoke.json');synthetic=read(ROOT/'synthetic/complete.json')
 if smoke['status']!='PASS' or synthetic['status']!='PASS' or smoke['optimizer_updates'] or synthetic['optimizer_updates']:raise ValueError('Shared smoke costs require actual completed artifacts')
 costs={a:dict(source_identity_NN=results[a]['source']['cost']['source_identity_NN'],prediction_identity_NN=results[a]['source']['cost']['prediction_identity_NN'],total_identity_NN=results[a]['source']['cost']['source_identity_NN']+results[a]['source']['cost']['prediction_identity_NN'],new_epochs=30,new_updates=6660) for a in ARMS}
 identity_cost=dict(per_model=costs,shared_preflight_NN=smoke['NN']+synthetic['NN'],matrix_total_identity_NN=sum(c['total_identity_NN'] for c in costs.values())+smoke['NN']+synthetic['NN'],new_model_epochs=90,new_updates=19980,diagnostic_parent_training_epochs_charged=0,each_own20_is_already_in30=True,analysis_NN=0)
 summary=dict(status='ANALYZED',run_id=RUN,model_seed=2026092701,model_seed_count=1,rows=results,identity_cost=identity_cost,pairs=pairs,fixed_ORD_clean_correct_retention=retention,source_decision=read(source_dir/'independent_source_decision.json'),urban_means={a:float(np.mean([results[a]['views'][v]['accuracy'] for v in base.VIEWS[-3:]])) for a in ARMS},suburban_gaps_pp={a:{v:100*(results[a]['views']['clean']['accuracy']-results[a]['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for a in ARMS},formal_confusion_matrices=sum(len(r['metrics']) for r in results.values()),source_confusion_matrices=len(source_summary['rows']),U_TX_truth_read=False,default_head_structure_changed=False,default_head_jointly_trained=True,additional_deployment_parameters=0,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,target_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False,interpretation_limits=['ORD clean-correct cohort is an As-trained control;not unaugmented father','Source states are not formal target state labels','L TX-by-state coverage not saved perpacket;L RX/day/scene/state aggregate and fullphysicalID exposure available','No perpacket gates or perbranch gradient trajectories saved;do not infer them from loss/accuracy','No extra U objective/teacher check;original U selection count is not pseudo-label precision','This tests As training organization;new IQ Ag is not implemented/proved'])
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in q.items() if k!='confusion'} for r in results.values() for q in r['metrics']]);write_csv(out/'paired_repairs.csv',pairs);write_csv(out/'fixed_ORD_retention.csv',retention);write(out/'figures.json',figures);write(out/'analysis_summary.json',summary);write(out/'complete.json',dict(status='ANALYZED',rows=list(ARMS),model_seed_count=1,formal_confusions=summary['formal_confusion_matrices'],source_confusions=summary['source_confusion_matrices'],figures=figures,analysis_NN=0,goal_complete=False));return summary

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
