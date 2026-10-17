"""Fixed L-smoothing x As analysis. Source mode never opens target/IQ/CP."""
import argparse,math,time
from collections import defaultdict
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
import numpy as np
import cvs_impulse_factorial_E30_results_20261018 as old
base=old.base
RUN='20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01'
REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
ROOT=base.PROJECT/'runs'/RUN
ARMS=('B00','H00','B01','H01');NEW=('H00','H01')
PAIRS=(('H00','B00'),('H01','B01'),('B01','B00'),('H01','H00'),('H01','B00'))
SCENES=old.SCENES;PANEL=old.PANEL
read,write,write_csv,lines=base.read,base.write,base.write_csv,old.lines
ROOTS={a:base.PROJECT/'runs'/(RUN if a in NEW else REF)/(a+'-s2026092701') for a in ARMS}
base.ROWS={a:(ROOTS[a].parent.name,ROOTS[a].name) for a in ARMS}
base.COLORS.update(B00='#64748b',H00='#f59e0b',B01='#2563eb',H01='#dc2626')
NAMES=dict(B00='LS.01 / no As (reused)',H00='L CE0 / no As (new)',B01='LS.01 / As (reused)',H01='L CE0 / As (new)')

@contextmanager
def run_scope(root):
 run,sat=old.RUN,old.satellite
 old.RUN=Path(root).parent.name;old.satellite=lambda a:a in ('B01','H01')
 try:yield
 finally:old.RUN=run;old.satellite=sat

def completed(root):
 with run_scope(root):return old.completed(root)

def parse_source(root,done=None):
 with run_scope(root):result=old.parse_source(root,done)
 arm=Path(root).name.split('-s')[0]
 result['cost'].update(original_training_epochs=30,original_training_updates=6660,new_epochs_in_this_experiment=30 if arm in NEW else 0,new_updates_in_this_experiment=6660 if arm in NEW else 0,reused_control=arm not in NEW)
 if arm in NEW and 'hard_L_recount' not in result:
  hard=[];maximum=0.
  for stage,offset in [('stage20',0),('source',4440)]:
   p=Path(root)/stage;n=0
   for i,(r,s,h) in enumerate(zip_longest(lines(p/'reference_steps.jsonl.gz'),lines(p/'shape_steps.jsonl.gz'),lines(p/'hard_L_steps.jsonl.gz')),1):
    if any(x is None for x in (r,s,h)) or h['step']!=i or h['cumulative_step']!=offset+i or len(h['pairs'])!=3 or h['extra_NN'] or not h['U_function_restored'] or r['L_label_smoothing']!=0. or r['native_U_label_smoothing']!=.01:raise ValueError('Incomplete/non-L hardCE record')
    pairs=h['pairs'];weights=(.5,.5,.25)
    if any(q['L_smoothing']!=0. or q['alpha']!=1. or q['NN'] for q in pairs):raise ValueError('Original KL/no new forward required')
    for q in pairs:
     for k in ('CE0','CE01','KL'):old.numeric(q[k],k)
    ce0=sum(w*q['CE0'] for w,q in zip(weights,pairs));ce01=sum(w*q['CE01'] for w,q in zip(weights,pairs));kl=sum(w*q['KL'] for w,q in zip(weights,pairs));aux=r['SupCon_weight']*r['SupCon']
    errors=[abs(ce0+kl+aux-h['L_total']),abs(ce01+kl+aux-h['original_L01_counterfactual_value']),abs(ce0-ce01-h['weighted_CE0_minus_CE01']),abs(h['L_total']-r['total_L'])]
    errors += [abs(q['CE0']-v) for q,v in zip(pairs,[r['common_CE'],r['extra_CE'],s['shape_CE']])]
    maximum=max(maximum,*errors)
    if max(errors)>3e-5:raise ValueError('Independent actual-L CE/KL correction differs')
    hard.append(dict(arm=arm,step=offset+i,epoch=(offset+i-1)//222+1,weighted_CE0=ce0,weighted_CE01_on_same_updated_logits=ce01,weighted_KL=kl,actual_total_L=h['L_total'],same_weights_L01_counterfactual=h['original_L01_counterfactual_value'],CE0_minus_CE01=ce0-ce01));n=i
   if n!=(4440 if stage=='stage20' else 2220):raise ValueError('Every real hardCE step required')
  if len(hard)!=6660:raise ValueError('Fixed 6660 hard-L updates required')
  result['hard_L_recount']=dict(records=len(hard),maximum_arithmetic_error=maximum,U_function_restored_every_step=True,extra_NN=0,native_U_smoothing=.01,actual_L_smoothing=0.,counterfactual_is_recalculated_at_H_updated_weights=True)
  result['hard_L_records']=hard
 return result

def source_required():
 return [ROOTS[a]/x for a in ARMS for x in ('stage20/completion.json','source/completion.json','source_qualification/complete.json')]+[ROOT/'source_matrix_decision.json']

def wait_source(pids):
 while not all(p.is_file() for p in source_required()):
  for p in [ROOT/'failure.json']+[ROOTS[a]/'source_qualification_failure.json' for a in NEW]:
   if p.exists():raise RuntimeError('Known missing/failed source artifacts;preserve '+str(p))
  if not any((Path('/proc')/str(pid)).exists() for pid in pids):raise RuntimeError('Owned handles gone before source completion;do not relaunch')
  time.sleep(20)

def independent_source(clean,urban,rx):
 if any(set(v)!=set(ARMS) for v in (clean,urban,rx)):raise ValueError('Exactly four source cells required')
 for a in ARMS:
  if len(urban[a])!=2 or any(len(d)!=3 for d in urban[a]) or set(rx[a])!={'1','3','4','6','8'}:raise ValueError('All source draws/scenes/RX required')
  vals=[clean[a]]+[v for d in urban[a] for v in d]+list(rx[a].values())
  if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in vals):raise ValueError('Finite source accuracies required')
 diffs=[[urban['H01'][d][s]-urban['B01'][d][s] for s in (1,2)] for d in range(2)]
 checks=dict(hard_clean_independent=clean['H00']-clean['B00']>=.002,joint_clean_net=clean['H01']>clean['B00'],joint_clean_preserves_As=clean['H01']>=clean['B01'],both_urban_each_draw=all(v>=.005 for row in diffs for v in row),suburban_preservation=all(urban['H01'][d][0]>=urban['B01'][d][0]-.0025 for d in range(2)),each_source_RX_preserved=all(rx['H01'][str(r)]>=rx['B01'][str(r)]-.0025 for r in (1,3,4,6,8)))
 return dict(status='SOURCE_QUALIFIED' if all(checks.values()) else 'SOURCE_REJECTED',source_qualified=all(checks.values()),checks=checks,urban_differences=diffs,clean=clean,source_accuracies=urban,Ag_proved=False,goal_complete=False,crossseed_confirmed=False,target_feedback=False)

def source_readout(out):
 data={};q={};role=read(PANEL/'data_roles.json');rows=[]
 for a in ARMS:
  b=ROOTS[a];proof=read(b/'source_qualification/complete.json')
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
 summary=dict(status='SOURCE_FOUR_ROW_COMPLETE',source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},rows=rows,source_qualification=q,model_seed_count=1,source_unique_V=1800,channel_draws=2,cohort='fixed matched B00 FINAL30 control clean-correct;not shared E20 father',added_noise_ratio_is_total_SNR=False,formal_state_accuracy_available=False,NN=0,training_updates=0,target_access=False)
 write(out/'source_summary.json',summary);write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);return summary

def analyze_source(out):
 if not all(p.is_file() for p in source_required()):raise ValueError('Source incomplete;target stays closed')
 clean={};urban={};rx={};source={}
 for a in ARMS:
  b=ROOTS[a]
  with run_scope(b):done=old.completed_training(b)
  source[a]=parse_source(b);clean[a]=done['final_source_metrics']['source_val_accuracy'];urban[a]=read(b/'source_qualification/complete.json')['source_accuracies'];rx[a]=done['final_source_metrics']['source_val_rx_accuracy']
 independent=independent_source(clean,urban,rx);saved=read(ROOT/'source_matrix_decision.json')
 if independent['checks']!=saved['checks'] or independent['source_qualified']!=saved['source_qualified'] or not np.allclose(independent['urban_differences'],saved['urban_differences'],atol=1e-12,rtol=0):raise ValueError('Independent prospective source decision differs')
 write(out/'independent_source_decision.json',independent);summary=source_readout(out);sequences={};ordinary={};As_counts={}
 for new,ref in [('H00','B00'),('H01','B01')]:
  sequences[new]={}
  for stage in ('stage20','source'):
   seq=(lines(ROOTS[a]/stage/'training_sequence.jsonl.gz') for a in (new,ref));n=diff=0
   for x,y in zip_longest(*seq):n+=1;diff+=x!=y
   sequences[new][stage]=dict(records=n,mismatched=diff)
  ordinary[new]=dict(records=6660,mismatched=sum(x!=y for x,y in zip_longest(source[new]['ordinary_observed'],source[ref]['ordinary_observed'])))
  if new=='H01':
   counts=dict(records=0,mismatched=0)
   for stage in ('stage20','source'):
    for x,y in zip_longest(lines(ROOTS[new]/stage/'reference_steps.jsonl.gz'),lines(ROOTS[ref]/stage/'reference_steps.jsonl.gz')):
     counts['records']+=1;counts['mismatched']+=any(x[k]!=y[k] for k in ('As_packets','scene_counts','state_counts'))
   As_counts[new]=counts
 for a in ARMS:write(out/(a+'_source.json'),source[a])
 write_csv(out/'training_curves.csv',[q for a in ARMS for q in source[a]['curve']]);write_csv(out/'hard_L_steps.csv',[q for a in NEW for q in source[a]['hard_L_records']])
 figures=source_plots(out,source,summary)
 result=dict(status='SOURCE_ANALYZED',source_decision=independent,source_cost={a:source[a]['cost'] for a in ARMS},source_full_log_recount={a:source[a]['full_records'] for a in ARMS},hard_L_recount={a:source[a]['hard_L_recount'] for a in NEW},ordinary_sequence_comparison=ordinary,physical_sequence_comparison=sequences,As_scene_state_comparison=As_counts,source_confusions=len(summary['rows']),all_source_RX=[1,3,4,6,8],new_NN=0,IQ_reads=0,checkpoint_reads=0,formal_scores_read=False,figures=figures,loss_only_ablation=True,Ag_proved=False,goal_complete=False)
 write(out/'complete.json',result);return result

def saver(out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 files=[]
 def save(fig,name):
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=180);files.append(str(p))
  plt.close(fig)
 return plt,files,save

def source_plots(out,source,summary):
 plt,files,save=saver(out);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for a in ARMS:
  c=source[a]['curve'];epochs=[q['epoch'] for q in c]
  for ax,key,title,ylabel in [(axes[0,0],'source_val_accuracy_pct','Full source V health','Source V accuracy (%)'),(axes[0,1],'full/RDrop_CE','Own L objective; CE0 vs CE.01 values are different targets','Weighted L CE'),(axes[1,0],'full/RDrop_KL','Unchanged RDrop KL objective','Weighted KL'),(axes[1,1],'grad_backbone','Native backbone gradient health','Gradient norm')]:
   ax.plot(epochs,[q[key] for q in c],label=NAMES[a],color=base.COLORS[a]);ax.set(title=title,xlabel='Cumulative epoch',ylabel=ylabel);ax.axvline(13.5,color='gray',ls=':');ax.axvline(20.5,color='gray',ls='--');ax.grid(alpha=.18)
 axes[0,0].legend(fontsize=7);fig.suptitle('All120 epoch records:60 new +60 reused; source V is not unseen-RX performance');save(fig,'full_training_curves')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for a in NEW:
  records=source[a]['hard_L_records'];means=[]
  for ep in range(1,31):
   group=[q for q in records if q['epoch']==ep];means.append({k:float(np.mean([q[k] for q in group])) for k in ('weighted_CE0','weighted_CE01_on_same_updated_logits','CE0_minus_CE01')})
  x=list(range(1,31));axes[0].plot(x,[q['CE0_minus_CE01'] for q in means],label=NAMES[a],color=base.COLORS[a]);axes[1].plot(x,[q['weighted_CE0'] for q in means],color=base.COLORS[a],label=a+' actual CE0');axes[1].plot(x,[q['weighted_CE01_on_same_updated_logits'] for q in means],color=base.COLORS[a],ls='--',label=a+' CE.01 at same logits')
 axes[0].axhline(0,color='gray',lw=1);axes[0].set(title='Calibration-objective value correction; not an accuracy gain',xlabel='Epoch',ylabel='Weighted CE0 minus CE.01');axes[1].set(title='Recalculation at H weights; not a control-training trajectory',xlabel='Epoch',ylabel='Weighted L CE')
 for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=7)
 save(fig,'hard_CE_value_recount')
 rows=summary['rows'];fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
 for ri,scene in enumerate(SCENES[1:]):
  for ci,dim in enumerate(('state_lock','added_noise_ratio')):
   subset=[r for r in rows if r['scene']==scene and r['draw']==-1 and r['dimension']==dim];keys=list(dict.fromkeys(r['stratum'] for r in subset));width=.19;x=np.arange(len(keys))
   for ai,a in enumerate(ARMS):
    own={r['stratum']:r for r in subset if r['arm']==a};axes[ri,ci].bar(x+(ai-1.5)*width,[100*own[k]['accuracy'] for k in keys],width,label=NAMES[a],color=base.COLORS[a])
   n={r['stratum']:r['count'] for r in subset if r['arm']=='B00'};labels=[k+'\nn='+str(n[k]) for k in keys];axes[ri,ci].set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Accuracy (%)',title=scene+' / '+dim);axes[ri,ci].tick_params(axis='x',labelrotation=20);axes[ri,ci].grid(axis='y',alpha=.15)
 axes[0,0].legend(fontsize=7);fig.suptitle('Frozen source V1800 x2 draws; all nonempty strata retained; added-noise ratio is not total SNR');save(fig,'source_state_quality')
 fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
 for a in ARMS:
  own=[next(r for r in rows if r['arm']==a and r['scene']==s and r['draw']==(-1 if s!='clean' else -1) and r['dimension']=='overall') for s in ('clean',*SCENES)]
  axes[0].plot(range(4),[r['mean_margin_over30'] for r in own],marker='o',label=NAMES[a],color=base.COLORS[a]);axes[1].plot(range(4),[100*r['retention_on_B00_clean_correct'] for r in own],marker='o',label=NAMES[a],color=base.COLORS[a])
 for ax in axes:ax.set(xticks=range(4),xticklabels=['clean','low suburban','mid urban','low urban']);ax.grid(alpha=.18)
 axes[0].set(title='Actual default-head margin / fixed cosine scale30',ylabel='Mean true-class margin /30');axes[1].set(title='Fixed B00 FINAL30 clean-correct source cohort',ylabel='Retention (%)');axes[0].legend(fontsize=7);save(fig,'source_margins_retention');return files

def effect_table(results):
 rows=[]
 for v in base.VIEWS:
  for stat in ('accuracy','macro_f1','worst_RX_accuracy'):
   x={a:float(results[a]['views'][v][stat]) for a in ARMS}
   rows.append(dict(view=v,metric=stat,hardL_without_As_pp=100*(x['H00']-x['B00']),As_under_LS01_pp=100*(x['B01']-x['B00']),joint_net_pp=100*(x['H01']-x['B00']),As_under_hardL_pp=100*(x['H01']-x['H00']),hardL_with_As_pp=100*(x['H01']-x['B01']),interaction_pp=100*(x['H01']-x['H00']-x['B01']+x['B00'])))
 return rows

def formal_plots(out,results,effects):
 plt,files,save=saver(out);labels=['clean','high sub','mid sub','low sub','high urban','mid urban','low urban'];x=np.arange(7)
 fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained')
 for a in ARMS:ax.plot(x,[100*results[a]['views'][v]['accuracy'] for v in base.VIEWS],marker='o',label=NAMES[a],color=base.COLORS[a])
 ax.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Accuracy (%)',title='All fixed views / default head / one modelseed;2 new +2 reused');ax.grid(alpha=.18);ax.legend(fontsize=8);save(fig,'formal_accuracy')
 fields=['hardL_without_As_pp','As_under_LS01_pp','joint_net_pp','As_under_hardL_pp','hardL_with_As_pp','interaction_pp'];matrix=np.asarray([[next(r for r in effects if r['view']==v and r['metric']=='accuracy')[f] for v in base.VIEWS] for f in fields]);lim=max(1.,float(np.max(np.abs(matrix))))
 fig,ax=plt.subplots(figsize=(13,5.8),layout='constrained');im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set(xticks=x,xticklabels=labels,yticks=range(len(fields)),yticklabels=fields,title='Loss definition x As effects (percentage points); not an IQ-Ag factor')
 for i in range(len(fields)):
  for j in range(7):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>lim*.55 else 'black')
 fig.colorbar(im,ax=ax,label='Percentage points');save(fig,'formal_effects')
 fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
 for ax,view in zip(axes,base.VIEWS[-2:]):
  values={a:{r['stratum']:r['accuracy'] for r in results[a]['metrics'] if r['view']==view and r['dimension']=='receiver'} for a in ARMS};keys=sorted(values['B00'])
  for a in ARMS:ax.plot(range(len(keys)),[100*values[a][k] for k in keys],marker='o',label=NAMES[a],color=base.COLORS[a])
  ax.set(xticks=range(len(keys)),xticklabels=keys,ylim=(0,100),xlabel='Actual target RX',ylabel='Accuracy (%)',title=view);ax.grid(alpha=.18)
 axes[0].legend(fontsize=7);save(fig,'formal_RX')
 fig,ax=plt.subplots(figsize=(12,4.7),layout='constrained')
 for a in ARMS:ax.plot(x,[results[a]['views'][v]['mean_true_margin_over30'] for v in base.VIEWS],marker='o',label=NAMES[a],color=base.COLORS[a])
 ax.set(xticks=x,xticklabels=labels,ylabel='True-class margin /30',title='Actual unchanged cosine scale30;no scale fit or extra classifier');ax.grid(alpha=.18);ax.legend(fontsize=7);save(fig,'formal_margins');return files

def final_analysis(out,source_dir,truth):
 source_done=read(source_dir/'complete.json');independent=read(source_dir/'independent_source_decision.json')
 if source_done['status']!='SOURCE_ANALYZED' or source_done['formal_scores_read'] or independent['checks']!=read(ROOT/'source_matrix_decision.json')['checks']:raise ValueError('Fixed independent source decision first')
 for a in ARMS:
  completed(ROOTS[a]);old.PARSED[a+'-s2026092701']=read(source_dir/(a+'_source.json'))
 base.completed=completed;base.parse_source=parse_source;results={};data={}
 for a in ARMS:results[a],data[a]=base.collect_row(a,truth)
 if any(not np.array_equal(data[a]['ids'],data['B00']['ids']) or not np.array_equal(data[a]['y'],data['B00']['y']) for a in ARMS):raise ValueError('Same physical target IDs/truth required')
 y=data['B00']['y'];cohort=data['B00']['pred']['clean']==y;pairs=[];retention=[]
 for new,reference in PAIRS:
  common=(data[new]['pred']['clean']==y)&(data[reference]['pred']['clean']==y)
  for view in base.VIEWS:
   n=data[new]['pred'][view]==y;o=data[reference]['pred'][view]==y;rep=int((n&~o).sum());br=int((~n&o).sum());pairs.append(dict(new=new,old=reference,view=view,repairs=rep,new_errors=br,net=rep-br,accuracy_delta_pp=100*(rep-br)/len(y),common_clean_count=int(common.sum()),common_clean_repairs=int((n&~o&common).sum()),common_clean_new_errors=int((~n&o&common).sum())))
 for a in ARMS:
  for view in base.VIEWS:retention.append(dict(arm=a,view=view,fixed_B00_clean_correct_count=int(cohort.sum()),retained=int(((data[a]['pred'][view]==y)&cohort).sum()),retention=float((data[a]['pred'][view][cohort]==y[cohort]).mean())))
 effects=effect_table(results);figures=formal_plots(out,results,effects);smoke=read(ROOT/'real_checkpoint_smoke.json');synthetic=read(ROOT/'synthetic_smoke.json')
 if smoke['status']!='PASS' or synthetic['status']!='PASS' or smoke['NN']!=2 or synthetic['NN']!=1536 or smoke['optimizer_updates'] or synthetic['optimizer_updates']:raise ValueError('Actual shared smoke evidence required')
 costs={a:dict(source_identity_NN=results[a]['source']['cost']['source_identity_NN'],prediction_identity_NN=results[a]['source']['cost']['prediction_identity_NN'],total_identity_NN=results[a]['source']['cost']['source_identity_NN']+results[a]['source']['cost']['prediction_identity_NN'],original_training_epochs=30,original_training_updates=6660,new_epochs=30 if a in NEW else 0,new_updates=6660 if a in NEW else 0,reused_control=a not in NEW) for a in ARMS}
 gaps={a:{v:100*(results[a]['views']['clean']['accuracy']-results[a]['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for a in ARMS};goals=dict(joint_clean_positive_over_B00=results['H01']['views']['clean']['accuracy']>results['B00']['views']['clean']['accuracy'],low_urban_above60=results['H01']['views']['practical_low_urban']['accuracy']>.60,each_suburban_gap_at_most2pp=all(v<=2+1e-12 for v in gaps['H01'].values()),crossseed_confirmed=False)
 summary=dict(status='ANALYZED',run_id=RUN,loss_only_ablation=True,IQ_Ag_tested=False,new_rows=list(NEW),reused_control_rows=['B00','B01'],model_seed=2026092701,model_seed_count=1,rows=results,source_decision=independent,effects=effects,pairs=pairs,fixed_B00_clean_correct_retention=retention,urban_means={a:float(np.mean([results[a]['views'][v]['accuracy'] for v in base.VIEWS[-3:]])) for a in ARMS},suburban_gaps_pp=gaps,terminal_exploratory_goals=goals,identity_cost=dict(per_model=costs,shared_new_smoke_NN=1538,new_identity_NN=sum(costs[a]['total_identity_NN'] for a in NEW)+1538,reused_control_identity_NN=sum(costs[a]['total_identity_NN'] for a in ARMS if a not in NEW),new_model_epochs=60,new_updates=13320,reused_control_epochs=60,each_own20_is_included_in30=True,analysis_NN=0),formal_confusion_matrices=sum(len(r['metrics']) for r in results.values()),source_confusion_matrices=source_done['source_confusions'],analysis_NN=0,IQ_reads=0,checkpoint_reads=0,U_TX_truth_read=False,default_head_structure_changed=False,additional_deployment_parameters=0,target_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False,interpretation_limits=['One modelseed only,no crossseed proof','B00 fixed FINAL30 cohort is a matched control,not a common pretrained parent','Source states/quality are not formal target state labels or controlled causal removal effects','No perpacket branch/gate trajectories saved;U selection counts are not precision','concat step telemetry is the inherited fourforward branch;use shape/reference/counters for actual6forward total','SixL costs50%moreLNN than4Lhistory;only these four cells match6L/1.25 scale','Different CE target values cannot establish better recognition;hardL records independently recover both targets at the same H weights'])
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in q.items() if k!='confusion'} for r in results.values() for q in r['metrics']]);write_csv(out/'factorial_effects.csv',effects);write_csv(out/'paired_repairs.csv',pairs);write_csv(out/'fixed_B00_retention.csv',retention);write(out/'figures.json',figures);write(out/'analysis_summary.json',summary);write(out/'complete.json',dict(status='ANALYZED',rows=list(ARMS),model_seed_count=1,formal_confusions=summary['formal_confusion_matrices'],source_confusions=summary['source_confusion_matrices'],figures=figures,analysis_NN=0,goal_complete=False));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('source','formal'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-dir',type=Path);p.add_argument('--truth',type=Path);p.add_argument('--wait-pids',type=int,nargs='*');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Keep immutable analysis outputs')
 if a.mode=='source':
  if a.wait_pids:wait_source(a.wait_pids)
  if not all(f.is_file() for f in source_required()):raise ValueError('Source incomplete')
 else:
  if a.source_dir is None or a.truth is None:raise ValueError('Formal mode needs complete source decision and frozen truth-last artifacts')
  source=read(a.source_dir/'complete.json')
  if source['status']!='SOURCE_ANALYZED' or source['formal_scores_read']:raise ValueError('Source-only decision first')
  for arm in ARMS:completed(ROOTS[arm])
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 result=analyze_source(a.output) if a.mode=='source' else final_analysis(a.output,a.source_dir,a.truth)
 write(a.output/'execution.json',dict(mode=a.mode,status=result['status'],wall_seconds=time.monotonic()-start,NN=0,IQ_reads=0,checkpoint_reads=0,head_fit_updates=0,target_read=a.mode=='formal',goal_complete=False))
 print(__import__('json').dumps(dict(status=result['status'],mode=a.mode,output=str(a.output),goal_complete=False)),flush=True)
if __name__=='__main__':main()
