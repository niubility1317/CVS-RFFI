"""Same-row independent four-factor analysis; no IQ/CP/model forward/training."""
import argparse,gzip,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
RUN='20261017-phase1-FcrossRX-fullsource-LA16xpractical-e30-s2026092701-m2-r01'
ORDER=('B00','B10','B01','B11')
PAIRS=(('B01','B00'),('B10','B00'),('B11','B00'),('B11','B10'),('B11','B01'))
base.ROWS={a:(RUN if a in ('B10','B11') else base.RUN30,a+'-s2026092701') for a in ORDER}
base.COLORS.update(B10='#16a34a',B11='#dc2626')
base.COMPARISON_PAIRS=PAIRS
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE_BASE=base.PROJECT/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def roots():return {a:base.PROJECT/'runs'/run/row for a,(run,row) in base.ROWS.items()}
def factorial(values):
 if set(values)!=set(ORDER) or any(set(v)!=set(base.VIEWS) for v in values.values()):raise ValueError('All four arms and seven views required')
 if any(not isinstance(x,(float,int)) or not np.isfinite(x) or not 0<=x<=1 for v in values.values() for x in v.values()):raise ValueError('Finite accuracy probabilities required')
 return {v:100*(values['B11'][v]-values['B10'][v]-values['B01'][v]+values['B00'][v]) for v in base.VIEWS}
def local_trace(root,result):
 if not result['config'].get('Ag'):return dict(applicable=False,packets=0)
 starts=Counter();physical=set();coverage=Counter();relative=[];removed=[];depth=[];zeros=0;records=0
 with gzip.open(root/'source/local_attenuation.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);s=r['stats'];ids=r['ids'];tx=r['TX'];expected=np.random.default_rng(np.random.SeedSequence([20261017111,i])).integers(0,241,size=128)
   if r['step']!=i or r['epoch']!=(i-1)//222+1 or len(ids)!=128 or len(tx)!=128 or not np.array_equal(s['starts'],expected) or s['outside_max_error']!=0:raise ValueError('Ordered private local Ag trace differs')
   for k,lo,hi,dst in [('relative_L2',0,.1000001,relative),('removed_power_fraction',-1e-12,1,removed),('depth',0,.5,depth)]:
    a=np.asarray(s[k]);
    if a.shape!=(128,) or not np.isfinite(a).all() or np.any((a<lo)|(a>hi)):raise ValueError('Fixed finite Ag bounds differ')
    dst.extend(a.tolist())
   if len(s['zero_energy'])!=128:raise ValueError('Missing zero-packet handling')
   zeros+=sum(s['zero_energy']);starts.update(expected.tolist());physical.update(tuple(p) for p in ids)
   for y,p in zip(tx,ids):
    if len(p)!=5 or p[0]!=1 or p[1] not in (1,3,4,6,8) or p[2] not in (1,2,3) or y not in range(6):raise ValueError('Ag outside allowed original L')
    coverage[(y,p[1],p[2])]+=1
   records+=1
 if records!=6660 or len(relative)!=852480 or len(physical)!=6300 or set(starts)!=set(range(241)) or set(coverage)!={(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}:raise ValueError('Complete original L local Ag coverage required')
 saved=result['local_attenuation']
 if saved['calls']!=records or saved['packets']!=len(relative) or saved['physical_IDs']!=len(physical) or saved['start_counts']!={str(k):v for k,v in sorted(starts.items())}:raise ValueError('Independent Ag trace recount differs')
 return dict(applicable=True,records=records,packets=len(relative),physical_IDs=len(physical),start_positions=len(starts),start_count_range=[min(starts.values()),max(starts.values())],private_positions_exact=True,zero_energy_packets_kept=zeros,relative_L2=dict(mean=float(np.mean(relative)),max=max(relative),quantiles=np.quantile(relative,[0,.1,.5,.9,1]).tolist()),removed_power_fraction=dict(mean=float(np.mean(removed)),max=max(removed),quantiles=np.quantile(removed,[0,.1,.5,.9,1]).tolist()),depth=dict(mean=float(np.mean(depth)),max=max(depth)),TX_RX_day_cells=len(coverage),coverage=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(coverage.items())],outside_max_error=0,no_waveform_reads=True,identity_preservation_proved=False)
def same_local_stream(aa,bb):
 from itertools import zip_longest
 records=positions=metadata=stats=0
 with gzip.open(aa/'source/local_attenuation.jsonl.gz','rt',encoding='utf8') as a,gzip.open(bb/'source/local_attenuation.jsonl.gz','rt',encoding='utf8') as b:
  for x,y in zip_longest(a,b):
   if x is None or y is None:raise ValueError('Two complete Ag streams required')
   x=json.loads(x);y=json.loads(y);records+=1
   positions+=x['stats']['starts']!=y['stats']['starts'];metadata+=any(x[k]!=y[k] for k in ('step','epoch','ids','TX'));stats+=x['stats']!=y['stats']
 return dict(records=records,position_mismatches=positions,physical_metadata_mismatches=metadata,numerical_stats_mismatches=stats)
def source_readout(out):
 data={};qualified={};role=base.read(PANEL/'data_roles.json');obs=base.read(PANEL/'observations/completion.json')
 if role['role']!='V' or role['V_panel']!=1800 or role['held_RX'] is not None or role['extra_Ldev'] or role['U_truth_used'] or role['target_access'] or obs['target_access'] or obs['draws']!=2 or obs['conditions'][0]!='full':raise ValueError('Fixed original source panel required')
 for a,root in roots().items():
  if a in ('B00','B01'):
   with np.load(SOURCE_BASE/a/'predictions.npz',allow_pickle=False) as z:data[a]={k:z[k].copy() for k in ('ids','TX','RX','day','clean')};data[a]['satellite']=z['satellite'][:,:,0].copy()
  else:
   p=root/'source_qualification/complete.json'
   if not p.is_file():qualified[a]=base.read(root/'source_qualification_failure.json');continue
   q=base.read(p)
   if q['status']!='SOURCE_QUALIFIED_READOUT' or q['NN']!=12602 or q['training_updates'] or q['head_fit_updates'] or q['target_access'] or q['formal_scores_read'] or not q['state_unchanged'] or q['new_channel_calls'] or q['device']!='cpu' or q['torch_threads']!=2:raise ValueError('Frozen CPU2 source qualification required')
   qualified[a]=q
   with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as z:data[a]={k:z[k].copy() for k in z.files}
 for a,x in data.items():
  if not np.array_equal(x['ids'],data['B00']['ids']) or not np.array_equal(x['ids'],np.asarray(role['physical_ids'])) or not np.array_equal(x['TX'],data['B00']['TX']) or x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Matched original finite source predictions required')
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as p:physics={k:p[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 y=data['B00']['TX'];rx=data['B00']['RX'];day=data['B00']['day'];cohort=data['B00']['clean'].argmax(1)==y;rows=[]
 for a,x in data.items():
  for si,scene in enumerate(SCENES):
   for dd in (-1,0,1):
    draws=(0,1) if dd==-1 else (dd,);yy=np.tile(y,len(draws));rr=np.tile(rx,len(draws));days=np.tile(day,len(draws));cc=np.tile(cohort,len(draws));z=np.concatenate([x['satellite'][d,si] for d in draws]);pred=z.argmax(1);old=np.concatenate([data['B01']['satellite'][d,si].argmax(1) for d in draws]);state=np.concatenate([physics['state'][d,si,0] for d in draws]);lock=np.concatenate([physics['lock'][d,si,0] for d in draws]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in draws]);others=z.copy();others[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-others.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in sorted(set(rx))]+[('day',str(d),days==d) for d in sorted(set(day))]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==k) for k,s in enumerate(('LOS','PARTIAL','BLOCKED'))]+[('lock',s,lock==k) for k,s in enumerate(('locked','degraded','unlocked'))]+[('state_lock',s+'/'+l,(state==k)&(lock==j)) for k,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dim,value,mask in masks:
     if not mask.any():continue
     q=base.metric(yy[mask],pred[mask]);den=int((mask&cc).sum());rows.append(dict(arm=a,scene=scene,draw=dd,dimension=dim,stratum=value,**q,mean_margin_over30=float(margin[mask].mean()),fixed_B00_clean_correct_count=den,retention_on_B00_clean_correct=float((pred==yy)[mask&cc].mean()) if den else None,repairs_vs_B01=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_B01=int(((pred!=yy)&(old==yy)&mask).sum())))
 base.write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);summary=dict(status='SOURCE_FACTORIAL_READOUT' if set(data)==set(ORDER) else 'SOURCE_FACTORIAL_PARTIAL',rows=rows,source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},source_qualification=qualified,fixed_B00_clean_correct_count=int(cohort.sum()),unique_source_V=1800,draws=2,source_observation='same original full reflect; no Ag at inference',added_noise_ratio_is_total_SNR=False,NN=0,IQ_reads=0,CP_reads=0,training_updates=0,target_feedback=False,crossseed_confirmed=False)
 base.write(out/'source_summary.json',summary);return summary

def source_plots(out,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=summary['rows'];arms=[a for a in ORDER if a in summary['source_clean']];files=[]
 def row(a,scene,dim='overall',stratum='ALL'):return next(r for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(a,scene,-1,dim,stratum))
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):
   path=out/(name+'.'+suffix);fig.savefig(path,dpi=200);files.append(str(path))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.7));bins=('<3','3-8','8-15','>=15')
 for ax,scene,title in zip(axes,SCENES[1:],('Mid urban','Low urban')):
  for a in arms:ax.plot(range(4),[100*row(a,scene,'added_noise_ratio',b)['accuracy'] for b in bins],'o-',label=a,color=base.COLORS[a])
  counts=[row('B01',scene,'added_noise_ratio',b)['count'] for b in bins];ax.set(xticks=range(4),xticklabels=[b+' dB\nn='+str(n) for b,n in zip(bins,counts)],ylim=(0,100),title=title,ylabel='Source accuracy (%)',xlabel='Signal / added noise ratio; not total SNR');ax.legend(frameon=False);ax.grid(alpha=.2)
 fig.suptitle('Same fixed source V1800 / two channel draws / default head');save(fig,'source_added_noise_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(13,5.1))
 for ax,scene,title in zip(axes,SCENES[1:],('Mid urban','Low urban')):
  strata=[r['stratum'] for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'])==('B01',scene,-1,'state_lock')];matrix=np.asarray([[100*row(a,scene,'state_lock',s)['accuracy'] for s in strata] for a in arms]);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set(yticks=range(len(arms)),yticklabels=arms,xticks=range(len(strata)),xticklabels=[s+'\nn='+str(row('B01',scene,'state_lock',s)['count']) for s in strata],title=title);ax.tick_params(axis='x',rotation=55)
  for j in range(len(arms)):
   for k in range(len(strata)):ax.text(k,j,f'{matrix[j,k]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>65 else 'black')
  fig.colorbar(im,ax=ax,fraction=.035,label='Source accuracy (%)')
 fig.suptitle('Source state/lock strata; tiny cells descriptive, not formal test states');save(fig,'source_state_lock_accuracy');return files

def wait_for_run(pid):
 if type(pid) is not int or pid<=0:raise ValueError('Positive existing run dispatcher PID required')
 root=base.PROJECT/'runs'/RUN
 while not (root/'complete.json').is_file():
  if (root/'failure.json').is_file():raise RuntimeError('Run terminal failure preserved; no analysis or restart')
  process=Path('/proc')/str(pid)
  if not process.is_dir():raise RuntimeError('Registered dispatcher no longer live; reconcile run before analysis')
  command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'cvs_fullsource_factorial27/runner.py' not in command:raise RuntimeError('Dispatcher handle no longer belongs to this run')
  time.sleep(15)

def main():
 p=argparse.ArgumentParser();p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--wait-dispatcher',type=int);a=p.parse_args();rs=roots()
 if a.wait_dispatcher is not None:wait_for_run(a.wait_dispatcher)
 # Verify every requested row before any target truth access; no target consumer controls training.
 for root in rs.values():base.completed(root)
 out=a.output;out.mkdir(exist_ok=False);results={};private={}
 for name in ORDER:results[name],private[name]=base.collect_row(name,a.truth)
 if any(not np.array_equal(private['B00']['ids'],private[n]['ids']) for n in ORDER):raise ValueError('Matched formal physical IDs required')
 comparison={};sequences={};cohort=private['B00']['pred']['clean']==private['B00']['y'];fixed_retention={}
 for name,ref in PAIRS:
  key=name+'-'+ref;sequences[key]=base.compare_training_sequences(name,ref);comparison[key]={};y=private[name]['y']
  for view in base.VIEWS:
   cc=private[name]['pred'][view]==y;old=private[ref]['pred'][view]==y;comparison[key][view]=dict(accuracy_gain_pp=100*(cc.mean()-old.mean()),repairs=int((cc&~old).sum()),breaks=int((~cc&old).sum()))
 for name in ORDER:fixed_retention[name]={v:float((private[name]['pred'][v]==private[name]['y'])[cohort].mean()) for v in base.VIEWS}
 interaction=factorial({n:{v:results[n]['views'][v]['accuracy'] for v in base.VIEWS} for n in ORDER});flat=[q for r in results.values() for q in r['metrics']]
 for n in ORDER:results[n]['source']['local_Ag_trace']=local_trace(rs[n],results[n]['source']['source_result'])
 local_pair=same_local_stream(rs['B10'],rs['B11']);base.write_csv(out/'per_row_seed.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in flat]);base.write_csv(out/'training_curves.csv',[dict(arm=n,**q) for n,r in results.items() for q in r['source']['curve']]);sources=source_readout(out)
 gaps={n:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for n,r in results.items()};costs={n:dict(r['source']['cost'],source_qualification_NN=(sources['source_qualification'].get(n,{}).get('NN') if n in ('B10','B11') else 0),source_qualification_wall_seconds=(sources['source_qualification'].get(n,{}).get('wall_seconds') if n in ('B10','B11') else 0),reference_frozen_source_probe_historical_NN=23402 if n in ('B00','B01') else 0) for n,r in results.items()}
 base.write(out/'analysis_summary.json',dict(code_commit=a.commit,rows=results,paired_comparisons=comparison,interaction_accuracy_pp=interaction,urban_mean_accuracy={n:sum(r['views'][v]['accuracy'] for v in base.VIEWS[4:])/3 for n,r in results.items()},suburban_clean_gaps_pp=gaps,fixed_B00_clean_correct_count=int(cohort.sum()),fixed_B00_clean_correct_satellite_retention=fixed_retention,training_sequences=sequences,local_Ag_stream_pair=local_pair,costs=costs,new_model_epochs=60,new_updates=13320,reused_reference_epochs=60,source_epoch_records=120,formal_confusion_matrices=len(flat),source_confusion_matrices=len(sources['rows'])+len(sources['source_clean']),NN=0,IQ_reads=0,CP_reads=0,training_updates=0,model_seed_count=1,crossseed_confirmed=False,Ag_proved=False,goal_complete=False,limitations=['Single model seed; two source channel draws do not establish cross-seed reproduction','Formal per-packet state metadata unavailable; source strata are not formal target states','No per-packet fusion activation trace saved; gradient or parameter movement alone is not identification gain','Source reference CPU2 batch90/newsource CPU2 batch256; numerical layout differs','U truth hidden; pseudo acceptance is not precision','Budget exhausted does not establish convergence; old LA16 standalone FAIL retained']))
 files=base.plots(out,results)+source_plots(out,sources);base.write(out/'figures.json',dict(files=files));base.write(out/'completion.json',dict(status='FACTORIAL_ANALYSIS_COMPLETE',rows=list(ORDER),model_seed_count=1,source_epochs_parsed=120,source_updates_parsed=26640,formal_confusion_matrices=len(flat),figures=len(files),NN=0,IQ_reads=0,CP_reads=0,new_updates=0,code_commit=a.commit,crossseed_confirmed=False,goal_complete=False));print(json.dumps(dict(status='FACTORIAL_ANALYSIS_COMPLETE',rows=list(ORDER),out=str(out),figures=len(files),NN=0)))
if __name__=='__main__':main()
