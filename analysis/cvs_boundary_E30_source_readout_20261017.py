"""Independent recount/figures after the completed frozen-source mechanism only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ARMS=('B00','B01','BCTX');SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban');CONDITIONS=('full','edge_fixed_AGC')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def require_done(d):
 if d.get('status')!='SOURCE_BOUNDARY_MECHANISM_COMPLETE' or d.get('NN')!=70206 or d.get('models')!=3 or d.get('model_seed_count')!=1 or d.get('channel_draws')!=2 or d.get('unique_source_V')!=1800 or d.get('training_updates')!=0 or d.get('head_fit_updates')!=0 or d.get('target_access') is not False or d.get('target_scores_or_truth_read') is not False or d.get('state_unchanged') is not True:raise ValueError('Completed frozen-source-only evidence required')
def metric(y,p):
 if len(y)==0 or y.shape!=p.shape or y.ndim!=1 or np.any((y<0)|(y>=6)|(p<0)|(p>=6)):raise ValueError('Invalid source decisions')
 cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den!=0)
 return dict(count=len(y),correct=int(cm.trace()),accuracy=float(cm.trace()/len(y)),macro_f1=float(f1.mean()),confusion=cm.tolist())
def one(rows,arm,scene,condition,draw=-1,dimension='overall',stratum='ALL'):
 selected=[r for r in rows if (r['arm'],r['scene'],r['condition'],r['draw'],r['dimension'],r['stratum'])==(arm,scene,condition,draw,dimension,stratum)]
 if len(selected)!=1:raise ValueError('Expected exactly one declared source cell')
 return selected[0]
def hypothesis(rows):
 cells=[]
 for scene in SCENES[1:]:
  for draw in (0,1):
   f0=one(rows,'B01',scene,'full',draw)['accuracy'];e0=one(rows,'B01',scene,'edge_fixed_AGC',draw)['accuracy'];f1=one(rows,'BCTX',scene,'full',draw)['accuracy'];e1=one(rows,'BCTX',scene,'edge_fixed_AGC',draw)['accuracy']
   recovery=100*((e1-f1)-(e0-f0));full_delta=100*(f1-f0)
   cells.append(dict(scene=scene,draw=draw,B01_edge_minus_full_pp=100*(e0-f0),BCTX_edge_minus_full_pp=100*(e1-f1),boundary_penalty_improvement_pp=recovery,full_accuracy_change_pp=full_delta,support=recovery>=3-1e-10 and full_delta>=-.5-1e-10))
 return dict(cells=cells,source_condition_supported=all(r['support'] for r in cells),scope='fixed receiver and AGC on source V; not total receiver effect, formal performance, Ag evidence or crossseed success')
def strata_mask(row,rx,state,lock,quality):
 dim,value=row['dimension'],row['stratum'];states=('LOS','PARTIAL','BLOCKED');locks=('locked','degraded','unlocked')
 if dim=='overall' and value=='ALL':return np.ones(len(rx),bool)
 if dim=='RX':return rx==int(value)
 if dim=='state':return state==states.index(value)
 if dim=='lock':return lock==locks.index(value)
 if dim=='state_lock':
  s,l=value.split('/');return (state==states.index(s))&(lock==locks.index(l))
 if dim=='added_noise_ratio':
  lo,hi={'<3':(-np.inf,3),'3-8':(3,8),'8-15':(8,15),'>=15':(15,np.inf)}[value];return (quality>=lo)&(quality<hi)
 raise ValueError('Unknown source stratum')
def figures(out,rows,results):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'B00':'#64748b','B01':'#2563eb','BCTX':'#a21caf'};files=[];labels=['Low suburban','Mid urban','Low urban']
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):
   path=out/(name+'.'+suffix);fig.savefig(path,dpi=200);files.append(str(path))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for i,arm in enumerate(ARMS):
  full=[100*one(rows,arm,s,'full')['accuracy'] for s in SCENES];edge=[100*one(rows,arm,s,'edge_fixed_AGC')['accuracy'] for s in SCENES]
  axes[0].plot(range(3),full,'o-',color=colors[arm],label=arm+' full');axes[0].plot(range(3),edge,'x--',color=colors[arm],label=arm+' fixed edge')
  axes[1].bar(np.arange(3)+(i-1)*.23,np.asarray(edge)-full,.23,color=colors[arm],label=arm)
 for ax in axes:ax.set(xticks=range(3),xticklabels=labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].set(ylabel='Source accuracy (%)',ylim=(0,100),title='Same V1800 / two channel draws');axes[0].legend(frameon=False,fontsize=7)
 axes[1].set(ylabel='Edge minus full (percentage points)',title='Fixed receiver / fixed AGC boundary effect');axes[1].axhline(0,color='black',lw=.7);axes[1].legend(frameon=False)
 fig.suptitle('Frozen E30 default heads: source mechanism, not formal target accuracy');save(fig,'source_boundary_sensitivity')
 fig,axes=plt.subplots(1,2,figsize=(11,4.5));quality_bins=['<3','3-8','8-15','>=15']
 for ax,scene,title in zip(axes,SCENES[1:],['Mid urban','Low urban']):
  counts=[one(rows,'B01',scene,'full',dimension='added_noise_ratio',stratum=b)['count'] for b in quality_bins]
  for arm in ARMS:ax.plot(range(4),[100*one(rows,arm,scene,'full',dimension='added_noise_ratio',stratum=b)['accuracy'] for b in quality_bins],'o-',color=colors[arm],label=arm)
  ax.set(xticks=range(4),xticklabels=[b+' dB\nn='+str(n) for b,n in zip(quality_bins,counts)],ylim=(0,100),title=title,ylabel='Source accuracy (%)',xlabel='Signal / added noise ratio; not total SNR');ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.suptitle('Original full observations; n counts two draws of fixed source packets');save(fig,'source_quality_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(13,4.9))
 for ax,scene,title in zip(axes,SCENES[1:],['Mid urban','Low urban']):
  strata=[r['stratum'] for r in rows if r['arm']=='B01' and r['scene']==scene and r['condition']=='full' and r['draw']==-1 and r['dimension']=='state_lock']
  counts=[one(rows,'B01',scene,'full',dimension='state_lock',stratum=s)['count'] for s in strata]
  z=np.asarray([[100*one(rows,arm,scene,'full',dimension='state_lock',stratum=s)['accuracy'] for s in strata] for arm in ARMS]);im=ax.imshow(z,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
  ax.set(yticks=range(3),yticklabels=ARMS,xticks=range(len(strata)),xticklabels=[s+'\nn='+str(n) for s,n in zip(strata,counts)],title=title);ax.tick_params(axis='x',rotation=55)
  for j in range(3):
   for k in range(len(strata)):ax.text(k,j,f'{z[j,k]:.1f}',ha='center',va='center',color='white' if z[j,k]>65 else 'black',fontsize=8)
  fig.colorbar(im,ax=ax,fraction=.035,label='Source accuracy (%)')
 fig.suptitle('Source propagation/lock strata; tiny strata are descriptive only');save(fig,'source_state_accuracy')
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,key in zip(axes,['PA','reference','signature']):
  for arm in ARMS:ax.plot(results[arm]['gates'][key],color=colors[arm],label=arm,alpha=.8,lw=1)
  ax.set(title=key,xlabel='Existing fusion coordinate',ylabel='Learned static gain');ax.grid(alpha=.2)
 axes[0].legend(frameon=False);fig.suptitle('Existing fusion gains; cosine logit scale remains30');save(fig,'source_static_gains')
 return files

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();root=a.source_root
 done=read(root/'completion.json');require_done(done)
 rows=read(root/'source_metrics.json')['records'];data={};results={};branch={}
 for arm in ARMS:
  with np.load(root/arm/'predictions.npz',allow_pickle=False) as z:data[arm]={k:z[k].copy() for k in z.files}
  results[arm]=read(root/arm/'result.json')
  with np.load(root/arm/'branch_metrics.npz',allow_pickle=False) as z:branch[arm]=z['values'].copy()
  x=data[arm]
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,2,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Completed defaulthead source logits differ')
  if any(not np.array_equal(x[k],data['B00'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Matched source physical samples required')
 y=data['B00']['TX'];ref_clean=data['B00']['clean'].argmax(1)==y;physics=np.load(Path(done['observation_source'])/'observations/physics.npz',allow_pickle=False);clean={};flat=[]
 for arm in ARMS:
  q=metric(y,data[arm]['clean'].argmax(1));old=read(root/arm/'clean_metrics.json')
  if any(q[k]!=old[k] for k in q):raise ValueError('Independent source clean recount differs')
  clean[arm]=q
 for row in rows:
  arm,scene,ci=row['arm'],SCENES.index(row['scene']),CONDITIONS.index(row['condition']);draws=(0,1) if row['draw']==-1 else (row['draw'],);yy=np.tile(y,len(draws));rx=np.tile(data[arm]['RX'],len(draws));cohort=np.tile(ref_clean,len(draws))
  state=np.concatenate([physics['state'][dd,scene,0] for dd in draws]);lock=np.concatenate([physics['lock'][dd,scene,0] for dd in draws]);quality=np.concatenate([physics['signal_to_added_noise_db'][dd,scene,0] for dd in draws]);mask=strata_mask(row,rx,state,lock,quality)
  logits=np.concatenate([data[arm]['satellite'][dd,scene,ci] for dd in draws]);pred=logits.argmax(1);full=np.concatenate([data[arm]['satellite'][dd,scene,0].argmax(1) for dd in draws]);ref=np.concatenate([data['B01']['satellite'][dd,scene,ci].argmax(1) for dd in draws]);q=metric(yy[mask],pred[mask])
  if any(q[k]!=row[k] for k in q):raise ValueError('Independent source stratified recount differs')
  repairs=int(((pred==yy)&(full!=yy)&mask).sum());breaks=int(((pred!=yy)&(full==yy)&mask).sum());den=int((cohort&mask).sum());retention=float((pred==yy)[cohort&mask].mean()) if den else None
  if row['repairs_vs_own_full']!=repairs or row['breaks_vs_own_full']!=breaks or row['fixed_B00_clean_correct_count']!=den or row['retention_on_B00_clean_correct']!=retention:raise ValueError('Source paired/referenced cohort recount differs')
  flat.append(dict(**{k:v for k,v in row.items() if k!='confusion'},repairs_vs_B01=int(((pred==yy)&(ref!=yy)&mask).sum()),breaks_vs_B01=int(((pred!=yy)&(ref==yy)&mask).sum())))
 out=a.output;out.mkdir(exist_ok=False);support=hypothesis(rows)
 with (out/'per_row_source.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 branch_summary=[]
 for arm in ARMS:
  if branch[arm].shape!=(2,3,2,1800,5,3) or not np.isfinite(branch[arm]).all():raise ValueError('Finite existing-branch summaries required')
  for si,scene in enumerate(SCENES):
   for ci,condition in enumerate(CONDITIONS):
    z=branch[arm][:,si,ci].reshape(-1,5,3)
    for ki,name in enumerate(('time','frequency','PA','reference','signature')):branch_summary.append(dict(arm=arm,scene=scene,condition=condition,branch=name,mean_clean_sat_cosine=float(z[:,ki,0].mean()),mean_clean_norm=float(z[:,ki,1].mean()),mean_satellite_norm=float(z[:,ki,2].mean()),cosine_q10_q50_q90=np.quantile(z[:,ki,0],[.1,.5,.9]).tolist()))
 files=figures(out,rows,results);write(out/'summary.json',dict(branch_summary=branch_summary,code_commit=a.commit,clean=clean,boundary_hypothesis=support,source_cost=done,model_seed_count=1,crossseed_confirmed=False,paired_count=len(rows),independent_confusion_matrices=len(rows)+3,source_B00_clean_correct_count=int(ref_clean.sum()),figures=files,NN=0,IQ_reads=0,CP_reads=0,training_updates=0,target_access=False,Ag_proved=False,goal_complete=False))
 write(out/'completion.json',dict(status='SOURCE_READOUT_COMPLETE',source_confusion_matrices=len(rows)+3,figures=len(files),NN=0,IQ_reads=0,CP_reads=0,updates=0,target_access=False,code_commit=a.commit));print(json.dumps(dict(status='SOURCE_READOUT_COMPLETE',source_confusion_matrices=len(rows)+3,figures=len(files),NN=0)))
if __name__=='__main__':main()
