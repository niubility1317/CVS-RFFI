"""Independent same-row source CM recount and figures; no IQ/CP/NN."""
import json,csv,time
from pathlib import Path
import numpy as np
P=Path('/home/szu2070436088/2510044040/CV-SincNet')
ROOT=P/'runs/20261017-diagnostic-FcrossRX-EMA-sourceV-e30-s2026092701-m2-r01'
OLD=P/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
PHYS=P/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic/observations/physics.npz'
SCENES=['practical_low_suburban','practical_mid_urban','practical_low_urban']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def calc(y,z):
 p=z.argmax(-1);cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1);fs=[2*cm[c,c]/den[c] if den[c] else 0. for c in range(6)]
 return dict(n=len(y),correct=int(cm.trace()),accuracy=float(cm.trace()/len(y)),macro_F1=float(sum(fs)/6),CM=cm.tolist())
def main():
 tic=time.perf_counter();done=read(ROOT/'completion.json')
 if done['status']!='SOURCE_EMA_DIAGNOSTIC_COMPLETE' or done['NN']!=25204 or done['updates'] or done['target_access']:raise ValueError('Complete source-only run required')
 out=ROOT/'analysis_r01';out.mkdir(exist_ok=False);physics=np.load(PHYS,allow_pickle=False);rows=[];table=[];checks=0;summary={}
 for arm in ('B00','B01'):
  a=np.load(ROOT/arm/'predictions.npz',allow_pickle=False);b=np.load(OLD/arm/'predictions.npz',allow_pickle=False);r=read(ROOT/arm/'result.json')
  if any(not np.array_equal(a[k],b[k]) for k in ('ids','TX','RX','day')):raise ValueError('Physical same-row mismatch')
  y=a['TX'];rx=a['RX'];deltas=np.zeros((2,2));newc=calc(y,a['clean']);oldc=calc(y,b['clean'])
  for key,got in [('clean_EMA',newc),('clean_student',oldc)]:
   for field in ('accuracy','macro_F1'):
    if abs(got[field]-r[key][field])>1e-12:raise ValueError('Independent clean metric differs')
   if got['CM']!=r[key]['CM']:raise ValueError('Independent CM differs')
   checks+=1
  pooled=[dict(view='clean',student=oldc,EMA=newc,delta_accuracy_pp=100*(newc['accuracy']-oldc['accuracy']))]
  for si,scene in enumerate(SCENES):
   truth=np.tile(y,2);student=b['satellite'][:,si,0];ema=a['satellite'][:,si]
   pp=student.argmax(-1);qq=ema.argmax(-1)
   aa=calc(truth,student.reshape(-1,6));bb=calc(truth,ema.reshape(-1,6));pool=dict(view=scene,student=aa,EMA=bb,delta_accuracy_pp=100*(bb['accuracy']-aa['accuracy']),repair=int(((pp!=y)&(qq==y)).sum()),new_error=int(((pp==y)&(qq!=y)).sum()));pooled.append(pool)
   qbin=np.digitize(physics['signal_to_added_noise_db'][:,si,0],[3,8,15]);state=physics['state'][:,si,0];lock=physics['lock'][:,si,0]
   masks=[('overall','ALL',np.ones((2,len(y)),bool))]+[('RX',str(rr),np.broadcast_to(rx==rr,(2,len(y)))) for rr in (1,3,4,6,8)]
   masks += [('state_lock',sn+'/'+ln,(state==sj)&(lock==lj)) for sj,sn in enumerate(['LOS','PARTIAL','BLOCKED']) for lj,ln in enumerate(['locked','degraded','unlocked'])]
   masks += [('quality',label,qbin==qi) for qi,label in enumerate(['<3','3-8','8-15','>=15'])]
   for dim,label,mask in masks:
    if not mask.any():continue
    yy=np.broadcast_to(y,(2,len(y)))[mask];mm=calc(yy,student[mask]);nn=calc(yy,ema[mask]);checks+=2
    rows.append(dict(arm=arm,scene=scene,dimension=dim,stratum=label,n=len(yy),student_accuracy=mm['accuracy'],EMA_accuracy=nn['accuracy'],delta_pp=100*(nn['accuracy']-mm['accuracy']),student_macro_F1=mm['macro_F1'],EMA_macro_F1=nn['macro_F1'],repair=int(((pp[mask]!=yy)&(qq[mask]==yy)).sum()),new_error=int(((pp[mask]==yy)&(qq[mask]!=yy)).sum())))
   for draw in range(2):
    old=calc(y,student[draw]);new=calc(y,ema[draw]);record=next(x for x in r['source_rows'] if x['draw']==draw and x['scene']==scene)
    if old['CM']!=record['student']['CM'] or new['CM']!=record['EMA']['CM']:raise ValueError('Independent draw CM differs')
    delta=100*(new['accuracy']-old['accuracy']);checks+=2
    if si:deltas[draw,si-1]=delta
  rx_delta=[100*(calc(y[rx==rr],a['clean'][rx==rr])['accuracy']-calc(y[rx==rr],b['clean'][rx==rr])['accuracy']) for rr in (1,3,4,6,8)]
  clean_delta=100*(newc['accuracy']-oldc['accuracy']);accepted=(clean_delta>=-1e-10 and min(rx_delta)>=-.5-1e-10 and np.all(deltas>=1-1e-10))
  if accepted!=(r['decision']['status']=='PASS_SOURCE_READOUT_ONLY') or int((deltas>=1-1e-10).sum())!=r['decision']['cells_passed']:raise ValueError('Independent fixed decision differs')
  summary[arm]=dict(pooled=pooled,decision=r['decision']);table += [dict(arm=arm,view=x['view'],n=x['student']['n'],student_accuracy=x['student']['accuracy'],EMA_accuracy=x['EMA']['accuracy'],delta_accuracy_pp=x['delta_accuracy_pp'],student_macro_F1=x['student']['macro_F1'],EMA_macro_F1=x['EMA']['macro_F1']) for x in pooled]
 with (out/'source_strata.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'pooled_results.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 result=dict(status='INDEPENDENT_SOURCE_ANALYSIS_COMPLETE',source=summary,recomputed_CM=checks,strata_rows=len(rows),source_V=1800,channel_draws=2,model_seed_count=1,NN=0,updates=0,IQ=0,CP=0,target_access=False,goal_completed=False)
 (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(4);values=summary['B01']['pooled']
 axes[0].bar(x-.18,[v['student']['accuracy']*100 for v in values],.36,label='Final student',color='#2563eb')
 axes[0].bar(x+.18,[v['EMA']['accuracy']*100 for v in values],.36,label='Own EMA',color='#f97316')
 axes[0].set(xticks=x,xticklabels=['Clean','Low suburban','Mid urban','Low urban'],ylabel='Source accuracy (%)',ylim=(0,105),title='B01 / same fixed source observations');axes[0].tick_params(axis='x',rotation=18);axes[0].legend()
 delta=np.asarray(summary['B01']['decision']['urban_delta_pp']);im=axes[1].imshow(delta,cmap='RdBu',vmin=-1,vmax=1)
 for i in range(2):
  for j in range(2):axes[1].text(j,i,f'{delta[i,j]:+.3f} pp',ha='center',va='center')
 axes[1].set(xticks=[0,1],xticklabels=['Mid urban','Low urban'],yticks=[0,1],yticklabels=['Draw 0','Draw 1'],title='EMA minus student / required each >= +1 pp');fig.colorbar(im,ax=axes[1],label='Percentage points')
 fig.suptitle('Frozen source V1800, one model seed, zero training updates');fig.tight_layout();fig.savefig(out/'EMA_source_comparison.png',dpi=180);fig.savefig(out/'EMA_source_comparison.pdf');plt.close(fig)
 result.update(wall_seconds=time.perf_counter()-tic,figures=['EMA_source_comparison.png','EMA_source_comparison.pdf']);(out/'completion.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
