"""Independent saved-logit/source-physics recount; no IQ, checkpoint or model call."""
import csv,gzip,json,time,hashlib,random
from pathlib import Path
import numpy as np
P=Path('/home/szu2070436088/2510044040/CV-SincNet')
ROOT=P/'runs/20261017-diagnostic-FcrossRX-cycleorder-placement-sourceV-e30-s2026092701-m2-r01'
POST=P/'runs/20261017-diagnostic-FcrossRX-cycleorder-sourceV-e30-s2026092701-m2-r01'
OLD=P/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
SCENES=['practical_low_suburban','practical_mid_urban','practical_low_urban']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def calc(y,z):
 if z.shape!=(len(y),6) or not np.isfinite(z).all():raise ValueError('Finite default six-class logits required')
 pred=z.argmax(-1);cm=np.zeros((6,6),np.int64);np.add.at(cm,(y,pred),1);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 margin=(z[np.arange(len(y)),y]-np.where(np.arange(6)[None,:]==y[:,None],-np.inf,z).max(1))/30.
 return dict(n=len(y),correct=int(cm.trace()),accuracy=float(cm.trace()/len(y)),macro_F1=float(f.mean()),margin_scale30=float(margin.mean()),CM=cm.tolist())
def compare(got,ref):
 if got['CM']!=ref['CM'] or any(abs(got[k]-ref[k])>1e-12 for k in ['accuracy','macro_F1','margin_scale30']):raise ValueError('Independent saved-logit metric mismatch')
def main():
 tic=time.perf_counter();done=read(ROOT/'completion.json');obs=read(ROOT/'observations/completion.json')
 if done['status']!='SOURCE_CYCLE_ORDER_PLACEMENT_COMPLETE' or done['NN']!=21604 or done['native_calls']!=21600 or done['updates'] or done['target_access']:raise ValueError('Complete fixed source-only experiment required')
 if obs['unique_existing_latent_pairs']!=10800 or obs['native_calls']!=21600 or obs['new_latent_seeds'] or len(obs['RX_sessions'])!=15 or obs['state_changes']:raise ValueError('Full paired original latent replay required')
 out=ROOT/'analysis_r01';out.mkdir(exist_ok=False);phys=np.load(ROOT/'observations/physics.npz',allow_pickle=False);trace=read(ROOT/'observations/iq_operator_trace.json');posttrace=read(POST/'B01/iq_operator_trace.json')
 for k,v in phys.items():
  if v.shape!=(2,3,2,1800) or not np.isfinite(v).all():raise ValueError('Full finite original/pre physical matrix required')
 if not np.array_equal(phys['state'][:,:,0],phys['state'][:,:,1]) or not np.array_equal(phys['added_noise_variance_reference'][:,:,0],phys['added_noise_variance_reference'][:,:,1]):raise ValueError('Physical state/noise variance changed')
 summaries={};strata=[];pooled_table=[];cmchecks=0;perms=0
 for arm in ('B00','B01'):
  orig=np.load(OLD/arm/'predictions.npz',allow_pickle=False);post=np.load(POST/arm/'predictions.npz',allow_pickle=False);pre=np.load(ROOT/arm/'predictions.npz',allow_pickle=False);result=read(ROOT/arm/'result.json');prior=read(POST/arm/'result.json')
  for key in ('ids','TX','RX','day'):
   if not np.array_equal(orig[key],post[key]) or not np.array_equal(orig[key],pre[key]):raise ValueError('Same full source physical cohort required')
  y=orig['TX'];rx=orig['RX'];truth=np.tile(y,2);clean_ok=all(v for k,v in prior['decision']['checks'].items() if k!='urban_protected');delta=np.zeros((2,2));pooled=[]
  cohort=orig['clean'].argmax(1)==y
  for si,scene in enumerate(SCENES):
   views={'original':orig['satellite'][:,si,0],'post_order':post['satellite'][:,si],'pre_order':pre['satellite'][:,si]};metrics={k:calc(truth,z.reshape(-1,6)) for k,z in views.items()};cmchecks+=3
   pp=views['original'].argmax(-1);qq=views['pre_order'].argmax(-1);rec=dict(scene=scene,**metrics,delta_pre_original_pp=100*(metrics['pre_order']['accuracy']-metrics['original']['accuracy']),delta_pre_post_pp=100*(metrics['pre_order']['accuracy']-metrics['post_order']['accuracy']),repair=int(((pp!=y)&(qq==y)).sum()),new_error=int(((pp==y)&(qq!=y)).sum()))
   for k,z in views.items():rec[k]['parent_clean_correct_retention']=float((z.argmax(-1)[:,cohort]==y[cohort]).mean())
   pooled.append(rec);pooled_table.append(dict(arm=arm,scene=scene,n=len(truth),**{k+'_'+metric:v[metric] for k,v in metrics.items() for metric in ('accuracy','macro_F1','margin_scale30')},delta_pre_original_pp=rec['delta_pre_original_pp'],delta_pre_post_pp=rec['delta_pre_post_pp'],repair=rec['repair'],new_error=rec['new_error']))
   masks=[('overall','ALL',np.ones((2,len(y)),bool))]+[('RX',str(r),np.broadcast_to(rx==r,(2,len(y)))) for r in (1,3,4,6,8)]
   state=phys['state'][:,si,0];lock=phys['lock'][:,si,0];qb=np.digitize(phys['signal_to_added_noise_db'][:,si,0],[3,8,15])
   masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(['LOS','PARTIAL','BLOCKED']) for j,l in enumerate(['locked','degraded','unlocked'])]
   masks += [('quality',lab,qb==j) for j,lab in enumerate(['<3','3-8','8-15','>=15'])]
   for dim,lab,mask in masks:
    if not mask.any():continue
    yy=np.broadcast_to(y,(2,len(y)))[mask];mm={k:calc(yy,z[mask]) for k,z in views.items()};cmchecks+=3
    strata.append(dict(arm=arm,scene=scene,dimension=dim,stratum=lab,n=len(yy),**{k+'_'+metric:v[metric] for k,v in mm.items() for metric in ('accuracy','macro_F1','margin_scale30')},delta_pre_original_pp=100*(mm['pre_order']['accuracy']-mm['original']['accuracy']),delta_pre_post_pp=100*(mm['pre_order']['accuracy']-mm['post_order']['accuracy'])))
   for draw in range(2):
    r=next(r for r in result['source_rows'] if r['draw']==draw and r['scene']==scene)
    for k,z in views.items():compare(calc(y,z[draw]),r[k]);cmchecks+=1
    change=100*(calc(y,views['pre_order'][draw])['accuracy']-calc(y,views['original'][draw])['accuracy'])
    if abs(change-r['delta_pre_vs_original_pp'])>1e-12:raise ValueError('Draw delta mismatch')
    if si:delta[draw,si-1]=change
  accepted=delta>=-.5-1e-10;status='PASS_SOURCE_PLACEMENT_ONLY' if clean_ok and (arm=='B00' or accepted.all()) else 'FAIL_FIXED_PRE_RECEIVER_CYCLE_ORDER'
  if status!=result['decision']['status'] or not np.allclose(delta,result['decision']['urban_delta_pp'],atol=1e-12,rtol=0):raise ValueError('Fixed source rule mismatch')
  summaries[arm]=dict(pooled=pooled,decision=result['decision'],prior_clean_guard=clean_ok)
  if arm=='B00':ids=pre['ids']
  elif not np.array_equal(ids,pre['ids']):raise ValueError('Physical IQ shared by models required')
 for tag,t in trace.items():
  if len(t['permutation'])!=1800 or t['permutation']!=posttrace[tag]['permutation']:raise ValueError('Order-paired permutations differ')
  for i,pid in enumerate(ids):
   seed=int.from_bytes(hashlib.blake2b(('202610173001/'+tag+'/'+str(pid)).encode('utf8'),digest_size=8).digest(),'little');perm=list(range(4));random.Random(seed).shuffle(perm)
   if not t['phase_defined'][i]:perm=list(range(4))
   if perm!=t['permutation'][i]:raise ValueError('Independent private permutation mismatch')
   perms+=1
 if perms!=10800:raise ValueError('Complete view accounting required')
 records=0;max_replay=0.;max_noise=0.;seen=set()
 with gzip.open(ROOT/'observations/physical_records.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);key=(r['draw'],r['scene'],r['physical_id'])
   if key in seen:raise ValueError('Duplicate physical observation')
   seen.add(key);records+=1;max_replay=max(max_replay,r['replay_max_abs']);max_noise=max(max_noise,r['noise_relative_error'])
 if records!=10800 or abs(max_replay-obs['original_replay_max_abs'])>1e-15 or abs(max_noise-obs['pre_agc_noise_max_relative'])>1e-15:raise ValueError('Independent native trace count mismatch')
 fields=['quality_snr_db','signal_to_added_noise_db','output_frequency_hz','estimated_frequency_hz','agc_gain_db','residual_scale','residual_kernel_energy','measured_added_noise_power_pre_agc']
 physics_summary={s:{k:dict(mean_delta=float((phys[k][:,si,1]-phys[k][:,si,0]).mean()),max_abs_delta=float(np.abs(phys[k][:,si,1]-phys[k][:,si,0]).max())) for k in fields} for si,s in enumerate(SCENES)}
 for name,rows in [('pooled_results.csv',pooled_table),('source_strata.csv',strata)]:
  with (out/name).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=dict(status='INDEPENDENT_PLACEMENT_SOURCE_ANALYSIS_COMPLETE',source=summaries,recomputed_CM=cmchecks,strata_rows=len(strata),permutations_verified=perms,native_records=records,native_observations=obs,physics_delta=physics_summary,NN=0,IQ=0,CP=0,updates=0,target_access=False,goal_completed=False,model_seed_count=1,channel_draws=2)
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.6));values=summaries['B01']['pooled'];x=np.arange(3)
 for shift,k,label,color in [(-.25,'original','Original As(x)','#2563eb'),(0.,'post_order','Ag(As(x))','#ea580c'),(.25,'pre_order','As(Ag(x))','#059669')]:axes[0].bar(x+shift,[r[k]['accuracy']*100 for r in values],.25,label=label,color=color)
 axes[0].set(xticks=x,xticklabels=['Low suburban','Mid urban','Low urban'],ylabel='Source accuracy (%)',ylim=(0,105),title='B01 / same 1800 physical source packets');axes[0].tick_params(axis='x',rotation=14);axes[0].legend(fontsize=8)
 delta=np.asarray(summaries['B01']['decision']['urban_delta_pp']);lim=max(1.,float(np.abs(delta).max()));im=axes[1].imshow(delta,cmap='RdBu',vmin=-lim,vmax=lim)
 for i in range(2):
  for j in range(2):axes[1].text(j,i,f'{delta[i,j]:+.3f} pp',ha='center',va='center')
 axes[1].set(xticks=[0,1],xticklabels=['Mid urban','Low urban'],yticks=[0,1],yticklabels=['Draw 0','Draw 1'],title='Pre-order minus original / each >= -0.5 pp');fig.colorbar(im,ax=axes[1],label='Percentage points');fig.suptitle('Frozen E30 students; source-only; zero training updates');fig.tight_layout()
 figs=['cycle_order_placement_source_comparison.png','cycle_order_placement_source_comparison.pdf']
 for name in figs:fig.savefig(out/name,dpi=180)
 plt.close(fig);summary.update(figures=figs,wall_seconds=time.perf_counter()-tic)
 (out/'completion.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(summary))
if __name__=='__main__':main()
