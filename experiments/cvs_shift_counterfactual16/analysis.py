"""Independent complete default-head prediction/logit recount, no Torch/IQ."""
import math
import numpy as np
from . import design as d
from experiments.cvs_iq_timing20.harvest import score
OUT=d.ROOT/d.LOCAL/'final'
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def metrics(y,p,z):
 z=np.asarray(z,dtype=float);assert z.shape==(len(y),6) and np.isfinite(z).all() and p==z.argmax(1).tolist();s=score(y,p);m=z.max(1);s['mean_CE']=float((m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]).mean());return s
def match(s,t):
 assert s['count']==t['count'] and s['confusion']==t['confusion']
 for k in ('accuracy','macro_F1','mean_CE'):close(s[k],t[k])
def paired_stats(y,f,b,nonzero,rx,day):
 take=np.flatnonzero(nonzero)
 def one(idx):
  repair=int(((f[idx]!=y[idx])&(b[idx]==y[idx])).sum());broken=int(((f[idx]==y[idx])&(b[idx]!=y[idx])).sum())
  return dict(count=len(idx),repairs=repair,broken=broken,recovery_pp=100*(repair-broken)/len(idx) if len(idx) else None)
 return dict(nonzero_shift=one(take),all=one(np.arange(len(y))),by_RX={str(v):one(take[rx[take]==v]) for v in d.FIT},by_day={str(v):one(take[day[take]==v]) for v in (1,2,3)},by_TX={str(v):one(take[y[take]==v]) for v in range(6)})
def analyze():
 r=d.read(OUT/'result.json');done=d.read(OUT/'completion.json');resolved=d.read(OUT/'resolved_config.json');smoke=d.read(OUT/'real_checkpoint_smoke.json');raw=d.read(OUT/'predictions/raw.json');ids=d.read(OUT/'fit_L_ids.json');ys=raw['labels'];y=np.array(ys);rx=np.array(raw['RX']);day=np.array(raw['day']);n=len(d.EPOCHS)
 assert r['status']==done['status']=='SOURCE_SHIFT_COUNTERFACTUAL_COMPLETE' and done['epochs']==n==13 and r['model_state_unchanged'] and r['default_head_only'] and r['NoExplicitCFO'] and not done['automatic_training'] and all(not r[k] for k in ('training_updates','head_fit_updates','teacher_forwards','As_generations','U_used','V_used','held_RX1_used','target_access','Ag_efficacy_proved','joint_goal_complete'))
 assert smoke['status']=='PASS' and smoke['source_IQ_reads']==0 and smoke['synthetic_NN_packets']==2 and not smoke['target_access'] and resolved['config']==d.config()
 parent=d.read(d.ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final/B00-RX1-s2026092701/source/source_contract.json')
 assert len(ids)==len(set(ids))==5040 and ids==raw['physical_ids'] and set(ids)==set(parent['role_ids']['L_s'])
 for rr in d.FIT:
  for dd in (1,2,3):
   for tx in range(6):assert int(((rx==rr)&(day==dd)&(y==tx)).sum())==70
 rawscore=metrics(ys,raw['predictions'],raw['logits']);match(rawscore,r['raw_score']);full=[];bypass=[];nonzero=[];changed=[];shifts=[];epochs=[];inspection=0;calls=0;canonical=0;allfullz=[];allbypassz=[]
 for epoch in d.EPOCHS:
  p=d.read(OUT/'predictions'/('epoch%02d.json'%epoch));t=r['epochs'][epoch-1];par=t['actual_time_parameters'];bound=par['max_time_shift']
  assert p['epoch']==epoch==t['epoch'] and p['augmentation_seed']==d.SEED+epoch and p['labels']==ys and p['physical_ids']==ids and p['RX']==raw['RX'] and p['day']==raw['day'] and all(len(p[k])==5040 for k in ('full_predictions','bypass_predictions','full_logits','bypass_logits','nonzero_shift','changed','shifts'))
  assert 3<=bound<=8 and 0<par['p_time_shift']<1 and par['p_time_shift']==t['state']['p_time_shift'] and bound==t['state']['max_time_shift']
  a=metrics(ys,p['full_predictions'],p['full_logits']);b=metrics(ys,p['bypass_predictions'],p['bypass_logits']);match(a,t['full_score']);match(b,t['bypass_score']);nz=np.array(p['nonzero_shift']);ch=np.array(p['changed']);ss=np.array(p['shifts'])
  assert np.array_equal(nz,ss!=0) and np.all(abs(ss)<=bound) and not (ch&~nz).any() and np.all(np.asarray(p['full_logits'])[~ch]==np.asarray(p['bypass_logits'])[~ch]);info=paired_stats(y,np.array(p['full_predictions']),np.array(p['bypass_predictions']),nz,rx,day);assert info==t['paired'] and int(nz.sum())==t['nonzero_shift'] and int(ch.sum())==t['changed']
  batches=p['batches'];assert [v['start'] for v in batches]==list(range(0,5040,128)) and sum(v['count'] for v in batches)==5040
  flatshifts=[];flatnz=[];flatch=[];mse=[];refmse=[]
  for i,v in enumerate(batches):
   keys=('shifts','nonzero_shift','changed','PA_strength','pair_MSE','reference_window_MSE','full_raw_MSE','bypass_raw_MSE','full_raw_power_ratio','bypass_raw_power_ratio')
   assert all(len(v[k])==v['count'] for k in keys) and v['same_native_draw'] and v['unshifted_exact'] and v['canonical_checked']==(i==0)
   assert all(math.isfinite(z) for k in keys for z in v[k]) and v['nonzero_shift']==[s!=0 for s in v['shifts']] and v['changed']==[m>0 for m in v['pair_MSE']] and v['shift_calls'] in (0,1) and v['shift_inspection_scalar_draws']==2*v['count']*v['shift_calls']
   assert all(abs(s)<=bound for s in v['shifts']) and (v['shift_calls'] or all(s==0 for s in v['shifts']))
   inspection+=v['shift_inspection_scalar_draws'];calls+=v['shift_calls'];canonical+=int(v['canonical_checked']);flatshifts+=v['shifts'];flatnz+=v['nonzero_shift'];flatch+=v['changed'];mse+=v['pair_MSE'];refmse+=v['reference_window_MSE']
  assert flatshifts==p['shifts'] and flatnz==p['nonzero_shift'] and flatch==p['changed']
  epochs.append(dict(epoch=epoch,actual_time_parameters=par,nonzero_shift=int(nz.sum()),changed=int(ch.sum()),full_score=a,bypass_score=b,paired=info,pair_MSE_mean=float(np.mean(mse)),reference_window_MSE_mean=float(np.mean(refmse))))
  full+=p['full_predictions'];bypass+=p['bypass_predictions'];allfullz+=p['full_logits'];allbypassz+=p['bypass_logits'];nonzero+=p['nonzero_shift'];changed+=p['changed'];shifts+=p['shifts']
 total=paired_stats(np.tile(y,n),np.array(full),np.array(bypass),np.array(nonzero),np.tile(rx,n),np.tile(day,n));assert total==r['pooled'] and sum(nonzero)==r['nonzero_shift_total'] and sum(changed)==r['changed_total'] and calls==r['shift_calls'] and inspection==r['shift_inspection_scalar_draws'] and canonical==r['canonical_whole_checks']==13
 c=total['nonzero_shift'];rr=total['by_RX'];conditions=dict(nonzero_coverage=c['count']>0,identity_recovery=c['recovery_pp'] is not None and c['recovery_pp']>=2.,RX_coverage=all(v['count']>0 for v in rr.values()),RX_positive=sum(v['recovery_pp'] is not None and v['recovery_pp']>0 for v in rr.values())>=3);decision='PASS' if all(conditions.values()) else 'FAIL'
 assert conditions==r['qualification']['conditions'] and decision==r['qualification']['decision']==done['decision']
 decisions=5040+2*len(full);assert decisions==136080==r['source_decisions']==done['source_decisions'] and decisions+2==r['actual_NN_packets']==done['actual_NN_packets'];assert r['ordinary_generated_packets']==2*len(full)==131040 and r['canonical_check_generated_packets']==13*128==1664 and r['total_augmentation_generated_packets']==132704
 log=(OUT/'ordinary_shift.stdout.log').read_text(encoding='utf8');assert log.index('SMOKE PASS')<log.index('INPUT PASS') and sum(v.startswith('EPOCH ') for v in log.splitlines())==13 and 'COMPLETE ' in log and 'Traceback (most recent call last)' not in log
 output=dict(status='ANALYZED_SOURCE_ONLY_VERIFIED',run_id=d.RUN,decision=decision,conditions=conditions,raw_score=rawscore,pooled_full_score=metrics(ys*n,full,allfullz),pooled_bypass_score=metrics(ys*n,bypass,allbypassz),epochs=epochs,pooled=total,nonzero_shift_total=sum(nonzero),changed_total=sum(changed),unique_nonzero_shift_physical_packets=int(np.array(nonzero).reshape(n,5040).any(0).sum()),shift_calls=calls,shift_inspection_scalar_draws=inspection,canonical_whole_checks=canonical,shift_histogram={str(v):shifts.count(v) for v in sorted(set(shifts))},paired_source_cells_recounted=n*72,all136080_default_head_predictions_and_logits_recounted=True,source_decisions=decisions,actual_NN_packets=decisions+2,ordinary_generated_packets=131040,canonical_check_generated_packets=1664,total_augmentation_generated_packets=132704,training_updates=0,head_fit_updates=0,model_state_unchanged=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,elapsed_seconds=r['elapsed_seconds'],full_stdout_lines=len(log.splitlines()),uncertainty='13 draws repeat same physical packets;no naive independent-packet significance claim')
 d.write(OUT/'analysis_summary.json',output);print({k:output[k] for k in ('status','decision','conditions','pooled','actual_NN_packets','elapsed_seconds')});return output
if __name__=='__main__':analyze()
