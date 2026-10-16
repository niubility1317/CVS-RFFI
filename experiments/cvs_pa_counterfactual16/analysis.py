"""Independent full prediction/logit/counterfactual recount;no Torch/IQ/model."""
import math
import numpy as np
from . import design as d
from experiments.cvs_iq_timing20.harvest import score
OUT=d.ROOT/d.LOCAL/'final'
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def metrics(y,p,z):
 assert p==np.asarray(z).argmax(1).tolist() and np.isfinite(z).all();s=score(y,p);z=np.asarray(z);m=z.max(1);s['mean_CE']=float((m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]).mean());return s
def match(s,t):
 assert s['count']==t['count'] and s['confusion']==t['confusion']
 for k in ('accuracy','macro_F1','mean_CE'):close(s[k],t[k])
def paired_stats(y,f,b,ch,rx,day):
 take=np.flatnonzero(ch)
 def one(idx):
  repair=int(((f[idx]!=y[idx])&(b[idx]==y[idx])).sum());broken=int(((f[idx]==y[idx])&(b[idx]!=y[idx])).sum())
  return dict(count=len(idx),repairs=repair,broken=broken,recovery_pp=100*(repair-broken)/len(idx) if len(idx) else None)
 return dict(changed=one(take),all=one(np.arange(len(y))),by_RX={str(r):one(take[rx[take]==r]) for r in d.FIT},by_day={str(v):one(take[day[take]==v]) for v in (1,2,3)},by_TX={str(v):one(take[y[take]==v]) for v in range(6)})
def analyze():
 r=d.read(OUT/'result.json');done=d.read(OUT/'completion.json');resolved=d.read(OUT/'resolved_config.json');smoke=d.read(OUT/'real_checkpoint_smoke.json');raw=d.read(OUT/'predictions/raw.json');ids=d.read(OUT/'fit_L_ids.json');ys=raw['labels'];rx=np.array(raw['RX']);day=np.array(raw['day']);y=np.array(ys)
 assert r['status']==done['status']=='SOURCE_PA_COUNTERFACTUAL_COMPLETE' and done['epochs']==20 and r['model_state_unchanged'] and r['default_head_only'] and r['NoExplicitCFO'] and not done['automatic_training'] and all(not r[k] for k in ('training_updates','head_fit_updates','teacher_forwards','As_generations','U_used','V_used','held_RX1_used','target_access','Ag_efficacy_proved','joint_goal_complete'))
 assert smoke['status']=='PASS' and smoke['source_IQ_reads']==0 and smoke['synthetic_NN_packets']==2 and not smoke['target_access'] and resolved['config']==d.config()
 parent=d.read(d.ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final/B00-RX1-s2026092701/source/source_contract.json');assert len(ids)==len(set(ids))==5040 and ids==raw['physical_ids'] and set(ids)==set(parent['role_ids']['L_s'])
 for rr in d.FIT:
  for dd in (1,2,3):
   for tx in range(6):assert int(((rx==rr)&(day==dd)&(y==tx)).sum())==70
 rawscore=metrics(ys,raw['predictions'],raw['logits']);match(rawscore,r['raw_score']);full=[];bypass=[];changed=[];triggered=0;epochs=[];counts={k:dict(calls=0,selected_packets=0) for k in d.SUBCHAINS};cells=0;max_diff=0.
 for epoch in d.EPOCHS:
  p=d.read(OUT/'predictions'/('epoch%02d.json'%epoch));t=r['epochs'][epoch-1]
  assert p['epoch']==epoch==t['epoch'] and p['augmentation_seed']==d.SEED+epoch and p['labels']==ys and p['physical_ids']==ids and p['RX']==raw['RX'] and p['day']==raw['day'] and all(len(p[k])==5040 for k in ('full_predictions','bypass_predictions','full_logits','bypass_logits','changed','triggered'))
  a=metrics(ys,p['full_predictions'],p['full_logits']);b=metrics(ys,p['bypass_predictions'],p['bypass_logits']);match(a,t['full_score']);match(b,t['bypass_score']);ch=np.array(p['changed']);tr=np.array(p['triggered']);assert not (ch&~tr).any() and np.all(np.asarray(p['full_logits'])[~ch]==np.asarray(p['bypass_logits'])[~ch]);info=paired_stats(y,np.array(p['full_predictions']),np.array(p['bypass_predictions']),ch,rx,day);assert info==t['paired'];assert int(ch.sum())==t['changed'] and int(tr.sum())==t['triggered']
  records=p['batches'];assert [v['start'] for v in records]==list(range(0,5040,128)) and sum(v['count'] for v in records)==5040
  flat_ch=[];flat_tr=[];strengths=[];relative=[];reference=[]
  for v in records:
   assert v['same_native_draw'] and v['untriggered_exact'] and all(len(v[k])==v['count'] for k in ('changed','triggered','strength','PA_MSE','PA_reference_window_MSE','prefix_raw_MSE','full_raw_MSE','full_raw_power_ratio','bypass_raw_power_ratio')) and set(v['subchains']).issubset(d.SUBCHAINS)
   assert v['triggered']==[s>0 for s in v['strength']] and v['changed']==[m>0 for m in v['PA_MSE']] and all(math.isfinite(n) for k in ('strength','PA_MSE','PA_reference_window_MSE','prefix_raw_MSE','full_raw_MSE','full_raw_power_ratio','bypass_raw_power_ratio') for n in v[k])
   for k in v['subchains']:counts[k]['calls']+=1;counts[k]['selected_packets']+=sum(v['triggered'])
   flat_ch+=v['changed'];flat_tr+=v['triggered'];strengths+=v['strength'];relative+=v['PA_MSE'];reference+=v['PA_reference_window_MSE']
  assert flat_ch==p['changed'] and flat_tr==p['triggered'];triggered+=int(tr.sum());full+=p['full_predictions'];bypass+=p['bypass_predictions'];changed+=p['changed']
  # Actual resolved probability is schedule-dependent,not a default assumption.
  assert 0.014-1e-12<=t['state']['p_pa']<=.049+1e-12 and t['state']['p_dac']==0
  epochs.append(dict(epoch=epoch,p_pa=t['state']['p_pa'],triggered=int(tr.sum()),changed=int(ch.sum()),full_score=a,bypass_score=b,paired=info,PA_MSE_mean=float(np.mean(relative)),PA_reference_window_MSE_mean=float(np.mean(reference)),strength_max=max(strengths)))
  for rr in d.FIT:
   for dd in (1,2,3):
    for tx in range(6):assert int(((rx==rr)&(day==dd)&(y==tx)).sum())==70;cells+=1
 total=paired_stats(np.tile(y,20),np.array(full),np.array(bypass),np.array(changed),np.tile(rx,20),np.tile(day,20));assert total==r['pooled'] and counts==r['subchains'] and triggered==r['triggered_total'] and sum(changed)==r['changed_total']
 c=total['changed'];rr=total['by_RX'];dd=total['by_day'];conditions=dict(sufficient_changed=c['count']>=1000,changed_identity_recovery=c['recovery_pp'] is not None and c['recovery_pp']>=2.,RX_coverage=all(rr[str(v)]['count']>0 for v in d.FIT),RX_positive=sum(v['recovery_pp'] is not None and v['recovery_pp']>0 for v in rr.values())>=3,RX_protection=all(v['recovery_pp'] is not None and v['recovery_pp']>=-1. for v in rr.values()),day_protection=all(v['recovery_pp'] is not None and v['recovery_pp']>=0 for v in dd.values()));decision='PASS' if all(conditions.values()) else 'FAIL';assert conditions==r['qualification']['conditions'] and decision==r['qualification']['decision']==done['decision']
 decisions=len(raw['predictions'])+2*len(full);assert decisions==206640==r['source_decisions']==done['source_decisions'] and decisions+2==r['actual_NN_packets']==done['actual_NN_packets'] and len(full)==r['ordinary_generated_packets']==100800
 log=(OUT/'ordinary_PA.stdout.log').read_text(encoding='utf8');assert log.index('SMOKE PASS')<log.index('INPUT PASS') and sum(v.startswith('EPOCH ') for v in log.splitlines())==20 and 'COMPLETE ' in log and 'Traceback (most recent call last)' not in log
 output=dict(status='ANALYZED_SOURCE_ONLY_VERIFIED',run_id=d.RUN,decision=decision,conditions=conditions,raw_score=rawscore,epochs=epochs,pooled=total,triggered_total=triggered,changed_total=sum(changed),unique_PA_changed_physical_packets=int(np.array(changed).reshape(20,5040).any(0).sum()),subchains=counts,paired_source_cells_recounted=cells,all206640_default_head_predictions_and_logits_recounted=True,source_decisions=decisions,actual_NN_packets=decisions+2,ordinary_generated_packets=len(full),training_updates=0,head_fit_updates=0,model_state_unchanged=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,elapsed_seconds=r['elapsed_seconds'],full_stdout_lines=len(log.splitlines()),uncertainty='Repeated physical observations across20 native draws;no naive independent-packet statistical significance claim')
 d.write(OUT/'analysis_summary.json',output);print({k:output[k] for k in ('status','decision','conditions','pooled','triggered_total','changed_total','actual_NN_packets','elapsed_seconds')});return output
if __name__=='__main__':analyze()
