"""Independent recount from completed source decisions;no IQ/model/runtime imports."""
import argparse,json
from pathlib import Path
import numpy as np

def recount(root):
 root=Path(root);result=json.loads((root/'result.json').read_text());done=json.loads((root/'completion.json').read_text())
 if done['status']!='ANALYZED_SOURCE_ONLY' or done['actual_NN']!=31502 or done['target_access'] or result['training_updates'] or not result['all_model_state_unchanged']:raise ValueError('Completed fixed source guard required')
 checked=0;max_error=0.;repaired={};strata=[]
 with np.load(root/'source_predictions.npz',allow_pickle=False) as a:
  names=('raw','white_draw0','color_draw0','white_draw1','color_draw1');truth=a['labels'];rx=a['RX'];day=a['day'];ids=a['ids']
  if len(ids)!=6300 or len(set(ids.tolist()))!=6300 or set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3} or set(truth.tolist())!=set(range(6)):raise ValueError('Full original source counts differ')
  raw=a['raw'].argmax(1);rawgood=raw==truth
  def cm(y,p):
   table=np.zeros((6,6),dtype=int);np.add.at(table,(y,p),1);den=table.sum(0)+table.sum(1)
   f=np.divide(2*np.diag(table),den,out=np.zeros(6),where=den>0)
   return table,float(np.trace(table)/len(y)),float(f.mean())
  for name in names:
   z=a[name]
   if z.shape!=(6300,6) or not np.isfinite(z).all():raise ValueError('Finite original defaulthead logits required')
   pred=z.argmax(1);q=result['scores'][name]
   selections=[('ALL','ALL',np.ones(6300,dtype=bool),q)]+[(dim,str(k),values==k,q['by_'+dim][str(k)]) for dim,values,keys in [('RX',rx,[1,3,4,6,8]),('day',day,[1,2,3])] for k in keys]
   for dim,k,mask,reported in selections:
    table,acc,f1=cm(truth[mask],pred[mask]);checked+=1
    if table.tolist()!=reported['confusion'] or len(truth[mask])!=reported['count']:raise ValueError('Source confusion/count mismatch')
    err=max(abs(acc-reported['accuracy']),abs(f1-reported['macro_F1']));max_error=max(max_error,err)
    if err>1e-12:raise ValueError('Source metric mismatch')
    strata.append(dict(view=name,dimension=dim,stratum=k,count=int(mask.sum()),accuracy=acc,macro_F1=f1))
   for tx in range(6):
    value=float(np.mean(pred[truth==tx]==tx));reported=q['by_TX'][str(tx)]
    if reported['count']!=1050 or abs(value-reported['accuracy'])>1e-12:raise ValueError('Full TX recall mismatch')
   retention=float(np.mean((pred==truth)[rawgood]))
   if abs(retention-q['raw_correct_retention'])>1e-12:raise ValueError('Source retention mismatch')
   repaired[name]=dict(repairs=int(np.sum((raw!=truth)&(pred==truth))),new_errors=int(np.sum(rawgood&(pred!=truth))),raw_correct_retention=retention)
  support={str(k):dict(count=int(np.sum(rx==k)),median=float(np.median(a['source_inband'][rx==k])),q10=float(np.quantile(a['source_inband'][rx==k],.1))) for k in (1,3,4,6,8)}
  if support!=result['source_support']:raise ValueError('Source spectral-prior summary mismatch')
  conditions={'all_RX_source_spectral_support':all(v['median']>=.90 for v in support.values())}
  noise=json.loads((root/'noise_draws.json').read_text());power={}
  for draw in (0,1):
   q=result['scores']['color_draw'+str(draw)];r=result['scores']['raw'];rows=noise[str(draw)];changes=[v for row in rows for v in row['color_relative_IQ_change']]
   if sum(row['count'] for row in rows)!=6300 or len(changes)!=6300:raise ValueError('All draw packets required')
   snr=np.asarray([v for row in rows for v in row['nominal_SNR_dB']]);wp=np.asarray([v for row in rows for v in row['white_noise_power']]);cp=np.asarray([v for row in rows for v in row['color_noise_power']]);expected=np.asarray([v for row in rows for v in row['expected_noise_power']])
   if snr.min()<20 or snr.max()>36 or not np.isfinite(expected).all() or expected.min()<=0:raise ValueError('Fixed finite SNR/power required')
   power[str(draw)]=dict(nominal_SNR_min=float(snr.min()),nominal_SNR_max=float(snr.max()),white_power_over_expected=float(np.mean(wp/expected)),color_power_over_expected=float(np.mean(cp/expected)),white_noise_power_mean=float(wp.mean()),color_noise_power_mean=float(cp.mean()),IQ_change_median=float(np.median(changes)),IQ_change_max=max(changes))
   conditions['draw'+str(draw)]=dict(accuracy=100*(q['accuracy']-r['accuracy'])>=-.5,macro_F1=100*(q['macro_F1']-r['macro_F1'])>=-.5,retention=q['raw_correct_retention']>=.99,all_groups=all(100*(q['by_'+dimension][key]['accuracy']-r['by_'+dimension][key]['accuracy'])>=-1. for dimension in ('RX','TX','day') for key in q['by_'+dimension]),effective_nonexcessive_noise=float(np.median(changes))>=.01 and max(changes)<=.15)
  passed=conditions['all_RX_source_spectral_support'] and all(all(conditions['draw'+str(k)].values()) for k in (0,1))
  decision='PASS_SOURCE_IDENTITY_AND_COARSE_SPECTRAL_PRIOR_ONLY' if passed else 'FAIL_FIXED_BAND_NOISE_SOURCE_GUARD_CLOSED'
  if conditions!=result['conditions'] or decision!=result['decision']:raise ValueError('Fixed source decision mismatch')
 return dict(status='VERIFIED',source_CM_recounts=checked,TX_accuracy_recounts=30,source_predictions=31500,max_metric_error=max_error,source_decision=decision,conditions=conditions,repairs_and_new_errors=repaired,source_support=support,power=power,strata=strata,IQ=0,CP=0,NN=0,raw_target_truth=0,updates=0,Ag_efficacy_proved=False,joint_goal_complete=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_root');p.add_argument('output');a=p.parse_args();v=recount(a.run_root);out=Path(a.output)
 with out.open('x',encoding='utf8') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v[k] for k in ('status','source_CM_recounts','TX_accuracy_recounts','source_decision','IQ','CP','NN','updates')}))
