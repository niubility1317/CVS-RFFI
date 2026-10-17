"""Independent frozen-prediction recount;no Torch,IQ,CP,NN or truth-side access."""
import numpy as np
from . import design as d

def run(root):
 root=d.resolve(d.BASE) if root is None else __import__('pathlib').Path(root);result=d.read(root/'result.json');rows=d.read(root/'coupled_draws.json');arr=np.load(root/'source_predictions.npz',allow_pickle=False);y=arr['labels'];rx=arr['RX'];day=arr['day'];ids=arr['ids'].tolist()
 if len(y)!=6300 or len(set(ids))!=6300 or sorted(set(arr.files)-{'ids','labels','RX','day'})!=sorted(d.VIEWS):raise ValueError('Full fixed source views required')
 ncm=0;ntx=0;maxerror=0.;pairs={};pred={}
 def verify(expected,truth,decisions):
  nonlocal ncm,maxerror
  cm=np.zeros((6,6),dtype=np.int64)
  for a,b in zip(truth,decisions):cm[int(a),int(b)]+=1
  if cm.tolist()!=expected['confusion'] or expected['count']!=len(truth):raise ValueError('Confusion/count differs')
  if len(truth):
   den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0);err=max(abs(np.trace(cm)/len(truth)-expected['accuracy']),abs(f.mean()-expected['macro_F1']));maxerror=max(maxerror,float(err))
   if err>1e-12:raise ValueError('Accuracy/F1 recount differs')
  ncm+=1
 for view in d.VIEWS:
  a=arr[view]
  if a.shape!=(6300,6) or not np.isfinite(a).all():raise ValueError('Finite defaulthead logits required')
  p=a.argmax(1);pred[view]=p;s=result['scores'][view];verify(s,y,p)
  for dim,v in (('RX',rx),('day',day)):
   for value,expected in s['by_'+dim].items():mask=v==int(value);verify(expected,y[mask],p[mask])
  for label,expected in s['by_TX'].items():
   mask=y==int(label)
   if expected['count']!=int(mask.sum()) or abs(expected['accuracy']-np.mean(p[mask]==y[mask]))>1e-12:raise ValueError('TX recall differs')
   ntx+=1
  if view!='raw':
   scene,k=view.split('/');rr=rows[scene]
   if [r['id'] for r in rr]!=ids or [r['RX'] for r in rr]!=rx.tolist() or [r['day'] for r in rr]!=day.tolist():raise ValueError('Coupled source IDs/RX/day differ')
   for dim in ('state','lock'):
    v=np.asarray([r[dim] for r in rr])
    for value,expected in s['by_'+dim].items():mask=v==value;verify(expected,y[mask],p[mask])
 for scene in d.SCENES:
  rr=rows[scene];base=pred[scene+'/noEQ'];wrong=base!=y;variants={}
  for key in ('MMSE24','MMSE64'):
   p=pred[scene+'/'+key];variants[key]=dict(accuracy_delta_pp=float(100*np.mean((p==y).astype(float)-(base==y).astype(float))),repairs=int(np.sum((p==y)&wrong)),new_errors=int(np.sum((p!=y)&~wrong)))
   if any(r['variants'][key]['EQ']!=(r['lock']!='unlocked') or r['variants'][key]['noise_filter_max_error']>1e-13 or not r['same_H'] or not r['same_receiver_quality_and_compensation'] for r in rr):raise ValueError('Coupled physical or receiver dependency differs')
   if any(p[i]!=base[i] for i,r in enumerate(rr) if r['lock']=='unlocked'):raise ValueError('Unlocked predictions changed')
  active=np.asarray([r['variants']['MMSE24']['relative_IQ_change']>1e-4 for r in rr]);variants['original_errors']=int(wrong.sum());variants['original_errors_affected_share']=float(np.mean(active[wrong])) if wrong.any() else 0.;pairs[scene]=variants
 sub,mid,low=d.SCENES;delta=lambda s:pairs[s]['MMSE24']['accuracy_delta_pp'];conditions=dict(low_gain=delta(low)>=.5,mid_gain=delta(mid)>=.5,suburban_protection=delta(sub)>=-.25,all_RX_low_protection=all(100*(result['scores'][low+'/MMSE24']['by_RX'][r]['accuracy']-result['scores'][low+'/noEQ']['by_RX'][r]['accuracy'])>=-1. for r in result['scores'][low+'/noEQ']['by_RX']),low_original_errors_affected=pairs[low]['original_errors_affected_share']>=.20)
 decision='PASS_SOURCE_RECEIVER_VIEW_ONLY' if all(conditions.values()) else 'FAIL_FIXED_MMSE24_SOURCE_CANDIDATE_CLOSED'
 if conditions!=result['conditions'] or decision!=result['decision'] or result['actual_NN']!=63002 or result['training_updates'] or result['target_access']:raise ValueError('Decision or cost mismatch')
 return dict(status='VERIFIED',CM_recounts=ncm,TX_recounts=ntx,source_predictions=63000,max_metric_error=maxerror,conditions=conditions,decision=decision,paired_source_results=pairs,IQ=0,CP=0,NN=0,training_updates=0,target_access=False,As_training_gain_proved=False,joint_goal_complete=False)
