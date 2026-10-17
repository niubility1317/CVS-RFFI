"""Independent complete source-prediction recount;no Torch/IQ/CP/NN/target."""
import numpy as np
from . import design as d

def run(root):
 root=__import__('pathlib').Path(root);result=d.read(root/'result.json');rows=d.read(root/'coupled_draws.json');arr=np.load(root/'source_predictions.npz',allow_pickle=False);y=arr['labels'];rx=arr['RX'];day=arr['day'];ids=arr['ids'].tolist()
 if result['status']!='ANALYZED_SOURCE_ONLY' or len(y)!=6300 or len(set(ids))!=6300 or sorted(set(arr.files)-{'ids','labels','RX','day'})!=sorted(d.VIEWS):raise ValueError('Complete fixed source predictions required')
 ncm=ntx=0;maxerror=0.;pred={};pairs={}
 def verify(e,yy,pp):
  nonlocal ncm,maxerror
  cm=np.zeros((6,6),dtype=np.int64)
  for t,p in zip(yy,pp):cm[int(t),int(p)]+=1
  if cm.tolist()!=e['confusion'] or e['count']!=len(yy):raise ValueError('CM/count differs')
  if len(yy):
   den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0);err=max(abs(np.trace(cm)/len(yy)-e['accuracy']),abs(f.mean()-e['macro_F1']));maxerror=max(maxerror,float(err))
   if err>1e-12:raise ValueError('Accuracy/F1 differs')
  ncm+=1
 for view in d.VIEWS:
  z=arr[view]
  if z.shape!=(6300,6) or not np.isfinite(z).all():raise ValueError('Defaulthead finite logits required')
  p=z.argmax(1);pred[view]=p;s=result['scores'][view];verify(s,y,p)
  for dim,v in (('RX',rx),('day',day)):
   for key,e in s['by_'+dim].items():mask=v==int(key);verify(e,y[mask],p[mask])
  for label,e in s['by_TX'].items():
   mask=y==int(label)
   if int(mask.sum())!=e['count'] or abs(np.mean(p[mask]==y[mask])-e['accuracy'])>1e-12:raise ValueError('TX recall differs')
   ntx+=1
  if view!='raw':
   scene,key=view.split('/');rr=rows[scene]
   if [r['id'] for r in rr]!=ids or [r['RX'] for r in rr]!=rx.tolist() or [r['day'] for r in rr]!=day.tolist():raise ValueError('Source identities orroles differ')
   for dim in ('state','lock','nominal_lock'):
    group=np.asarray([r['variants'][key]['lock'] if dim=='lock' else r[dim] for r in rr])
    for k,e in s['by_'+dim].items():mask=group==k;verify(e,y[mask],p[mask])
 conditions={}
 for scene in d.SCENES:
  base=pred[scene+'/nominal'];wrong=base!=y;pr={}
  for key in d.VARIANTS[1:]:
   p=pred[scene+'/'+key];pr[key]=dict(repairs=int(np.sum((p==y)&wrong)),new_errors=int(np.sum((p!=y)&~wrong)),accuracy_delta_pp=float(100*np.mean((p==y).astype(float)-(base==y).astype(float))))
  pp=np.concatenate([pred[scene+'/'+k] for k in ('N0_half','N0_1p5')]);yy=np.tile(y,2);rr=np.tile(rx,2);dd=np.tile(day,2);e=result['scores'][scene+'/expected_mixture'];verify(e,yy,pp)
  for dim,v in (('RX',rr),('day',dd)):
   for k,ev in e['by_'+dim].items():mask=v==int(k);verify(ev,yy[mask],pp[mask])
  delta=100*(float(np.mean(pp==yy))-float(np.mean(base==y)))
  if abs(delta-.5*sum(pr[k]['accuracy_delta_pp'] for k in ('N0_half','N0_1p5')))>1e-12:raise ValueError('Single-draw expected risk differs')
  pr['expected_single_draw']=dict(accuracy_delta_pp=delta,expected_repairs=.5*sum(pr[k]['repairs'] for k in ('N0_half','N0_1p5')),expected_new_errors=.5*sum(pr[k]['new_errors'] for k in ('N0_half','N0_1p5')),not_logit_ensemble=True);pairs[scene]=pr
  for r in rows[scene]:
   if not r['same_H'] or not r['same_hardware']:raise ValueError('Latent differs')
   for k,v in r['variants'].items():
    if v['N0_factor']!=d.FACTORS[k] or (k.endswith('fixedQ') and (v['fixed_quality_signal_error']>1e-13 or v['fixed_quality_noise_error']>1e-13 or abs(v['quality']-r['nominal_quality'])>1e-10)):raise ValueError('Noise physicalcontrol differs')
 sub,mid,low=d.SCENES;delta=lambda s:pairs[s]['expected_single_draw']['accuracy_delta_pp'];conditions=dict(low_gain=delta(low)>=.5,mid_gain=delta(mid)>=.5,suburban_protection=delta(sub)>=-.25,all_RX_low_protection=all(100*(result['scores'][low+'/expected_mixture']['by_RX'][str(r)]['accuracy']-result['scores'][low+'/nominal']['by_RX'][str(r)]['accuracy'])>=-1. for r in d.FIT))
 decision='PASS_SOURCE_N0_OBSERVATION_LAW_ONLY' if all(conditions.values()) else 'FAIL_FIXED_N0_PRIOR_SOURCE_CANDIDATE_CLOSED'
 if conditions!=result['conditions'] or decision!=result['decision'] or result['actual_NN']!=100802 or result['training_updates'] or result['target_access']:raise ValueError('Decision or cost differs')
 return dict(status='VERIFIED',CM_recounts=ncm,TX_recounts=ntx,source_predictions=100800,max_metric_error=maxerror,conditions=conditions,decision=decision,paired_source_results=pairs,IQ=0,CP=0,NN=0,training_updates=0,target_access=False,Ag_efficacy_proved=False,As_training_gain_proved=False,joint_goal_complete=False)
