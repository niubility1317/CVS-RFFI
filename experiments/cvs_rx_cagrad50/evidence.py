import gzip,json,math,statistics as st
import numpy as np
import torch
from . import design as d
from experiments.cvs_source_layer_order.probe import gradient_stats
def recount(row,*,design=None):
 dd=design or d;c=dd.config(row);p=dd.BASE/row['row_id']/'source';obs=dd.read(p/'receiver_risk_training_observation.json')
 if obs['status']!='VERIFIED' or obs['steps']!=11100 or obs['U_packets']!=963900 or not obs['U_loss_returned_unchanged']:raise ValueError('NativeU/exposure differs')
 grouped={};fallback=changed=0;worst_gain=[];packets=0
 with gzip.open(p/'receiver_risk_steps.jsonl.gz','rt',encoding='utf8') as f,gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf8') as nf:
  for step,line in enumerate(f,1):
   r=json.loads(line);native=json.loads(next(nf));A=np.asarray(r['gram']);b=np.asarray(r['weights']);a=np.asarray(r['coefficients']);counts=np.asarray(r['group_counts']);means=np.asarray(r['group_mean_CE']);n=float(b@A@b)
   if r['step']!=step or native['step']!=step or r['epoch']!=(step-1)//222+1 or r['mode']!=c['receiver_gradient']['mode'] or len(r['groups'])!=len(counts) or not set(r['groups'])<=set(dd.RX) or int(counts.sum())!=r['source_packets'] or (counts<=0).any() or not np.isfinite(A).all() or not np.allclose(b,counts/counts.sum(),atol=1e-7):raise ValueError('Actual step/Gram/groups differ')
   actual=float(counts@means/counts.sum())
   if not all(math.isclose(actual,v,abs_tol=3e-6,rel_tol=1e-6) for v in [r['original_CE'],r['selected_CE'],native['metrics']['train/loss_tx_labeled']]):raise ValueError('Native CE scalar differs')
   if n>1e-24:
    norm=float(np.sqrt(max(a@A@a,0.)/n));trust=float(np.sqrt(max((a-b)@A@(a-b),0.)/n));gain=float((A@a).min()-r['norm_control_scale']*(A@b).min());worst_gain.append(gain/n)
    if not math.isclose(norm,r['norm_control_scale'],abs_tol=1e-6,rel_tol=1e-5) or trust>.50002 or gain/n< -1e-4:raise ValueError('Trustregion/worstlocalgain formula differs')
   fallback+=bool(r['fallback']);changed+=r['direction_change']>1e-6;grouped.setdefault(r['epoch'],[]).append(r);packets+=r['source_packets']
  if next(nf,None) is not None or step!=11100:raise ValueError('Step streams incomplete')
 curves=[]
 for ep,rr in grouped.items():
  if len(rr)!=222:raise ValueError('Epoch incomplete')
  curves.append(dict(epoch=ep,original_L_CE=st.mean(v['original_CE'] for v in rr),solver_fallbacks=sum(v['fallback'] for v in rr),mean_norm_ratio=st.mean(v['norm_ratio'] for v in rr),mean_direction_change=st.mean(v['direction_change'] for v in rr)))
 records=dd.read(p/'receiver_risk_gradient_records.json')['batch_records'];layers=0;body_signal=True
 for ep in dd.DIAGNOSTIC_EPOCHS:
  raw=torch.load(p/f'receiver_risk_gradients_e{ep}.pt',map_location='cpu',weights_only=False);expected=[r for r in records if r['epoch']==ep]
  if len(raw)!=13 or len(expected)!=13:raise ValueError('Gradientperiod incomplete')
  for j,(a,b) in enumerate(zip(raw,expected),1):
   if a['step']!=(ep-1)*222+j*16 or a['step']!=b['step']:raise ValueError('Gradient identity differs')
   for k,name in enumerate([*dd.PARAMETERS,'existing_default_W']):
    if b['layers'][k]['parameter']!=name:raise ValueError('Gradientobject differs')
    for key in ('selected','delta'):
     if gradient_stats(a['vectors']['original'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Savedgradient statistics differ')
    orig,sel,delta,target=(a['vectors'][key][k] for key in ('original','selected','delta','target'))
    if not torch.allclose(sel,orig+delta,atol=1e-6,rtol=1e-5) or not torch.allclose(sel,target,atol=1e-6,rtol=1e-5):raise ValueError('Actualselected vector not consumed')
    if k==3 and torch.count_nonzero(delta):raise ValueError('DefaultW received correction')
    if k<3 and delta.norm()<=1e-12:body_signal=False
    layers+=1
 if layers!=208:raise ValueError('Gradient count differs')
 return dict(status='VERIFIED',row_id=row['row_id'],steps=step,actual_L_packet_observations=packets,curves=curves,last5_unweighted_CE=st.mean(v['original_L_CE'] for v in curves[-5:]),solver_fallbacks=fallback,solver_fallback_fraction=fallback/step,direction_changed_steps=changed,min_relative_worst_local_gain_vs_norm_control=min(worst_gain),body_signal=body_signal,native_U_objective_unchanged=True,gradient_layer_records=layers,saved_vectors=layers*4,head_fit_updates=0,query_read=False)
