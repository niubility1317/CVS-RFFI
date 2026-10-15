import gzip,json,math,statistics as st
import torch
from . import design as d
from experiments.cvs_source_layer_order.probe import gradient_stats
def loss_close(a,b):return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-6,abs_tol=3e-6)
def recount(row):
 c=d.config(row);p=d.BASE/row['row_id']/'source';observed=d.read(p/'receiver_risk_training_observation.json');alpha=c['receiver_risk']['coefficient']
 if observed['status']!='VERIFIED' or observed['steps']!=11100 or observed['U_packets']!=963900 or observed['coefficient']!=alpha or not observed['U_loss_returned_unchanged']:raise ValueError('VREXcompletedexposure differs')
 grouped={};packets=steps=0
 with gzip.open(p/'receiver_risk_steps.jsonl.gz','rt',encoding='utf8') as f,gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf8') as n:
  for steps,line in enumerate(f,1):
   r=json.loads(line);native=json.loads(next(n));counts=torch.tensor(r['group_counts'],dtype=torch.float64);means=torch.tensor(r['group_mean_CE'],dtype=torch.float64);present=counts>0
   if r['step']!=steps or native['step']!=steps or r['epoch']!=(steps-1)//222+1 or r['mode']!=c['receiver_risk']['mode'] or counts.shape!=(5,) or means.shape!=(5,) or int(counts.sum())!=r['source_packets'] or not present.any() or (counts<0).any() or (means<0).any() or not torch.isfinite(means).all() or r['coefficient']!=alpha:raise ValueError('SavedVREXidentity/counts differ')
   balanced=float(means[present].mean());var=float(means[present].var(unbiased=False));raw=float((counts*means).sum()/counts.sum());selected=balanced+alpha*var
   for a,b in [(balanced,r['balanced_CE']),(var,r['risk_variance']),(raw,r['original_CE']),(selected,r['selected_CE']),(selected,native['metrics']['train/loss_tx_labeled'])]:
    if not loss_close(a,b):raise ValueError('ActualnativeVREXlossrecountdiffers')
   grouped.setdefault(r['epoch'],[]).append(r);packets+=r['source_packets']
  if next(n,None) is not None or steps!=11100:raise ValueError('NativeVREXstreamincomplete')
 curves=[]
 for epoch,rr in grouped.items():
  if len(rr)!=222:raise ValueError('VREXepochincomplete')
  counts=[sum(r['group_counts'][k] for r in rr) for k in range(5)]
  if not all(counts):raise ValueError('Sourcereceivernotobserved')
  means=[sum(r['group_counts'][k]*r['group_mean_CE'][k] for r in rr)/counts[k] for k in range(5)]
  curves.append(dict(epoch=epoch,group_counts=counts,group_CE=means,unweighted_CE=sum(a*b for a,b in zip(counts,means))/sum(counts),worst_RX_CE=max(means),mean_batch_risk_variance=st.mean(r['risk_variance'] for r in rr)))
 records=d.read(p/'receiver_risk_gradient_records.json')['batch_records'];layers=0
 for epoch in d.DIAGNOSTIC_EPOCHS:
  raw=torch.load(p/f'receiver_risk_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);expected=[r for r in records if r['epoch']==epoch]
  if len(raw)!=13 or len(expected)!=13:raise ValueError('VREXderivativeperiodincomplete')
  for j,(a,b) in enumerate(zip(raw,expected),1):
   if a['step']!=(epoch-1)*222+j*16 or a['step']!=b['step']:raise ValueError('Derivativeidentitydiffers')
   for k,name in enumerate([*d.PARAMETERS,'existing_default_W']):
    if b['layers'][k]['parameter']!=name:raise ValueError('Derivativeobjectdiffers')
    for key in ('selected','variance'):
     if gradient_stats(a['vectors']['balanced'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('SavedVREXderivativerecountdiffers')
    bal,aux,sel=(a['vectors'][key][k] for key in ('balanced','variance','selected'))
    if not torch.allclose(sel,bal+alpha*aux,atol=1e-6,rtol=1e-5) or alpha==0 and not torch.equal(sel,bal):raise ValueError('Actualpenaltygradientnotconsumed')
    layers+=1
 if layers!=208 or len(records)!=52:raise ValueError('VREXgradientbudgetdiffers')
 last=curves[-5:]
 return dict(status='VERIFIED',row_id=row['row_id'],arm=row['arm'],steps=steps,actual_L_packet_observations=packets,curves=curves,last5_unweighted_CE=st.mean(r['unweighted_CE'] for r in last),last5_worst_RX_CE=max(st.mean(r['group_CE'][k] for r in last) for k in range(5)),last5_mean_batch_risk_variance=st.mean(r['mean_batch_risk_variance'] for r in last),native_U_objective_unchanged=True,body_signal=all(any(r['epoch']==ep and r['layers'][k]['variance']['auxiliary_norm']>1e-12 for r in records) for ep in (17,33,50) for k in range(3)),gradient_layer_records=layers,saved_vectors=layers*3,extra_model_forwards=0,head_fit_updates=0,query_read=False)
