import gzip,json,statistics as st,time
import torch
from . import design as d,model as api
from .observer import gradient_stats
from experiments.cvs_signature_stats50.audit import run as shared

def run():
 torch.set_num_threads(2);start=time.time();audit=shared(design=d,build_fn=api.build);observations=[]
 for row in d.training_rows():
  p=d.BASE/row['row_id']/'source';v=d.read(p/'ssl_training_observation.json');saved=d.read(p/'ssl_gradient_records.json');sampling=d.read(p/'source_sampling_epoch_diagnostic.json')['records'];records=saved['batch_records'];count=0
  if v['status']!='VERIFIED' or v['steps']!=11100 or v['U_packets']!=963900 or v['gradient_epochs']!=[33,34,50] or v['gradient_batches']!=666 or len(records)!=666 or v['head_fit_updates'] or v['U_truth_read'] or v['extra_model_forwards'] or not v['loss_returned_unchanged']:raise ValueError('Actual fixedSIG SSL budget differs')
  for epoch in d.DIAGNOSTIC_EPOCHS:
   raw=torch.load(p/f'ssl_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);rr=[r for r in records if r['epoch']==epoch]
   if len(raw)!=222 or len(rr)!=222:raise ValueError('Actual SSL gradient epoch incomplete')
   for j,(a,b) in enumerate(zip(raw,rr),1):
    if a['step']!=(epoch-1)*222+j or a['step']!=b['step'] or a['epoch']!=epoch:raise ValueError('Actual SSL gradient step identity differs')
    for k,name in enumerate(d.PARAMETERS):
     if b['layers'][k]['parameter']!=name:raise ValueError('Actual SSL parameter object differs')
     for key in ('weighted_U_CE','weighted_U_CE_entropy'):
      if gradient_stats(a['vectors']['main_L_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved SSL gradient arithmetic differs')
      if epoch==33 and b['layers'][k][key]['auxiliary_norm']!=0:raise ValueError('Unexpected U signal before pseudo stage')
     count+=1
  if len(sampling)!=50 or [r['epoch'] for r in sampling]!=list(range(1,51)):raise ValueError('Complete sampler epoch history required')
  for r in sampling:
   if r['epoch']<=33:
    if r['sampler'] is not None or r['U_IQ_fetched']:raise ValueError('Label epoch fetched U')
   else:
    s=r['sampler']
    if not r['U_IQ_fetched'] or s['delivered_count']!=56700 or s['generated_count']!=56700 or not s['physical_packets_once'] or s['U_truth_read'] or s['IQ_read_for_order']:raise ValueError('Actual hiddenU once-per-epoch contract differs')
  summaries=[]
  for ep in d.DIAGNOSTIC_EPOCHS:
   rr=[r for r in records if r['epoch']==ep]
   for k,name in enumerate(d.PARAMETERS):
    item=dict(epoch=ep,parameter=name,batches=222)
    for key in ('weighted_U_CE','weighted_U_CE_entropy'):
     vv=[r['layers'][k][key] for r in rr];cs=[r['cosine'] for r in vv if r['cosine'] is not None];item[key]=dict(nonzero_batches=sum(r['auxiliary_norm']>1e-12 for r in vv),mean_ratio=st.mean(r['auxiliary_over_CE'] or 0 for r in vv),mean_cosine=st.mean(cs) if cs else None)
    summaries.append(item)
  if count!=1998:raise ValueError('Actual layer/batch gradient count differs')
  observations.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],observation=v,independent_layer_gradient_records=count,sampler_epoch_curve=sampling,gradient_summary=summaries,saved_gradient_recount='VERIFIED',scope='Actual L versus U separate batches, not samepacket or cumulativeAdamW contribution.'))
 return dict(status='VERIFIED',audit=audit,actual_training_observations=observations,gradient_layer_records_recounted=1998*len(observations),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True,seconds=time.time()-start)
