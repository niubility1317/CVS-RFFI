import gzip,json,statistics as st
import torch
from . import design as d,model as api
from .objective import coefficient
from experiments.cvs_sigtime_pair50.audit import run as shared
from experiments.cvs_source_layer_order.probe import gradient_stats
def run(*,design=None,build_fn=None):
 d=design or globals()['d']
 base=shared(design=d,build_fn=build_fn or api.build);values=[]
 for row in d.training_rows():
  p=d.BASE/row['row_id']/'source';obs=d.read(p/'feature_mean_teacher_training_observation.json');records=d.read(p/'feature_mean_teacher_gradient_records.json')['batch_records']
  with gzip.open(p/'feature_mean_teacher_steps.jsonl.gz','rt',encoding='utf8') as stream:steps=[json.loads(t) for t in stream]
  if len(steps)!=11100 or len(records)!=666 or obs['gradient_epochs']!=[33,34,50] or obs['U_feature_packets']!=963900 or obs['extra_model_forwards'] or obs['head_fit_updates'] or obs['auxiliary_classW_dependency']:raise ValueError('Complete feature observations differ')
  for j,v in enumerate(steps,1):
   ep=(j-1)//222+1;batch=(j-1)%222+1;n=0 if ep<34 else 124 if batch==222 else 256
   if (v['step'],v['epoch'],v['batch'],v['U_packets'],v['coefficient'])!=(j,ep,batch,n,coefficient(ep,obs['mode'])) or abs(v['effective_loss']-v['coefficient']*v['feature_loss'])>1e-8:raise ValueError('Actual allU feature schedule differs')
  count=0;summaries=[]
  for ep in d.DIAGNOSTIC_EPOCHS:
   raw=torch.load(p/f'feature_mean_teacher_gradients_e{ep}.pt',map_location='cpu',weights_only=False);rr=[r for r in records if r['epoch']==ep]
   if len(raw)!=222 or len(rr)!=222:raise ValueError('Feature gradient period incomplete')
   for j,(a,b) in enumerate(zip(raw,rr),1):
    if a['step']!=(ep-1)*222+j or a['step']!=b['step'] or a['epoch']!=ep:raise ValueError('Feature gradient step differs')
    for k,pname in enumerate(d.PARAMETERS):
     if b['layers'][k]['parameter']!=pname:raise ValueError('Actual feature gradient object differs')
     for key in ('feature','effective'):
      if gradient_stats(a['vectors']['main_L_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Independent saved feature gradient recount differs')
     count+=1
   for k,pname in enumerate(d.PARAMETERS):
    item=dict(epoch=ep,parameter=pname,batches=222)
    for key in ('feature','effective'):
     z=[r['layers'][k][key] for r in rr];cs=[r['cosine'] for r in z if r['cosine'] is not None];item[key]=dict(nonzero_batches=sum(r['auxiliary_norm']>1e-12 for r in z),mean_ratio=st.mean(r['auxiliary_over_CE'] or 0 for r in z),mean_cosine=st.mean(cs) if cs else None)
    summaries.append(item)
  if count!=1998:raise ValueError('Feature gradient layer count differs')
  values.append(dict(row_id=row['row_id'],observation=obs,gradient_summary=summaries,saved_gradient_recount='VERIFIED',independent_layer_gradient_records=count))
 return dict(status='VERIFIED',audit=base,actual_feature_consistency=values,feature_layer_records_recounted=1998*len(values),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True)
