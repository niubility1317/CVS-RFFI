import gzip,json,statistics as st,time
import torch
from . import design as d,model as api
from .objective import validate_layout,variance_terms
from experiments.cvs_source_layer_order.probe import gradient_stats
from experiments.cvs_signature_stats50.audit import run as shared

def run(*,design=None,build_fn=None):
 d=design or globals()['d']
 torch.set_num_threads(2);started=time.time();base=shared(design=d,build_fn=build_fn or api.build);observations=[]
 for row in d.training_rows():
  c=d.config(row);p=d.BASE/row['row_id']/'source';done=d.read(p/'backbone_fishr_training_observation.json');saved=d.read(p/'backbone_fishr_gradient_records.json');batches=saved['batch_records'];count=0;matrix_count=0
  if done['status']!='VERIFIED' or done['steps']!=11100 or done['extra_batches']!=650 or done['extra_source_packet_forwards']!=117000 or done['U_packets']!=963900 or done['gradient_epochs']!=[1,17,33,50] or done['gradient_batches']!=52 or len(batches)!=52 or done['head_fit_updates'] or done['U_truth_read']:raise ValueError('Actual backbone-gradient budget differs')
  for epoch in d.DIAGNOSTIC_EPOCHS:
   values=torch.load(p/f'backbone_fishr_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);reference=[v for v in batches if v['epoch']==epoch]
   if len(values)!=13 or len(reference)!=13:raise ValueError('Actual gradient epoch incomplete')
   for j,(a,b) in enumerate(zip(values,reference),1):
    if a['step']!=(epoch-1)*222+j*16 or a['epoch']!=epoch or a['step']!=b['step']:raise ValueError('Actual gradient identity differs')
    validate_layout(a['y'],a['rx'],a['day'],a['source_physical_ids']);recomputed={key:[] for key in d.LEVELS}
    for k,G in enumerate(a['parameter_gradient_matrices']):
     if G.shape!=(180,[3456,6912][k]):raise ValueError('Saved actual per-packet derivative object differs')
     losses,_=variance_terms(G,a['y'],a['rx'],a['day']);matrix_count+=1
     for key in d.LEVELS:recomputed[key].append(float(losses[key]))
    for key in d.LEVELS:
     value=sum(recomputed[key])/2
     if abs(value-b['raw_losses'][key])>1e-8*max(1.,abs(value)):raise ValueError('Saved derivative variance objective differs')
    for k,name in enumerate(d.PARAMETERS):
     if b['layers'][k]['parameter']!=name:raise ValueError('Observed derivative object differs')
     for key in (*d.LEVELS,'effective'):
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved second-order loss gradient differs')
     if (epoch==1 or c['backbone_fishr']['mode']=='ZERO') and b['layers'][k]['effective']['auxiliary_norm']!=0:raise ValueError('Zero/warmup effective gradient differs')
     count+=1
  summaries=[]
  for epoch in d.DIAGNOSTIC_EPOCHS:
   rr=[r for r in batches if r['epoch']==epoch]
   for k,name in enumerate(d.PARAMETERS):
    item=dict(epoch=epoch,parameter=name,batches=13)
    for key in (*d.LEVELS,'effective'):
     vv=[r['layers'][k][key] for r in rr];cc=[x['cosine'] for x in vv if x['cosine'] is not None];item[key]=dict(nonzero_batches=sum(x['auxiliary_norm']>1e-12 for x in vv),mean_ratio=st.mean(x['auxiliary_over_CE'] or 0 for x in vv),mean_cosine=st.mean(cc) if cc else None)
    summaries.append(item)
  local_count=0;per_epoch={};accepted=0
  with gzip.open(p/'backbone_fishr_steps.jsonl.gz','rt') as auxiliary,gzip.open(p/'step_metrics.jsonl.gz','rt') as native:
   for index,line in enumerate(native,1):
    v=json.loads(line);epoch=(index-1)//222+1;batch=(index-1)%222+1;metrics=v['metrics'];accepted+=int(metrics['train/pseudo_selected'])
    if batch%16:
     if metrics.get('backbone_fishr_extra_forward')!=0:raise ValueError('Unscheduled auxiliary forward')
     continue
    w=json.loads(next(auxiliary));local_count+=1;mode=c['backbone_fishr']['mode'];weight=0. if mode=='ZERO' else .01*min(1.,max(0.,(epoch-5)/10));selected=w['raw_losses']['RAW' if mode=='ZERO' else mode]
    if w['step']!=index or w['cohort_rank']!=(local_count-1)%35 or w['mode']!=mode or w['coefficient']!=weight or w['extra_source_packets']!=180 or metrics.get('backbone_fishr_extra_forward')!=1 or w['detail']['packet_forwards']!=180:raise ValueError('Applied source exposure differs')
    if abs(w['effective_loss']-weight*selected)>2e-6 or abs(metrics['backbone_fishr_effective']-w['effective_loss'])>2e-6:raise ValueError('Native applied regularizer differs')
    per_epoch.setdefault(epoch,[]).append(w)
   if next(auxiliary,None) is not None:raise ValueError('Extra source stream longer than native')
  if local_count!=650 or count!=156 or matrix_count!=104 or accepted!=done['U_accepted']:raise ValueError('Independent complete budget differs')
  curve=[]
  for epoch,rr in per_epoch.items():
   if len(rr)!=13:raise ValueError('Epoch auxiliary exposure differs')
   curve.append(dict(epoch=epoch,batches=13,coefficient=rr[0]['coefficient'],effective_loss=st.mean(x['effective_loss'] for x in rr),auxiliary_seconds=sum(x['detail']['seconds'] for x in rr),raw_losses={key:st.mean(x['raw_losses'][key] for x in rr) for key in d.LEVELS}))
  observations.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],observation=done,independent_auxiliary_steps=local_count,independent_gradient_layer_cohorts=count,independent_parameter_gradient_matrices=matrix_count,gradient_summary=summaries,auxiliary_epoch_curve=curve,saved_gradient_recount='VERIFIED',scope='Actual backbone-gradient regularizer vs separate mainL CE; no extra head fitting/cumulativeAdamW/newTX claim'))
 return dict(status='VERIFIED',audit=base,actual_training_observations=observations,auxiliary_steps_recounted=650*len(observations),gradient_layer_cohorts_recounted=156*len(observations),parameter_gradient_matrices_recounted=104*len(observations),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True,seconds=time.time()-started)
