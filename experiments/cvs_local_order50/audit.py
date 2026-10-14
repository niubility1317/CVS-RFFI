import gzip,json,math,statistics,time
import torch
from . import design as default_design,model as api
from .observer import gradient_stats
from experiments.cvs_source_layer_order.design import PARAMETERS
from experiments.cvs_signature_stats50.audit import run as shared

def run(*,design=None,build_fn=None):
 d=design or default_design;torch.set_num_threads(2);started=time.time();audit=shared(design=d,build_fn=build_fn or api.build);observations=[]
 for row in d.training_rows():
  c=d.config(row);p=d.BASE/row['row_id']/'source';done=d.read(p/'local_training_observation.json');saved=d.read(p/'local_gradient_records.json');batches=saved['batch_records']
  if done['status']!='VERIFIED' or done['steps']!=11100 or done['extra_batches']!=2750 or done['extra_source_packet_forwards']!=247500 or done['U_packets']!=963900 or done['gradient_epochs']!=[1,17,33,50] or done['gradient_batches']!=220 or done['layer_gradient_records']!=660 or done['head_fit_updates'] or done['U_truth_read'] or saved['epochs']!=[1,17,33,50] or len(batches)!=220:raise ValueError('Actual local source training budget differs')
  gradient_count=0
  for epoch in (1,17,33,50):
   values=torch.load(p/f'local_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);reference=[r for r in batches if r['epoch']==epoch]
   if len(values)!=55 or len(reference)!=55:raise ValueError('Saved actual gradient epoch incomplete')
   for j,(a,b) in enumerate(zip(values,reference),1):
    if a['step']!=(epoch-1)*222+j*4 or a['epoch']!=epoch or a['step']!=b['step']:raise ValueError('Actual gradient step identity differs')
    for k,name in enumerate(PARAMETERS):
     if b['layers'][k]['parameter']!=name:raise ValueError('Actual parameter object differs')
     for mode,key in [('raw_ORDER','ORDER'),('raw_SUPCON','SUPCON'),('effective','effective')]:
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][mode][k])!=b['layers'][k][key]:raise ValueError('Independent saved gradient arithmetic differs')
     if (epoch==1 or c['local_order']['mode']=='ZERO') and b['layers'][k]['effective']['auxiliary_norm']!=0:raise ValueError('Zero/warmup auxiliary gradient differs')
     if k==2 and any(b['layers'][k][key]['auxiliary_norm']!=0 for key in ('ORDER','SUPCON','effective')):raise ValueError('Local auxiliary unexpectedly enters t3')
     gradient_count+=1
  summaries=[]
  for epoch in (1,17,33,50):
   rr=[r for r in batches if r['epoch']==epoch]
   for k,name in enumerate(PARAMETERS):
    record=dict(epoch=epoch,parameter=name,batches=55)
    for mode in ('ORDER','SUPCON','effective'):
     vals=[r['layers'][k][mode] for r in rr];cos=[r['cosine'] for r in vals if r['cosine'] is not None];record[mode]=dict(nonzero_batches=sum(r['auxiliary_norm']>1e-12 for r in vals),mean_aux_norm=statistics.mean(r['auxiliary_norm'] for r in vals),mean_ratio=statistics.mean(r['auxiliary_over_CE'] or 0 for r in vals),mean_cosine=statistics.mean(cos) if cos else None)
    summaries.append(record)
  local_steps=0;epoch_summary=[];per_epoch={};U_accepted=0
  with gzip.open(p/'local_order_steps.jsonl.gz','rt') as local,gzip.open(p/'step_metrics.jsonl.gz','rt') as native:
   for index,line in enumerate(native,1):
    v=json.loads(line);epoch=(index-1)//222+1;batch=(index-1)%222+1;metrics=v['metrics'];U_accepted+=int(metrics['train/pseudo_selected'])
    if batch%4:
     if metrics.get('local_order_extra_forward')!=0:raise ValueError('Unexpected extra forward on unscheduled batch')
     continue
    w=json.loads(next(local));local_steps+=1;ramp=min(1.,max(0.,(epoch-5)/10));coefficient=0. if c['local_order']['mode']=='ZERO' else .1*ramp
    if w['step']!=index or w['epoch']!=epoch or w['batch']!=batch or w['cohort_rank']!=(local_steps-1)%70 or w['mode']!=c['local_order']['mode'] or w['coefficient']!=coefficient or w['extra_source_packets']!=90 or not w['source_only'] or metrics.get('local_order_extra_forward')!=1:raise ValueError('Native/source relation exposure differs')
    selected=w['raw_supcon'] if c['local_order']['mode']=='SUPCON' else w['raw_order']
    if abs(w['effective_loss']-coefficient*selected)>2e-6 or abs(metrics['local_order_effective']-w['effective_loss'])>2e-6:raise ValueError('Actual applied source objective differs')
    group=per_epoch.setdefault(epoch,[]);group.append(w)
   if next(local,None) is not None:raise ValueError('Extra source stream longer than native budget')
  if local_steps!=2750 or gradient_count!=660 or U_accepted!=done['U_accepted']:raise ValueError('Independent auxiliary/gradient budget differs')
  for epoch,rr in per_epoch.items():
   if len(rr)!=55:raise ValueError('Actual epoch relation exposure differs')
   record=dict(epoch=epoch,batches=55,extra_source_packets=4950,raw_ORDER=statistics.mean(r['raw_order'] for r in rr),raw_SUPCON=statistics.mean(r['raw_supcon'] for r in rr),effective_loss=statistics.mean(r['effective_loss'] for r in rr),coefficient=rr[0]['coefficient'],layers=[])
   for name in ('t1','t2'):
    vals=[r['layers'][name] for r in rr]
    if any(r['triplets']!=1800 or r['cohort_packets']!=90 for r in vals):raise ValueError('Actual relation cardinality differs')
    record['layers'].append(dict(level=name,relations=99000,unsatisfied=sum(r['active'] for r in vals),wrong_order=sum(r['wrong_order'] for r in vals),active_packet_occurrences=sum(r['active_packets'] for r in vals),packet_occurrences=4950))
   epoch_summary.append(record)
  observations.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],observation=done,independent_native_steps=11100,independent_auxiliary_steps=2750,independent_layer_cohort_gradients=660,gradient_batch_arithmetic_means=summaries,auxiliary_epoch_curve=epoch_summary,saved_gradient_recount='VERIFIED',scope='Actual mainL CE and separateL90 auxiliary gradients, not samepacket CE or cumulative AdamW contribution. Packet occurrences repeat across batches; not unique physical coverage or independent modelseeds.'))
 return dict(status='VERIFIED',audit=audit,actual_training_observations=observations,auxiliary_steps_recounted=2750*len(observations),gradient_layer_cohorts_recounted=660*len(observations),no_new_target_forwards=True,head_fit_updates=0,U_truth_read=False,seconds=time.time()-started)
