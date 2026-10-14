import gzip,json,statistics as st,time
import torch
from . import design as d,model as api
from .observer import gradient_stats
from experiments.cvs_signature_stats50.audit import run as shared

def run():
 torch.set_num_threads(2);start=time.time();audit=shared(design=d,build_fn=api.build);observations=[]
 for row in d.training_rows():
  c=d.config(row);p=d.BASE/row['row_id']/'source';done=d.read(p/'signature_training_observation.json');saved=d.read(p/'signature_gradient_records.json');batches=saved['batch_records'];gradient_count=0
  if done['status']!='VERIFIED' or done['steps']!=11100 or done['extra_batches']!=2750 or done['extra_source_packet_forwards']!=247500 or done['U_packets']!=963900 or done['gradient_epochs']!=[1,17,33,50] or done['gradient_batches']!=220 or len(batches)!=220 or done['head_fit_updates'] or done['U_truth_read']:raise ValueError('Actual signature training budget differs')
  for epoch in d.DIAGNOSTIC_EPOCHS:
   values=torch.load(p/f'signature_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);reference=[v for v in batches if v['epoch']==epoch]
   if len(values)!=55 or len(reference)!=55:raise ValueError('Actual gradient epoch incomplete')
   for j,(a,b) in enumerate(zip(values,reference),1):
    if a['step']!=(epoch-1)*222+j*4 or a['epoch']!=epoch or a['step']!=b['step']:raise ValueError('Actual gradient identity differs')
    for k,name in enumerate(d.PARAMETERS):
     if b['layers'][k]['parameter']!=name:raise ValueError('Actual parameter object differs')
     for key in (*d.LEVELS,'effective'):
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved gradient arithmetic differs')
     if (epoch==1 or c['signature_order']['mode']=='ZERO') and b['layers'][k]['effective']['auxiliary_norm']!=0:raise ValueError('Zero/warmup gradient differs')
     gradient_count+=1
  summaries=[]
  for epoch in d.DIAGNOSTIC_EPOCHS:
   rr=[r for r in batches if r['epoch']==epoch]
   for k,name in enumerate(d.PARAMETERS):
    record=dict(epoch=epoch,parameter=name,batches=55)
    for key in (*d.LEVELS,'effective'):
     vals=[r['layers'][k][key] for r in rr];cos=[r['cosine'] for r in vals if r['cosine'] is not None];record[key]=dict(nonzero_batches=sum(r['auxiliary_norm']>1e-12 for r in vals),mean_aux_norm=st.mean(r['auxiliary_norm'] for r in vals),mean_ratio=st.mean(r['auxiliary_over_CE'] or 0 for r in vals),mean_cosine=st.mean(cos) if cos else None)
    summaries.append(record)
  local_steps=0;per_epoch={};U_accepted=0
  with gzip.open(p/'signature_order_steps.jsonl.gz','rt') as local,gzip.open(p/'step_metrics.jsonl.gz','rt') as native:
   for index,line in enumerate(native,1):
    v=json.loads(line);epoch=(index-1)//222+1;batch=(index-1)%222+1;m=v['metrics'];U_accepted+=int(m['train/pseudo_selected'])
    if batch%4:
     if m.get('signature_order_extra_forward')!=0:raise ValueError('Unscheduled extra forward')
     continue
    w=json.loads(next(local));local_steps+=1;mode=c['signature_order']['mode'];weight=0. if mode=='ZERO' else .1*min(1.,max(0.,(epoch-5)/10));selected=w['raw_losses']['SIG_RAW' if mode=='ZERO' else mode]
    if w['step']!=index or w['cohort_rank']!=(local_steps-1)%70 or w['mode']!=mode or w['coefficient']!=weight or w['extra_source_packets']!=90 or not w['source_only'] or m.get('signature_order_extra_forward')!=1 or abs(w['clock']['clock_min']-1)>2e-6 or abs(w['clock']['clock_max']-1)>2e-6:raise ValueError('Actual signature/source exposure differs')
    if abs(w['effective_loss']-weight*selected)>2e-6 or abs(m['signature_order_effective']-w['effective_loss'])>2e-6:raise ValueError('Native applied auxiliary differs')
    per_epoch.setdefault(epoch,[]).append(w)
   if next(local,None) is not None:raise ValueError('Extra source stream longer than native')
  if local_steps!=2750 or gradient_count!=660 or U_accepted!=done['U_accepted']:raise ValueError('Independent source budget differs')
  curve=[]
  for epoch,rr in per_epoch.items():
   if len(rr)!=55:raise ValueError('Actual epoch exposure differs')
   curve.append(dict(epoch=epoch,batches=55,coefficient=rr[0]['coefficient'],effective_loss=st.mean(r['effective_loss'] for r in rr),clock_energy_fraction_mean=st.mean(r['clock']['mean_clock_energy_fraction'] for r in rr),views={key:dict(relations=99000,unsatisfied=sum(r['views'][key]['active'] for r in rr),wrong_order=sum(r['views'][key]['wrong_order'] for r in rr),active_packet_occurrences=sum(r['views'][key]['active_packets'] for r in rr),packet_occurrences=4950,mean_hinge=st.mean(r['views'][key]['hinge'] for r in rr)) for key in d.LEVELS}))
  observations.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],observation=done,independent_auxiliary_steps=local_steps,independent_layer_cohort_gradients=gradient_count,gradient_batch_arithmetic_means=summaries,auxiliary_epoch_curve=curve,saved_gradient_recount='VERIFIED',scope='Actual mainL128CE versus separate extraL90. Physical packet occurrences repeat, not unique coverage or independent modelseeds.'))
 return dict(status='VERIFIED',audit=audit,actual_training_observations=observations,auxiliary_steps_recounted=2750*len(observations),gradient_layer_cohorts_recounted=660*len(observations),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True,seconds=time.time()-start)
