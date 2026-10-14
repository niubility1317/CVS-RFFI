"""Read frozen complete logs, actual gradients and fixed default predictions only."""
import gzip,json,math,time
import torch
from . import design as d,model as api
from experiments.cvs_trusted_weight50.observer import records as gradient_records
from experiments.cvs_signature_stats50.audit import run as shared
def run():
 started=time.time();audit=shared(design=d,build_fn=api.build);observations=[]
 for row in d.training_rows():
  c=d.config(row);p=d.BASE/row['row_id']/'source';done=d.read(p/'trusted_training_observation.json');saved=d.read(p/'trusted_gradient_records.json')
  if done['status']!='VERIFIED' or done['steps']!=11100 or done['U_steps']!=3774 or done['U_packets']!=963900 or done['epochs']!=[33,34,50] or done['layer_gradient_records']!=1998 or done['head_fit'] or done['U_truth_read'] or done['weight_active']!=c['trusted_weight']['active'] or saved['epochs']!=[33,34,50] or len(saved['batch_records'])!=666:raise ValueError('Actual training observation differs')
  batches=saved['batch_records'];vectors_recount=0
  for epoch in (33,34,50):
   raw=torch.load(p/f'trusted_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);ref=[x for x in batches if x['epoch']==epoch]
   if len(raw)!=222 or len(ref)!=222:raise ValueError('Saved epoch gradient matrix incomplete')
   for batch,(x,y) in enumerate(zip(raw,ref),start=1):
    if x['batch']!=batch or x['step']!=(epoch-1)*222+batch or x['epoch']!=epoch or x['active']!=(epoch>=34) or gradient_records(x['vectors'])!=y['records'] or y['step']!=x['step'] or y['accepted']!=x['accepted']:raise ValueError('Independent saved gradient norm/count recount differs')
    if epoch==33 and (x['accepted']!=0 or any(r['lambdaU_uniform_norm']!=0 or r['lambdaU_weighted_norm']!=0 or r['lambdaEnt_norm']!=0 for r in y['records'])):raise ValueError('Label-only epoch has U gradients')
    vectors_recount+=len(y['records'])
   del raw
  step_count=U_packets=U_accepted=weight_varying_steps=loss_different_steps=0;step_weight=[]
  with gzip.open(p/'trusted_weight_steps.jsonl.gz','rt',encoding='utf-8') as weights,gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf-8') as native:
   for text in weights:
    w=json.loads(text);other=next(native,None)
    if other is None:raise ValueError('Native step stream shorter than trusted log')
    other=json.loads(other);step_count+=1;epoch=(step_count-1)//222+1;m=other['metrics']
    if w['step']!=step_count or w['epoch']!=epoch or w['batch']!=(step_count-1)%222+1 or w['active']!=(epoch>=34) or w['weighted_training']!=c['trusted_weight']['active'] or w['lambda_u']!=.16 or w['lambda_ent']!=.01 or w['U_count']!=m['train/pseudo_total'] or w['accepted']!=m['train/pseudo_selected'] or m.get('train/pseudo_truth_available',0)!=0:raise ValueError('Native/trusted step or gate counts differ')
    if epoch>=34:
     actual=w['weighted_CE'] if c['trusted_weight']['active'] else w['uniform_CE']
     if not math.isfinite(actual) or abs(actual-m['train/loss_unlabeled'])>2e-6:raise ValueError('Actual native U loss differs from preregistered mode')
     U_packets+=w['U_count'];U_accepted+=w['accepted']
     if w['accepted']:
      if not .75<=w['raw_min']<=w['raw_max']<=1.25 or not 0<=w['accepted_h_mean']<=1:raise ValueError('Raw weight/hardness bounds differ')
      weight_varying_steps+=int(w['raw_max']-w['raw_min']>1e-8);loss_different_steps+=int(abs(w['weighted_CE']-w['uniform_CE'])>1e-8)
     step_weight.append(w)
   if next(native,None) is not None:raise ValueError('Native step stream longer than trusted log')
  if step_count!=11100 or U_packets!=963900 or U_accepted!=done['U_accepted'] or vectors_recount!=1998:raise ValueError('Trusted weight complete budgets differ')
  agg=[]
  for epoch in (33,34,50):
   for name in saved['batch_records'][0]['records'][0]['parameter'],saved['batch_records'][0]['records'][1]['parameter'],saved['batch_records'][0]['records'][2]['parameter']:
    rr=[r for b in batches if b['epoch']==epoch for r in b['records'] if r['parameter']==name];means={};valid={}
    for key in ('L_CE_norm','lambdaU_uniform_norm','lambdaU_weighted_norm','lambdaEnt_norm','uniform_U_to_L','weighted_U_to_L','uniform_totalU_to_L','weighted_totalU_to_L','weighted_uniform_cosine','uniformU_L_cosine','weightedU_L_cosine'):
     vv=[r[key] for r in rr if r[key] is not None];means[key]=sum(vv)/len(vv) if vv else None;valid[key]=len(vv)
    agg.append(dict(epoch=epoch,parameter=name,batches=222,means=means,valid_batches=valid))
  observations.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],observation=done,independent_gradient_records=vectors_recount,native_loss_and_mask_steps_recount=step_count,weight_varying_U_steps=weight_varying_steps,weighted_uniform_loss_different_steps=loss_different_steps,gradient_batch_arithmetic_means=agg,gradient_batch_records=batches,U_step_weight_records=step_weight,scope='Actual prebackward parameter gradients on existing native L/U batches. Within-model uniform/weighted counterfactuals; models diverge after weighting. No Utruth or newforward/RNG. Cross-layer norm rank is not importance.'))
 return dict(status='VERIFIED',audit=audit,actual_training_observations=observations,gradient_layer_records_recounted=len(observations)*1998,native_weight_loss_steps_recounted=len(observations)*11100,no_new_target_forwards=True,head_fit=False,U_truth_read=False,seconds=time.time()-started)
