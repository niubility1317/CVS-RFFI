"""Original physical optimizer and source budgets, plus full memory counters."""
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN or tuple(v[1] for v in contract['domain_map'])!=(1,2,3)*5:raise ValueError('Actual source RX/day map differs')
 def finish(self):
  value=super().finish();m=self.model;records=[r for r in m._supcon_records if r['details']];details=[v for r in records for v in r['details']]
  if len(records)!=9990 or len(details)!=39960 or any(r['contrastive_mode']!=self.c['treatment'] for r in records) or any(v['mode']!=self.c['treatment'] or not 0<=v['historical_keys']<=384 or v['key_count']!=128+v['historical_keys'] or not 0<=v['max_history_age']<=3 for v in details):raise ValueError('Fixed memory objective budget differs')
  expected=9990 if self.c['treatment']=='MEM4' else 0
  if any(b.updates!=expected or len(b.entries)!=(3 if expected else 0) for phase in m._crossbatch_banks.values() for b in phase.values()):raise ValueError('Memory only updated on actual activeL steps')
  total=sum(v['valid_anchors'] for v in details);per=[]
  for c in range(6):
   count=sum(v['per_TX_keys'][c]['count'] for v in details);p=sum(v['per_TX_keys'][c]['count']*(v['per_TX_keys'][c]['positive_mean'] or 0.) for v in details);n=sum(v['per_TX_keys'][c]['count']*(v['per_TX_keys'][c]['negative_mean'] or 0.) for v in details);per.append(dict(TX=d.CLASSES[c],valid_anchor_occurrences=count,positive_mean=p/count if count else None,negative_mean=n/count if count else None))
  d.write(self.out/'crossbatch_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,valid_anchors=total,mean_historical_keys=sum(v['historical_keys'] for v in details)/len(details),max_historical_keys=max(v['historical_keys'] for v in details),max_history_age=max(v['max_history_age'] for v in details),per_TX_keys=per,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,temperature=.2,layers=['time_down','t2.pool']))
  return value
