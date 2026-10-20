"""Original physical Sinc/full paired budgets; scalar denominator diagnostics."""
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
  if len(records)!=9990 or len(details)!=39960 or any(r['contrastive_mode']!=self.c['treatment'] for r in records) or any(v['mode']!=self.c['treatment'] for v in details):raise ValueError('Fixed decoupling objective budget differs')
  total=sum(v['valid_anchors'] for v in details);per=[]
  for c in range(6):
   count=sum(v['per_TX_q'][c]['count'] for v in details);qsum=sum(v['per_TX_q'][c]['count']*(v['per_TX_q'][c]['q_mean'] or 0.) for v in details);per.append(dict(TX=d.CLASSES[c],valid_anchor_occurrences=count,q_mean=qsum/count if count else None))
  d.write(self.out/'decoupled_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,valid_anchors=total,q_mean=sum(v['valid_anchors']*(v['q_mean'] or 0.) for v in details)/total if total else None,per_TX_q=per,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,temperature=.2,layers=['time_down','t2.pool']))
  return value
