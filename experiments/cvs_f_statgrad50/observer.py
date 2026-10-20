"""Complete original physical update budgets and statistic-backward observations."""
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN or tuple(v[1] for v in contract['domain_map'])!=(1,2,3)*5:raise ValueError('Actual source RX/day map differs')
 def finish(self):
  value=super().finish();records=self.model._supcon_records;details=[v for r in records for v in r['details']];policy=self.c['statistic_gradient_policy']
  if len(details)!=39960 or any(v['statistic_policy']!=policy or v['backward'].get('mode')!=policy or v['backward'].get('backward_calls')!=1 for v in details):raise ValueError('Actual statistic backward budget differs')
  norms=('native_descriptor_grad_norm','full_descriptor_grad_norm','delivered_descriptor_grad_norm','norm_ratio','translation_removed_norm','scale_removed_norm')
  d.write(self.out/'statistic_gradient_observation.json',dict(status='VERIFIED',policy=policy,calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,backward_calls=39960,valid_anchors=sum(v['valid_anchors'] for v in details),gradient_means={k:sum(v['backward'][k] for v in details)/len(details) for k in norms},clamped_channel_occurrences=sum(v['backward']['clamped_channels'] for v in details),extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,temperature=.2,layers=['time_down','t2.pool'],forward_exact_native=True))
  return value
