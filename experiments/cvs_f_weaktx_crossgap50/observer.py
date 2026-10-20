"""Original physical Sinc/RDrop/SupCon budgets plus disjoint-gap observations."""
import gzip,json
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN or tuple(v[1] for v in contract['domain_map'])!=(1,2,3)*5:raise ValueError('Actual source RX/day map differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;scheduled=max(0,steps-1110)
  if m._gap_calls!=steps or len(m._gap_records)!=steps or m._gap_scheduled!=scheduled or not 0<=m._gap_active<=m._gap_eligible<=scheduled:raise ValueError('Original ordered source gap budget differs')
  d.write(out/'gap_progress.json',dict(epoch=len(rows),calls=steps,scheduled=scheduled,eligible=m._gap_eligible,active=m._gap_active,cell_calls=m._gap_cell_calls,current_L_packets_in_cells=m._gap_used_packets,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  value=super().finish();m=self.model
  if m._gap_calls!=11100 or len(m._gap_records)!=11100 or m._gap_scheduled!=9990 or not 0<=m._gap_active<=m._gap_eligible<=9990 or len(m._gap_cell_calls)!=15:raise ValueError('Full original50 plus source gap observations required')
  with gzip.open(self.out/'gap_training_steps.jsonl.gz','xt',encoding='utf8') as f:
   for row in m._gap_records:f.write(json.dumps(row,allow_nan=False)+'\n')
  d.write(self.out/'gap_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,scheduled=9990,eligible_calls=m._gap_eligible,active_calls=m._gap_active,eligible_fraction=m._gap_eligible/9990,active_fraction=m._gap_active/9990,cell_calls=m._gap_cell_calls,RX_day_cell_coverage=sum(n>=20 for n in m._gap_cell_calls)/15,current_L_packets_in_valid_cells=m._gap_used_packets,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,weak_indices=[1,3],head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,feature='actualjoint160',margin=.4,grouping='sameRX_day',disjoint_physical_packets=True,pair_average_before_partition=True,interpretation='Source geometric penalty only. Signed cross product is not certified unbiased for dependent finite-pool halves; no fitted classifier or target guarantee.'))
  return value
