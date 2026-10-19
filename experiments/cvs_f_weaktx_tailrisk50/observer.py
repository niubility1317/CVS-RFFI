"""Original50 witnesses plus complete weak feature-tail exposure accounting."""
import gzip,json
from collections import Counter
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original source physical RX/day map differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;scheduled=max(steps-1110,0)
  if m._tail_calls!=steps or len(m._tail_records)!=steps or sum(r['scheduled'] for r in m._tail_records)!=scheduled or m._tail_applied>scheduled:raise ValueError('Actual tail calendar differs')
  d.write(out/'feature_tail_progress.json',dict(epoch=len(rows),steps=steps,scheduled=scheduled,applied=m._tail_applied,cell_coverage=m._tail_cell_total/(scheduled*20) if scheduled else 0.,available_anchor_objectives=m._tail_available,selected_anchor_objectives=m._tail_selected,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  result=super().finish();m=self.model;scheduled=sum(r['scheduled'] for r in m._tail_records)
  if m._tail_calls!=11100 or len(m._tail_records)!=11100 or scheduled!=9990 or not 0<=m._tail_applied<=9990:raise ValueError('Full original50/tail ordered budget required')
  with gzip.open(self.out/'feature_tail_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._tail_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  cells=Counter()
  for r in m._tail_records:
   if r['eligible']:
    for level in r['levels']:
     for cell in level['cells']:cells[str(cell['TX'])+'@'+str(cell['RX'])]+=1
  d.write(self.out/'feature_tail_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,scheduled=9990,applied=m._tail_applied,eligible_fraction=m._tail_applied/9990,weak_RX_cell_coverage=m._tail_cell_total/(9990*20),available_weak_anchor_objectives=m._tail_available,selected_weak_anchor_objectives=m._tail_selected,selection_fraction=m._tail_selected/m._tail_available if m._tail_available else 0.,cell_level_calls=dict(cells),levels=['time_down','t2.pool'],pair_average_before_selection=True,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,weak_indices=[1,3],head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False))
  return result
