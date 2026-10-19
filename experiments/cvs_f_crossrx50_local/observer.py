"""Original optimizer/RDrop witnesses, with feature-treatment counts."""
from experiments.cvs_f_supcon50.observer import Observer as Original
from . import design as d
from .mechanism import RX_BY_DOMAIN
class Observer(Original):
 def __init__(self,c):
  d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx
  contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original RX/day domain mapping differs')
 def finish(self):
  result=super().finish();path=self.out/'supcon_training_observation.json';value=d.read(path)
  value.update(treatment=self.c['treatment'],actual_layer_forward_objectives=sum(len(r['details']) for r in self.model._supcon_records),class_weights=self.c['class_weights'],joint160_geometry=self.c['joint160_geometry'],joint160_max_weight=.05)
  expected=59940 if self.c['joint160_geometry'] else 39960
  if value['actual_layer_forward_objectives']!=expected:raise ValueError('Feature-objective coverage differs')
  d.write(path,value);return result
