"""Exact original RDrop graph; optional reused X3 pool and centered Sinc."""
from . import design as d
from .coordinate import CoordinateSinc
from .x3_features import CompositePool
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier=original.classifier;source_domain_count=original.source_domain_count

def configure(m,c):
 original.attach(m,c);policy=c['combination'];ref=m.id_backbone.encoder
 if policy['SIG_TIME']=='SIG_TIME_X3':
  oldpool=ref.sequence_pool;ref.sequence_pool=CompositePool(oldpool.channels,'SIG_TIME_X3',c['model_seed'])
  m._feature_learning_report.update(arm='SIG_TIME_X3',modes=['SIG_TIME']*3,parameter_matched=False,independent_local_projections=True,local_dimension_each=6,descriptor_width_each=28,extra_parameters=sum(p.numel() for p in ref.sequence_pool.parameters()),combination='Three independent local views jointly feed one actual joint160; no voting')
 if policy['Sinc_coordinates']=='CENTER_25K':ref.core.id_backbone.sinc=CoordinateSinc(ref.core.id_backbone.sinc,'CENTER_25K')
 m._feature_learning_report.update(sinc_coordinates=policy['Sinc_coordinates'],Sinc_optimizer=policy['Sinc_optimizer'],A_F_arm=c['arm'])
 return m

def install(m,c):
 d.validate(c);return configure(original.old.install(m,original.d.legacy_config(c['model_seed'])),c)
def build(c,device='cpu'):
 d.validate(c);return configure(original.old.build(original.d.legacy_config(c['model_seed']),'cpu'),c).to(device)

def physical_state(state):
 """Derived native-Hz state for kernel analysis only; checkpoint retains actual q/origins."""
 result=dict(state);prefix='id_backbone.encoder.core.id_backbone.sinc.'
 if prefix+'low_coordinate' in state:
  for short,native in (('low','low_hz_'),('band','band_hz_')):
   result[prefix+native]=state[prefix+short+'_origin_hz']+25000.*state[prefix+short+'_coordinate']
   del result[prefix+short+'_coordinate'];del result[prefix+short+'_origin_hz']
 return result

class FeatureExport(__import__('torch').nn.Module):
 """Actual per-packet f_theta; copy the coordinate/origin leaves and discard W only."""
 def __init__(self,m):
  super().__init__()
  from copy import deepcopy
  self.encoder=deepcopy(m.id_backbone.encoder);del self.encoder.core.id_backbone.cls_head.weight
 def forward(self,x):return self.encoder.features(x)
def export(m):return FeatureExport(m)
