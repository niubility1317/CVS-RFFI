from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from . import design as d
from .relations import CrossFrequencyRelations
from experiments.cvs_reference_identity.model import ReferenceCVS
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
from experiments.cvs_source_identity_readout.input_policy import apply,NoExplicitCFO
from experiments.cvs_phase1_stack.runtime import installed as native_installed

class CrossFrequencyReference(nn.Module):
 def __init__(self,reference,mode,seed):
  super().__init__()
  if type(reference) is not ReferenceCVS or type(reference.response) is not NoExplicitCFO:raise ValueError('Expected unchanged NoExplicitCFO ReferenceCVS')
  self.variant=reference.variant
  self.core=reference.core;self.response=reference.response;self.response_projection=reference.response_projection;self.response_gain=reference.response_gain
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(seed+51503)
   self.spectral=CrossFrequencyRelations(mode)
  self.spectral_gain=nn.Parameter(torch.full((160,),.01))
 def contract(self):return dict(self.response.contract(),cross_frequency_mode=self.spectral.mode,initial_gain=.01)
 def features(self,x):
  base=self.core.features(x);physical=self.response_projection(self.response(x))
  return base+self.response_gain.tanh()*physical+self.spectral_gain.tanh()*self.spectral(x)
 def classify_features(self,z):return self.core.id_backbone.cls_head.classify(z)
 def forward(self,x):return self.classify_features(self.features(x))
 @torch.no_grad()
 def diagnostics(self,x):
  result=ReferenceCVS.diagnostics(self,x)
  result.update(spectral_gain_abs_mean=float(self.spectral_gain.tanh().abs().mean()),spectral_gain_gradient_norm=float(self.spectral_gain.grad.norm()) if self.spectral_gain.grad is not None else None)
  return result

class CrossFrequencyFeatures(nn.Module):
 dimension=160;normalization_epsilon=1e-4
 def __init__(self,reference):
  super().__init__()
  if type(reference) is not CrossFrequencyReference:raise ValueError('Own cross-frequency feature boundary required')
  h=reference.core.id_backbone.cls_head
  if reference.core.id_backbone.nmfdu_gate is not None:raise ValueError('Unexpected identity gate')
  if type(h) is not ResidualPhysicalHead or h.weight.shape!=(6,160) or h.scale!=30.:raise ValueError('Classifier interface differs')
  self.encoder=deepcopy(reference);del self.encoder.core.id_backbone.cls_head.weight
  self.train(reference.training)
 def forward_features(self,x):
  if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ[B,2,256]')
  with torch.autocast(device_type=x.device.type,enabled=False):return self.encoder.features(x.float())
 def forward(self,x):return self.forward_features(x)
 def normalized_features(self,x):return torch.nn.functional.normalize(self.forward_features(x),dim=1,eps=self.normalization_epsilon)

def from_phase1(model):
 if type(model) is CrossFrequencyReference:return CrossFrequencyFeatures(model)
 from experiments.cvs_phase1_overlay.model import Phase1IdentityAdapter
 if type(getattr(model,'id_backbone',None)) is not Phase1IdentityAdapter:raise ValueError('Own Phase1 identity adapter required')
 return CrossFrequencyFeatures(model.id_backbone.encoder)

def install(model,c):
 d.validate(c)
 if not c['cross_frequency']['enabled']:raise ValueError('Baseline comparison belongs to original model')
 model._identity_input_policy=apply(model)
 reference=model.id_backbone.encoder;head=reference.core.id_backbone.cls_head
 model.id_backbone.encoder=CrossFrequencyReference(reference,c['cross_frequency']['mode'],c['model_seed'])
 if model.id_backbone.encoder.core.id_backbone.cls_head is not head:raise ValueError('Classifier changed')
 model._cross_frequency_contract=dict(c['cross_frequency'],head='unchanged cosine6x160',pseudo_EMA='historical unmodified')
 return model

@contextmanager
def installed(c,training=False):
 d.validate(c)
 with native_installed(c) as n:
  previous=n.build_baseline_model;hook=n._source_feature_auxiliary_hook
  if hook is not None:raise ValueError('Unregistered auxiliary loss hook')
  def build(args,device):return install(previous(args,device),c).to(device)
  n.build_baseline_model=build
  try:yield n
  finally:n.build_baseline_model=previous

def build(c,device='cpu'):
 a=d.make_args(c,str(device));a.num_domains=5;a.input_len=256
 with installed(c) as n:return n.build_baseline_model(a,torch.device(device))
