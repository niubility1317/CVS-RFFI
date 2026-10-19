import torch
from experiments.cvs_reference_identity.model import ReferenceCVS
from .operators import MODES,from_coordinates,identities
class Probe:
 def __init__(self,model):
  self.model=model;self.ref=model.id_backbone.encoder;self.pool=self.ref.sequence_pool;self.base=None;self.native=None;self.coords={};self.handles=[];self.calls=0;self.max_displacement=0.;self.max_area=0.;self.closed=False
  if type(self.ref).__name__!='SequenceReference' or self.pool.arm!='SIG_TIME_X3' or len(self.pool.views)!=3 or model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen original F/SIG_TIME_X3/default head required')
 def __enter__(self):
  self.original=ReferenceCVS.features;owner=self
  def capture_base(ref,x):
   v=owner.original(ref,x)
   if ref is owner.ref:
    if owner.base is not None:raise ValueError('Unconsumed native base')
    owner.base=v
   return v
  ReferenceCVS.features=capture_base
  for i,v in enumerate(self.pool.views):
   def capture(module,args,output,index=i):
    if index in self.coords or output.ndim!=3 or output.shape[1:]!=(6,32):raise ValueError('Exactly one original local projection per native input')
    self.coords[index]=torch.tanh(output).transpose(1,2)
   self.handles.append(v.local_projection.register_forward_hook(capture))
  def capture_pool(module,args,out):
   if self.native is not None:raise ValueError('Unconsumed native pool')
   self.native=out
  self.handles.append(self.pool.register_forward_hook(capture_pool));return self
 def take(self,output):
  if self.base is None or self.native is None or set(self.coords)!={0,1,2} or torch.is_grad_enabled():raise ValueError('One complete no-grad original source forward required')
  coords=[self.coords[i] for i in range(3)];native_joint=self.base+self.native
  if not torch.equal(native_joint,output['id_feat_joint']) or not torch.equal(self.ref.classify_features(native_joint),output['tx_logits']) or not torch.equal(from_coordinates(self.pool,coords,'NATIVE'),self.native):raise ValueError('Native base/pool/joint/default head replay differs')
  for u in coords:
   z=identities(u.reshape(-1,8,6));self.max_displacement=max(self.max_displacement,z['displacement_max_error']);self.max_area=max(self.max_area,z['time_area_max_error'])
  if max(self.max_displacement,self.max_area)>2e-5:raise ValueError('Actual native signature endpoint identities differ')
  logits={'NATIVE':output['tx_logits']};joint_change={}
  for mode in MODES[1:]:
   v=self.base+from_coordinates(self.pool,coords,mode);logits[mode]=self.ref.classify_features(v);joint_change[mode]=float((v-native_joint).square().sum(1).sqrt().mean())
  if not all(torch.isfinite(v).all() and v.shape==output['tx_logits'].shape for v in logits.values()):raise ValueError('Nonfinite or changed default class schema')
  self.base=self.native=None;self.coords={};self.calls+=1;return dict(logits=logits,joint_L2_mean=joint_change)
 def __exit__(self,*exc):
  ReferenceCVS.features=self.original
  for h in self.handles:h.remove()
  self.base=self.native=None;self.coords={};self.closed=True
