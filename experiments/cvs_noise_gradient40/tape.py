"""Read existing tensors and exact small parameter gradients;never update a model."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .maths import head_weight_vectors,conv_vectors,OBJECTIVES
FEATURES=('joint','time','frequency','PA','reference','SIG_TIME','SIG0','SIG1','SIG2')
PARAMETERS=('head_W','head_PA_gain','freq_gate','PA_gate')
class Tape:
 def __init__(self,m):
  self.ref=m.id_backbone.encoder;self.core=self.ref.core.id_backbone;self.head=self.core.cls_head;self.values={};self.inputs={};self.handles=[];self.local_leaves=[];self.head_logits=None
 def put(self,name,value):
  if name in self.values or not torch.is_tensor(value) or value.ndim<2 or not bool(torch.isfinite(value).all()):raise ValueError('One finite existing activation per identity forward:'+name)
  if not value.requires_grad:value.requires_grad_(True);self.local_leaves.append(name)
  self.values[name]=value
 def hook(self,name,module,gate=False):
  def capture(mod,args,out):
   self.put(name,out)
   if gate:self.inputs[name]=args[0]
  self.handles.append(module.register_forward_hook(capture))
 def __enter__(self):
  h=self.head;r=self.ref;b=self.core
  for name,module in [('identity',h.base_norm),('PA',h.pa_norm),('reference',r.response_projection),('SIG_TIME',r.sequence_pool),('freq_gate_z',b.freq_gate.conv),('PA_gate_z',b.pa_gate.net)]+[('SIG'+str(i),v) for i,v in enumerate(r.sequence_pool.views)]:self.hook(name,module,name.endswith('_z'))
  self.handles.append(b.fuse.register_forward_pre_hook(lambda mod,args:self.put('base_input',args[0])))
  self.had_override='classify' in h.__dict__;self.previous=h.__dict__.get('classify');self.original=h.classify
  def classified(joint):
   self.put('joint',joint);self.head_logits=self.original(joint);return self.head_logits
  h.classify=classified
  return self
 def __exit__(self,*error):
  for handle in self.handles:handle.remove()
  if self.had_override:self.head.classify=self.previous
  else:delattr(self.head,'classify')
 def parameter_objects(self):
  h=self.head;a=self.core.freq_gate.conv;b=self.core.pa_gate.net
  return (h.weight,h.gain,a.weight,a.bias,b.weight,b.bias)

def observe(m,x,y,*,smoke=False):
 if m.training or x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=3 or x.shape[1:]!=(2,256) or y.dtype!=torch.long or y.shape!=(len(x),) or not bool(((y>=0)&(y<6)).all()):raise ValueError('Frozen true-label L-only CPU IQ required')
 t=Tape(m);params=t.parameter_objects();flags=[p.requires_grad for p in params]
 if any(flags) or any(p.grad is not None for p in m.parameters()):raise ValueError('Readonly checkpoint parameters required')
 try:
  if smoke:
   for p in params:p.requires_grad_(True)
  with torch.enable_grad(),t:
   xx=x.detach().clone().requires_grad_(True);out=m.forward_identity_only(xx);z=out['tx_logits']
   required={'identity','PA','reference','SIG_TIME','SIG0','SIG1','SIG2','base_input','joint','freq_gate_z','PA_gate_z'}
   if set(t.values)!=required or t.values['base_input'].shape!=(len(x),321) or z.shape!=(len(x),6) or not torch.allclose(z,t.head_logits,atol=1e-6,rtol=1e-6):raise ValueError('Actual F/defaulthead branch path differs')
   competitors=z.detach().clone();competitors[torch.arange(len(y)),y]=-torch.inf;other=competitors.argmax(1);margin=(z[torch.arange(len(y)),y]-z[torch.arange(len(y)),other])/30
   losses=dict(margin=-margin,CE0=F.cross_entropy(z,y,reduction='none'),CE01=F.cross_entropy(z,y,reduction='none',label_smoothing=.01));keys=list(t.values);targets=[t.values[k] for k in keys]+(list(params) if smoke else []);groups={};checks={}
   for oi,name in enumerate(OBJECTIVES):
    grad=torch.autograd.grad(losses[name].sum(),targets,allow_unused=False,retain_graph=oi<len(OBJECTIVES)-1);g=dict(zip(keys,grad[:len(keys)]))
    if not all(bool(torch.isfinite(v).all()) for v in g.values()):raise ValueError('Nonfinite local gradient;no sample dropping')
    with torch.no_grad():
     pv=dict(head_W=head_weight_vectors(t.values['joint'],z,y,t.head.weight,name),head_PA_gain=g['identity']*t.values['PA']*(1-t.head.gain.tanh().square()),freq_gate=conv_vectors(t.inputs['freq_gate_z'],g['freq_gate_z'],t.core.freq_gate.conv),PA_gate=conv_vectors(t.inputs['PA_gate_z'],g['PA_gate_z'],t.core.pa_gate.net))
     fg={k:g[k].flatten(1).detach().clone() for k in FEATURES if k not in ('time','frequency')};fg['time']=g['base_input'][:,:160].detach().clone();fg['frequency']=g['base_input'][:,160:320].detach().clone()
     groups[name]=dict(feature_gradients=fg,parameter_gradients={k:v.detach().clone() for k,v in pv.items()})
     if smoke:
      actual=grad[len(keys):];expected=(pv['head_W'].sum(0).reshape_as(params[0]),pv['head_PA_gain'].sum(0),pv['freq_gate'].sum(0)[:-params[3].numel()].reshape_as(params[2]),pv['freq_gate'].sum(0)[-params[3].numel():],pv['PA_gate'].sum(0)[:-params[5].numel()].reshape_as(params[4]),pv['PA_gate'].sum(0)[-params[5].numel():])
      errors=[float((a-b).abs().max()) for a,b in zip(actual,expected)]
      if not all(torch.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(actual,expected)):raise ValueError('Exact parameter-gradient smoke differs:'+name+str(errors))
      checks[name]=errors
  with torch.no_grad():
   fv={k:t.values[k].flatten(1).detach().clone() for k in FEATURES if k not in ('time','frequency')};fv['time']=t.values['base_input'][:,:160].detach().clone();fv['frequency']=t.values['base_input'][:,160:320].detach().clone();gates={}
   for name,module,axis in [('freq_gate',t.core.freq_gate,1),('PA_gate',t.core.pa_gate,2)]:
    scale=1+module.alpha*(2*t.values[name+'_z'].sigmoid()-1);flat=scale.flatten(1);gates[name]=dict(mean=flat.mean(1).detach().clone(),std=flat.std(1,unbiased=False).detach().clone(),min=flat.min(1).values.detach().clone(),max=flat.max(1).values.detach().clone(),position_profile=scale.mean(axis).detach().clone())
   return dict(logits=z.detach().clone(),losses={k:v.detach().clone() for k,v in losses.items()},margin_over30=margin.detach().clone(),competitor=other,features=fv,objectives=groups,gates=gates,local_activation_leaves=list(t.local_leaves),smoke_check=checks,autograd_calls=3)
 finally:
  for p,flag in zip(params,flags):p.requires_grad_(flag)
