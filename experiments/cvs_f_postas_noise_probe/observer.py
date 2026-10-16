"""Original forward hooks only;no intervention or fitted diagnosis classifier."""
import torch
from torch.nn import functional as F
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
PATHS=dict(time_projection='core.id_backbone.t_proj',frequency_projection='core.id_backbone.f_proj',reference_projection='response_projection',SIG_TIME='sequence_pool',frequency_gate_logits='core.id_backbone.freq_gate.conv',PA_gate_logits='core.id_backbone.pa_gate.net',raw_reference='response')
FEATURES=('time_projection','frequency_projection','reference_projection','SIG_TIME','base_identity','PA_physical','joint160')
def norm(x):return x.detach().flatten(1).double().norm(dim=1)
def run(model,x,y,domains):
 if model.training or x.ndim!=3 or x.shape[1:]!=(2,256) or y.shape!=(len(x),) or bool(((y<0)|(y>5)).any()):raise ValueError('Frozen eval legal labeled IQ256 required')
 enc=model.id_backbone.encoder;head=enc.core.id_backbone.cls_head;seen={};handles=[]
 def hook(name):
  def capture(module,args,out):
   if name in seen:raise ValueError('Expected exactly one actual identity-path call')
   seen[name]=out
  return capture
 try:
  for name,path in PATHS.items():handles.append(enc.get_submodule(path).register_forward_hook(hook(name)))
  with fresh_parameter_caches(model):
   out=model.forward_identity_only(x,domain_labels=domains);z=out['tx_logits'];seen.update(base_identity=out['id_feat_cls'],PA_physical=out['id_feat_pa'],joint160=out['id_feat_joint'])
   if set(seen)!=set(PATHS)|{'base_identity','PA_physical','joint160'} or z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()) or seen['joint160'].shape!=(len(x),160) or head.scale!=30.:raise ValueError('Actual full fixedCVS/defaultcos30 path differs')
   if seen['raw_reference'].shape!=(len(x),29) or not bool((seen['raw_reference'][:,27]==0).all()):raise ValueError('NoExplicitCFO actual coordinate27 must be masked')
   per_ce=F.cross_entropy(z,y,reduction='none');loss=per_ce.sum();watch=FEATURES+('frequency_gate_logits','PA_gate_logits');grad=torch.autograd.grad(loss,[seen[k] for k in watch],allow_unused=True)
   values={k:seen[k].detach().clone() for k in FEATURES};records=[];other=z.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values;margins=(z.gather(1,y[:,None]).flatten()-other)/30
   if any(not bool(torch.isfinite(v).all()) for v in (*values.values(),*([a for a in grad if a is not None]))):raise FloatingPointError('Nonfinite native branch value or diagnostic gradient')
   norms={k:norm(seen[k]).cpu().tolist() for k in FEATURES};gnorms={k:None if a is None else norm(a).cpu().tolist() for k,a in zip(watch,grad)};gated={}
   for name,parent in (('frequency',enc.core.id_backbone.freq_gate),('PA',enc.core.id_backbone.pa_gate)):
    h=seen['frequency_gate_logits' if name=='frequency' else 'PA_gate_logits'].detach().double().sigmoid().flatten(1);s=1+parent.alpha*(2*h-1);gated[name]=dict(mean=h.mean(1).cpu().tolist(),std=h.std(1,unbiased=False).cpu().tolist(),scale_mean=s.mean(1).cpu().tolist(),scale_min=s.min(1).values.cpu().tolist(),scale_max=s.max(1).values.cpu().tolist(),alpha=parent.alpha)
   logit_values=z.detach().double().cpu().tolist();predictions=z.detach().argmax(1).cpu().tolist();margin_values=margins.detach().cpu().tolist();ce_values=per_ce.detach().cpu().tolist()
   for i in range(len(x)):
    gates={name:{k:(v if k=='alpha' else v[i]) for k,v in a.items()} for name,a in gated.items()}
    records.append(dict(logits=logit_values[i],prediction=predictions[i],margin_over30=margin_values[i],CE=ce_values[i],feature_norms={k:v[i] for k,v in norms.items()},feature_gradient_L2={k:None if v is None else v[i] for k,v in gnorms.items()},gates=gates,raw_reference_CFO_coordinate=0.,gradient_scope='sum unsmoothed default-head CE derivative to actual intermediate outputs;no parameter.grad accumulation'))
 finally:
  for h in handles:h.remove()
 return records,values

def add_clean_cosines(records,features,clean):
 for k in FEATURES:
  c=F.cosine_similarity(features[k].flatten(1).double(),clean[k].flatten(1).double(),dim=1,eps=1e-12)
  for i,r in enumerate(records):r.setdefault('feature_cosine_to_clean',{})[k]=float(c[i])
 return records
