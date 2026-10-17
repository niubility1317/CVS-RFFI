"""Alternative AdamW second moment, no change to objective or first moment.

O and S are the already weighted half-view objectives. Their summed gradient
is the native supervised mean. For uniform selection of the two full view
gradients, the coordinate variance is (grad(O)-grad(S))**2. Add this nonnegative
term only to AdamW's second moment, after applying the SAME native global clip
factor. Stored Sinc coordinates are retained here; native physical clip/decay
and per-group learning-rate/epsilon mapping remain responsible for units.
This is a different preconditioner, not a correction of an AdamW bug or an
unbiased estimate of total L+U stochastic variance.
"""
import math,time
import torch
from experiments.cvs_rdrop_sinc_units.optimizer import QNAMES
MODES=('MEASURE','VIEW_SECOND')

def vector(named,gradients):
 if len(named)!=len(gradients):raise ValueError('Gradient/parameter alignment differs')
 values=[]
 for (name,p),g in zip(named,gradients):
  if g is not None and (g.shape!=p.shape or not bool(torch.isfinite(g).all())):raise FloatingPointError('Invalid view gradient')
  v=torch.zeros_like(p).detach() if g is None else g.detach()
  values.append(v.reshape(-1).double()/(25000. if name in QNAMES else 1.))
 return torch.cat(values)
def geometry(named,gO,gS):
 a=vector(named,gO);b=vector(named,gS);oo=torch.dot(a,a);ss=torch.dot(b,b);os=torch.dot(a,b)
 if not bool(torch.isfinite(torch.stack((oo,ss,os))).all()):raise FloatingPointError('Nonfinite view gradient geometry')
 alpha=float(-os/oo) if bool(os<0) and bool(oo>0) else 0.
 return dict(dot=float(os),ordinary_norm=float(oo.sqrt()),satellite_norm=float(ss.sqrt()),cosine=float(os/(oo*ss).sqrt()) if bool(oo>0) and bool(ss>0) else None,coefficient=alpha,conflict=bool(os<0),projected_satellite_dot=float(os+alpha*oo),projection_delta_norm=alpha*float(oo.sqrt()))


def initialize(model,mode):
 if mode not in MODES:raise ValueError('Only fixed native-measure/view-second mode')
 model._vm_mode=mode;model._vm_pending=None;model._vm_ready=None;model._vm_steps=0;model._vm_clips=0;model._vm_vjp_calls=0;model._vm_records=[]
 return model

def validate_optimizer(opt):
 if not isinstance(opt,torch.optim.AdamW):raise ValueError('Existing joint AdamW required')
 for group in opt.param_groups:
  b1,b2=group['betas']
  if not (0<=b1<1 and 0<b2<1) or any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable','fused')):raise ValueError('Unsupported AdamW flags/betas')
  if any(p.is_complex() or p.dtype not in (torch.float32,torch.float64) for p in group['params']):raise ValueError('Real original parameter coordinates required')
 return opt

def capture(model,ordinary,secondary,step):
 if model._vm_pending is not None or model._vm_ready is not None or step!=model._vm_steps+1 or model._vm_clips!=model._vm_steps:raise ValueError('One ordered actual L pair per update')
 start=time.perf_counter();named=tuple((n,p) for n,p in model.named_parameters() if p.requires_grad);params=tuple(p for _,p in named);before=tuple(p.grad for p in params)
 a=torch.autograd.grad(ordinary,params,retain_graph=True,allow_unused=True);b=torch.autograd.grad(secondary,params,retain_graph=True,allow_unused=True)
 if any(p.grad is not g for p,g in zip(params,before)):raise ValueError('VJP changed native gradient accumulation')
 stats=geometry(named,a,b);delta=[]
 for (_,p),g,h in zip(named,a,b):
  if g is None and h is None:delta.append(None);continue
  g=torch.zeros_like(p) if g is None else g.detach();h=torch.zeros_like(p) if h is None else h.detach();v=(g-h).square()
  if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite second-moment contribution')
  delta.append(v)
 model._vm_pending=(named,tuple(delta),stats,step,time.perf_counter()-start);model._vm_vjp_calls+=2

def after_clip(model,parameters,max_norm,total_norm):
 params=list(parameters);pending=model._vm_pending
 if pending is None or model._vm_ready is not None or model._vm_clips!=model._vm_steps:raise ValueError('One native clip after capture required')
 named,delta,stats,step,elapsed=pending
 if step!=model._vm_steps+1 or {id(p) for _,p in named}!={id(p) for p in params}:raise ValueError('Stale or different jointly trained parameters')
 norm=float(total_norm);limit=float(max_norm)
 if not math.isfinite(norm) or norm<0 or not math.isfinite(limit) or limit<=0:raise ValueError('Finite original global clip required')
 factor=min(1.,limit/(norm+1e-6))
 model._vm_last_norm=norm;model._vm_ready=(named,delta,stats,step,elapsed,factor);model._vm_pending=None;model._vm_clips=step

def attach_optimizer(opt,model):
 validate_optimizer(opt)
 if getattr(opt,'_vm_attached',False):raise ValueError('One optimizer association only')
 opt._vm_attached=True
 def pre(optimizer,args,kwargs):
  validate_optimizer(optimizer)
  if len(args)!=1 or args[0] is not optimizer or kwargs:raise ValueError('Native optimizer step must not use closure')
  if model._vm_ready is None:raise ValueError('Optimizer step without captured/clipped real L views')
  named,delta,stats,step,elapsed,factor=model._vm_ready
  if step!=model._vm_steps+1 or model._vm_clips!=step:raise ValueError('Optimizer lifecycle mismatch')
  actual={id(p) for g in optimizer.param_groups for p in g['params']}
  if actual!={id(p) for _,p in named}:raise ValueError('Optimizer omitted or added parameter')
  names={id(p):n for n,p in named};dv={id(p):v for (_,p),v in zip(named,delta)};witness=[]
  selected=set(QNAMES)|{'id_backbone.encoder.core.id_backbone.cls_head.weight'}
  if not any(n in selected for n,_ in named):selected={named[0][0]}
  with torch.no_grad():
   for group in optimizer.param_groups:
    beta1,beta2=group['betas']
    for p in group['params']:
     var=dv[id(p)];state=optimizer.state[p]
     if p.grad is None:
      if var is not None and bool((var!=0).any()):raise ValueError('Native backward omitted a nonzero L contribution')
      continue
     n=names[id(p)]
     if n in selected:
      witness.append(dict(parameter=n,gradient=float(p.grad.reshape(-1)[0]),view_variance_postclip=0. if var is None else float(var.reshape(-1)[0])*factor**2,old_first=0. if not state else float(state['exp_avg'].reshape(-1)[0]),old_second=0. if not state else float(state['exp_avg_sq'].reshape(-1)[0]),beta1=beta1,beta2=beta2))
     if model._vm_mode=='VIEW_SECOND' and var is not None:
      if not state:
       state['step']=torch.tensor(0.,dtype=torch.float32,device='cpu');state['exp_avg']=torch.zeros_like(p,memory_format=torch.preserve_format);state['exp_avg_sq']=torch.zeros_like(p,memory_format=torch.preserve_format)
      # Native AdamW multiplies the state by beta2 next. Preaddition yields
      # beta2*v_old+(1-beta2)*(g_postclip**2+variance_postclip).
      state['exp_avg_sq'].add_(var,alpha=(1-beta2)/beta2*factor**2)
  model._vm_step_record=dict(step=step,mode=model._vm_mode,extra_VJP_calls=2,clip_factor=factor,physical_global_norm=float(getattr(model,'_vm_last_norm',0.)),capture_host_enqueue_seconds=elapsed,variance_physical_trace_postclip=max(0.,stats['ordinary_norm']**2+stats['satellite_norm']**2-2*stats['dot'])*factor**2,weighted_view_geometry=stats,witness=witness)
 def post(optimizer,args,kwargs):
  record=model._vm_step_record;named=model._vm_ready[0];lookup={n:p for n,p in named}
  for value in record['witness']:
   state=optimizer.state[lookup[value['parameter']]];value.update(actual_first=float(state['exp_avg'].reshape(-1)[0]),actual_second=float(state['exp_avg_sq'].reshape(-1)[0]),optimizer_state_step=float(state['step']))
  model._vm_steps+=1;model._vm_records.append(record);model._vm_ready=None;model._vm_step_record=None
 opt.register_step_pre_hook(pre);opt.register_step_post_hook(post);return opt
