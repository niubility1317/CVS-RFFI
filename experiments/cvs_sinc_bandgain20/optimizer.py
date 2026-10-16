"""Scale only Sinc adaptive displacement; physical origin decay stays intact."""
import torch
MODES=('FULL','NORM','BAND')
PREFIX='id_backbone.encoder.core.id_backbone.sinc'
def gains(sinc):
 if type(sinc).__name__!='CoordinateSinc' or sinc.scale_hz!=25000. or sinc.sample_rate!=25000000. or sinc.kernel_size!=79:raise ValueError('Fixed centered ManySig identity Sinc required')
 if sinc.low_coordinate.shape!=(24,1) or sinc.band_coordinate.shape!=(24,1) or torch.count_nonzero(sinc.low_coordinate) or torch.count_nonzero(sinc.band_coordinate):raise ValueError('Own fresh centered initialization required')
 low=(sinc.min_low_hz+sinc.low_origin_hz.abs()).clamp(sinc.min_low_hz,sinc.sample_rate/2-sinc.min_band_hz-1)
 high=torch.minimum(low+sinc.min_band_hz+sinc.band_origin_hz.abs(),torch.full_like(low,sinc.sample_rate/2-1));bw=high-low
 # Fs/K is the existing K-point frequency-grid spacing, a fixed scale heuristic,
 # not a theorem claiming that narrower Sinc filters cannot carry information.
 w=torch.maximum(bw,torch.full_like(bw,sinc.sample_rate/sinc.kernel_size))/sinc.sample_rate
 if not torch.isfinite(w).all() or not bool(((w>0)&(w<1)).all()) or torch.unique(w).numel()<2:raise ValueError('Nontrivial fixed positive channel scales required')
 return w.detach().clone(),bw.detach().clone()
def adjusted(adaptive,weights,mode):
 if mode not in MODES or len(adaptive)!=2 or len(weights)!=2 or any(a.shape!=(24,1) or w.shape!=a.shape or not torch.isfinite(a).all() or not torch.isfinite(w).all() or not bool(((w>0)&(w<1)).all()) for a,w in zip(adaptive,weights)):raise ValueError('Invalid fixed Sinc adaptive displacement')
 source=torch.cat([v.reshape(-1).double() for v in adaptive]);weighted=torch.cat([(a*w).reshape(-1).double() for a,w in zip(adaptive,weights)]);raw_norm=source.norm();weighted_norm=weighted.norm()
 scalar=weighted_norm/raw_norm if float(raw_norm)>0 else raw_norm.new_tensor(1.)
 if mode=='FULL':result=[v for v in adaptive]
 elif mode=='BAND':result=[a*w for a,w in zip(adaptive,weights)]
 else:result=[a*scalar.to(a.dtype) for a in adaptive]
 return result,dict(raw_norm=raw_norm,weighted_norm=weighted_norm,scalar_gain=scalar)
def attach(opt,model,mode):
 if mode not in MODES or getattr(opt,'_band_gain_attached',False):raise ValueError('One registered Sinc gain policy per optimizer')
 policy=opt._sinc_unit_policy
 if policy['mode']!='CENTER_UNITS' or policy['base_weight_decay']!=.0001 or policy['scale']!=25000.:raise ValueError('Original unified physical clipping/decay policy required')
 w,bw=gains(model.get_submodule(PREFIX));opt._band_gain_attached=True;opt._band_gain_steps=0;opt._band_gain_records=[];opt._band_gain_raw=[];opt._band_gain_pending=None;opt._band_gain_weights=[w,w];opt._band_gain_initial_bandwidth=bw
 def pre(opt,args,kwargs):
  # The existing unit-policy pre hook already performed exactly one physical
  # decay including origin. Capture that state, never scale or repeat decay.
  if opt._band_gain_pending is not None or opt._sinc_unit_steps!=opt._band_gain_steps:raise ValueError('Sinc gain step ownership mismatch')
  opt._band_gain_pending=[q.detach().clone() for q in policy['sinc']]
 def post(opt,args,kwargs):
  if opt._band_gain_pending is None or opt._sinc_unit_steps!=opt._band_gain_steps+1:raise ValueError('Native AdamW pre/post order changed')
  before=opt._band_gain_pending;adaptive=[q.detach()-b for q,b in zip(policy['sinc'],before)];update,metrics=adjusted(adaptive,opt._band_gain_weights,mode)
  step=opt._sinc_unit_steps;epoch=(step-1)//222+1;batch=(step-1)%222+1;witness=None
  if epoch in (1,13,14,20) and batch%16==0:
   witness=dict(step=step,epoch=epoch,mode=mode,alpha=float(opt.param_groups[0]['lr']),decay=policy['base_weight_decay'],scale=policy['scale'],parameters=[dict(name=n,before_decay=opt._sinc_pre_step[n].detach().cpu().clone(),after_decay=b.detach().cpu().clone(),after_native=q.detach().cpu().clone(),origin=o.detach().cpu().clone(),gain=w.detach().cpu().clone()) for n,q,b,o,w in zip(policy['names'],policy['sinc'],before,policy['origins'],opt._band_gain_weights)])
  with torch.no_grad():
   if mode!='FULL':
    for q,b,u in zip(policy['sinc'],before,update):q.copy_(b+u)
  actual=torch.cat([(q.detach()-b).reshape(-1).double() for q,b in zip(policy['sinc'],before)]).norm()*policy['scale']
  if witness is not None:
   for item,q in zip(witness['parameters'],policy['sinc']):item['after_gain']=q.detach().cpu().clone()
   opt._band_gain_raw.append(witness)
  opt._band_gain_steps=step;opt._band_gain_pending=None
  opt._band_gain_records.append(dict(step=step,mode=mode,base_alpha=float(opt.param_groups[0]['lr']),raw_adaptive_L2_Hz=float(metrics['raw_norm']*policy['scale']),counterfactual_band_L2_Hz=float(metrics['weighted_norm']*policy['scale']),scalar_gain=float(metrics['scalar_gain']),actual_adaptive_L2_Hz=float(actual),physical_decay_unscaled=True,head_fit_updates=0))
 opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
 return dict(mode=mode,formula='w_i=max(initial_effective_bandwidth_i,Fs/K)/Fs; K79, Fs25e6',weights=w.detach().cpu().flatten().tolist(),initial_bandwidth_Hz=bw.detach().cpu().flatten().tolist(),extra_parameters=0,physical_decay='Original alpha lambda p once, including origin; unscaled',clipping='Original global clip5 in physical Hz',norm_control='NORM matches BAND adaptive-displacement L2 at its own incoming Adam displacement; physical decay and cross-trajectory displacement are not matched')
