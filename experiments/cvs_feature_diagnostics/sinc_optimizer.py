"""Read saved AdamW state for actual Sinc parameters, without an update."""
from types import SimpleNamespace
import torch
from .sinc_change import PREFIX

def inspect_state(model,checkpoint,native):
 args=checkpoint['args']
 if args.get('freeze_backbone') or args.get('use_muse_ssdg') or native._fasttrust_lr_enabled(SimpleNamespace(**args)):
  raise ValueError('This mapping is restricted to the verified ordinary AdamW recipe')
 parameters=native._optimizer_parameters(model,None)
 names={id(p):name for name,p in model.named_parameters()}
 if any(id(p) not in names for p in parameters):raise ValueError('Unknown optimizer parameter')
 groups=checkpoint['optimizer']['param_groups'];ids=[index for group in groups for index in group['params']]
 if len(groups)!=1 or len(ids)!=len(parameters) or len(set(ids))!=len(ids):raise ValueError('Saved optimizer parameter order/count differs')
 states=checkpoint['optimizer']['state']
 for index,p in zip(ids,parameters):
  if index in states:
   for key in ('exp_avg','exp_avg_sq'):
    if states[index][key].shape!=p.shape or not torch.isfinite(states[index][key]).all():raise ValueError('Saved moment shape/value differs')
 mapping={names[id(p)]:(index,p) for index,p in zip(ids,parameters)}
 group=groups[0];records={}
 for suffix in ('low_hz_','band_hz_'):
  name=PREFIX+suffix
  if name not in mapping:raise ValueError('Actual identity Sinc is absent from optimizer')
  index,p=mapping[name]
  if index not in states:raise ValueError('Actual identity Sinc has no saved optimizer state')
  state=states[index];step=float(state['step'])
  if step!=checkpoint['a1_ema_successful_updates'] or step!=11100:raise ValueError('Actual Sinc optimizer steps differ from full budget')
  m=state['exp_avg'].double();v=state['exp_avg_sq'].double()
  if (v<0).any():raise ValueError('Negative second moment')
  beta1,beta2=group['betas'];denominator=(v/(1-beta2**step)).sqrt()+group['eps']
  adaptive=-(group['lr']/(1-beta1**step))*m/denominator
  value=p.detach().cpu().double()
  ulp=(torch.nextafter(p.detach().cpu(),torch.full_like(p.detach().cpu(),float('inf')))-p.detach().cpu()).double()
  records[suffix]=dict(saved_optimizer_index=index,actual_step=int(step),elements=p.numel(),nonzero_first_moments=int((m!=0).sum()),nonzero_second_moments=int((v!=0).sum()),first_moment_abs_max=float(m.abs().max()),second_moment_sqrt_max=float(v.sqrt().max()),first_moment_abs_mean=float(m.abs().mean()),
   final_saved_lr=group['lr'],eps=group['eps'],betas=list(group['betas']),weight_decay=group['weight_decay'],final_moment_adaptive_term_Hz_max=float(adaptive.abs().max()),per_channel_final_moment_adaptive_term_Hz=adaptive.flatten().tolist(),
   initial_for_this_comparison='actual final parameter value, not original initialization',
   final_float32_ULP_Hz_min=float(ulp.min()),final_float32_ULP_Hz_max=float(ulp.max()),adaptive_term_below_half_ULP_channels=int((adaptive.abs()<ulp.abs()/2).sum()),
   limitation='Reconstructed algebraic adaptive term using final saved moments and LR; not a recorded history of attempted updates and not an executed optimizer step')
 return dict(status='VERIFIED',actual_optimizer_parameter_count=len(parameters),saved_parameter_count=len(ids),all_saved_moment_shapes_match=True,Sinc=records,optimizer_updates_executed_by_diagnostic=0,IQ_read=False,query_read=False,head_updates=0)

def stationary_ema(initial,SincType,update_fn):
 """Numeric reconstruction with fixed actual initial filters, not training."""
 import copy
 selected={k:initial[PREFIX+k] for k in ('low_hz_','band_hz_','t_','window_')}
 if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in selected.values()):raise ValueError('Replay requires finite float32 source Sinc state')
 student=SincType(24,79,sample_rate=25e6,dataset='wisig',input_len=256).eval().requires_grad_(False)
 student.load_state_dict(selected,strict=True);teacher=copy.deepcopy(student)
 for step in range(11100):update_fn(teacher,student,.999,versioned=True)
 if any(not torch.equal(v,student.state_dict()[k]) for k,v in selected.items()):raise ValueError('Stationary replay changed fixed student')
 return {PREFIX+k:v for k,v in teacher.state_dict().items()}


def checks():
 from torch import nn
 class Sinc(nn.Module):
  def __init__(self):
   super().__init__();self.low_hz_=nn.Parameter(torch.full((24,1),1e6));self.band_hz_=nn.Parameter(torch.full((24,1),5e5))
 model=nn.Module();module=model
 for part in PREFIX.rstrip('.').split('.'):
  q=nn.Module() if part!='sinc' else Sinc();module.add_module(part,q);module=q
 names=list(model.named_parameters());native=SimpleNamespace(_optimizer_parameters=lambda model,state:[p for p in model.parameters() if p.requires_grad],_fasttrust_lr_enabled=lambda args:False)
 states={i:dict(step=torch.tensor(11100.),exp_avg=torch.full_like(p,1e-11),exp_avg_sq=torch.full_like(p,1e-22)) for i,(n,p) in enumerate(names)}
 checkpoint=dict(args=dict(freeze_backbone=False,use_muse_ssdg=False),optimizer=dict(state=states,param_groups=[dict(params=list(states),betas=(.9,.999),eps=1e-8,lr=1e-6,weight_decay=.0001)]),a1_ema_successful_updates=11100)
 result=inspect_state(model,checkpoint,native);assert result['Sinc']['low_hz_']['adaptive_term_below_half_ULP_channels']==24
 import copy
 negative=0
 for change in ('state_missing','wrong_count','wrong_shape','wrong_step','frozen'):
  bad=copy.deepcopy(checkpoint)
  if change=='state_missing':del bad['optimizer']['state'][0]
  elif change=='wrong_count':bad['optimizer']['param_groups'][0]['params'].pop()
  elif change=='wrong_shape':bad['optimizer']['state'][0]['exp_avg']=torch.zeros(1)
  elif change=='wrong_step':bad['optimizer']['state'][0]['step']=torch.tensor(1)
  else:bad['args']['freeze_backbone']=True
  try:inspect_state(model,bad,native)
  except ValueError:negative+=1
  else:raise AssertionError('Incorrect saved mapping accepted')
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules();from model import SincConv1d
 sinc=SincConv1d(24,79,sample_rate=25e6,dataset='wisig',input_len=256)
 initial={PREFIX+k:v for k,v in sinc.state_dict().items()}
 def update(teacher,student,decay,versioned):
  with torch.no_grad():
   for a,b in zip(teacher.parameters(),student.parameters()):a.mul_(decay).add_(b,alpha=1-decay)
   for a,b in zip(teacher.buffers(),student.buffers()):a.copy_(b)
 replay=stationary_ema(initial,SincConv1d,update)
 assert any(not torch.equal(v,replay[k]) for k,v in initial.items() if k.endswith(('low_hz_','band_hz_')))
 return dict(status='PASS',synthetic_only=True,negative_checks=negative,stationary_float32_EMA_drift_reproduced=True,optimizer_steps=0,real_checkpoint_evidence=False)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))
