import torch,sys,json,pathlib
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer'))
from phase1_axial_spectral_mask import AxialSpectralMask
torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(123);m=AxialSpectralMask().train();events=[]
def stat(items):
 a=[x for x in items if isinstance(x,torch.Tensor)];return max([float(x.detach().double().abs().nan_to_num(posinf=1e300,neginf=1e300).max()) for x in a] or [0.])
def callback(name):
 def cb(module,gin,gout):events.append({'name':name,'out_grad_max':stat(gout),'input_grad_max':stat(gin)})
 return cb
modules=dict(m.named_modules());names=['input','local.0','local.1','local.2','time_norm','time_attention','freq_norm','freq_attention','output','temporal_output','temporal_input'];handles=[modules[n].register_full_backward_hook(callback(n)) for n in names]
for i in range(8):
 events.clear();m.zero_grad(set_to_none=True);x=torch.randn(16,2,256);out=m(x);loss=out.square().mean()*0;loss.backward();rows=[(float(p.grad.detach().double().abs().nan_to_num(posinf=1e300,neginf=1e300).max()),n,int((~torch.isfinite(p.grad)).sum())) for n,p in m.named_parameters() if p.grad is not None];mx=max(x[0] for x in rows)
 if mx!=0 or any(x[2] for x in rows):
  print(json.dumps({'iteration':i,'zero_grad_max':mx,'top_parameters':sorted(rows,reverse=True)[:8],'backward_events':events},allow_nan=False));break
else:print(json.dumps({'iterations':8,'status':'no_violation_with_hooks'}))
