import torch,sys,json,pathlib
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-phase-shift-student-s2026092701-r02-release/experiments/phase1_phase_shift_restorer'))
from phase1_axial_spectral_mask import AxialSpectralMask
torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(123);m=AxialSpectralMask().train();events=[];inputs=[]
def info(t):
 if not isinstance(t,torch.Tensor):return None
 return {'shape':list(t.shape),'stride':list(t.stride()),'finite':bool(torch.isfinite(t).all()),'nonfinite':int((~torch.isfinite(t)).sum()),'max':float(t.detach().double().abs().nan_to_num(nan=0,posinf=1e300,neginf=1e300).max())}
def cb(name):
 def hook(module,gin,gout):events.append({'name':name,'gin':[info(x) for x in gin],'gout':[info(x) for x in gout]})
 return hook
mods=dict(m.named_modules());names=['input','local.0','local.1','local.2','time_norm','time_attention','freq_norm','freq_attention','output'];handles=[mods[n].register_full_backward_hook(cb(n)) for n in names]
for i in range(8):
 events.clear();m.zero_grad(set_to_none=True);x=torch.randn(16,2,256);loss=m(x).square().mean()*0;loss.backward();rows=[(float(p.grad.detach().double().abs().nan_to_num(nan=0,posinf=1e300,neginf=1e300).max()),n,int((~torch.isfinite(p.grad)).sum())) for n,p in m.named_parameters() if p.grad is not None]
 if max(x[0] for x in rows)!=0 or any(x[2] for x in rows):
  print(json.dumps({'iteration':i,'top':sorted(rows,reverse=True)[:5],'events':events},allow_nan=False));break
else:print(json.dumps({'status':'no_violation_in_eight'}))
